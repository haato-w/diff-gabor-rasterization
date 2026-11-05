#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C
from .projector import project_errors


def set_profiling_enabled(enabled: bool):
    """Opt into CUDA stage events; rebuild the extension to add this API."""
    _C.set_profiling_enabled(bool(enabled))


def collect_profiling() -> dict:
    """Drain CUDA stage timings in milliseconds, summing repeated stage names."""
    return _C.collect_profiling()


def cpu_deep_copy_tuple(input_tuple):
    return tuple(item.cpu().clone() if isinstance(item, torch.Tensor) else item
                 for item in input_tuple)


def rasterize_gabors(
    means3D, means2D, sh, sh2, colors_precomp, colors_precomp2, opacities,
    frequency_vectors, phases, scales,
    rotations, cov3Ds_precomp, raster_settings, sh_rest=None, sh2_rest=None,
    raw_inputs=False, split_sh=False, specular_precomp=None,
    attributes_precomp=None,
):
    if sh_rest is None:
        sh_rest = means3D.new_empty((0,))
    if sh2_rest is None:
        sh2_rest = means3D.new_empty((0,))
    if specular_precomp is None:
        specular_precomp = means3D.new_empty((0,))
    if attributes_precomp is None:
        attributes_precomp = means3D.new_empty((0,))
    has_attributes = attributes_precomp.numel() != 0 or (
        means3D.shape[0] == 0 and attributes_precomp.shape == (0, 6)
    )
    if has_attributes:
        if attributes_precomp.shape != (means3D.shape[0], 6):
            raise ValueError("attributes_precomp must have shape [N, 6]")
        if attributes_precomp.dtype != torch.float32:
            raise RuntimeError("attributes_precomp must use float32")
        if not attributes_precomp.is_cuda or attributes_precomp.device != means3D.device:
            raise RuntimeError("attributes_precomp must be a CUDA tensor on the same device as means3D")
        if not attributes_precomp.is_contiguous():
            raise RuntimeError("attributes_precomp must be contiguous")
    return _RasterizeGabors.apply(
        means3D, means2D, sh, sh2, colors_precomp, colors_precomp2, opacities,
        frequency_vectors, phases, scales,
        rotations, cov3Ds_precomp, raster_settings, sh_rest, sh2_rest,
        bool(raw_inputs), bool(split_sh), specular_precomp, attributes_precomp,
    )


class _RasterizeGabors(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, means3D, means2D, sh, sh2, colors_precomp, colors_precomp2,
        opacities, frequency_vectors, phases,
        scales, rotations, cov3Ds_precomp, raster_settings, sh_rest, sh2_rest,
        raw_inputs, split_sh, specular_precomp, attributes_precomp,
    ):
        args = (
            raster_settings.bg, means3D, frequency_vectors,
            phases, colors_precomp, colors_precomp2,
            opacities, scales, rotations, raster_settings.scale_modifier,
            cov3Ds_precomp, raster_settings.viewmatrix, raster_settings.projmatrix,
            raster_settings.tanfovx, raster_settings.tanfovy,
            raster_settings.image_height, raster_settings.image_width, sh, sh2,
            raster_settings.sh_degree, raster_settings.campos,
            raster_settings.prefiltered, raster_settings.debug,
        )
        fused = split_sh or raw_inputs
        native_forward = _C.rasterize_gabors
        culled = raster_settings.tile_culling or raster_settings.load_balance
        has_attributes = attributes_precomp.numel() != 0 or (
            means3D.shape[0] == 0 and attributes_precomp.shape == (0, 6)
        )
        # Extra attributes use pixel/warp backward rather than allocating
        # six more channels of Gaussian-owned compositing checkpoints.
        gaussian_backward = raster_settings.gaussian_backward and not has_attributes
        warp_culling = raster_settings.warp_culling
        separate_sort = raster_settings.separate_sort
        if fused or raster_settings.tight_bbox or culled or gaussian_backward or warp_culling or separate_sort:
            args += (sh_rest, sh2_rest, split_sh, raw_inputs)
            native_forward = _C.rasterize_gabors_fused
        if raster_settings.tight_bbox or culled or gaussian_backward or warp_culling or separate_sort:
            args += (raster_settings.tight_bbox,)
            native_forward = _C.rasterize_gabors_bounded
        if culled or gaussian_backward or warp_culling or separate_sort:
            args += (raster_settings.tile_culling, raster_settings.load_balance,
                     raster_settings.tile_culling_threshold)
            native_forward = _C.rasterize_gabors_culled
        if gaussian_backward or warp_culling or separate_sort:
            args += (gaussian_backward,)
            native_forward = _C.rasterize_gabors_gaussian_backward
        if warp_culling or separate_sort:
            args += (warp_culling,)
            native_forward = _C.rasterize_gabors_warp_culled
        if separate_sort:
            args += (separate_sort,)
            native_forward = _C.rasterize_gabors_separate_sorted
        has_specular = specular_precomp.numel() != 0
        if has_specular or has_attributes:
            # Shaded/attribute exports preserve all optimization flags using
            # the same single-vector prefix as the other native entrypoints.
            args = args[:23] + (
                sh_rest, sh2_rest, split_sh, raw_inputs,
                raster_settings.tight_bbox, raster_settings.tile_culling,
                raster_settings.load_balance, raster_settings.tile_culling_threshold,
                gaussian_backward, warp_culling, separate_sort, specular_precomp,
            )
            native_forward = _C.rasterize_gabors_specular
            if has_attributes:
                args += (attributes_precomp,)
                native_forward = _C.rasterize_gabors_attributes

        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                result = native_forward(*args)
            except Exception:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occurred in forward. Writing snapshot_fw.dump for debugging.")
                raise
        else:
            result = native_forward(*args)
        num_rendered, color, depth, radii, geomBuffer, binningBuffer, imgBuffer = result

        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.raw_inputs = raw_inputs
        ctx.split_sh = split_sh
        ctx.fused = fused
        ctx.has_specular = has_specular
        ctx.has_attributes = has_attributes
        ctx.gaussian_backward = gaussian_backward
        ctx.backward_reduction = raster_settings.backward_reduction and not gaussian_backward
        ctx.backward_reduction_threshold = raster_settings.backward_reduction_threshold
        ctx.save_for_backward(
            colors_precomp, colors_precomp2, means3D, frequency_vectors,
            phases, scales, rotations, cov3Ds_precomp, radii,
            sh, sh2, geomBuffer, binningBuffer, imgBuffer, sh_rest, sh2_rest, specular_precomp,
            *((color, depth) if gaussian_backward else ()),
            attributes_precomp,
        )
        return color, radii, depth

    @staticmethod
    def backward(ctx, grad_out_color, grad_radii, grad_depth):
        settings = ctx.raster_settings
        (colors, colors2, means3D, frequency_vectors, phases, scales, rotations,
         covariance, radii, sh, sh2, geometry, binning, image,
         sh_rest, sh2_rest) = ctx.saved_tensors[:16]
        specular = ctx.saved_tensors[16]
        attributes = ctx.saved_tensors[-1]
        # Autograd may provide expanded or strided image gradients.
        args = (
            settings.bg, means3D, frequency_vectors, phases, radii,
            colors, colors2, scales, rotations, settings.scale_modifier,
            covariance, settings.viewmatrix, settings.projmatrix,
            settings.tanfovx, settings.tanfovy, grad_out_color.contiguous(),
            grad_depth.contiguous(), sh, sh2, settings.sh_degree, settings.campos,
            geometry, ctx.num_rendered, binning, image, settings.debug,
        )
        native_backward = _C.rasterize_gabors_backward
        if ctx.fused or ctx.gaussian_backward or settings.separate_sort or ctx.backward_reduction:
            args += (sh_rest, sh2_rest, ctx.split_sh, ctx.raw_inputs)
            native_backward = _C.rasterize_gabors_backward_fused
        if ctx.gaussian_backward or settings.separate_sort or ctx.backward_reduction:
            checkpoints = (ctx.saved_tensors[17:19] if ctx.gaussian_backward
                           else (means3D.new_empty((0,)), means3D.new_empty((0,))))
            args += (ctx.gaussian_backward, *checkpoints)
            native_backward = _C.rasterize_gabors_backward_gaussian
        if settings.separate_sort or ctx.backward_reduction:
            args += (settings.separate_sort,)
            native_backward = _C.rasterize_gabors_backward_separate_sorted
        if ctx.backward_reduction:
            args += (True, ctx.backward_reduction_threshold)
            native_backward = _C.rasterize_gabors_backward_butterfly
        if ctx.has_specular or ctx.has_attributes:
            checkpoints = (ctx.saved_tensors[17:19] if ctx.gaussian_backward
                           else (means3D.new_empty((0,)), means3D.new_empty((0,))))
            args = args[:26] + (
                sh_rest, sh2_rest, ctx.split_sh, ctx.raw_inputs,
                ctx.gaussian_backward, *checkpoints, settings.separate_sort,
                ctx.backward_reduction, ctx.backward_reduction_threshold, specular,
            )
            native_backward = _C.rasterize_gabors_backward_specular
            if ctx.has_attributes:
                args += (attributes,)
                native_backward = _C.rasterize_gabors_backward_attributes
        if settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                result = native_backward(*args)
            except Exception:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occurred in backward. Writing snapshot_bw.dump for debugging.")
                raise
        else:
            result = native_backward(*args)

        (grad_means2D, grad_colors, grad_colors2, grad_opacities, grad_frequency_vectors,
         grad_phases, grad_means3D, grad_covariance,
         grad_sh, grad_sh2, grad_scales, grad_rotations, _) = result[:13]
        grad_sh_rest, grad_sh2_rest = result[13:15] if ctx.split_sh else (None, None)
        grad_specular = result[15] if ctx.has_specular or ctx.has_attributes else torch.zeros_like(specular)
        grad_attributes = result[16] if ctx.has_attributes else torch.zeros_like(attributes)
        return (
            grad_means3D, grad_means2D, grad_sh, grad_sh2, grad_colors, grad_colors2,
            grad_opacities, grad_frequency_vectors, grad_phases,
            grad_scales, grad_rotations, grad_covariance, None,
            grad_sh_rest, grad_sh2_rest, None, None, grad_specular, grad_attributes,
        )


class GaborRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool
    tight_bbox: bool = False
    tile_culling: bool = False
    load_balance: bool = False
    tile_culling_threshold: int = 4
    gaussian_backward: bool = False
    warp_culling: bool = False
    separate_sort: bool = False
    backward_reduction: bool = False
    backward_reduction_threshold: int = 16


class GaborRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        with torch.no_grad():
            settings = self.raster_settings
            return _C.mark_visible(positions, settings.viewmatrix, settings.projmatrix)

    def forward(
        self, means3D, means2D, opacities, frequency_vectors,
        phases, shs=None, shs2=None, colors_precomp=None,
        colors_precomp2=None, scales=None, rotations=None, cov3D_precomp=None,
        shs_rest=None, shs2_rest=None, raw_inputs=False, specular_precomp=None,
        attributes_precomp=None,
    ):
        """Rasterize one Cartesian Gabor carrier per primitive.

        ``frequency_vectors`` is [N,2] in local surfel coordinates, and
        ``phases`` is [N,1]. Endpoints blend with (1 +/- cos(2*pi*k.s+phase))/2.
        There are no frequency-bin weights or a frequency-count argument.

        With ``shs_rest``/``shs2_rest``, each SH tensor is the DC coefficient
        (P, 1, 3), and each rest tensor contains the remaining coefficients.
        ``raw_inputs`` accepts log scales, unnormalized quaternions and opacity
        logits; CUDA preprocessing performs the activations and their gradients.
        ``raster_settings.tight_bbox`` enables conservative opacity-dependent
        tile bounds independently of either input-fusion option.
        ``tile_culling`` rejects tiles outside the projected surfel and filter
        support. ``load_balance`` cooperatively processes large tile rectangles;
        ``tile_culling_threshold`` controls the first 0 to 32 tiles per thread.
        ``gaussian_backward`` records compositing checkpoints in forward for
        primitive-owned backward accumulation; False retains per-pixel backward.
        ``warp_culling`` skips a primitive for a whole pixel warp when its
        conservative projected support misses that warp's pixel rectangle.
        ``separate_sort`` orders primitives by depth before duplication, then
        stably sorts duplicated entries using only their tile identifiers.
        ``backward_reduction`` aggregates per-pixel gradients within each warp
        before atomic additions; it is ignored for Gaussian-owned backward.
        ``backward_reduction_threshold`` selects aggregation when at least this
        many lanes contribute to the same primitive (0 to 32).
        ``specular_precomp`` optionally adds RGB [P, 3] independently of the
        Gabor carrier, before alpha compositing.
        ``attributes_precomp`` optionally composites six [P, 6] attributes
        independently of the carrier into channels 7:13 of the returned
        auxiliary map. RGB and its original seven auxiliary channels retain
        their existing behavior. Attributes use pixel/warp backward even when
        ``gaussian_backward`` is enabled, avoiding extra Gaussian checkpoints.
        """
        if frequency_vectors.shape != (means3D.shape[0], 2):
            raise ValueError("frequency_vectors must have shape [N, 2]")
        if phases.shape != (means3D.shape[0], 1):
            raise ValueError("phases must have shape [N, 1]")
        for name, value in (("frequency_vectors", frequency_vectors), ("phases", phases)):
            if value.dtype != means3D.dtype or value.device != means3D.device:
                raise ValueError(f"{name} must match means3D dtype and device")
            if not value.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        threshold = self.raster_settings.tile_culling_threshold
        if type(threshold) is not int or not 0 <= threshold <= 32:
            raise ValueError("tile_culling_threshold must be an integer between 0 and 32")
        reduction_threshold = self.raster_settings.backward_reduction_threshold
        if type(reduction_threshold) is not int or not 0 <= reduction_threshold <= 32:
            raise ValueError("backward_reduction_threshold must be an integer between 0 and 32")
        if (shs is None) == (colors_precomp is None):
            raise ValueError("Please provide exactly one of either SHs or precomputed colors!")
        if (shs2 is None) == (colors_precomp2 is None):
            raise ValueError("Please provide exactly one of either SHs2 or precomputed colors2!")
        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise ValueError("Please provide exactly one of scale/rotation or precomputed 3D covariance!")
        if raw_inputs and cov3D_precomp is not None:
            raise ValueError("raw_inputs requires scales and rotations, without precomputed covariance")
        if specular_precomp is not None and specular_precomp.numel():
            if specular_precomp.shape != (means3D.shape[0], 3):
                raise ValueError("specular_precomp must have shape [N, 3]")

        split_sh = shs_rest is not None or shs2_rest is not None
        if split_sh:
            if shs is not None and shs_rest is None:
                raise ValueError("Split SH input requires shs_rest for the first SH set")
            if shs2 is not None and shs2_rest is None:
                raise ValueError("Split SH input requires shs2_rest for the second SH set")
            if shs is None and shs_rest is not None:
                raise ValueError("shs_rest requires shs DC coefficients")
            if shs2 is None and shs2_rest is not None:
                raise ValueError("shs2_rest requires shs2 DC coefficients")

        def or_empty(tensor):
            return means3D.new_empty((0,)) if tensor is None else tensor

        return rasterize_gabors(
            means3D, means2D, or_empty(shs), or_empty(shs2),
            or_empty(colors_precomp), or_empty(colors_precomp2), opacities,
            frequency_vectors, phases,
            or_empty(scales), or_empty(rotations), or_empty(cov3D_precomp),
            self.raster_settings, or_empty(shs_rest), or_empty(shs2_rest),
            raw_inputs, split_sh, specular_precomp, attributes_precomp,
        )
