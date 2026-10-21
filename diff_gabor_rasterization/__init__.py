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
    weights_for_frequency_angle, frequency_lengths, phases, F, scales,
    rotations, cov3Ds_precomp, raster_settings, sh_rest=None, sh2_rest=None,
    raw_inputs=False, split_sh=False,
):
    if sh_rest is None:
        sh_rest = means3D.new_empty((0,))
    if sh2_rest is None:
        sh2_rest = means3D.new_empty((0,))
    return _RasterizeGabors.apply(
        means3D, means2D, sh, sh2, colors_precomp, colors_precomp2, opacities,
        weights_for_frequency_angle, frequency_lengths, phases, F, scales,
        rotations, cov3Ds_precomp, raster_settings, sh_rest, sh2_rest,
        bool(raw_inputs), bool(split_sh),
    )


class _RasterizeGabors(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, means3D, means2D, sh, sh2, colors_precomp, colors_precomp2,
        opacities, weights_for_frequency_angle, frequency_lengths, phases, F,
        scales, rotations, cov3Ds_precomp, raster_settings, sh_rest, sh2_rest,
        raw_inputs, split_sh,
    ):
        args = (
            raster_settings.bg, means3D, weights_for_frequency_angle,
            frequency_lengths, phases, F, colors_precomp, colors_precomp2,
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
        gaussian_backward = raster_settings.gaussian_backward
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
        ctx.F = F
        ctx.raw_inputs = raw_inputs
        ctx.split_sh = split_sh
        ctx.fused = fused
        ctx.save_for_backward(
            colors_precomp, colors_precomp2, means3D, weights_for_frequency_angle,
            frequency_lengths, phases, scales, rotations, cov3Ds_precomp, radii,
            sh, sh2, geomBuffer, binningBuffer, imgBuffer, sh_rest, sh2_rest,
            *((color, depth) if gaussian_backward else ()),
        )
        return color, radii, depth

    @staticmethod
    def backward(ctx, grad_out_color, grad_radii, grad_depth):
        settings = ctx.raster_settings
        (colors, colors2, means3D, weights, lengths, phases, scales, rotations,
         covariance, radii, sh, sh2, geometry, binning, image,
         sh_rest, sh2_rest) = ctx.saved_tensors[:17]
        # Autograd may provide expanded or strided image gradients.
        args = (
            settings.bg, means3D, weights, lengths, phases, ctx.F, radii,
            colors, colors2, scales, rotations, settings.scale_modifier,
            covariance, settings.viewmatrix, settings.projmatrix,
            settings.tanfovx, settings.tanfovy, grad_out_color.contiguous(),
            grad_depth.contiguous(), sh, sh2, settings.sh_degree, settings.campos,
            geometry, ctx.num_rendered, binning, image, settings.debug,
        )
        native_backward = _C.rasterize_gabors_backward
        if ctx.fused or settings.gaussian_backward or settings.separate_sort:
            args += (sh_rest, sh2_rest, ctx.split_sh, ctx.raw_inputs)
            native_backward = _C.rasterize_gabors_backward_fused
        if settings.gaussian_backward or settings.separate_sort:
            checkpoints = (ctx.saved_tensors[17:] if settings.gaussian_backward
                           else (means3D.new_empty((0,)), means3D.new_empty((0,))))
            args += (settings.gaussian_backward, *checkpoints)
            native_backward = _C.rasterize_gabors_backward_gaussian
        if settings.separate_sort:
            args += (settings.separate_sort,)
            native_backward = _C.rasterize_gabors_backward_separate_sorted
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

        (grad_means2D, grad_colors, grad_colors2, grad_opacities, grad_weights,
         grad_lengths, grad_phases, grad_means3D, grad_covariance,
         grad_sh, grad_sh2, grad_scales, grad_rotations, _) = result[:14]
        grad_sh_rest, grad_sh2_rest = result[14:] if ctx.split_sh else (None, None)
        return (
            grad_means3D, grad_means2D, grad_sh, grad_sh2, grad_colors, grad_colors2,
            grad_opacities, grad_weights, grad_lengths, grad_phases, None,
            grad_scales, grad_rotations, grad_covariance, None,
            grad_sh_rest, grad_sh2_rest, None, None,
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


class GaborRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        with torch.no_grad():
            settings = self.raster_settings
            return _C.mark_visible(positions, settings.viewmatrix, settings.projmatrix)

    def forward(
        self, means3D, means2D, opacities, weights_for_frequency_angle,
        frequency_lengths, phases, F, shs=None, shs2=None, colors_precomp=None,
        colors_precomp2=None, scales=None, rotations=None, cov3D_precomp=None,
        shs_rest=None, shs2_rest=None, raw_inputs=False,
    ):
        """Rasterize packed SH or separate DC/rest tensors.

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
        """
        threshold = self.raster_settings.tile_culling_threshold
        if type(threshold) is not int or not 0 <= threshold <= 32:
            raise ValueError("tile_culling_threshold must be an integer between 0 and 32")
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
            weights_for_frequency_angle, frequency_lengths, phases, F,
            or_empty(scales), or_empty(rotations), or_empty(cov3D_precomp),
            self.raster_settings, or_empty(shs_rest), or_empty(shs2_rest),
            raw_inputs, split_sh,
        )
