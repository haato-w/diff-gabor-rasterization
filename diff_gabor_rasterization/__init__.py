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

import numpy as np
from matplotlib import pyplot as plt

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gabors(
    means3D,
    means2D,
    sh,
    sh2,
    colors_precomp,
    colors_precomp2,
    opacities,
    frequencies, 
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGabors.apply(
        means3D,
        means2D,
        sh,
        sh2,
        colors_precomp,
        colors_precomp2,
        opacities,
        frequencies, 
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGabors(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        sh2,
        colors_precomp,
        colors_precomp2,
        opacities,
        frequencies, 
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):
        
        # print("means3D: ", means3D)
        # print("frequencies: ", frequencies)
        # print("colors_precomp: ", colors_precomp)
        # print("colors_precomp2: ", colors_precomp2)
        # print("opacities: ", opacities)
        # print("scales: ", scales)
        # print("rotations: ", rotations)

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            frequencies, 
            colors_precomp,
            colors_precomp2,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            sh2,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, depth, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gabors(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, depth, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gabors(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, colors_precomp2, means3D, frequencies, scales, rotations, cov3Ds_precomp, radii, sh, sh2, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, depth

    @staticmethod
    def backward(ctx, grad_out_color, grad_radii, grad_depth):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, colors_precomp2, means3D, frequencies, scales, rotations, cov3Ds_precomp, radii, sh, sh2, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # print("frequencies: \n", frequencies)

        # for debug
        # print("in backward")
        # print("frequencies: \n", frequencies)
        # print("colors_precomp: \n", colors_precomp)
        # print("colors_precomp2: \n", colors_precomp2)
        # print("sh: \n", sh)
        # print("sh2: \n", sh2)

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                frequencies, 
                radii, 
                colors_precomp, 
                colors_precomp2, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color,
                grad_depth,
                sh, 
                sh2, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_colors_precomp2, grad_opacities, grad_frequencies, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_sh2, grad_scales, grad_rotations, d_image = _C.rasterize_gabors_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_colors_precomp2, grad_opacities, grad_frequencies, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_sh2, grad_scales, grad_rotations, d_image = _C.rasterize_gabors_backward(*args)

        # print("d_image.shape: ", d_image.shape)
        # save_image = d_image.clone().detach()
        # save_image = save_image.permute(1, 2, 0)
        # save_image = save_image.squeeze(-1)
        # # save_image = save_image * 1000000
        # print("save_image.shape: ", save_image.shape)

        # # save_image = save_image.clamp(max=1.0, min=0.0)
        # print("save_image: \n", save_image[200:300, 200:300])
        # img = save_image.cpu().numpy()

        # max_ = 50
        # img = (img + max_) / (2 * max_)
        # img = np.clip(img, a_min = 0, a_max = 1)

        # import matplotlib 
        # img = matplotlib.colormaps['seismic'](img)[..., :3]

        # plt.imsave("d_image.png", img)

        # grad_frequencies = torch.zeros((grad_means2D.shape[0], 2)).float().cuda() # tmp
        # grad_colors_precomp2 = torch.zeros(grad_colors_precomp.shape).float().cuda() # tmp
        # grad_sh2 = torch.zeros(grad_sh.shape).float().cuda() # tmp

        # for debug
        # print("out backward")
        # print("grad_frequencies: \n", grad_frequencies)
        # print("grad_colors_precomp2: \n", grad_colors_precomp2)
        # print("grad_sh2: \n", grad_sh2)

        # grads = (
        #     torch.zeros_like(grad_means3D),
        #     torch.zeros_like(grad_means2D),
        #     torch.zeros_like(grad_sh),
        #     torch.zeros_like(grad_sh2),
        #     torch.zeros_like(grad_colors_precomp),
        #     torch.zeros_like(grad_colors_precomp2),
        #     torch.zeros_like(grad_opacities),
        #     grad_frequencies, 
        #     torch.zeros_like(grad_scales),
        #     torch.zeros_like(grad_rotations),
        #     torch.zeros_like(grad_cov3Ds_precomp),
        #     None,
        # )

        # grads = (
        #     torch.zeros_like(grad_means3D),
        #     torch.zeros_like(grad_means2D),
        #     torch.zeros_like(grad_sh),
        #     torch.zeros_like(grad_sh2),
        #     torch.zeros_like(grad_colors_precomp),
        #     torch.zeros_like(grad_colors_precomp2),
        #     torch.zeros_like(grad_opacities),
        #     grad_frequencies, 
        #     torch.zeros_like(grad_scales),
        #     torch.zeros_like(grad_rotations),
        #     torch.zeros_like(grad_cov3Ds_precomp),
        #     None,
        # )

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_sh2,
            grad_colors_precomp,
            grad_colors_precomp2,
            grad_opacities,
            grad_frequencies, 
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

class GaborRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

class GaborRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, frequencies, shs = None, shs2 = None, colors_precomp = None, colors_precomp2 = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if (shs2 is None and colors_precomp2 is None) or (shs2 is not None and colors_precomp2 is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([]).cuda()
        if shs2 is None:
            shs2 = torch.Tensor([]).cuda()
        if colors_precomp is None:
            colors_precomp = torch.Tensor([]).cuda()
        if colors_precomp2 is None:
            colors_precomp2 = torch.Tensor([]).cuda()

        if scales is None:
            scales = torch.Tensor([]).cuda()
        if rotations is None:
            rotations = torch.Tensor([]).cuda()
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([]).cuda()
        

        # Invoke C++/CUDA rasterization routine
        return rasterize_gabors(
            means3D,
            means2D,
            shs,
            shs2,
            colors_precomp,
            colors_precomp2,
            opacities,
            frequencies, 
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )

