/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

using RasterizeResult = std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;
using RasterizeBackwardResult = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;
using RasterizeFusedBackwardResult = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;

RasterizeResult RasterizeGaborsCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& weights_for_frequency_angle,
	const torch::Tensor& frequency_lengths,
	const torch::Tensor& phases,
	const int F,
	const torch::Tensor& colors,
	const torch::Tensor& colors2,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const torch::Tensor& sh2,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug);

RasterizeResult RasterizeGaborsFusedCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& weights_for_frequency_angle,
	const torch::Tensor& frequency_lengths,
	const torch::Tensor& phases,
	const int F,
	const torch::Tensor& colors,
	const torch::Tensor& colors2,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const torch::Tensor& sh2,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug,
	const torch::Tensor& sh_rest,
	const torch::Tensor& sh2_rest,
	const bool split_sh,
	const bool raw_inputs);

// Keep the existing forward exports unchanged; bounds are an independent option.
RasterizeResult RasterizeGaborsBoundedCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& weights_for_frequency_angle,
	const torch::Tensor& frequency_lengths,
	const torch::Tensor& phases,
	const int F,
	const torch::Tensor& colors,
	const torch::Tensor& colors2,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const torch::Tensor& sh2,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug,
	const torch::Tensor& sh_rest,
	const torch::Tensor& sh2_rest,
	const bool split_sh,
	const bool raw_inputs,
	const bool tight_bbox);

// Tile culling and cooperative scheduling have a separate forward entry point,
// preserving the positional ABI of the legacy, fused and bounded exports.
RasterizeResult RasterizeGaborsCulledCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& weights_for_frequency_angle,
	const torch::Tensor& frequency_lengths,
	const torch::Tensor& phases,
	const int F,
	const torch::Tensor& colors,
	const torch::Tensor& colors2,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const torch::Tensor& sh2,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug,
	const torch::Tensor& sh_rest,
	const torch::Tensor& sh2_rest,
	const bool split_sh,
	const bool raw_inputs,
	const bool tight_bbox,
	const bool tile_culling,
	const bool load_balance,
	const int tile_culling_threshold);

RasterizeBackwardResult RasterizeGaborsBackwardCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& weights_for_frequency_angle,
	const torch::Tensor& frequency_lengths,
	const torch::Tensor& phases,
	const int F,
	const torch::Tensor& radii,
	const torch::Tensor& colors,
	const torch::Tensor& colors2,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_others,
	const torch::Tensor& sh,
	const torch::Tensor& sh2,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug);

RasterizeFusedBackwardResult RasterizeGaborsBackwardFusedCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& weights_for_frequency_angle,
	const torch::Tensor& frequency_lengths,
	const torch::Tensor& phases,
	const int F,
	const torch::Tensor& radii,
	const torch::Tensor& colors,
	const torch::Tensor& colors2,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_others,
	const torch::Tensor& sh,
	const torch::Tensor& sh2,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug,
	const torch::Tensor& sh_rest,
	const torch::Tensor& sh2_rest,
	const bool split_sh,
	const bool raw_inputs);

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix);
