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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

struct TileSupport;

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const float* weights_for_frequency_angle, 
		const float* frequency_lengths, 
		const float* phases, 
		const int F, 
		const glm::vec2* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		const float* shs2,
		bool* clamped,
		bool* clamped2,
		const float* transMat_precomp,
		const float* colors_precomp,
		const float* colors_precomp2,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* geo_weights_for_frequency_angle, 
		float* geo_frequency_lengths, 
		float* geo_phases, 
		float* depths,
		// float* isovals,
		// float3* normals,
		float* transMats,
		float* colors,
		float* colors2,
		float4* normal_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered,
		const float* shs_rest,
		const float* shs2_rest,
		bool split_sh,
		bool raw_inputs,
		uint4* tile_bounds,
		bool tight_bbox,
		TileSupport* tile_supports,
		bool tile_culling,
		bool load_balance,
		int tile_culling_threshold,
		bool warp_culling,
		float4* pixel_bounds,
		uint32_t* visible_ids,
		uint32_t* visible_depth_keys,
		uint32_t* visibility_counts);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		float focal_x, float focal_y,
		const float2* points_xy_image,
		const float* geo_weights_for_frequency_angle, 
		const float* geo_frequency_lengths, 
		const float* geo_phases, 
		const int F, 
		const float* features,
		const float* features2,
		const float* transMats,
		const float* depths,
		const float4* normal_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		float* out_others,
		bool gaussian_backward,
		const uint32_t* bucket_offsets,
		uint32_t* bucket_to_tile,
		uint32_t* tile_max_contrib,
		float4* bucket_color_T,
		float4* bucket_depth_normal,
		float2* bucket_moments,
		bool warp_culling,
		const float4* pixel_bounds);
}


#endif
