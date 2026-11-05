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

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace BACKWARD
{
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		float focal_x, float focal_y,
		const float* bg_color,
		const float2* means2D,
		const float* frequency_vectors,

		const float* phases, 
		const float4* normal_opacity,
		const float* transMats,
		const float* colors,
		const float* colors2,
		const float* depths,
		const float* final_Ts,
		const uint32_t* n_contrib,
		const float* dL_dpixels,
		const float* dL_depths,
		float * dL_dtransMat,
		float3* dL_dmean2D,
		float* dL_dnormal3D,
		float* dL_dopacity,
		float* dL_dfrequency_vectors,

		float* dL_dphases, 
		float* dL_dcolors,
		float* dL_dcolors2, 
		float* d_image,
		bool backward_reduction = false,
		int backward_reduction_threshold = 16,
		const float* specular = nullptr,
		float* dL_dspecular = nullptr,
		const float* attributes = nullptr,
		float* dL_dattributes = nullptr);

	// Ordinary arguments follow render's implementation order: color endpoints
	// precede the projective transforms. The additional buffers are saved by
	// the forward pass only when Gaussian-owned backward rendering is enabled.
	void render_gaussian(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		float focal_x, float focal_y,
		const float* bg_color,
		const float2* means2D,
		const float* frequency_vectors,

		const float* phases,
		const float4* normal_opacity,
		const float* colors,
		const float* colors2,
		const float* transMats,
		const float* depths,
		const float* final_Ts,
		const uint32_t* n_contrib,
		const float* dL_dpixels,
		const float* dL_depths,
		float* dL_dtransMat,
		float3* dL_dmean2D,
		float* dL_dnormal3D,
		float* dL_dopacity,
		float* dL_dfrequency_vectors,

		float* dL_dphases,
		float* dL_dcolors,
		float* dL_dcolors2,
		float* d_image,
		int bucket_capacity, int num_tiles,
		const uint32_t* bucket_offsets,
		const uint32_t* bucket_to_tile,
		const uint32_t* tile_max_contrib,
		const float4* bucket_color_T,
		const float4* bucket_depth_normal,
		const float2* bucket_moments,
		const float* final_image,
		const float* final_aux,
		const float* specular = nullptr,
		float* dL_dspecular = nullptr);

	void preprocess(
		int P, int D, int M,
		const float3* means,
		const int* radii,
		const float* shs,
		const float* shs2,
		const bool* clamped,
		const bool* clamped2,
		const glm::vec2* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float* transMats,
		const float* view,
		const float* proj,
		const float focal_x, const float focal_y,
		const float tan_fovx, const float tan_fovy,
		const glm::vec3* campos,
		float3* dL_dmean2D,
		const float* dL_dnormal3D,
		float* dL_dtransMat,
		float* dL_dcolor,
		float* dL_dcolor2,
		float* dL_dsh,
		float* dL_dsh2,
		glm::vec3* dL_dmeans,
		glm::vec2* dL_dscale,
		glm::vec4* dL_drot,
		const float* shs_rest,
		const float* shs2_rest,
		bool split_sh,
		bool raw_inputs,
		float* dL_dsh_rest,
		float* dL_dsh2_rest,
		const float4* normal_opacity,
		float* dL_dopacity
		);
}

#endif
