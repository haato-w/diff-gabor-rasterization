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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		// split_sh keeps DC (P, 1, 3) and rest (P, M - 1, 3) separate.
		// raw_inputs fuses exp(scale), sigmoid(opacity) and normalize(rotation)
		// into preprocessing; backward returns gradients of those raw inputs.
		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* weights_for_frequency_angle, 
			const float* frequency_lengths, 
			const float* phases, 
			const int F, // the number of frequency candidates
			const float* shs,
			const float* shs2,
			const float* colors_precomp,
			const float* colors_precomp2,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* transMat_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_color,
			float* out_others,
			int* radii = nullptr,
			bool debug = false,
			const float* shs_rest = nullptr,
			const float* shs2_rest = nullptr,
			bool split_sh = false,
			bool raw_inputs = false,
			bool tight_bbox = false,
			bool tile_culling = false,
			bool load_balance = false,
			int tile_culling_threshold = 4,
			bool gaussian_backward = false);

		static void backward(
			const int P, int D, int M, int R,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* weights_for_frequency_angle, 
			const float* frequency_lengths, 
			const float* phases, 
			const int F, // the number of frequency candidates
			const float* shs,
			const float* shs2,
			const float* colors_precomp,
			const float* colors_precomp2,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* transMat_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float tan_fovx, float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,
			const float* dL_depths,
			float* dL_dmean2D,
			float* dL_dnormal,
			float* dL_dopacity,
			float* dL_dweights_for_frequency_angle, 
			float* dL_dfrequency_lengths, 
			float* dL_dphases, 
			float* dL_dcolor,
			float* dL_dcolor2,
			float* dL_dmean3D,
			float* dL_dtransMat,
			float* dL_dsh,
			float* dL_dsh2,
			float* dL_dscale,
			float* dL_drot,
			float* d_image, 
			bool debug,
			const float* shs_rest = nullptr,
			const float* shs2_rest = nullptr,
			bool split_sh = false,
			bool raw_inputs = false,
			float* dL_dsh_rest = nullptr,
			float* dL_dsh2_rest = nullptr,
			bool gaussian_backward = false,
			const float* final_image = nullptr,
			const float* final_aux = nullptr);
	};
};

#endif
