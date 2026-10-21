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

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

struct TileSupport;

namespace CudaRasterizer
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	struct GeometryState
	{
		size_t scan_size;
		float* depths;
		char* scanning_space;
		bool* clamped;
		bool* clamped2;
		int* internal_radii;
		float2* means2D;
		float* weights_for_frequency_angle;
		float* frequency_lengths;
		float* phases;
		float* transMat;
		float4* normal_opacity;
		float* rgb;
		float* rgb2;
		uint32_t* point_offsets;
		uint32_t* tiles_touched;
		// Tile rectangle, stored as (min_x, min_y, max_x, max_y), upper exclusive.
		uint4* tile_bounds;
		// Opacity-cutoff projective ellipse and unchanged low-pass disk.
		TileSupport* tile_supports;

		static GeometryState fromChunk(char*& chunk, size_t P, int F);
	};

	struct ImageState
	{
		uint2* ranges;
		uint32_t* n_contrib;
		float* accum_alpha;

		static ImageState fromChunk(char*& chunk, size_t N, int F);
	};

	struct BinningState
	{
		size_t sorting_size;
		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;
		uint16_t* tile_keys16_unsorted;
		uint16_t* tile_keys16;
		uint32_t* tile_keys32_unsorted;
		uint32_t* tile_keys32;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		char* list_sorting_space;

		static BinningState fromChunk(char*& chunk, size_t P, int F,
			bool separate_sort = false, int tiles = 0);
	};

	// Optional forward scratch. Render intermediates retain original primitive
	// indices so no extra parameter/gradient indirection is needed in backward.
	struct DepthSortState
	{
		size_t selection_size;
		size_t reduction_size;
		size_t sorting_size;
		uint32_t* visible_ids;
		uint32_t* sorted_ids;
		uint32_t* depth_keys;
		uint32_t* depth_keys_unsorted;
		uint32_t* ordered_tiles;
		uint32_t* counts; // [primitive/tile instance count, visible primitive count]
		char* workspace;

		static DepthSortState fromChunk(char*& chunk, size_t P, int F);
	};

	// Optional training checkpoints, appended after BinningState. A bucket is
	// 32 consecutive entries of one tile's sorted primitive list.
	struct BucketState
	{
		size_t scan_size;
		uint32_t* counts;
		uint32_t* offsets;
		uint32_t* to_tile;
		uint32_t* tile_max_contrib;
		char* scanning_space;
		float4* color_T;
		float4* depth_normal;
		float2* moments;

		static BucketState fromChunk(char*& chunk, size_t tiles, size_t capacity);
	};

	template<typename T, typename... Args>
	size_t required(size_t P, int F, Args... args)
	{
		char* size = nullptr;
		T::fromChunk(size, P, F, args...);
		return ((size_t)size) + 128;
	}
};
