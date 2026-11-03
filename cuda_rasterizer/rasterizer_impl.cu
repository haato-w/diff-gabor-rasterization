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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <type_traits>
#include <limits>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "tile_culling.h"
#include "forward.h"
#include "backward.h"
#include "profiling.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
template<bool TileCulling, bool LoadBalance>
__global__ void duplicateWithKeys(
	int P,
	const uint4* tile_bounds,
	const TileSupport* tile_supports,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	const int* radii,
	dim3 grid,
	bool /*tile_culling*/,
	bool /*load_balance*/,
	int tile_culling_threshold)
{
	const uint32_t idx = cg::this_grid().thread_rank();
	// Do not return individual lanes: inactive lanes assist heavy neighbors.
	const bool active = idx < P && radii[idx] > 0;
	const uint4 bounds = active ? tile_bounds[idx] : make_uint4(0, 0, 0, 0);
	const TileSupport support = active && TileCulling ? tile_supports[idx] : unrestricted_tile_support();
	const float depth = active ? depths[idx] : 0.0f;
	const uint32_t offset = active && idx > 0 ? offsets[idx - 1] : 0;
	enumerate_tiles<true>(idx, bounds, support, depth, offset, grid.x,
		active, TileCulling, LoadBalance, tile_culling_threshold,
		gaussian_keys_unsorted, gaussian_values_unsorted);
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
template<typename KeyT>
__global__ void identifyTileRanges(int L, const KeyT* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	const KeyT key = point_list_keys[idx];
	const uint32_t currtile = sizeof(KeyT) == 8 ? static_cast<uint64_t>(key) >> 32 : static_cast<uint32_t>(key);
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		const KeyT previous = point_list_keys[idx - 1];
		const uint32_t prevtile = sizeof(KeyT) == 8 ? static_cast<uint64_t>(previous) >> 32 : static_cast<uint32_t>(previous);
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

__global__ void gatherOrderedTileCounts(int count, const uint32_t* ids,
	const uint32_t* tiles_touched, uint32_t* ordered_tiles)
{
	const unsigned int rank = blockIdx.x * blockDim.x + threadIdx.x;
	if (rank < count) ordered_tiles[rank] = tiles_touched[ids[rank]];
}

__global__ void scatterDepthOffsets(int count, const uint32_t* sorted_ids,
	const uint32_t* ordered_inclusive_offsets, uint32_t* starts_by_id)
{
	const unsigned int rank = blockIdx.x * blockDim.x + threadIdx.x;
	if (rank < count)
		starts_by_id[sorted_ids[rank]] = rank ? ordered_inclusive_offsets[rank - 1] : 0;
}

// Faster-GS separate sorting (Apache-2.0 reference; see THIRD_PARTY_LICENSES):
// contiguous emission segments follow global depth order, then a stable sort
// groups by tile only. Primitive IDs still address the original 2DGS arrays.
template<typename KeyT, bool TileCulling, bool LoadBalance>
__global__ void duplicateDepthOrdered(
	int count, const uint32_t* tiles_touched, const uint4* tile_bounds,
	const TileSupport* tile_supports, const uint32_t* offsets,
	KeyT* keys, uint32_t* values, dim3 grid, int threshold)
{
	const unsigned int primitive = blockIdx.x * blockDim.x + threadIdx.x;
	const bool active = primitive < count && tiles_touched[primitive] > 0;
	const uint4 bounds = active ? tile_bounds[primitive] : make_uint4(0, 0, 0, 0);
	const TileSupport support = active && TileCulling ? tile_supports[primitive] : unrestricted_tile_support();
	const uint32_t offset = active ? offsets[primitive] : 0;
	enumerate_tiles<true>(primitive, bounds, support, 0.0f, offset, grid.x,
		active, TileCulling, LoadBalance, threshold, keys, values);
}

__global__ void countTileBuckets(int tiles, const uint2* ranges, uint32_t* counts)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < tiles)
	{
		const uint2 range = ranges[idx];
		counts[idx] = (range.y - range.x + 31) / 32;
	}
}

// Sum ceil(tile_instances/32) <= ceil(R/32) + nonempty_tiles - 1.
// Reserve this bound instead of downloading another GPU count to the CPU.
static int bucketCapacity(int instances, int tiles)
{
	return instances == 0 ? 0 : static_cast<int>(std::min<size_t>(
		instances, (static_cast<size_t>(instances) + 31) / 32 + tiles - 1));
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P, int F)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.clamped2, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.weights_for_frequency_angle, P * F, 128);
	obtain(chunk, geom.frequency_lengths, P * F, 128);
	obtain(chunk, geom.phases, P * F, 128);
	obtain(chunk, geom.transMat, P * 9, 128);
	obtain(chunk, geom.normal_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.rgb2, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	obtain(chunk, geom.tile_bounds, P, 128);
	obtain(chunk, geom.tile_supports, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N, int F)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N * 3, 128);
	obtain(chunk, img.n_contrib, N * 2, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(
	char*& chunk, size_t P, int F, bool separate_sort, int tiles)
{
	BinningState binning{};
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	if (!separate_sort)
	{
		obtain(chunk, binning.point_list_keys, P, 128);
		obtain(chunk, binning.point_list_keys_unsorted, P, 128);
		cub::DeviceRadixSort::SortPairs(nullptr, binning.sorting_size,
			binning.point_list_keys_unsorted, binning.point_list_keys,
			binning.point_list_unsorted, binning.point_list, P);
	}
	else if (tiles <= 65536)
	{
		obtain(chunk, binning.tile_keys16, P, 128);
		obtain(chunk, binning.tile_keys16_unsorted, P, 128);
		cub::DeviceRadixSort::SortPairs(nullptr, binning.sorting_size,
			binning.tile_keys16_unsorted, binning.tile_keys16,
			binning.point_list_unsorted, binning.point_list, P, 0, getHigherMsb(tiles - 1));
	}
	else
	{
		obtain(chunk, binning.tile_keys32, P, 128);
		obtain(chunk, binning.tile_keys32_unsorted, P, 128);
		cub::DeviceRadixSort::SortPairs(nullptr, binning.sorting_size,
			binning.tile_keys32_unsorted, binning.tile_keys32,
			binning.point_list_unsorted, binning.point_list, P, 0, getHigherMsb(tiles - 1));
	}
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

CudaRasterizer::DepthSortState CudaRasterizer::DepthSortState::fromChunk(char*& chunk, size_t P, int F)
{
	DepthSortState state;
	obtain(chunk, state.visible_ids, P, 128);
	obtain(chunk, state.sorted_ids, P, 128);
	obtain(chunk, state.depth_keys, P, 128);
	obtain(chunk, state.depth_keys_unsorted, P, 128);
	obtain(chunk, state.ordered_tiles, P, 128);
	obtain(chunk, state.counts, 2, 128);
	cub::DoubleBuffer<uint32_t> depth_keys(state.depth_keys_unsorted, state.depth_keys);
	cub::DoubleBuffer<uint32_t> ids(state.visible_ids, state.sorted_ids);
	cub::DeviceRadixSort::SortPairs(nullptr, state.sorting_size, depth_keys, ids, P);
	obtain(chunk, state.workspace, state.sorting_size, 128);
	return state;
}

CudaRasterizer::BucketState CudaRasterizer::BucketState::fromChunk(char*& chunk, size_t tiles, size_t capacity)
{
	BucketState state;
	obtain(chunk, state.counts, tiles, 128);
	obtain(chunk, state.offsets, tiles, 128);
	obtain(chunk, state.to_tile, capacity, 128);
	obtain(chunk, state.tile_max_contrib, tiles, 128);
	cub::DeviceScan::InclusiveSum(nullptr, state.scan_size, state.counts, state.offsets, tiles);
	obtain(chunk, state.scanning_space, state.scan_size, 128);
	const size_t samples = capacity * static_cast<size_t>(BLOCK_SIZE);
	obtain(chunk, state.color_T, samples, 128);
	obtain(chunk, state.depth_normal, samples, 128);
	obtain(chunk, state.moments, samples, 128);
	return state;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	// const float* frequencies,
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
	int* radii,
	bool debug,
	const float* shs_rest,
	const float* shs2_rest,
	bool split_sh,
	bool raw_inputs,
	bool tight_bbox,
	bool tile_culling,
	bool load_balance,
	int tile_culling_threshold,
	bool gaussian_backward,
	bool warp_culling,
	bool separate_sort,
	const float* specular,
	const float* attributes)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	Profiling::Scope allocate_timing("forward.allocate");
	size_t chunk_size = required<GeometryState>(P, F);
	// Forward-only bounds follow the established geometry layout. Backward can
	// reconstruct its unchanged state and ignore these trailing checkpoints.
	if (warp_culling) chunk_size += size_t(P) * sizeof(float4) + 128;
	if (separate_sort) chunk_size += required<DepthSortState>(P, F);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P, F);
	float4* pixel_bounds = nullptr;
	if (warp_culling) obtain(chunkptr, pixel_bounds, P, 128);
	DepthSortState depthState{};
	if (separate_sort) depthState = DepthSortState::fromChunk(chunkptr, P, F);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height, F);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height, F);
	allocate_timing.stop();

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	Profiling::Scope preprocess_timing("forward.preprocess");
	if (separate_sort)
		CHECK_CUDA(cudaMemset(depthState.counts, 0, 2 * sizeof(uint32_t)), debug)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		weights_for_frequency_angle, 
		frequency_lengths, 
		phases, 
		F, 
		(glm::vec2*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		shs2,
		geomState.clamped,
		geomState.clamped2,
		transMat_precomp,
		colors_precomp,
		colors_precomp2,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.weights_for_frequency_angle, // geomStateに入れる必要ないかも？
		geomState.frequency_lengths, // geomStateに入れる必要ないかも？
		geomState.phases, // geomStateに入れる必要ないかも？
		geomState.depths,
		geomState.transMat,
		geomState.rgb,
		geomState.rgb2,
		geomState.normal_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered,
		shs_rest,
		shs2_rest,
		split_sh,
		raw_inputs,
		geomState.tile_bounds,
		tight_bbox,
		geomState.tile_supports,
		tile_culling,
		load_balance,
		tile_culling_threshold,
		warp_culling,
		pixel_bounds,
		depthState.visible_ids,
		depthState.depth_keys_unsorted,
		depthState.counts
	), debug)
	preprocess_timing.stop();

	if (!separate_sort)
	{
		Profiling::Scope scan_timing("forward.scan");
		CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size,
			geomState.tiles_touched, geomState.point_offsets, P), debug)
	}

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	int num_visible = 0;
	Profiling::Scope count_timing("forward.count_download");
	if (separate_sort)
	{
		uint32_t counts[2];
		CHECK_CUDA(cudaMemcpy(counts, depthState.counts, sizeof(counts), cudaMemcpyDeviceToHost), debug)
		if (counts[0] > std::numeric_limits<int>::max())
			throw std::runtime_error("Primitive/tile instance count exceeds the supported int32 range");
		num_rendered = counts[0];
		num_visible = counts[1];
	}
	else
	{
		CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1,
			sizeof(int), cudaMemcpyDeviceToHost), debug)
	}
	count_timing.stop();

	if (separate_sort && num_visible > 0)
	{
		Profiling::Scope depth_timing("forward.depth_sort");
		cub::DoubleBuffer<uint32_t> depth_keys(depthState.depth_keys_unsorted, depthState.depth_keys);
		cub::DoubleBuffer<uint32_t> ids(depthState.visible_ids, depthState.sorted_ids);
		CHECK_CUDA(cub::DeviceRadixSort::SortPairs(depthState.workspace, depthState.sorting_size,
			depth_keys, ids, num_visible), debug)
		depthState.sorted_ids = ids.Current();
		depth_timing.stop();
		Profiling::Scope order_timing("forward.depth_order");
		gatherOrderedTileCounts<<<(num_visible + 255) / 256, 256>>>(num_visible,
			depthState.sorted_ids, geomState.tiles_touched, depthState.ordered_tiles);
		CHECK_CUDA(, debug)
		order_timing.stop();
		Profiling::Scope depth_scan_timing("forward.depth_scan");
		CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size,
			depthState.ordered_tiles, depthState.ordered_tiles, num_visible), debug)
		depth_scan_timing.stop();
		// Segment addresses stay in depth order; reads of large per-primitive
		// surfel records stay coalesced in the original array order.
		Profiling::Scope scatter_timing("forward.depth_order");
		scatterDepthOffsets<<<(num_visible + 255) / 256, 256>>>(num_visible,
			depthState.sorted_ids, depthState.ordered_tiles, geomState.point_offsets);
		CHECK_CUDA(, debug)
	}

	Profiling::Scope binning_timing("forward.allocate_binning");
	const int num_tiles = tile_grid.x * tile_grid.y;
	size_t binning_chunk_size = required<BinningState>(num_rendered, F, separate_sort, num_tiles);
	const int bucket_capacity = gaussian_backward ? bucketCapacity(num_rendered, num_tiles) : 0;
	if (gaussian_backward)
		binning_chunk_size += required<BucketState>(num_tiles, bucket_capacity);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered, F, separate_sort, num_tiles);
	BucketState bucketState{};
	if (gaussian_backward)
		bucketState = BucketState::fromChunk(binning_chunkptr, num_tiles, bucket_capacity);
	binning_timing.stop();

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	Profiling::Scope duplicate_timing("forward.duplicate");
	auto launch_duplicate = [&](auto culling_mode, auto balance_mode)
	{
		if (separate_sort)
		{
			if (num_visible == 0) return;
			auto launch_depth_ordered = [&](auto* keys)
			{
				using KeyT = std::remove_pointer_t<decltype(keys)>;
				duplicateDepthOrdered<KeyT, decltype(culling_mode)::value, decltype(balance_mode)::value>
					<<<(P + 255) / 256, 256>>>(P, geomState.tiles_touched,
						geomState.tile_bounds, geomState.tile_supports, geomState.point_offsets,
						keys, binningState.point_list_unsorted, tile_grid, tile_culling_threshold);
			};
			if (num_tiles <= 65536) launch_depth_ordered(binningState.tile_keys16_unsorted);
			else launch_depth_ordered(binningState.tile_keys32_unsorted);
			return;
		}
		duplicateWithKeys<decltype(culling_mode)::value, decltype(balance_mode)::value> << <(P + 255) / 256, 256 >> > (
		P,
		geomState.tile_bounds,
		geomState.tile_supports,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid,
		tile_culling,
		load_balance,
		tile_culling_threshold);
	};
	if (tile_culling)
	{
		if (load_balance) launch_duplicate(std::true_type{}, std::true_type{});
		else launch_duplicate(std::true_type{}, std::false_type{});
	}
	else
	{
		if (load_balance) launch_duplicate(std::false_type{}, std::true_type{});
		else launch_duplicate(std::false_type{}, std::false_type{});
	}
	CHECK_CUDA(, debug)
	duplicate_timing.stop();

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	Profiling::Scope sort_timing("forward.sort");
	if (separate_sort)
	{
		auto sort_tiles = [&](auto* keys_in, auto* keys_out)
		{
			if (num_rendered > 0)
				CHECK_CUDA(cub::DeviceRadixSort::SortPairs(binningState.list_sorting_space,
					binningState.sorting_size, keys_in, keys_out, binningState.point_list_unsorted,
					binningState.point_list, num_rendered, 0, getHigherMsb(num_tiles - 1)), debug)
		};
		if (num_tiles <= 65536) sort_tiles(binningState.tile_keys16_unsorted, binningState.tile_keys16);
		else sort_tiles(binningState.tile_keys32_unsorted, binningState.tile_keys32);
	}
	else
	{
		CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
			binningState.list_sorting_space, binningState.sorting_size,
			binningState.point_list_keys_unsorted, binningState.point_list_keys,
			binningState.point_list_unsorted, binningState.point_list,
			num_rendered, 0, 32 + bit), debug)
	}
	sort_timing.stop();

	Profiling::Scope tile_ranges_timing("forward.tile_ranges");
	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	auto identify_ranges = [&](auto* keys)
	{
		if (num_rendered > 0)
			identifyTileRanges<<<(num_rendered + 255) / 256, 256>>>(num_rendered, keys, imgState.ranges);
	};
	if (!separate_sort) identify_ranges(binningState.point_list_keys);
	else if (num_tiles <= 65536) identify_ranges(binningState.tile_keys16);
	else identify_ranges(binningState.tile_keys32);
	CHECK_CUDA(, debug)
	tile_ranges_timing.stop();

	if (gaussian_backward)
	{
		Profiling::Scope bucket_timing("forward.backward_buckets");
		countTileBuckets<<<(num_tiles + 255) / 256, 256>>>(num_tiles, imgState.ranges, bucketState.counts);
		CHECK_CUDA(cub::DeviceScan::InclusiveSum(bucketState.scanning_space,
			bucketState.scan_size, bucketState.counts, bucketState.offsets, num_tiles), debug)
	}

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	const float* feature_ptr2 = colors_precomp2 != nullptr ? colors_precomp2 : geomState.rgb2;
	const float* transMat_ptr = transMat_precomp != nullptr ? transMat_precomp : geomState.transMat;
	Profiling::Scope render_timing("forward.render");
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		focal_x, focal_y,
		geomState.means2D,
		geomState.weights_for_frequency_angle, 
		geomState.frequency_lengths, 
		geomState.phases, 
		F, 
		feature_ptr,
		feature_ptr2,
		transMat_ptr,
		geomState.depths,
		geomState.normal_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		out_others,
		gaussian_backward,
		bucketState.offsets,
		bucketState.to_tile,
		bucketState.tile_max_contrib,
		bucketState.color_T,
		bucketState.depth_normal,
		bucketState.moments,
		warp_culling,
		pixel_bounds,
		specular, attributes), debug)
	render_timing.stop();

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
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
	char* img_buffer,
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
	const float* shs_rest,
	const float* shs2_rest,
	bool split_sh,
	bool raw_inputs,
	float* dL_dsh_rest,
	float* dL_dsh2_rest,
	bool gaussian_backward,
	const float* final_image,
	const float* final_aux,
	bool separate_sort,
	bool backward_reduction,
	int backward_reduction_threshold,
	const float* specular,
	float* dL_dspecular,
	const float* attributes,
	float* dL_dattributes)
{
	Profiling::Scope allocate_timing("backward.allocate");
	const int num_tiles = ((width + BLOCK_X - 1) / BLOCK_X) * ((height + BLOCK_Y - 1) / BLOCK_Y);
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P, F);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R, F, separate_sort, num_tiles);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height, F);
	allocate_timing.stop();

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);
	const int bucket_capacity = gaussian_backward ? bucketCapacity(R, num_tiles) : 0;
	BucketState bucketState{};
	if (gaussian_backward)
		bucketState = BucketState::fromChunk(binning_buffer, num_tiles, bucket_capacity);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	const float* color_ptr2 = (colors_precomp2 != nullptr) ? colors_precomp2 : geomState.rgb2;
	const float* depth_ptr = geomState.depths;
	const float* transMat_ptr = (transMat_precomp != nullptr) ? transMat_precomp : geomState.transMat;
	Profiling::Scope render_timing("backward.render");
	if (gaussian_backward)
	{
		BACKWARD::render_gaussian(
			tile_grid, block, imgState.ranges, binningState.point_list,
			width, height, focal_x, focal_y, background,
			geomState.means2D, geomState.weights_for_frequency_angle,
			geomState.frequency_lengths, geomState.phases, F,
			geomState.normal_opacity, color_ptr, color_ptr2, transMat_ptr,
			depth_ptr, imgState.accum_alpha, imgState.n_contrib,
			dL_dpix, dL_depths, dL_dtransMat, (float3*)dL_dmean2D,
			dL_dnormal, dL_dopacity, dL_dweights_for_frequency_angle,
			dL_dfrequency_lengths, dL_dphases, dL_dcolor, dL_dcolor2,
			d_image, bucket_capacity, num_tiles, bucketState.offsets,
			bucketState.to_tile, bucketState.tile_max_contrib,
			bucketState.color_T, bucketState.depth_normal,
			bucketState.moments, final_image, final_aux, specular, dL_dspecular);
	}
	else
	{
		BACKWARD::render(
			tile_grid, block, imgState.ranges, binningState.point_list,
			width, height, focal_x, focal_y, background,
			geomState.means2D, geomState.weights_for_frequency_angle,
			geomState.frequency_lengths, geomState.phases, F,
			geomState.normal_opacity, color_ptr, color_ptr2, transMat_ptr,
			depth_ptr, imgState.accum_alpha, imgState.n_contrib,
			dL_dpix, dL_depths, dL_dtransMat, (float3*)dL_dmean2D,
			dL_dnormal, dL_dopacity, dL_dweights_for_frequency_angle,
			dL_dfrequency_lengths, dL_dphases, dL_dcolor, dL_dcolor2,
			d_image, backward_reduction, backward_reduction_threshold, specular, dL_dspecular,
			attributes, dL_dattributes);
	}
	CHECK_CUDA(, debug)
	render_timing.stop();

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	// const float* transMat_ptr = (transMat_precomp != nullptr) ? transMat_precomp : geomState.transMat;
	Profiling::Scope preprocess_timing("backward.preprocess");
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		shs2,
		geomState.clamped,
		geomState.clamped2,
		(glm::vec2*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		transMat_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D, // gradient inputs
		dL_dnormal,		     // gradient inputs
		dL_dtransMat,
		dL_dcolor,
		dL_dcolor2,
		dL_dsh,
		dL_dsh2,
		(glm::vec3*)dL_dmean3D,
		(glm::vec2*)dL_dscale,
		(glm::vec4*)dL_drot,
		shs_rest,
		shs2_rest,
		split_sh,
		raw_inputs,
		dL_dsh_rest,
		dL_dsh2_rest,
		geomState.normal_opacity,
		dL_dopacity), debug)
	preprocess_timing.stop();
}
