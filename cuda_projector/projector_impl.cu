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

#include "projector_impl.h"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include "../cuda_rasterizer/auxiliary.h"
#include "../cuda_rasterizer/tile_culling.h"
#include "../cuda_rasterizer/profiling.h"
#include "forward.h"

namespace
{
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

} // namespace

CudaProjector::GeometryState CudaProjector::GeometryState::fromChunk(
    char*& chunk, size_t P, bool tile_culling, bool warp_culling)
{
    GeometryState geom{};
    obtain(chunk, geom.depths, P, 128);
    obtain(chunk, geom.radii, P, 128);
    obtain(chunk, geom.means2D, P, 128);
    obtain(chunk, geom.transMat, P * 9, 128);
    obtain(chunk, geom.opacities, P, 128);
    obtain(chunk, geom.tiles_touched, P, 128);
    obtain(chunk, geom.tile_bounds, P, 128);
    if (tile_culling) obtain(chunk, geom.tile_supports, P, 128);
    if (warp_culling) obtain(chunk, geom.pixel_bounds, P, 128);
    cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size,
        geom.tiles_touched, geom.tiles_touched, P);
    obtain(chunk, geom.scanning_space, geom.scan_size, 128);
    obtain(chunk, geom.point_offsets, P, 128);
    return geom;
}

CudaProjector::ImageState CudaProjector::ImageState::fromChunk(char*& chunk, size_t tiles)
{
    ImageState img{};
    obtain(chunk, img.ranges, tiles, 128);
    return img;
}

CudaProjector::BinningState CudaProjector::BinningState::fromChunk(
    char*& chunk, size_t P, bool separate_sort, int tiles)
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

CudaProjector::DepthSortState CudaProjector::DepthSortState::fromChunk(char*& chunk, size_t P)
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

// Standalone projection only: no RGB inputs, render outputs or backward state.
int CudaProjector::Projector::project(
    std::function<char* (size_t)> geometryBuffer,
    std::function<char* (size_t)> binningBuffer,
    std::function<char* (size_t)> imageBuffer,
    int P, int width, int height,
    const float* means3D, const float* opacities, const float* scales,
    float scale_modifier, const float* rotations,
    const float* viewmatrix, const float* projmatrix,
    const float* error_map, float* gaussian_errors, float* gaussian_effects,
    const float* hair_vote_map, float* hair_votes,
    bool raw_inputs, bool tight_bbox, bool tile_culling, bool load_balance,
    int tile_culling_threshold, bool warp_culling, bool separate_sort,
    bool frontmost_only, bool debug)
{
    namespace Profiling = CudaRasterizer::Profiling;
    if (P == 0) return 0;
    Profiling::Scope allocate_timing("error_projection.allocate");
    size_t chunk_size = required<GeometryState>(P, tile_culling, warp_culling);
    if (separate_sort) chunk_size += required<DepthSortState>(P);
    char* chunkptr = geometryBuffer(chunk_size);
    GeometryState geomState = GeometryState::fromChunk(chunkptr, P, tile_culling, warp_culling);
    const float4* pixel_bounds = geomState.pixel_bounds;
    DepthSortState depthState{};
    if (separate_sort) depthState = DepthSortState::fromChunk(chunkptr, P);
    int* radii = geomState.radii;

    dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    dim3 block(BLOCK_X, BLOCK_Y, 1);

    // Only tile ranges are needed; no image outputs or backward checkpoints.
    const size_t image_entries = tile_grid.x * tile_grid.y;
    size_t img_chunk_size = required<ImageState>(image_entries);
    char* img_chunkptr = imageBuffer(img_chunk_size);
    ImageState imgState = ImageState::fromChunk(img_chunkptr, image_entries);
    allocate_timing.stop();

    // Fuse raw parameter activation, exact bounds and warp tile counting.
    Profiling::Scope preprocess_timing("error_projection.preprocess");
    if (separate_sort)
        CHECK_CUDA(cudaMemset(depthState.counts, 0, 2 * sizeof(uint32_t)), debug)
    CHECK_CUDA(PROJECTOR_FORWARD::preprocess(
        P, means3D, (const glm::vec2*)scales, scale_modifier,
        (const glm::vec4*)rotations, opacities, viewmatrix, projmatrix,
        width, height, radii, geomState.means2D, geomState.depths,
        geomState.transMat, geomState.opacities, tile_grid,
        geomState.tiles_touched, raw_inputs, geomState.tile_bounds,
        tight_bbox, geomState.tile_supports, tile_culling, load_balance,
        tile_culling_threshold, warp_culling, geomState.pixel_bounds,
        depthState.visible_ids, depthState.depth_keys_unsorted,
        depthState.counts), debug)

    preprocess_timing.stop();

    if (!separate_sort)
    {
        Profiling::Scope scan_timing("error_projection.scan");
        CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size,
            geomState.tiles_touched, geomState.point_offsets, P), debug)
    }

    // Retrieve total number of Gaussian instances to launch and resize aux buffers
    int num_rendered;
    int num_visible = 0;
    Profiling::Scope count_timing("error_projection.count_download");
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
        Profiling::Scope depth_timing("error_projection.depth_sort");
        cub::DoubleBuffer<uint32_t> depth_keys(depthState.depth_keys_unsorted, depthState.depth_keys);
        cub::DoubleBuffer<uint32_t> ids(depthState.visible_ids, depthState.sorted_ids);
        CHECK_CUDA(cub::DeviceRadixSort::SortPairs(depthState.workspace, depthState.sorting_size,
            depth_keys, ids, num_visible), debug)
        depthState.sorted_ids = ids.Current();
        depth_timing.stop();
        Profiling::Scope order_timing("error_projection.depth_order");
        gatherOrderedTileCounts<<<(num_visible + 255) / 256, 256>>>(num_visible,
            depthState.sorted_ids, geomState.tiles_touched, depthState.ordered_tiles);
        CHECK_CUDA(, debug)
        order_timing.stop();
        Profiling::Scope depth_scan_timing("error_projection.depth_scan");
        CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size,
            depthState.ordered_tiles, depthState.ordered_tiles, num_visible), debug)
        depth_scan_timing.stop();
        // Segment addresses stay in depth order; reads of large per-primitive
        // surfel records stay coalesced in the original array order.
        Profiling::Scope scatter_timing("error_projection.depth_order");
        scatterDepthOffsets<<<(num_visible + 255) / 256, 256>>>(num_visible,
            depthState.sorted_ids, depthState.ordered_tiles, geomState.point_offsets);
        CHECK_CUDA(, debug)
    }

    Profiling::Scope binning_timing("error_projection.allocate_binning");
    const int num_tiles = tile_grid.x * tile_grid.y;
    size_t binning_chunk_size = required<BinningState>(num_rendered, separate_sort, num_tiles);
    char* binning_chunkptr = binningBuffer(binning_chunk_size);
    BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered, separate_sort, num_tiles);
    binning_timing.stop();

    // For each instance to be rendered, produce adequate [ tile | depth ] key
    // and corresponding duplicated Gaussian indices to be sorted
    Profiling::Scope duplicate_timing("error_projection.duplicate");
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
    Profiling::Scope sort_timing("error_projection.sort");
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

    Profiling::Scope tile_ranges_timing("error_projection.tile_ranges");
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

    // Each tile accumulates error/effect and optional hair evidence in one pass.
    Profiling::Scope render_timing("error_projection.render");
    CHECK_CUDA(PROJECTOR_FORWARD::projectErrors(tile_grid, block, imgState.ranges,
        binningState.point_list, width, height, geomState.means2D, geomState.transMat,
        geomState.opacities, error_map, gaussian_errors, gaussian_effects,
        hair_vote_map, hair_votes, frontmost_only, warp_culling, pixel_bounds), debug)
    render_timing.stop();
    return num_rendered;
}
