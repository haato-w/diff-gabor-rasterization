/*
 * Private inverse-projector scratch, independent of rasterizer state and
 * backward layouts. See LICENSE.md for the original rasterizer license.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include "projector.h"

struct TileSupport;

namespace CudaProjector
{
template <typename T>
static void obtain(char*& chunk, T*& ptr, size_t count, size_t alignment)
{
    const uintptr_t offset = (reinterpret_cast<uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
    ptr = reinterpret_cast<T*>(offset);
    chunk = reinterpret_cast<char*>(ptr + count);
}

struct GeometryState
{
    size_t scan_size;
    float* depths;
    int* radii;
    float2* means2D;
    float* transMat;
    float* opacities;
    uint32_t* tiles_touched;
    uint4* tile_bounds;
    TileSupport* tile_supports;
    float4* pixel_bounds;
    char* scanning_space;
    uint32_t* point_offsets;

    static GeometryState fromChunk(char*& chunk, size_t P,
        bool tile_culling, bool warp_culling);
};

struct ImageState
{
    uint2* ranges;
    static ImageState fromChunk(char*& chunk, size_t tiles);
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

    static BinningState fromChunk(char*& chunk, size_t instances,
        bool separate_sort, int tiles);
};

struct DepthSortState
{
    size_t sorting_size;
    uint32_t* visible_ids;
    uint32_t* sorted_ids;
    uint32_t* depth_keys;
    uint32_t* depth_keys_unsorted;
    uint32_t* ordered_tiles;
    uint32_t* counts;
    char* workspace;

    static DepthSortState fromChunk(char*& chunk, size_t P);
};

template<typename T, typename... Args>
size_t required(size_t count, Args... args)
{
    char* size = nullptr;
    T::fromChunk(size, count, args...);
    return reinterpret_cast<size_t>(size) + 128;
}
} // namespace CudaProjector
