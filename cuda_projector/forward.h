/*
 * Dedicated inverse-projection kernels. See LICENSE.md for the original
 * optimized rasterizer license.
 */
#pragma once

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

struct TileSupport;

namespace PROJECTOR_FORWARD
{
void preprocess(int P, const float* means3D,
    const glm::vec2* scales, float scale_modifier, const glm::vec4* rotations,
    const float* opacities, const float* viewmatrix, const float* projmatrix,
    int W, int H, int* radii, float2* means2D, float* depths,
    float* transMats, float* activated_opacities, const dim3 grid,
    uint32_t* tiles_touched, bool raw_inputs, uint4* tile_bounds,
    bool tight_bbox, TileSupport* tile_supports, bool tile_culling,
    bool load_balance, int tile_culling_threshold, bool warp_culling,
    float4* pixel_bounds, uint32_t* visible_ids,
    uint32_t* visible_depth_keys, uint32_t* visibility_counts);

void projectErrors(const dim3 grid, dim3 block, const uint2* ranges,
    const uint32_t* point_list, int W, int H, const float2* means2D,
    const float* transMats, const float* opacities, const float* error_map,
    float* gaussian_errors, float* gaussian_effects, const float* hair_vote_map,
    float* hair_votes, bool frontmost_only, bool warp_culling,
    const float4* pixel_bounds);
} // namespace PROJECTOR_FORWARD
