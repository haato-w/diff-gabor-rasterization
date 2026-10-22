/*
 * Dedicated visibility-only inverse projector, copied from the optimized
 * 2DGS rasterizer. See LICENSE.md for the original rasterizer license.
 */
#pragma once

#include <cstddef>
#include <functional>

namespace CudaProjector
{
class Projector
{
public:
    static int project(
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
        bool frontmost_only, bool debug);
};
} // namespace CudaProjector
