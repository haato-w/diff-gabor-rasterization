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

#include "forward.h"
#include "../cuda_rasterizer/auxiliary.h"
#include "../cuda_rasterizer/tile_culling.h"
#include <type_traits>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// This geometry-only pipeline is intentionally independent of RGB rendering.
// Its visibility math is copied from the optimized rasterizer; shared stable
// headers provide activation functions and conservative tile support tests.
namespace
{
// Compute a 2D-to-2D mapping matrix from a tangent plane into a image plane
// given a 2D gaussian parameters.
__device__ void compute_transmat(
    const float3& p_orig,
    const glm::vec2 scale,
    float mod,
    const glm::vec4 rot,
    const float* projmatrix,
    const float* viewmatrix,
    const int W,
    const int H,
    glm::mat3 &T,
    float3 &normal
) {

    glm::mat3 R = quat_to_rotmat(rot);
    glm::mat3 S = scale_to_mat(scale, mod);
    glm::mat3 L = R * S;

    // center of Gaussians in the camera coordinate
    glm::mat3x4 splat2world = glm::mat3x4(
        glm::vec4(L[0], 0.0),
        glm::vec4(L[1], 0.0),
        glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1)
    );

    glm::mat4 world2ndc = glm::mat4(
        projmatrix[0], projmatrix[4], projmatrix[8], projmatrix[12],
        projmatrix[1], projmatrix[5], projmatrix[9], projmatrix[13],
        projmatrix[2], projmatrix[6], projmatrix[10], projmatrix[14],
        projmatrix[3], projmatrix[7], projmatrix[11], projmatrix[15]
    );

    glm::mat3x4 ndc2pix = glm::mat3x4(
        glm::vec4(float(W) / 2.0, 0.0, 0.0, float(W-1) / 2.0),
        glm::vec4(0.0, float(H) / 2.0, 0.0, float(H-1) / 2.0),
        glm::vec4(0.0, 0.0, 0.0, 1.0)
    );

    T = glm::transpose(splat2world) * world2ndc * ndc2pix;
    normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix);

}

// Fix rounding for this legacy AABB expression across kernel specializations.
// Otherwise support construction can change compiler FMA contraction/CSE and
// subtly move the low-pass filter center when tile culling is enabled.
__forceinline__ __device__ glm::vec3 aabb_product(const glm::vec3 a, const glm::vec3 b)
{
    return glm::vec3(__fmul_rn(a.x, b.x), __fmul_rn(a.y, b.y), __fmul_rn(a.z, b.z));
}

__forceinline__ __device__ float aabb_dot(const glm::vec3 a, const glm::vec3 b)
{
    return __fadd_rn(__fadd_rn(__fmul_rn(a.x, b.x), __fmul_rn(a.y, b.y)), __fmul_rn(a.z, b.z));
}

// Computing the bounding box of the 2D Gaussian and its center.
// The center of the bounding box is used to create a low pass filter.
__device__ bool compute_aabb(
    glm::mat3 T,
    float cutoff,
    float2& point_image,
    float2& extent
) {
    const glm::vec3 t = glm::vec3(__fmul_rn(cutoff, cutoff), __fmul_rn(cutoff, cutoff), -1.0f);
    const float d = aabb_dot(t, aabb_product(T[2], T[2]));
    if (d == 0.0f) return false;
    const float inv_d = __fdiv_rn(1.0f, d);
    const glm::vec3 f = aabb_product(glm::vec3(inv_d), t);
    const float px = aabb_dot(f, aabb_product(T[0], T[2]));
    const float py = aabb_dot(f, aabb_product(T[1], T[2]));
    const float hx = __fsub_rn(__fmul_rn(px, px), aabb_dot(f, aabb_product(T[0], T[0])));
    const float hy = __fsub_rn(__fmul_rn(py, py), aabb_dot(f, aabb_product(T[1], T[1])));
    point_image = {px, py};
    extent = {sqrtf(fmaxf(1.0e-4f, hx)), sqrtf(fmaxf(1.0e-4f, hy))};
    return true;
}

// Bound one coordinate of the perspective projection of a local disk. Splitting
// its numerator into components parallel/perpendicular to the denominator avoids
// the cancellation in the usual extent^2 = center^2 - quadratic expression.
__forceinline__ __device__ bool project_disk_axis(
    const glm::vec3 a, const glm::vec3 w, const float r2,
    const float denominator, const float w_length,
    float& lower, float& upper)
{
    const float dot_aw = fmaf(a.x, w.x, a.y * w.y);
    const float center = fmaf(-r2, dot_aw, a.z * w.z) / denominator;
    float extent;
    if (w_length > 0.0f)
    {
        const float parallel = dot_aw / w_length;
        const float perpendicular = fmaf(a.x, w.y, -a.y * w.x) / w_length;
        const float projected_parallel = fmaf(parallel, w.z, -a.z * w_length);
        extent = sqrtf(r2) * hypotf(projected_parallel,
            perpendicular * sqrtf(denominator)) / denominator;
    }
    else
    {
        extent = sqrtf(r2) * hypotf(a.x, a.y) / fabsf(w.z);
    }
    if (!isfinite(center) || !isfinite(extent)) return false;
    // Cover rounding in the cutoff, perspective arithmetic and integer pixels.
    const float guard = 1.0e-3f + 1.0e-4f * (fabsf(center) + extent + 1.0f);
    lower = center - extent - guard;
    upper = center + extent + guard;
    return isfinite(lower) && isfinite(upper);
}

__forceinline__ __device__ uint32_t bound_tile(
    const float coordinate, const int tile_size, const uint32_t tile_count,
    const bool upper)
{
    // Our renderer samples integer pixels, unlike the half-pixel convention in
    // Faster-GS. Include the tile containing the upper endpoint as well.
    const float tile = floorf(coordinate / tile_size) + (upper ? 1.0f : 0.0f);
    if (tile <= 0.0f) return 0;
    if (tile >= static_cast<float>(tile_count)) return tile_count;
    return static_cast<uint32_t>(tile);
}

__device__ void tighten_tile_rect(
    const glm::mat3 T, const float opacity, const float2 legacy_center,
    const dim3 grid, uint2& rect_min, uint2& rect_max,
    const bool tight_bbox, float4* pixel_bounds)
{
    // Unstable projective geometry retains the entire original tile rectangle.
    // Bounds use inclusive integer pixel centers, unlike Faster-GS's +0.5 grid.
    if (pixel_bounds)
        *pixel_bounds = make_float4(rect_min.x * BLOCK_X, rect_min.y * BLOCK_Y,
            rect_max.x * BLOCK_X - 1, rect_max.y * BLOCK_Y - 1);
    if (!isfinite(opacity) || !isfinite(legacy_center.x) || !isfinite(legacy_center.y)) return;
    if (opacity < 1.0f / 255.0f)
    {
        // Preserve the legacy scalar radius/visibility even when no pixel can
        // contribute. Only the duplication rectangle becomes empty.
        if (tight_bbox) rect_max = rect_min;
        if (pixel_bounds) *pixel_bounds = make_float4(0.0f, 0.0f, -1.0f, -1.0f);
        return;
    }

    float r2 = 2.0f * logf(opacity * 255.0f);
    if (!isfinite(r2)) return;
    r2 = fmaxf(r2, 0.0f);
    r2 += 1.0e-4f + 1.0e-5f * r2;
    const glm::vec3 w = T[2];
    const float w_length2 = fmaf(w.x, w.x, w.y * w.y);
    const float w_depth2 = w.z * w.z;
    const float denominator = fmaf(-r2, w_length2, w_depth2);
    // A disk crossing the projective horizon does not have a finite ellipse
    // AABB. Ill-conditioned and nonfinite cases retain the legacy rectangle.
    if (!isfinite(denominator) || !isfinite(w_depth2) ||
        !(denominator > 1.0e-3f * w_depth2)) return;

    float x_min, x_max, y_min, y_max;
    const float w_length = sqrtf(w_length2);
    if (!project_disk_axis(T[0], w, r2, denominator, w_length, x_min, x_max) ||
        !project_disk_axis(T[1], w, r2, denominator, w_length, y_min, y_max)) return;

    // rho = min(rho_surfel, rho_filter), hence support is the UNION of the
    // projected disk and the low-pass disk at the unchanged legacy center.
    const float filter_radius = sqrtf(r2 / FilterInvSquare) + 1.0e-3f;
    x_min = fminf(x_min, legacy_center.x - filter_radius);
    x_max = fmaxf(x_max, legacy_center.x + filter_radius);
    y_min = fminf(y_min, legacy_center.y - filter_radius);
    y_max = fmaxf(y_max, legacy_center.y + filter_radius);
    if (!isfinite(x_min) || !isfinite(x_max) || !isfinite(y_min) || !isfinite(y_max)) return;
    if (pixel_bounds) *pixel_bounds = make_float4(x_min, y_min, x_max, y_max);
    if (!tight_bbox) return;

    // Intersect with the baseline rectangle: high opacities can have a cutoff
    // larger than 3 sigma, and must not introduce new baseline contributions.
    rect_min.x = max(rect_min.x, bound_tile(x_min, BLOCK_X, grid.x, false));
    rect_min.y = max(rect_min.y, bound_tile(y_min, BLOCK_Y, grid.y, false));
    rect_max.x = max(rect_min.x, min(rect_max.x, bound_tile(x_max, BLOCK_X, grid.x, true)));
    rect_max.y = max(rect_min.y, min(rect_max.y, bound_tile(y_max, BLOCK_Y, grid.y, true)));
}

__device__ __forceinline__ void preprocessPrimitive(int P,
    const float* orig_points, const glm::vec2* scales,
    const float scale_modifier, const glm::vec4* rotations,
    const float* opacities, const float* viewmatrix, const float* projmatrix,
    const int W, const int H, int* radii, float2* points_xy_image,
    float* depths, float* transMats, float* activated_opacities,
    const dim3 grid, uint32_t* tiles_touched, bool raw_inputs,
    uint4* tile_bounds, bool tight_bbox, TileSupport* tile_supports,
    bool tile_culling, bool warp_culling, float4* pixel_bounds)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    // Initialize radius and touched tiles to 0. If this isn't changed,
    // this Gaussian will not be processed further.
    radii[idx] = 0;
    tiles_touched[idx] = 0;
    tile_bounds[idx] = make_uint4(0, 0, 0, 0);
    if (tile_culling) tile_supports[idx] = unrestricted_tile_support();

    // Perform near culling, quit if outside.
    float3 p_view;
    if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view))
        return;

    // Compute transformation matrix
    glm::mat3 T;
    float3 normal;
    const glm::vec2 scale = raw_inputs ? activate_scale(scales[idx]) : scales[idx];
    const glm::vec4 rotation = raw_inputs ? activate_rotation(rotations[idx]) : rotations[idx];
    compute_transmat(((const float3*)orig_points)[idx], scale, scale_modifier,
        rotation, projmatrix, viewmatrix, W, H, T, normal);
    float3* T_ptr = (float3*)transMats;
    T_ptr[idx * 3 + 0] = {T[0][0], T[0][1], T[0][2]};
    T_ptr[idx * 3 + 1] = {T[1][0], T[1][1], T[1][2]};
    T_ptr[idx * 3 + 2] = {T[2][0], T[2][1], T[2][2]};

#if DUAL_VISIABLE
    float cos = -sumf3(p_view * normal);
    if (cos == 0) return;
#endif

    const float opacity = raw_inputs ? activate_opacity(opacities[idx]) : opacities[idx];

    // Keep the same 3-sigma center and tile coverage as RGB rendering: the
    // low-pass filter depends on this center even with opacity-aware bounds.
    const float cutoff = 3.0f;

    // Compute center and radius
    float2 point_image;
    float radius;
    {
        float2 extent;
        bool ok = compute_aabb(T, cutoff, point_image, extent);
        if (!ok) return;
        radius = ceil(max(max(extent.x, extent.y), cutoff * FilterSize));
    }

    uint2 rect_min, rect_max;
    getRect(point_image, radius, rect_min, rect_max, grid);
    if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
        return;
    if (tight_bbox || warp_culling)
        tighten_tile_rect(T, opacity, point_image, grid, rect_min, rect_max,
            tight_bbox, warp_culling ? pixel_bounds + idx : nullptr);
    tile_bounds[idx] = make_uint4(rect_min.x, rect_min.y, rect_max.x, rect_max.y);
    if (tile_culling) tile_supports[idx] = make_tile_support(T, opacity, point_image);

    depths[idx] = p_view.z;
    radii[idx] = (int)radius;
    points_xy_image[idx] = point_image;
    activated_opacities[idx] = opacity;
    tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Faster-GS collects visible IDs/depth keys and the instance total during
// preprocessing. Reserve one contiguous segment per nonempty warp instead of
// issuing two atomics per primitive. All lanes, including culled/out-of-range
// lanes, must participate after the tile-counting collectives have finished.
__device__ __forceinline__ void collect_visible_primitives(
    const uint32_t primitive, const uint32_t tile_count, const float depth,
    uint32_t* visible_ids, uint32_t* depth_keys, uint32_t* counts)
{
    constexpr uint32_t mask = 0xffffffffu;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t visible = __ballot_sync(mask, tile_count > 0);
    if (visible == 0) return;
    uint32_t warp_tiles = tile_count;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        warp_tiles += __shfl_down_sync(mask, warp_tiles, offset);
    uint32_t base = 0;
    if (lane == 0)
    {
        base = atomicAdd(counts + 1, __popc(visible));
        atomicAdd(counts, warp_tiles);
    }
    base = __shfl_sync(mask, base, 0);
    if (tile_count > 0)
    {
        const uint32_t rank = __popc(visible & ((1u << lane) - 1u));
        visible_ids[base + rank] = primitive;
        depth_keys[base + rank] = __float_as_uint(depth);
    }
}

template<bool TileCulling, bool LoadBalance, bool CollectVisibility>
__global__ void preprocessCUDA(int P, const float* orig_points,
    const glm::vec2* scales, const float scale_modifier,
    const glm::vec4* rotations, const float* opacities,
    const float* viewmatrix, const float* projmatrix,
    const int W, const int H, int* radii, float2* points_xy_image,
    float* depths, float* transMats, float* activated_opacities,
    const dim3 grid, uint32_t* tiles_touched, bool raw_inputs,
    uint4* tile_bounds, bool tight_bbox, TileSupport* tile_supports,
    int tile_culling_threshold, bool warp_culling, float4* pixel_bounds,
    uint32_t* visible_ids, uint32_t* visible_depth_keys,
    uint32_t* visibility_counts)
{
    preprocessPrimitive(P, orig_points, scales, scale_modifier, rotations,
        opacities, viewmatrix, projmatrix, W, H, radii, points_xy_image,
        depths, transMats, activated_opacities, grid, tiles_touched,
        raw_inputs, tile_bounds, tight_bbox, tile_supports, TileCulling,
        warp_culling, pixel_bounds);
    // Early exits live inside the per-primitive routine. Every lane still enters
    // enumeration, so partially occupied warps safely assist heavy primitives.
    if (TileCulling)
    {
        const uint32_t idx = cg::this_grid().thread_rank();
        const bool active = idx < P && radii[idx] > 0;
        const uint4 bounds = active ? tile_bounds[idx] : make_uint4(0, 0, 0, 0);
        const TileSupport support = active ? tile_supports[idx] : unrestricted_tile_support();
        const uint32_t count = enumerate_tiles<false>(idx, bounds, support, 0.0f,
            0, grid.x, active, true, LoadBalance, tile_culling_threshold);
        if (idx < P) tiles_touched[idx] = count;
    }
    if (CollectVisibility)
    {
        const uint32_t idx = cg::this_grid().thread_rank();
        const uint32_t count = idx < P ? tiles_touched[idx] : 0;
        // A positive radius is insufficient: exact culling may retain zero tiles.
        const float depth = count > 0 ? depths[idx] : 0.0f;
        collect_visible_primitives(idx, count, depth, visible_ids,
            visible_depth_keys, visibility_counts);
    }
}

// Match RGB visibility: projective surfel, low-pass filter, alpha cutoff
// and near-plane test. No Gabor color or auxiliary output is evaluated.
__device__ __forceinline__ bool surfel_contribution(
    const uint2 pix, const float2 xy, const float3 Tu, const float3 Tv,
    const float3 Tw, const float opacity, float2& s, float& depth, float& alpha)
{
    const float3 k = pix.x * Tw - Tu;
    const float3 l = pix.y * Tw - Tv;
    const float3 p = cross(k, l);
    if (p.z == 0.0f) return false;
    s = make_float2(p.x / p.z, p.y / p.z);
    const float rho3d = s.x * s.x + s.y * s.y;
    const float2 d = {xy.x - float(pix.x), xy.y - float(pix.y)};
    const float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y);
    const float rho = min(rho3d, rho2d);
    depth = rho3d <= rho2d ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z;
    if (depth < near_n) return false;
    const float power = -0.5f * rho;
    if (power > 0.0f) return false;
    alpha = min(0.99f, opacity * exp(power));
    if (alpha < 1.0f / 255.0f) return false;
    return true;
}

// Error projection needs visibility alone. No colors, harmonics, image outputs,
// differentiable checkpoints, normals, depth maps or distortion moments exist.
template<bool WarpCulling, bool Frontmost>
__global__ void __launch_bounds__(BLOCK_SIZE) projectErrorsCUDA(
    const uint2* __restrict__ ranges, const uint32_t* __restrict__ point_list,
    int W, int H, const float2* __restrict__ means2D,
    const float* __restrict__ transMats, const float* __restrict__ opacities,
    const float* __restrict__ error_map, float* __restrict__ gaussian_errors,
    float* __restrict__ gaussian_effects, const float* __restrict__ hair_vote_map,
    float* __restrict__ hair_votes, const float4* __restrict__ pixel_bounds)
{
    auto block = cg::this_thread_block();
    const uint32_t thread = block.thread_rank();
    const uint32_t lane = thread & 31;
    const uint32_t warp = thread / 32;
    const uint2 origin = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
    const uint2 local = WarpCulling
        ? make_uint2((warp % 2) * 8 + lane % 8, (warp / 2) * 4 + lane / 8)
        : make_uint2(block.thread_index().x, block.thread_index().y);
    const uint2 pix = {origin.x + local.x, origin.y + local.y};
    const uint2 subtile = {origin.x + (warp % 2) * 8, origin.y + (warp / 2) * 4};
    const bool inside = pix.x < W && pix.y < H;
    const uint32_t pixel = pix.y * W + pix.x;
    const uint2 range = ranges[block.group_index().y * ((W + BLOCK_X - 1) / BLOCK_X) + block.group_index().x];
    const int rounds = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;
    __shared__ int ids[BLOCK_SIZE];
    __shared__ float2 centers[BLOCK_SIZE];
    __shared__ float opacity[BLOCK_SIZE];
    __shared__ float3 Tu[BLOCK_SIZE], Tv[BLOCK_SIZE], Tw[BLOCK_SIZE];
    __shared__ float4 bounds[WarpCulling ? BLOCK_SIZE : 1];
    const float pixel_error = inside && error_map ? error_map[pixel] : 0.0f;
    const float hair = inside && hair_vote_map ? hair_vote_map[pixel] : 0.0f;
    const float confidence = inside && hair_vote_map ? hair_vote_map[H * W + pixel] : 0.0f;
    // Pixels with no observed error or hair evidence need no visibility work.
    bool done = !inside || (!error_map && !(confidence > 0.0f));
    float T = 1.0f;
    float closest_depth = INFINITY;
    int closest_id = -1;
    int remaining = range.y - range.x;
    for (int batch = 0; batch < rounds; ++batch, remaining -= BLOCK_SIZE)
    {
        if (__syncthreads_count(done) == BLOCK_SIZE) break;
        const uint32_t entry = range.x + batch * BLOCK_SIZE + thread;
        if (entry < range.y)
        {
            const int id = point_list[entry];
            ids[thread] = id;
            centers[thread] = means2D[id];
            opacity[thread] = opacities[id];
            Tu[thread] = {transMats[9 * id], transMats[9 * id + 1], transMats[9 * id + 2]};
            Tv[thread] = {transMats[9 * id + 3], transMats[9 * id + 4], transMats[9 * id + 5]};
            Tw[thread] = {transMats[9 * id + 6], transMats[9 * id + 7], transMats[9 * id + 8]};
            if (WarpCulling) bounds[thread] = pixel_bounds[id];
        }
        block.sync();
        uint32_t pending = 0;
        for (int j = 0; (WarpCulling || !done) && j < min(BLOCK_SIZE, remaining); ++j)
        {
            if (WarpCulling)
            {
                if ((j & 31) == 0)
                {
                    if (__all_sync(0xffffffffu, done)) break;
                    bool overlap = false;
                    if (j + lane < min(BLOCK_SIZE, remaining))
                    {
                        const float4 b = bounds[j + lane];
                        overlap = b.x <= subtile.x + 7 && b.z >= subtile.x &&
                            b.y <= subtile.y + 3 && b.w >= subtile.y;
                    }
                    pending = __ballot_sync(0xffffffffu, overlap);
                }
                if (pending == 0) { j |= 31; continue; }
                j = (j & ~31) + __ffs(pending) - 1;
                pending &= pending - 1;
                if (done) continue;
            }
            float2 s;
            float depth, alpha;
            if (!surfel_contribution(pix, centers[j], Tu[j], Tv[j], Tw[j], opacity[j], s, depth, alpha)) continue;
            const int id = ids[j];
            if constexpr (Frontmost)
            {
                // Sorting by mean depth does not order intersections of tilted surfels.
                if (isfinite(depth) && (closest_id < 0 || depth < closest_depth ||
                    (depth == closest_depth && id < closest_id)))
                {
                    closest_depth = depth;
                    closest_id = id;
                }
            }
            else
            {
                const float test_T = T * (1.0f - alpha);
                if (test_T < 0.0001f) { done = true; continue; }
                const float weight = alpha * T;
                if (error_map)
                {
                    atomicAdd(gaussian_errors + id, pixel_error * weight);
                    atomicAdd(gaussian_effects + id, weight);
                }
                if (hair_vote_map && confidence > 0.0f)
                {
                    atomicAdd(hair_votes + 2 * id, hair * weight);
                    atomicAdd(hair_votes + 2 * id + 1, confidence * weight);
                }
                T = test_T;
            }
        }
        // Even done/edge lanes must finish reading before the shared batch is reused.
        block.sync();
    }
    if constexpr (Frontmost)
    {
        if (inside && closest_id >= 0 && confidence > 0.0f)
        {
            atomicAdd(hair_votes + 2 * closest_id, hair);
            atomicAdd(hair_votes + 2 * closest_id + 1, confidence);
        }
    }
}

} // namespace

void PROJECTOR_FORWARD::projectErrors(const dim3 grid, dim3 block, const uint2* ranges,
    const uint32_t* point_list, int W, int H, const float2* means2D,
    const float* transMats, const float* opacities, const float* error_map,
    float* gaussian_errors, float* gaussian_effects, const float* hair_vote_map,
    float* hair_votes, bool frontmost_only, bool warp_culling, const float4* pixel_bounds)
{
    auto launch = [&](auto warp_mode, auto front_mode)
    {
        projectErrorsCUDA<decltype(warp_mode)::value, decltype(front_mode)::value><<<grid, block>>>(
            ranges, point_list, W, H, means2D, transMats, opacities, error_map,
            gaussian_errors, gaussian_effects, hair_vote_map, hair_votes, pixel_bounds);
    };
    if (frontmost_only)
    {
        if (warp_culling) launch(std::true_type{}, std::true_type{});
        else launch(std::false_type{}, std::true_type{});
    }
    else
    {
        if (warp_culling) launch(std::true_type{}, std::false_type{});
        else launch(std::false_type{}, std::false_type{});
    }
}

void PROJECTOR_FORWARD::preprocess(int P, const float* means3D,
    const glm::vec2* scales, float scale_modifier, const glm::vec4* rotations,
    const float* opacities, const float* viewmatrix, const float* projmatrix,
    int W, int H, int* radii, float2* means2D, float* depths,
    float* transMats, float* activated_opacities, const dim3 grid,
    uint32_t* tiles_touched, bool raw_inputs, uint4* tile_bounds,
    bool tight_bbox, TileSupport* tile_supports, bool tile_culling,
    bool load_balance, int tile_culling_threshold, bool warp_culling,
    float4* pixel_bounds, uint32_t* visible_ids,
    uint32_t* visible_depth_keys, uint32_t* visibility_counts)
{
    auto launch = [&](auto culling_mode, auto balance_mode, auto compact_mode)
    {
        preprocessCUDA<decltype(culling_mode)::value,
            decltype(balance_mode)::value, decltype(compact_mode)::value>
            <<<(P + 255) / 256, 256>>>(P, means3D, scales, scale_modifier,
                rotations, opacities, viewmatrix, projmatrix, W, H, radii,
                means2D, depths, transMats, activated_opacities, grid,
                tiles_touched, raw_inputs, tile_bounds, tight_bbox,
                tile_supports, tile_culling_threshold, warp_culling,
                pixel_bounds, visible_ids, visible_depth_keys, visibility_counts);
    };
    auto launch_culling = [&](auto culling_mode, auto balance_mode)
    {
        if (visibility_counts) launch(culling_mode, balance_mode, std::true_type{});
        else launch(culling_mode, balance_mode, std::false_type{});
    };
    if (tile_culling)
    {
        if (load_balance) launch_culling(std::true_type{}, std::true_type{});
        else launch_culling(std::true_type{}, std::false_type{});
    }
    else
    {
        if (load_balance) launch_culling(std::false_type{}, std::true_type{});
        else launch_culling(std::false_type{}, std::false_type{});
    }
}
