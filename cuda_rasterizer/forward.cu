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
#include "auxiliary.h"
#include "tile_culling.h"
#include <type_traits>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const float* shs_rest, bool split_sh, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	const auto sh = sh_coefficients((const glm::vec3*)shs, (const glm::vec3*)shs_rest, idx, max_coeffs, split_sh);
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

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

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__device__ __forceinline__ void preprocessPrimitive(int P, int D, int M,
	const float* orig_points,
	const float* frequency_vectors,

	const float* phases, 
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
	const float tan_fovx, const float tan_fovy,
	const float focal_x, const float focal_y,
	int* radii,
	float2* points_xy_image,
	float* geo_frequency_vectors,

	float* geo_phases, 
	float* depths,
	float* transMats,
	float* rgb,
	float* rgb2,
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
	bool warp_culling,
	float4* pixel_bounds)
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
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;
	
	// Compute transformation matrix
	glm::mat3 T;
	glm::mat4 object2world_mat;
	glm::mat4 world2ndc_mat;
	glm::mat3x4 ndc2pix_mat;
	float3 normal;
	if (transMat_precomp == nullptr)
	{
		const glm::vec2 scale = raw_inputs ? activate_scale(scales[idx]) : scales[idx];
		const glm::vec4 rotation = raw_inputs ? activate_rotation(rotations[idx]) : rotations[idx];
		compute_transmat(((float3*)orig_points)[idx], scale, scale_modifier, rotation, projmatrix, viewmatrix, W, H, T, normal);
		float3 *T_ptr = (float3*)transMats;
		T_ptr[idx * 3 + 0] = {T[0][0], T[0][1], T[0][2]};
		T_ptr[idx * 3 + 1] = {T[1][0], T[1][1], T[1][2]};
		T_ptr[idx * 3 + 2] = {T[2][0], T[2][1], T[2][2]};
	} else {
		glm::vec3 *T_ptr = (glm::vec3*)transMat_precomp;
		T = glm::mat3(
			T_ptr[idx * 3 + 0], 
			T_ptr[idx * 3 + 1],
			T_ptr[idx * 3 + 2]
		);
		normal = make_float3(0.0, 0.0, 1.0);
	}

#if DUAL_VISIABLE
	float cos = -sumf3(p_view * normal);
	if (cos == 0) return;
	float multiplier = cos > 0 ? 1: -1;
	normal = multiplier * normal;
#endif

	const float opacity = raw_inputs ? activate_opacity(opacities[idx]) : opacities[idx];

	// Keep this center and radius unchanged: the pixel filter and its backward
	// derivative use the 3-sigma center, and densification consumes the radius.
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

	// Compute colors 
	if (colors_precomp == nullptr) {
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, shs_rest, split_sh, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}
	if (colors_precomp2 == nullptr) {
		glm::vec3 result2 = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs2, shs2_rest, split_sh, clamped2);
		rgb2[idx * C + 0] = result2.x;
		rgb2[idx * C + 1] = result2.y;
		rgb2[idx * C + 2] = result2.z;
	} 

	geo_frequency_vectors[2 * idx] = frequency_vectors[2 * idx];
	geo_frequency_vectors[2 * idx + 1] = frequency_vectors[2 * idx + 1];
	geo_phases[idx] = phases[idx];

	depths[idx] = p_view.z;
	radii[idx] = (int)radius;
	points_xy_image[idx] = point_image;
	normal_opacity[idx] = {normal.x, normal.y, normal.z, opacity};
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

template<int C, bool TileCulling, bool LoadBalance, bool CollectVisibility>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const float* frequency_vectors,

	const float* phases,
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
	const float tan_fovx, const float tan_fovy,
	const float focal_x, const float focal_y,
	int* radii,
	float2* points_xy_image,
	float* geo_frequency_vectors,

	float* geo_phases,
	float* depths,
	float* transMats,
	float* rgb,
	float* rgb2,
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
	bool /*tile_culling*/,
	bool /*load_balance*/,
	int tile_culling_threshold,
	bool warp_culling,
	float4* pixel_bounds,
	uint32_t* visible_ids,
	uint32_t* visible_depth_keys,
	uint32_t* visibility_counts)
{
	preprocessPrimitive<C>(
		P, D, M, orig_points, frequency_vectors,
		phases, scales, scale_modifier, rotations, opacities, shs, shs2,
		clamped, clamped2, transMat_precomp, colors_precomp, colors_precomp2,
		viewmatrix, projmatrix, cam_pos, W, H, tan_fovx, tan_fovy, focal_x,
		focal_y, radii, points_xy_image, geo_frequency_vectors,
		 geo_phases, depths, transMats, rgb, rgb2,
		normal_opacity, grid, tiles_touched, prefiltered, shs_rest, shs2_rest,
		split_sh, raw_inputs, tile_bounds, tight_bbox, tile_supports, TileCulling,
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

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS, bool GaussianBackward, bool WarpCulling, bool HasSpecular, bool HasAttributes>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ geo_frequency_vectors,

	const float* __restrict__ geo_phases, 
	const float* __restrict__ features,
	const float* __restrict__ features2,
	const float* __restrict__ transMats,
	const float* __restrict__ depths,
	const float4* __restrict__ normal_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_others,
	const uint32_t* __restrict__ bucket_offsets,
	uint32_t* __restrict__ bucket_to_tile,
	uint32_t* __restrict__ tile_max_contrib,
	float4* __restrict__ bucket_color_T,
	float4* __restrict__ bucket_depth_normal,
	float2* __restrict__ bucket_moments,
	const float4* __restrict__ pixel_bounds,
	const float* __restrict__ specular,
	const float* __restrict__ attributes)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint32_t thread = block.thread_rank();
	const uint32_t lane = thread & 31;
	const uint32_t warp = thread / 32;
	const uint2 local_pixel = WarpCulling
		? make_uint2((warp % 2) * 8 + lane % 8, (warp / 2) * 4 + lane / 8)
		: make_uint2(block.thread_index().x, block.thread_index().y);
	const uint32_t pixel_slot = local_pixel.y * BLOCK_X + local_pixel.x;
	uint2 pix = { pix_min.x + local_pixel.x, pix_min.y + local_pixel.y };
	const uint2 subtile_min = make_uint2(pix_min.x + (warp % 2) * 8, pix_min.y + (warp / 2) * 4);
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y};

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const uint32_t tile = block.group_index().y * horizontal_blocks + block.group_index().x;
	uint32_t first_bucket = 0;
	if (GaussianBackward)
	{
		first_bucket = tile ? bucket_offsets[tile - 1] : 0;
		const uint32_t buckets = (range.y - range.x + 31) / 32;
		// Initialize even buckets that no pixel reaches due to early termination.
		for (uint32_t b = block.thread_rank(); b < buckets; b += BLOCK_SIZE)
			bucket_to_tile[first_bucket + b] = tile;
	}
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	// __shared__ float2 collected_frequency[BLOCK_SIZE];
	// __shared__ float collected_wave_length_in_pix[BLOCK_SIZE];
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];
	__shared__ float4 collected_bounds[WarpCulling ? BLOCK_SIZE : 1];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	float attributes_accum[HasAttributes ? 6 : 1] = { 0 };


#if RENDER_AXUTILITY
	// render axutility ouput
	float N[3] = {0};
	float D = { 0 };
	float M1 = {0};
	float M2 = {0};
	float distortion = {0};
	float median_depth = {0};
	// float median_weight = {0};
	float median_contributor = {-1};

#endif

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			// collected_frequency[block.thread_rank()] = frequency[coll_id];
			// collected_wave_length_in_pix[block.thread_rank()] = wave_length_in_pix[coll_id];
			collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id+0], transMats[9 * coll_id+1], transMats[9 * coll_id+2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id+3], transMats[9 * coll_id+4], transMats[9 * coll_id+5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};
			if (WarpCulling) collected_bounds[thread] = pixel_bounds[coll_id];
		}
		block.sync();

		// Iterate over current batch
		uint32_t pending = 0;
		for (int j = 0; (WarpCulling || !done) && j < min(BLOCK_SIZE, toDo); j++)
		{
			// All lanes, including terminated/edge pixels, reach each ballot.
			// Warp-wide work sharing follows Faster-GS (Apache-2.0; see
			// THIRD_PARTY_LICENSES/Faster-GS-LICENSE), adapted to 2DGS support.
			if (WarpCulling && (j & 31) == 0 && __all_sync(0xffffffffu, done)) break;
			if (GaussianBackward && !done && (j & 31) == 0)
			{
				const size_t b = first_bucket + (i * BLOCK_SIZE + j) / 32;
				const size_t snapshot = b * BLOCK_SIZE + pixel_slot;
				bucket_color_T[snapshot] = make_float4(C[0], C[1], C[2], T);
#if RENDER_AXUTILITY
				bucket_depth_normal[snapshot] = make_float4(D, N[0], N[1], N[2]);
				bucket_moments[snapshot] = make_float2(M1, M2);
#endif
			}
			if (WarpCulling)
			{
				if ((j & 31) == 0)
				{
					bool overlaps = false;
					if (j + lane < min(BLOCK_SIZE, toDo))
					{
						const float4 bounds = collected_bounds[j + lane];
						overlaps = bounds.x <= subtile_min.x + 7 && bounds.z >= subtile_min.x &&
							bounds.y <= subtile_min.y + 3 && bounds.w >= subtile_min.y;
					}
					pending = __ballot_sync(0xffffffffu, overlaps);
				}
				if (pending == 0)
				{
					j |= 31;
					continue;
				}
				j = (j & ~31) + __ffs(pending) - 1;
				pending &= pending - 1;
				if (done) continue;
			}
			// Keep track of current position in range
			contributor = i * BLOCK_SIZE + j + 1;

			// Fisrt compute two homogeneous planes, See Eq. (8)
			const float2 xy = collected_xy[j];
			// const float2 frequency_vec = collected_frequency[j];
			// const float wave_length = collected_wave_length_in_pix[j];
			const float3 Tu = collected_Tu[j];
			const float3 Tv = collected_Tv[j];
			const float3 Tw = collected_Tw[j];
			float3 k = pix.x * Tw - Tu;
			float3 l = pix.y * Tw - Tv;
			float3 p = cross(k, l);
			if (p.z == 0.0) continue;
			float2 s = {p.x / p.z, p.y / p.z};
			float rho3d = (s.x * s.x + s.y * s.y); 
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y); 

			// compute intersection and depth
			float rho = min(rho3d, rho2d);
			float depth = (rho3d <= rho2d) ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z; 
			if (depth < near_n) continue;
			float4 nor_o = collected_normal_opacity[j];
			float normal[3] = {nor_o.x, nor_o.y, nor_o.z};
			float opa = nor_o.w;

			float power = -0.5f * rho;
			if (power > 0.0f)
				continue;
			// for easiness of visualization
			// if (rho >= 1.0)
			// 	continue;
			
			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, opa * exp(power)); // harmonic1 + harmonic2 = 1
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Alpha and termination do not depend on Gabor color. Evaluate
			// harmonics only for contributions that reach RGB accumulation.
			const int id = collected_id[j];
			const float argument = 2.0f * PI * (geo_frequency_vectors[2 * id] * s.x
				+ geo_frequency_vectors[2 * id + 1] * s.y) + geo_phases[id];
			const float harmonic1 = 0.5f * (1.0f + __cosf(argument));
			const float harmonic2 = 1.0f - harmonic1;

			float w = alpha * T;
#if RENDER_AXUTILITY
			// Render depth distortion map
			// Efficient implementation of distortion loss, see 2DGS' paper appendix.
			float A = 1-T;
			float m = far_n / (far_n - near_n) * (1 - near_n / depth);
			distortion += (m * m * A + M2 - 2 * m * M1) * w;
			D  += depth * w;
			M1 += m * w;
			M2 += m * m * w;

			if (T > 0.5) {
				median_depth = depth;
				// median_weight = w;
				median_contributor = contributor;
			}
			// Render normal map
			for (int ch=0; ch<3; ch++) N[ch] += normal[ch] * w;
#endif

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++) {
				C[ch] += features[collected_id[j] * CHANNELS + ch] * w * harmonic1;
				C[ch] += features2[collected_id[j] * CHANNELS + ch] * w * harmonic2;
				if constexpr (HasSpecular)
					C[ch] += specular[collected_id[j] * CHANNELS + ch] * w;
				// C[ch] += features[collected_id[j] * CHANNELS + ch] * w;
			}
			if constexpr (HasAttributes)
			{
#pragma unroll
				for (int ch = 0; ch < 6; ++ch)
					attributes_accum[ch] += attributes[collected_id[j] * 6 + ch] * w;
			}
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}
	if (GaussianBackward)
	{
		__shared__ uint32_t warp_max[BLOCK_SIZE / 32];
		uint32_t value = inside ? last_contributor : 0;
		for (int offset = 16; offset > 0; offset /= 2)
			value = max(value, __shfl_down_sync(0xffffffffu, value, offset));
		if ((block.thread_rank() & 31) == 0)
			warp_max[block.thread_rank() / 32] = value;
		block.sync();
		if (block.thread_rank() == 0)
		{
			value = 0;
			for (int warp = 0; warp < BLOCK_SIZE / 32; ++warp)
				value = max(value, warp_max[warp]);
			tile_max_contrib[tile] = value;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		if constexpr (HasAttributes)
		{
#pragma unroll
			for (int ch = 0; ch < 6; ++ch)
				out_others[(7 + ch) * H * W + pix_id] = attributes_accum[ch];
		}

#if RENDER_AXUTILITY
		n_contrib[pix_id + H * W] = median_contributor;
		final_T[pix_id + H * W] = M1;
		final_T[pix_id + 2 * H * W] = M2;
		out_others[pix_id + DEPTH_OFFSET * H * W] = D;
		out_others[pix_id + ALPHA_OFFSET * H * W] = 1 - T;
		for (int ch=0; ch<3; ch++) out_others[pix_id + (NORMAL_OFFSET+ch) * H * W] = N[ch];
		out_others[pix_id + MIDDEPTH_OFFSET * H * W] = median_depth;
		out_others[pix_id + DISTORTION_OFFSET * H * W] = distortion;
		// out_others[pix_id + MEDIAN_WEIGHT_OFFSET * H * W] = median_weight;
#endif
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float2* means2D,
	const float* geo_frequency_vectors,

	const float* geo_phases, 
	const float* colors,
	const float* colors2,
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
	const float4* pixel_bounds,
	const float* specular,
	const float* attributes)
{
	auto launch = [&](auto gaussian_mode, auto warp_mode, auto specular_mode, auto attributes_mode)
	{
	renderCUDA<NUM_CHANNELS, decltype(gaussian_mode)::value, decltype(warp_mode)::value, decltype(specular_mode)::value, decltype(attributes_mode)::value> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		focal_x, focal_y,
		means2D,
		geo_frequency_vectors,

		geo_phases, 
		colors,
		colors2,
		transMats,
		depths,
		normal_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		out_others,
		bucket_offsets,
		bucket_to_tile,
		tile_max_contrib,
		bucket_color_T,
		bucket_depth_normal,
		bucket_moments,
		pixel_bounds, specular, attributes);
	};
	auto launch_mode = [&](auto gaussian_mode, auto warp_mode)
	{
		auto launch_attributes = [&](auto specular_mode)
		{
			if (attributes) launch(gaussian_mode, warp_mode, specular_mode, std::true_type{});
			else launch(gaussian_mode, warp_mode, specular_mode, std::false_type{});
		};
		if (specular) launch_attributes(std::true_type{});
		else launch_attributes(std::false_type{});
	};
	if (gaussian_backward)
	{
		if (warp_culling) launch_mode(std::true_type{}, std::true_type{});
		else launch_mode(std::true_type{}, std::false_type{});
	}
	else
	{
		if (warp_culling) launch_mode(std::false_type{}, std::true_type{});
		else launch_mode(std::false_type{}, std::false_type{});
	}
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const float* frequency_vectors,

	const float* phases, 
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
	const int W, const int H,
	const float focal_x, const float focal_y,
	const float tan_fovx, const float tan_fovy,
	int* radii,
	float2* means2D,
	float* geo_frequency_vectors,

	float* geo_phases, 
	float* depths,
	float* transMats,
	float* rgb,
	float* rgb2,
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
	uint32_t* visibility_counts)
{
	auto launch = [&](auto culling_mode, auto balance_mode, auto compact_mode)
	{
		preprocessCUDA<NUM_CHANNELS, decltype(culling_mode)::value,
			decltype(balance_mode)::value, decltype(compact_mode)::value> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		frequency_vectors,

		phases, 
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		shs2,
		clamped,
		clamped2,
		transMat_precomp,
		colors_precomp,
		colors_precomp2,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		geo_frequency_vectors,

		geo_phases, 
		depths,
		transMats,
		rgb,
		rgb2,
		normal_opacity,
		grid,
		tiles_touched,
		prefiltered,
		shs_rest,
		shs2_rest,
		split_sh,
		raw_inputs,
		tile_bounds,
		tight_bbox,
		tile_supports,
		tile_culling,
		load_balance,
		tile_culling_threshold,
		warp_culling,
		pixel_bounds,
		visible_ids,
		visible_depth_keys,
		visibility_counts
		);
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
