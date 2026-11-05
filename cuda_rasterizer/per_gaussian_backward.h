/*
 * Per-Gaussian backward rendering, adapted to projective 2D Gabor surfels.
 *
 * The 32-instance bucket and systolic warp traversal follow Faster-GS
 * (FasterGSCudaBackend, rasterization/include/kernels_backward.cuh). Each
 * lane owns one sorted Gaussian/tile instance and accumulates its gradients
 * in registers across every pixel in that tile. Global atomics are issued
 * once per instance rather than once per contributing pixel/instance pair.
 * The surrounding renderer retains its original research license.
 * Faster-GS reference: Copyright 2026 The Faster-GS Authors, Apache-2.0;
 * see THIRD_PARTY_LICENSES/Faster-GS-LICENSE in this extension's root.
 */
#ifndef CUDA_RASTERIZER_PER_GAUSSIAN_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_PER_GAUSSIAN_BACKWARD_H_INCLUDED

namespace
{
constexpr int GAUSSIAN_BACKWARD_WARPS = 4;

// Pixel state moves from the first to the last Gaussian in a bucket. Combining
// the loss-weighted RGB, depth, normal and distortion tails saves carrying all
// nine separate accumulated outputs through the warp.
struct GaussianBackwardPixel
{
	float T;
	float remaining_loss;
	float3 grad_rgb;
	float grad_depth;
	float3 grad_normal;
	float grad_median;
	float distortion_A;
	float distortion_D;
	float distortion_D2;
	float alpha_common;
	uint32_t last_contributor;
	uint32_t median_contributor;
};
static_assert(sizeof(GaussianBackwardPixel) == 64, "Keep the pixel pipeline compact");

__forceinline__ __device__ GaussianBackwardPixel shuffle_gaussian_pixel(
	GaussianBackwardPixel state)
{
	constexpr unsigned mask = 0xffffffffu;
	state.T = __shfl_up_sync(mask, state.T, 1);
	state.remaining_loss = __shfl_up_sync(mask, state.remaining_loss, 1);
	state.grad_rgb.x = __shfl_up_sync(mask, state.grad_rgb.x, 1);
	state.grad_rgb.y = __shfl_up_sync(mask, state.grad_rgb.y, 1);
	state.grad_rgb.z = __shfl_up_sync(mask, state.grad_rgb.z, 1);
	state.grad_depth = __shfl_up_sync(mask, state.grad_depth, 1);
	state.grad_normal.x = __shfl_up_sync(mask, state.grad_normal.x, 1);
	state.grad_normal.y = __shfl_up_sync(mask, state.grad_normal.y, 1);
	state.grad_normal.z = __shfl_up_sync(mask, state.grad_normal.z, 1);
	state.grad_median = __shfl_up_sync(mask, state.grad_median, 1);
	state.distortion_A = __shfl_up_sync(mask, state.distortion_A, 1);
	state.distortion_D = __shfl_up_sync(mask, state.distortion_D, 1);
	state.distortion_D2 = __shfl_up_sync(mask, state.distortion_D2, 1);
	state.alpha_common = __shfl_up_sync(mask, state.alpha_common, 1);
	state.last_contributor = __shfl_up_sync(mask, state.last_contributor, 1);
	state.median_contributor = __shfl_up_sync(mask, state.median_contributor, 1);
	return state;
}

__forceinline__ __device__ GaussianBackwardPixel load_gaussian_pixel(
	int pixel_in_tile, uint2 tile_origin, uint32_t first_instance,
	uint32_t bucket_idx, int W, int H,
	const float* bg_color, const float* final_Ts,
	const uint32_t* n_contrib,
	const float* dL_dpixels, const float* dL_depths,
	const float4* bucket_color_T, const float4* bucket_depth_normal,
	const float2* bucket_moments, const float* final_image,
	const float* final_aux)
{
	GaussianBackwardPixel state = {};
	const int x = tile_origin.x + pixel_in_tile % BLOCK_X;
	const int y = tile_origin.y + pixel_in_tile / BLOCK_X;
	if (x >= W || y >= H)
		return state;
	const int n_pixels = W * H;
	const int pixel = y * W + x;
	state.last_contributor = n_contrib[pixel];
	// Forward stops saving checkpoints once a pixel is done. In particular,
	// do not read an uninitialized snapshot for an occluded bucket.
	if (state.last_contributor <= first_instance)
		return state;

	const size_t snapshot_idx = size_t(bucket_idx) * BLOCK_SIZE + pixel_in_tile;
	const float4 color_T = bucket_color_T[snapshot_idx];
	const float4 depth_normal = bucket_depth_normal[snapshot_idx];
	const float2 moments = bucket_moments[snapshot_idx];
	const float final_T = final_Ts[pixel];
	const float final_D = final_Ts[n_pixels + pixel];
	const float final_D2 = final_Ts[2 * n_pixels + pixel];
	const float final_A = 1.0f - final_T;
	state.T = color_T.w;
	state.grad_rgb = {
		dL_dpixels[pixel],
		dL_dpixels[n_pixels + pixel],
		dL_dpixels[2 * n_pixels + pixel]
	};
	state.grad_depth = dL_depths[DEPTH_OFFSET * n_pixels + pixel];
	state.grad_normal = {
		dL_depths[(NORMAL_OFFSET + 0) * n_pixels + pixel],
		dL_depths[(NORMAL_OFFSET + 1) * n_pixels + pixel],
		dL_depths[(NORMAL_OFFSET + 2) * n_pixels + pixel]
	};
	state.grad_median = dL_depths[MIDDEPTH_OFFSET * n_pixels + pixel];
	state.median_contributor = n_contrib[n_pixels + pixel];
	const float distortion_gradient = dL_depths[DISTORTION_OFFSET * n_pixels + pixel];
	state.distortion_A = final_A * distortion_gradient;
	state.distortion_D = final_D * distortion_gradient;
	state.distortion_D2 = final_D2 * distortion_gradient;

	const float tail_r = final_image[pixel] - final_T * bg_color[0] - color_T.x;
	const float tail_g = final_image[n_pixels + pixel] - final_T * bg_color[1] - color_T.y;
	const float tail_b = final_image[2 * n_pixels + pixel] - final_T * bg_color[2] - color_T.z;
	state.remaining_loss = tail_r * state.grad_rgb.x + tail_g * state.grad_rgb.y + tail_b * state.grad_rgb.z;
	state.remaining_loss += (final_aux[DEPTH_OFFSET * n_pixels + pixel] - depth_normal.x) * state.grad_depth;
	state.remaining_loss += (final_aux[(NORMAL_OFFSET + 0) * n_pixels + pixel] - depth_normal.y) * state.grad_normal.x;
	state.remaining_loss += (final_aux[(NORMAL_OFFSET + 1) * n_pixels + pixel] - depth_normal.z) * state.grad_normal.y;
	state.remaining_loss += (final_aux[(NORMAL_OFFSET + 2) * n_pixels + pixel] - depth_normal.w) * state.grad_normal.z;
#if !DETACH_WEIGHT
	// Sum_{j in the bucket and behind it} w_j * d(distortion)/dw_j.
	state.remaining_loss += state.distortion_D2 * (color_T.w - final_T)
		+ state.distortion_A * (final_D2 - moments.y)
		- 2.0f * state.distortion_D * (final_D - moments.x);
#endif
	const float background_gradient = bg_color[0] * state.grad_rgb.x
		+ bg_color[1] * state.grad_rgb.y + bg_color[2] * state.grad_rgb.z;
	state.alpha_common = final_T * (dL_depths[ALPHA_OFFSET * n_pixels + pixel] - background_gradient);
	return state;
}

template <bool HasSpecular = false>
__global__ void __launch_bounds__(32 * GAUSSIAN_BACKWARD_WARPS)
renderGaussianCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ means2D,
	const float* __restrict__ frequency_vectors,

	const float* __restrict__ phases,
	const float4* __restrict__ normal_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ colors2,
	const float* __restrict__ transMats,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_depths,
	float* __restrict__ dL_dtransMat,
	float3* __restrict__ dL_dmean2D,
	float* __restrict__ dL_dnormal3D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dfrequency_vectors,

	float* __restrict__ dL_dphases,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_dcolors2,
	int bucket_capacity, int num_tiles,
	const uint32_t* __restrict__ bucket_offsets,
	const uint32_t* __restrict__ bucket_to_tile,
	const uint32_t* __restrict__ tile_max_contrib,
	const float4* __restrict__ bucket_color_T,
	const float4* __restrict__ bucket_depth_normal,
	const float2* __restrict__ bucket_moments,
	const float* __restrict__ final_image,
	const float* __restrict__ final_aux,
	const float* __restrict__ specular,
	float* __restrict__ dL_dspecular)
{
	const unsigned lane = threadIdx.x & 31;
	const unsigned warp_in_block = threadIdx.x / 32;
	const unsigned bucket_idx = blockIdx.x * GAUSSIAN_BACKWARD_WARPS + warp_in_block;
	// Every guard is uniform within its warp: all shuffle participants survive.
	if (bucket_idx >= bucket_capacity || num_tiles == 0
		|| bucket_idx >= bucket_offsets[num_tiles - 1])
		return;
	const unsigned tile = bucket_to_tile[bucket_idx];
	const unsigned first_bucket = tile ? bucket_offsets[tile - 1] : 0;
	const unsigned first_instance = (bucket_idx - first_bucket) * 32;
	if (first_instance >= tile_max_contrib[tile])
		return;
	const uint2 range = ranges[tile];
	const unsigned instance = first_instance + lane;
	const bool valid_primitive = instance < range.y - range.x;
	unsigned primitive = 0;
	float2 xy = {};
	float4 normal_opa = {};
	float3 Tu = {}, Tv = {}, Tw = {};
	float color1[3] = {}, color2[3] = {};
	float specular_color[3] = {};
	float frequency_x = 0.f, frequency_y = 0.f, phase = 0.f;
	if (valid_primitive)
	{
		primitive = point_list[range.x + instance];
		xy = means2D[primitive];
		normal_opa = normal_opacity[primitive];
		Tu = {transMats[9 * primitive], transMats[9 * primitive + 1], transMats[9 * primitive + 2]};
		Tv = {transMats[9 * primitive + 3], transMats[9 * primitive + 4], transMats[9 * primitive + 5]};
		Tw = {transMats[9 * primitive + 6], transMats[9 * primitive + 7], transMats[9 * primitive + 8]};
		#pragma unroll
		for (int ch = 0; ch < 3; ++ch)
		{
			color1[ch] = colors[3 * primitive + ch];
			color2[ch] = colors2[3 * primitive + ch];
			if constexpr (HasSpecular) specular_color[ch] = specular[3 * primitive + ch];
		}
		frequency_x = frequency_vectors[2 * primitive];
		frequency_y = frequency_vectors[2 * primitive + 1];
		phase = phases[primitive];
	}

	float grad_color1[3] = {}, grad_color2[3] = {};
	float grad_specular[3] = {};
	float grad_frequency_x = 0.f, grad_frequency_y = 0.f, grad_phase = 0.f;
	float grad_T[9] = {};
	float3 grad_normal = {};
	float2 grad_mean = {};
	float grad_opacity = 0.0f;
	bool contributed = false;
	const unsigned grid_width = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 tile_origin = {(tile % grid_width) * BLOCK_X, (tile / grid_width) * BLOCK_Y};
	__shared__ GaussianBackwardPixel collected[GAUSSIAN_BACKWARD_WARPS][32];
	GaussianBackwardPixel state = {};

	// A lane retains its Gaussian while pixels advance one lane per cycle.
	// Every 32 cycles a coalesced warp load fills the next shared-memory group.
	for (int cycle = 0; cycle < BLOCK_SIZE + 31; ++cycle)
	{
		if ((cycle & 31) == 0 && cycle < BLOCK_SIZE)
		{
			// The leader may still be reading the previous group's final
			// shared entry on GPUs with independent thread scheduling.
			__syncwarp();
			collected[warp_in_block][lane] = load_gaussian_pixel(
				cycle + lane, tile_origin, first_instance, bucket_idx, W, H,
				bg_color, final_Ts, n_contrib, dL_dpixels, dL_depths,
				bucket_color_T, bucket_depth_normal, bucket_moments,
				final_image, final_aux);
			__syncwarp();
		}
		if (cycle)
			state = shuffle_gaussian_pixel(state);
		if (lane == 0 && cycle < BLOCK_SIZE)
			state = collected[warp_in_block][cycle & 31];
		const int pixel_in_tile = cycle - int(lane);
		if (!valid_primitive || pixel_in_tile < 0 || pixel_in_tile >= BLOCK_SIZE
			|| instance >= state.last_contributor)
			continue;
		const float px = float(tile_origin.x + pixel_in_tile % BLOCK_X);
		const float py = float(tile_origin.y + pixel_in_tile / BLOCK_X);
		const float3 k = px * Tw - Tu;
		const float3 l = py * Tw - Tv;
		const float3 p = cross(k, l);
		if (p.z == 0.0f)
			continue;
		const float2 s = {p.x / p.z, p.y / p.z};
		const float rho3d = s.x * s.x + s.y * s.y;
		const float2 delta = {xy.x - px, xy.y - py};
		const float rho2d = FilterInvSquare * (delta.x * delta.x + delta.y * delta.y);
		const float depth = rho3d <= rho2d ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z;
		if (depth < near_n)
			continue;
		const float power = -0.5f * min(rho3d, rho2d);
		if (power > 0.0f)
			continue;
		const float gaussian = exp(power);
		const float alpha = min(0.99f, normal_opa.w * gaussian);
		if (alpha < 1.0f / 255.0f)
			continue;
		contributed = true;
		const float weight = alpha * state.T;
		const float argument = 2.0f * PI * (frequency_x * s.x + frequency_y * s.y) + phase;
		const float sine = __sinf(argument);
		const float harmonic1 = 0.5f * (1.f + __cosf(argument));
		const float harmonic2 = 1.f - harmonic1;
		const float pixel_gradient[3] = {state.grad_rgb.x, state.grad_rgb.y, state.grad_rgb.z};
		float local_loss = 0.0f;
		#pragma unroll
		for (int ch = 0; ch < 3; ++ch)
		{
			float color = harmonic1 * color1[ch] + harmonic2 * color2[ch];
			if constexpr (HasSpecular)
			{
				color += specular_color[ch];
				grad_specular[ch] += pixel_gradient[ch] * weight;
			}
			local_loss += color * pixel_gradient[ch];
			grad_color1[ch] += pixel_gradient[ch] * weight * harmonic1;
			grad_color2[ch] += pixel_gradient[ch] * weight * harmonic2;
		}
		local_loss += depth * state.grad_depth;
		local_loss += normal_opa.x * state.grad_normal.x + normal_opa.y * state.grad_normal.y + normal_opa.z * state.grad_normal.z;
		const float m = far_n / (far_n - near_n) * (1.0f - near_n / depth);
#if !DETACH_WEIGHT
		local_loss += state.distortion_D2 + m * m * state.distortion_A - 2.0f * m * state.distortion_D;
#endif
		state.remaining_loss -= weight * local_loss;
		// There is no suffix behind the last accepted instance. Enforce this
		// identity instead of amplifying accumulated subtraction roundoff.
		if (instance + 1 == state.last_contributor)
			state.remaining_loss = 0.0f;
		const float one_minus_alpha = 1.0f - alpha;
		const float grad_alpha = state.T * local_loss
			+ (state.alpha_common - state.remaining_loss) / one_minus_alpha;
		const float grad_gaussian = normal_opa.w * grad_alpha;
		grad_opacity += gaussian * grad_alpha;
		grad_normal.x += weight * state.grad_normal.x;
		grad_normal.y += weight * state.grad_normal.y;
		grad_normal.z += weight * state.grad_normal.z;
		const float dmd_dd = far_n * near_n / ((far_n - near_n) * depth * depth);
		float grad_depth = weight * state.grad_depth
			+ 2.0f * weight * (m * state.distortion_A - state.distortion_D) * dmd_dd;
		if (instance + 1 == state.median_contributor)
			grad_depth += state.grad_median;
		const float part2 = pixel_gradient[0] * (color1[0] - color2[0])
			+ pixel_gradient[1] * (color1[1] - color2[1])
			+ pixel_gradient[2] * (color1[2] - color2[2]);
		const float vector_gradient = -part2 * PI * sine * weight;
		grad_frequency_x += vector_gradient * s.x;
		grad_frequency_y += vector_gradient * s.y;
		grad_phase += -.5f * part2 * sine * weight;
		const float tmp_dsx = sine * frequency_x, tmp_dsy = sine * frequency_y;
		{
			const bool projected_envelope = rho3d <= rho2d;
			if (!projected_envelope) {
				grad_mean.x += grad_gaussian * (-gaussian * FilterInvSquare * delta.x);
				grad_mean.y += grad_gaussian * (-gaussian * FilterInvSquare * delta.y);
			}
			const float2 grad_s = {
				weight * -part2 * PI * tmp_dsx + (projected_envelope ? grad_gaussian * -gaussian * s.x + grad_depth * Tw.x : 0.f),
				weight * -part2 * PI * tmp_dsy + (projected_envelope ? grad_gaussian * -gaussian * s.y + grad_depth * Tw.y : 0.f)
			};
			const float dsx_pz = grad_s.x / p.z, dsy_pz = grad_s.y / p.z;
			const float3 grad_p = {dsx_pz, dsy_pz, -(dsx_pz * s.x + dsy_pz * s.y)};
			const float3 grad_k = cross(l, grad_p), grad_l = cross(grad_p, k);
			grad_T[0] += -grad_k.x;
			grad_T[1] += -grad_k.y;
			grad_T[2] += -grad_k.z;
			grad_T[3] += -grad_l.x;
			grad_T[4] += -grad_l.y;
			grad_T[5] += -grad_l.z;
			grad_T[6] += px * grad_k.x + py * grad_l.x + (projected_envelope ? grad_depth * s.x : 0.f);
			grad_T[7] += px * grad_k.y + py * grad_l.y + (projected_envelope ? grad_depth * s.y : 0.f);
			grad_T[8] += px * grad_k.z + py * grad_l.z + grad_depth;
		}
		state.T *= one_minus_alpha;
	}

	if (valid_primitive && contributed)
	{
		#pragma unroll
		for (int ch = 0; ch < 3; ++ch)
		{
			atomicAdd(dL_dcolors + primitive * 3 + ch, grad_color1[ch]);
			atomicAdd(dL_dcolors2 + primitive * 3 + ch, grad_color2[ch]);
			if constexpr (HasSpecular)
				atomicAdd(dL_dspecular + primitive * 3 + ch, grad_specular[ch]);
		}
		atomicAdd(dL_dfrequency_vectors + 2 * primitive, grad_frequency_x);
		atomicAdd(dL_dfrequency_vectors + 2 * primitive + 1, grad_frequency_y);
		atomicAdd(dL_dphases + primitive, grad_phase);
		#pragma unroll
		for (int coordinate = 0; coordinate < 9; ++coordinate)
			atomicAdd(dL_dtransMat + primitive * 9 + coordinate, grad_T[coordinate]);
		atomicAdd(&dL_dmean2D[primitive].x, grad_mean.x);
		atomicAdd(&dL_dmean2D[primitive].y, grad_mean.y);
		atomicAdd(dL_dnormal3D + primitive * 3, grad_normal.x);
		atomicAdd(dL_dnormal3D + primitive * 3 + 1, grad_normal.y);
		atomicAdd(dL_dnormal3D + primitive * 3 + 2, grad_normal.z);
		atomicAdd(dL_dopacity + primitive, grad_opacity);
	}
}
}

#endif
