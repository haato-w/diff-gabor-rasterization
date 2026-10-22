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

#include "project_errors.h"
#include "cuda_projector/projector.h"
#include "cuda_rasterizer/profiling.h"
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <functional>

namespace {
std::function<char*(size_t)> resizeBuffer(torch::Tensor& buffer) {
    return [&buffer](size_t bytes) {
        buffer.resize_({static_cast<int64_t>(bytes)});
        return reinterpret_cast<char*>(buffer.data_ptr());
    };
}

void checkTensor(const torch::Tensor& tensor, const torch::Tensor& reference,
                 const char* name, bool require_contiguous = true) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must have dtype float32");
    TORCH_CHECK(!require_contiguous || tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.device() == reference.device(), name,
                " must be on the same device as means3D");
}

void checkShape(const torch::Tensor& tensor, int64_t points, int64_t columns,
                const char* name) {
    TORCH_CHECK(tensor.dim() == 2 && tensor.size(0) == points && tensor.size(1) == columns,
                name, " must have shape (", points, ", ", columns, ")");
}
} // namespace

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor> ProjectErrorsCUDA(
	const torch::Tensor& error_map, const torch::Tensor& means3D,
	const torch::Tensor& opacity, const torch::Tensor& scales,
	const torch::Tensor& rotations, const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	int image_height, int image_width, float scale_modifier, bool raw_inputs,
	bool tight_bbox, bool tile_culling, bool load_balance,
	int tile_culling_threshold, bool warp_culling, bool separate_sort,
	const torch::Tensor& hair_vote_map, bool frontmost_only, bool debug)
{
	checkTensor(means3D, means3D, "means3D");
	TORCH_CHECK(means3D.dim() == 2 && means3D.size(1) == 3,
		"means3D must have shape (num_points, 3)");
	const c10::cuda::CUDAGuard device_guard(means3D.device());
	const int P = means3D.size(0);
	TORCH_CHECK(image_height > 0 && image_width > 0, "Image dimensions must be positive");
	TORCH_CHECK(std::isfinite(scale_modifier) && scale_modifier > 0,
		"scale_modifier must be finite and positive");
	TORCH_CHECK(tile_culling_threshold >= 0 && tile_culling_threshold <= 32,
		"tile_culling_threshold must be between 0 and 32");
	checkTensor(opacity, means3D, "opacity");
	checkTensor(scales, means3D, "scales");
	checkTensor(rotations, means3D, "rotations");
	checkTensor(viewmatrix, means3D, "viewmatrix", false);
	checkTensor(projmatrix, means3D, "projmatrix", false);
	checkShape(opacity, P, 1, "opacity");
	checkShape(scales, P, 2, "scales");
	checkShape(rotations, P, 4, "rotations");
	TORCH_CHECK(viewmatrix.numel() == 16 && projmatrix.numel() == 16,
		"viewmatrix and projmatrix must contain 16 elements");
	checkTensor(error_map, means3D, "error_map");
	checkTensor(hair_vote_map, means3D, "hair_vote_map");
	const bool has_error = error_map.numel() != 0;
	const bool has_votes = hair_vote_map.numel() != 0;
	TORCH_CHECK(has_error || has_votes, "Provide an error map or hair vote map");
	TORCH_CHECK(!has_error || (error_map.dim() == 2 && error_map.size(0) == image_height &&
		error_map.size(1) == image_width), "error_map must have shape (height, width)");
	TORCH_CHECK(!has_votes || (hair_vote_map.dim() == 3 && hair_vote_map.size(0) == 2 &&
		hair_vote_map.size(1) == image_height && hair_vote_map.size(2) == image_width),
		"hair_vote_map must have shape (2, height, width): hair evidence, confidence");
	TORCH_CHECK(!frontmost_only || has_votes, "frontmost_only requires a hair vote map");
	TORCH_CHECK(!frontmost_only || !has_error, "frontmost_only projects hair votes without an error map");

	CudaRasterizer::Profiling::Scope allocation("error_projection.allocate");
	auto errors = torch::zeros({P}, means3D.options());
	auto effects = torch::zeros({P}, means3D.options());
	auto votes = torch::zeros({has_votes ? P : 0, 2}, means3D.options());
	auto scratch_options = means3D.options().dtype(torch::kUInt8);
	auto geometry = torch::empty({0}, scratch_options);
	auto binning = torch::empty({0}, scratch_options);
	auto image = torch::empty({0}, scratch_options);
	allocation.stop();
	int rendered = 0;
	if (P > 0)
	{
		rendered = CudaProjector::Projector::project(
			resizeBuffer(geometry), resizeBuffer(binning), resizeBuffer(image),
			P, image_width, image_height,
			means3D.data_ptr<float>(), opacity.data_ptr<float>(),
			scales.data_ptr<float>(), scale_modifier, rotations.data_ptr<float>(),
			viewmatrix.contiguous().data_ptr<float>(), projmatrix.contiguous().data_ptr<float>(),
			has_error ? error_map.data_ptr<float>() : nullptr, errors.data_ptr<float>(),
			effects.data_ptr<float>(), has_votes ? hair_vote_map.data_ptr<float>() : nullptr,
			has_votes ? votes.data_ptr<float>() : nullptr,
			raw_inputs, tight_bbox, tile_culling, load_balance, tile_culling_threshold,
			warp_culling, separate_sort, frontmost_only, debug);
	}
	return std::make_tuple(rendered, errors, effects, votes);
}

