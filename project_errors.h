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

#include <torch/extension.h>
#include <tuple>

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor> ProjectErrorsCUDA(
    const torch::Tensor& error_map, const torch::Tensor& means3D,
    const torch::Tensor& opacity, const torch::Tensor& scales,
    const torch::Tensor& rotations, const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    int image_height, int image_width, float scale_modifier, bool raw_inputs,
    bool tight_bbox, bool tile_culling, bool load_balance,
    int tile_culling_threshold, bool warp_culling, bool separate_sort,
    const torch::Tensor& hair_vote_map, bool frontmost_only, bool debug);
