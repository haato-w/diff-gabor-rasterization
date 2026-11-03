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

#include <torch/extension.h>
#include <pybind11/stl.h>
#include "rasterize_points.h"
#include "project_errors.h"
#include "cuda_rasterizer/profiling.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gabors", &RasterizeGaborsCUDA);
  m.def("rasterize_gabors_backward", &RasterizeGaborsBackwardCUDA);
  m.def("rasterize_gabors_fused", &RasterizeGaborsFusedCUDA);
  m.def("rasterize_gabors_bounded", &RasterizeGaborsBoundedCUDA);
  m.def("rasterize_gabors_culled", &RasterizeGaborsCulledCUDA);
  m.def("rasterize_gabors_backward_fused", &RasterizeGaborsBackwardFusedCUDA);
  m.def("rasterize_gabors_gaussian_backward", &RasterizeGaborsGaussianBackwardCUDA);
  m.def("rasterize_gabors_warp_culled", &RasterizeGaborsWarpCulledCUDA);
  m.def("rasterize_gabors_separate_sorted", &RasterizeGaborsSeparateSortedCUDA);
  m.def("rasterize_gabors_backward_gaussian", &RasterizeGaborsBackwardGaussianCUDA);
  m.def("rasterize_gabors_backward_separate_sorted", &RasterizeGaborsBackwardSeparateSortedCUDA);
  m.def("rasterize_gabors_backward_butterfly", &RasterizeGaborsBackwardButterflyCUDA);
  m.def("rasterize_gabors_specular", &RasterizeGaborsSpecularCUDA);
  m.def("rasterize_gabors_backward_specular", &RasterizeGaborsBackwardSpecularCUDA);
  m.def("rasterize_gabors_attributes", &RasterizeGaborsAttributesCUDA);
  m.def("rasterize_gabors_backward_attributes", &RasterizeGaborsBackwardAttributesCUDA);
  m.def("project_errors", &ProjectErrorsCUDA);
  m.def("mark_visible", &markVisible);
  m.def("set_profiling_enabled", &CudaRasterizer::Profiling::setEnabled);
  m.def("collect_profiling", &CudaRasterizer::Profiling::collect);
}
