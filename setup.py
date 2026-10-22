#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
from pathlib import Path
os.path.dirname(os.path.abspath(__file__))


class HeaderAwareBuildExtension(BuildExtension):
    def build_extensions(self):
        # Some supported PyTorch versions emit a Ninja CUDA depfile rule without
        # telling nvcc to generate that file. A stale GeometryState/TileSupport
        # layout across translation units can then corrupt GPU memory. Invalidate
        # this extension's derived objects when any local CUDA header changes.
        source_root = Path(__file__).resolve().parent
        headers = []
        for directory in ("cuda_rasterizer", "cuda_projector"):
            headers.extend((source_root / directory).glob("*.h"))
            headers.extend((source_root / directory).glob("*.cuh"))
        headers.extend(source_root / name for name in ("rasterize_points.h", "project_errors.h"))
        objects = list(Path(self.build_temp).rglob("*.o"))
        newest_header = max(header.stat().st_mtime_ns for header in headers)
        if any(obj.stat().st_mtime_ns < newest_header for obj in objects):
            for obj in objects:
                obj.unlink()
        super().build_extensions()


setup(
    name="diff_gabor_rasterization",
    packages=['diff_gabor_rasterization'],
    version='0.0.1',
    ext_modules=[
        CUDAExtension(
            name="diff_gabor_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/profiling.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "cuda_projector/projector_impl.cu",
            "cuda_projector/forward.cu",
            "project_errors.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': HeaderAwareBuildExtension
    }
)
