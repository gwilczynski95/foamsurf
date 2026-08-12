from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent

setup(
    name="foamsurf-renderer",
    version="0.1.0",
    description="Standalone differentiable Gaussian/Voronoi renderer for FoamSurf",
    packages=find_packages(where="."),
    python_requires=">=3.10",
    install_requires=["torch"],
    ext_modules=[
        CUDAExtension(
            name="foamsurf._C",
            sources=[
                str(ROOT / "native" / "bindings.cpp"),
                str(ROOT / "native" / "hybrid_cuda.cu"),
            ],
            include_dirs=[str(ROOT / "native")],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-lineinfo"],
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(
            no_python_abi_suffix=True,
            use_ninja=True,
        )
    },
)
