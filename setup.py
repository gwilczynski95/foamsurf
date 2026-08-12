import os
import re
import sys
from pathlib import Path

import cmake_build_extension
import setuptools
import sysconfig

lib_path = sysconfig.get_path("purelib")
assert os.path.exists(
    f"{lib_path}/torch"
), "Could not find PyTorch; please make sure it is installed in your environment before installing FoamSurf topology."

cmake_options = []

if "CUDA_HOME" in os.environ:
    cmake_options.append(f"-DCUDA_TOOLKIT_ROOT_DIR={os.environ['CUDA_HOME']}")

source_dir = Path(__file__).parent.absolute()
cmake = (source_dir / "CMakeLists.txt").read_text()
version = re.search(r"project\(\S+ VERSION (\S+)\)", cmake).group(1)

install_requirements = []


setuptools.setup(
    version=version,
    install_requires=install_requirements,
    ext_modules=[
        cmake_build_extension.CMakeExtension(
            name="FoamSurfTopologyBindings",
            install_prefix="foamsurf_topology",
            cmake_depends_on=["pybind11"],
            write_top_level_init=None,
            source_dir=str(source_dir),
            cmake_configure_options=[
                f"-DPython3_ROOT_DIR={Path(sys.prefix)}",
                "-DBUILD_SHARED_LIBS:BOOL=OFF",
                "-DGPU_DEBUG:BOOL=OFF",
                f"-DTorch_DIR={lib_path}/torch/share/cmake/Torch",
            ]
            + cmake_options,
        ),
    ],
    cmdclass=dict(
        build_ext=cmake_build_extension.BuildExtension,
    ),
)
