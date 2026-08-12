#!/usr/bin/env bash

# Local CUDA/OptiX build for the standalone FoamSurf renderer:
#   - /shared/results/gmwilczn/conda/envs/foamsurf
#   - CUDA 12.4
#   - NVIDIA OptiX SDK 8.0.0
#   - RTX 4090 (compute capability 8.9)

set -Eeuo pipefail

trap 'echo "Build failed at line ${LINENO}." >&2' ERR

PROJECT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RENDERER_DIRECTORY="$PROJECT_DIRECTORY/third_party/flare_renderer"
OUTPUT_DIRECTORY="$PROJECT_DIRECTORY/foamsurf/_optix"

CONDA_ENV_PATH="${CONDA_ENV_PATH:-/shared/results/gmwilczn/conda/envs/foamsurf}"
OPTIX_DIRECTORY="${OPTIX_DIRECTORY:-${FOAMSURF_OPTIX_DIRECTORY:-/shared/results/gmwilczn/drivers/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64}}"
DRIVER_LIBRARY_DIRECTORY="${FOAMSURF_DRIVER_LIBRARY_DIRECTORY:-/usr/lib/x86_64-linux-gnu}"
CUDA_ARCH="${CUDA_ARCH:-89}"
CXX="${CXX:-g++}"

if [[ ! -d "$CONDA_ENV_PATH" ]]; then
    echo "Error: conda environment does not exist: $CONDA_ENV_PATH" >&2
    echo "Set CONDA_ENV_PATH to the foamsurf environment and retry." >&2
    exit 1
fi

CONDA_ENV_PATH="$(cd -- "$CONDA_ENV_PATH" && pwd)"
PYTHON="${PYTHON:-$CONDA_ENV_PATH/bin/python}"
CUDA_DIRECTORY="${CUDA_DIRECTORY:-${CUDA_HOME:-$CONDA_ENV_PATH}}"
NVCC="$CUDA_DIRECTORY/bin/nvcc"

export PATH="$CONDA_ENV_PATH/bin:$PATH"
export CUDA_HOME="$CUDA_DIRECTORY"
export CUDA_PATH="$CUDA_DIRECTORY"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"

if [[ ! -x "$PYTHON" ]]; then
    echo "Error: Python was not found in the conda environment: $PYTHON" >&2
    exit 1
fi
if [[ ! -x "$NVCC" ]]; then
    echo "Error: nvcc was not found in the conda environment: $NVCC" >&2
    exit 1
fi
if ! "$NVCC" --version | grep -q 'release 12\.4'; then
    echo "Error: this renderer build expects CUDA 12.4." >&2
    "$NVCC" --version >&2
    exit 1
fi
if [[ ! -f "$OPTIX_DIRECTORY/include/optix.h" ]]; then
    echo "Error: OptiX headers were not found in $OPTIX_DIRECTORY/include" >&2
    exit 1
fi
if [[ ! -e "$DRIVER_LIBRARY_DIRECTORY/libcuda.so.1" ]]; then
    echo "Error: system libcuda.so.1 was not found in $DRIVER_LIBRARY_DIRECTORY" >&2
    exit 1
fi
if [[ ! -e "$DRIVER_LIBRARY_DIRECTORY/libnvoptix.so.1" ]]; then
    echo "Error: system libnvoptix.so.1 was not found in $DRIVER_LIBRARY_DIRECTORY" >&2
    exit 1
fi
if ! command -v "$CXX" >/dev/null 2>&1; then
    echo "Error: C++ compiler was not found: $CXX" >&2
    exit 1
fi

OPTIX_HEADER_VERSION="$(awk '$1 == "#define" && $2 == "OPTIX_VERSION" { print $3; exit }' "$OPTIX_DIRECTORY/include/optix.h")"
OPTIX_HEADER_ABI="$(awk '$1 == "#define" && $2 == "OPTIX_ABI_VERSION" { print $3; exit }' "$OPTIX_DIRECTORY/include/optix_function_table.h")"
if [[ "$OPTIX_HEADER_VERSION" != "80000" || "$OPTIX_HEADER_ABI" != "87" ]]; then
    echo "Error: the renderer requires OptiX 8.0 headers (version 80000, ABI 87)." >&2
    echo "Selected headers report version ${OPTIX_HEADER_VERSION:-unknown}, ABI ${OPTIX_HEADER_ABI:-unknown}." >&2
    exit 1
fi

TORCH_DIRECTORY="$("$PYTHON" -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent)')"
TORCH_VERSION="$("$PYTHON" -c 'import torch; print(torch.__version__)')"
TORCH_CUDA_VERSION="$("$PYTHON" -c 'import torch; print(torch.version.cuda)')"
PYTHON_INCLUDE="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
TORCH_ABI="$("$PYTHON" -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')"

if [[ "$TORCH_CUDA_VERSION" != "12.4" ]]; then
    echo "Error: PyTorch must use CUDA 12.4; found torch $TORCH_VERSION with CUDA $TORCH_CUDA_VERSION." >&2
    exit 1
fi

CUDA_INCLUDE="$CUDA_DIRECTORY/include"
if [[ -d "$CUDA_DIRECTORY/lib64" ]]; then
    CUDA_LIBRARY="$CUDA_DIRECTORY/lib64"
else
    CUDA_LIBRARY="$CUDA_DIRECTORY/lib"
fi
if [[ -d "$CUDA_LIBRARY/stubs" ]]; then
    CUDA_STUB_LIBRARY="$CUDA_LIBRARY/stubs"
elif [[ -d "$CUDA_DIRECTORY/targets/x86_64-linux/lib/stubs" ]]; then
    CUDA_STUB_LIBRARY="$CUDA_DIRECTORY/targets/x86_64-linux/lib/stubs"
else
    echo "Error: CUDA driver stub directory was not found under $CUDA_DIRECTORY" >&2
    exit 1
fi

TORCH_INCLUDE="$TORCH_DIRECTORY/include"
TORCH_API_INCLUDE="$TORCH_INCLUDE/torch/csrc/api/include"
TORCH_LIBRARY="$TORCH_DIRECTORY/lib"
OPTIX_INCLUDE="$OPTIX_DIRECTORY/include"

for directory in \
    "$RENDERER_DIRECTORY" \
    "$CUDA_INCLUDE" \
    "$CUDA_LIBRARY" \
    "$CUDA_STUB_LIBRARY" \
    "$TORCH_INCLUDE" \
    "$TORCH_API_INCLUDE" \
    "$TORCH_LIBRARY" \
    "$PYTHON_INCLUDE"; do
    if [[ ! -d "$directory" ]]; then
        echo "Error: required directory does not exist: $directory" >&2
        exit 1
    fi
done

export LD_LIBRARY_PATH="$TORCH_LIBRARY:$CUDA_LIBRARY:$DRIVER_LIBRARY_DIRECTORY:${LD_LIBRARY_PATH:-}"

COMPUTE="compute_$CUDA_ARCH"
SM="sm_$CUDA_ARCH"
NVCC_FLAGS=(
    -std=c++17
    -Xcompiler -fPIC
    -arch="$COMPUTE"
    -code="$SM"
    -I"$OPTIX_INCLUDE"
    -I"$CUDA_INCLUDE"
)
CPP_FLAGS=(
    -std=c++17
    -fPIC
    -D_GLIBCXX_USE_CXX11_ABI="$TORCH_ABI"
    -I"$OPTIX_INCLUDE"
    -I"$CUDA_INCLUDE"
    -I"$TORCH_INCLUDE"
    -I"$TORCH_API_INCLUDE"
    -I"$PYTHON_INCLUDE"
)

CUDA_SOURCES=(
    CPyOptiXFLARERenderer_CUDA_Backward
    CPyOptiXFLARERenderer_CUDA_Backward_base
    CPyOptiXFLARERenderer_CUDA_Core
    CPyOptiXFLARERenderer_CUDA_Forward_training
    CPyOptiXFLARERenderer_CUDA_Forward_training_base
    CPyOptiXFLARERenderer_CUDA_Forward_inference
    CPyOptiXFLARERenderer_CUDA_Forward_inference_base
    GenerateRays
)

for source_name in "${CUDA_SOURCES[@]}"; do
    if [[ ! -f "$RENDERER_DIRECTORY/${source_name}.cu" ]]; then
        echo "Error: CUDA source does not exist: $RENDERER_DIRECTORY/${source_name}.cu" >&2
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIRECTORY"

echo "Project:          $PROJECT_DIRECTORY"
echo "Conda env:        $CONDA_ENV_PATH"
echo "Python:           $PYTHON"
echo "PyTorch:          $TORCH_VERSION (CUDA $TORCH_CUDA_VERSION)"
echo "CUDA toolkit:     $CUDA_DIRECTORY"
echo "OptiX SDK:        $OPTIX_DIRECTORY"
echo "OptiX headers:    version $OPTIX_HEADER_VERSION / ABI $OPTIX_HEADER_ABI"
echo "NVIDIA libraries: $DRIVER_LIBRARY_DIRECTORY"
echo "CUDA target:      $COMPUTE / $SM"

echo "Building OptiX PTX..."
rm -f "$OUTPUT_DIRECTORY/shaders.cu.optixir"
"$NVCC" -std=c++17 -ptx -arch="$COMPUTE" \
    "$RENDERER_DIRECTORY/shaders.cu" \
    -o "$OUTPUT_DIRECTORY/shaders.cu.ptx" \
    -I"$OPTIX_INCLUDE"

CUDA_OBJECTS=()
echo "Compiling CUDA sources..."
for source_name in "${CUDA_SOURCES[@]}"; do
    object="$OUTPUT_DIRECTORY/${source_name}.cu.o"
    "$NVCC" "${NVCC_FLAGS[@]}" \
        -c "$RENDERER_DIRECTORY/${source_name}.cu" \
        -o "$object"
    CUDA_OBJECTS+=("$object")
done

if ! nm "$OUTPUT_DIRECTORY/CPyOptiXFLARERenderer_CUDA_Core.cu.o" \
    | grep -E '[[:space:]]g_optixFunctionTable$' >/dev/null; then
    echo "Error: renderer core was not compiled against the selected OptiX ABI $OPTIX_HEADER_ABI headers." >&2
    echo "Check include-path overrides such as CPATH and CPLUS_INCLUDE_PATH." >&2
    exit 1
fi

echo "Linking CUDA device code..."
"$NVCC" -std=c++17 -Xcompiler -fPIC -dlink \
    "${CUDA_OBJECTS[@]}" \
    -o "$OUTPUT_DIRECTORY/dlink.o" \
    -arch="$COMPUTE" -code="$SM" \
    -L"$CUDA_STUB_LIBRARY" \
    -L"$CUDA_LIBRARY" \
    -lcuda -lcudart

echo "Compiling C++ sources..."
"$CXX" "${CPP_FLAGS[@]}" \
    -c "$RENDERER_DIRECTORY/GenerateRays.cpp" \
    -o "$OUTPUT_DIRECTORY/GenerateRays.cpp.o"
"$CXX" "${CPP_FLAGS[@]}" \
    -c "$RENDERER_DIRECTORY/CPyOptiXFLARERenderer.cpp" \
    -o "$OUTPUT_DIRECTORY/CPyOptiXFLARERenderer.cpp.o"

echo "Linking PYOPTIXFLARERENDERER.so..."
"$CXX" "${CPP_FLAGS[@]}" -shared \
    "$RENDERER_DIRECTORY/PyOptiXFLARERenderer.cpp" \
    "${CUDA_OBJECTS[@]}" \
    "$OUTPUT_DIRECTORY/dlink.o" \
    "$OUTPUT_DIRECTORY/GenerateRays.cpp.o" \
    "$OUTPUT_DIRECTORY/CPyOptiXFLARERenderer.cpp.o" \
    -o "$OUTPUT_DIRECTORY/PYOPTIXFLARERENDERER.so" \
    -L"$CUDA_STUB_LIBRARY" \
    -L"$CUDA_LIBRARY" \
    -L"$TORCH_LIBRARY" \
    -lcuda -lcudart -ldl -ltorch_python -ltorch -ltorch_cpu -lc10 \
    -Wl,--enable-new-dtags \
    -Wl,-rpath,"$DRIVER_LIBRARY_DIRECTORY" \
    -Wl,-rpath,"$CUDA_LIBRARY" \
    -Wl,-rpath,"$TORCH_LIBRARY"

echo "Build completed successfully."
echo "Created: $OUTPUT_DIRECTORY/PYOPTIXFLARERENDERER.so"
echo "Created: $OUTPUT_DIRECTORY/shaders.cu.ptx"

