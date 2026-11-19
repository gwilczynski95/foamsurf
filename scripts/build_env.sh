#!/usr/bin/env bash
# Shared environment for in-place builds; source this file from the build scripts.
CONDA_ENV_PATH="${CONDA_ENV_PATH:-${CONDA_PREFIX:-}}"
PYTHON="${FOAMSURF_PYTHON:-${PYTHON:-${CONDA_ENV_PATH:+$CONDA_ENV_PATH/bin/}python}}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Python not found: $PYTHON. Activate the build environment first." >&2
    exit 1
fi
CONDA_ENV_PATH="${CONDA_ENV_PATH:-$("$PYTHON" -c 'import sys; print(sys.prefix)')}"
export CONDA_ENV_PATH PYTHON
export CUDA_HOME="${CUDA_HOME:-$CONDA_ENV_PATH}"
export PATH="$CONDA_ENV_PATH/bin:$PATH"
export OPTIX_DIRECTORY="${OPTIX_ROOT:-${OPTIX_DIRECTORY:-${FOAMSURF_OPTIX_DIRECTORY:-}}}"
if [[ ! -f "$OPTIX_DIRECTORY/include/optix.h" ]]; then
    echo "Set OPTIX_ROOT to the NVIDIA OptiX SDK 8.0 directory (containing include/optix.h)." >&2
    exit 1
fi
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    TORCH_CUDA_ARCH_LIST="$("$PYTHON" -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability())')"
fi
export TORCH_CUDA_ARCH_LIST
# The PyTorch extension can contain multiple architectures; OptiX needs one target.
if [[ -z "${CUDA_ARCH:-}" ]]; then
    if [[ "$TORCH_CUDA_ARCH_LIST" =~ ^([0-9]+)\.([0-9]+)(\+PTX)?$ ]]; then
        CUDA_ARCH="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
    else
        echo "Set CUDA_ARCH (e.g. 89) when TORCH_CUDA_ARCH_LIST contains multiple targets." >&2
        exit 1
    fi
fi
export CUDA_ARCH
# Load the installed driver, never a CUDA stub library, at runtime.
export FOAMSURF_DRIVER_LIBRARY_DIRECTORY="${FOAMSURF_DRIVER_LIBRARY_DIRECTORY:-/usr/lib/x86_64-linux-gnu}"
export LD_LIBRARY_PATH="$FOAMSURF_DRIVER_LIBRARY_DIRECTORY:${LD_LIBRARY_PATH:-}"
