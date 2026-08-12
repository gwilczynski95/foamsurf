#!/usr/bin/env bash

set -Eeuo pipefail

RENDERER_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-${CONDA_PREFIX:-/shared/results/gmwilczn/conda/envs/foamsurf}}"
PYTHON="${PYTHON:-$CONDA_ENV_PATH/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
    echo "Python not found: $PYTHON" >&2
    exit 1
fi
if [[ ! -x "$CONDA_ENV_PATH/bin/nvcc" ]]; then
    echo "nvcc not found: $CONDA_ENV_PATH/bin/nvcc" >&2
    exit 1
fi

export PATH="$CONDA_ENV_PATH/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME="${CUDA_HOME:-$CONDA_ENV_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"

"$RENDERER_DIRECTORY/build_optix.sh"

"$PYTHON" -m pip install -e "$RENDERER_DIRECTORY" \
    --no-build-isolation \
    --no-deps

(
    cd "$RENDERER_DIRECTORY"
    "$PYTHON" - <<'PY'
import importlib

import foamsurf

bindings = importlib.import_module("foamsurf._C")
if not hasattr(bindings, "hybrid_forward"):
    raise RuntimeError("foamsurf._C does not export hybrid_forward")
print(f"foamsurf package: {foamsurf.__file__}")
print("standalone FoamSurf CUDA/OptiX renderer: OK")
PY
)
