#!/usr/bin/env bash

set -Eeuo pipefail

REPOSITORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-${CONDA_PREFIX:-/shared/results/gmwilczn/conda/envs/foamsurf}}"
PYTHON="${PYTHON:-$CONDA_ENV_PATH/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
    echo "Python not found: $PYTHON" >&2
    exit 1
fi

export CUDA_HOME="${CUDA_HOME:-$CONDA_ENV_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export PATH="$CONDA_ENV_PATH/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

"$PYTHON" -m pip install -r "$REPOSITORY/requirements.txt"

"$PYTHON" -m pip install -e "$REPOSITORY"     --no-build-isolation     --no-deps

CONDA_ENV_PATH="$CONDA_ENV_PATH" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$REPOSITORY/renderer/build.sh"

(
    cd "$REPOSITORY"
    "$PYTHON" - <<'PY'
import foamsurf
import foamsurf._C
import foamsurf_topology
import wandb

assert hasattr(foamsurf_topology, "Triangulation")
assert hasattr(foamsurf._C, "hybrid_forward")
print("FoamSurf B0 build: OK")
print("renderer package:", foamsurf.__file__)
PY
)
