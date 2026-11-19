#!/usr/bin/env bash

set -Eeuo pipefail

REPOSITORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$REPOSITORY/scripts/build_env.sh"

cd "$REPOSITORY"
if [[ ! -f external/submodules/eigen/Eigen/Core ]]; then
    git submodule update --init --recursive
fi
"$PYTHON" setup.py build_ext --inplace
export PYTHONPATH="$REPOSITORY/renderer:$REPOSITORY/src:$REPOSITORY:${PYTHONPATH:-}"

CONDA_ENV_PATH="$CONDA_ENV_PATH" "$REPOSITORY/renderer/build.sh"

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
