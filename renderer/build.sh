#!/usr/bin/env bash

set -Eeuo pipefail

RENDERER_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$RENDERER_DIRECTORY/../scripts/build_env.sh"

# Training uses K=64; the paper's FPS measurements use K=32.
for hit_buffer in 64 32; do
    HIT_BUFFER_SIZE="$hit_buffer" OPTIX_OUTPUT_DIRECTORY="$RENDERER_DIRECTORY/foamsurf/_optix_k$hit_buffer" \
        "$RENDERER_DIRECTORY/build_optix.sh"
done

cd "$RENDERER_DIRECTORY"
"$PYTHON" setup.py build_ext --inplace

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
