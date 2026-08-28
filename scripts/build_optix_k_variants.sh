#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
for K in 16 32 64 128; do
    OUTPUT="$ROOT/renderer/foamsurf/_optix_k$K"
    echo "Building FoamSurf OptiX backend K=$K"
    HIT_BUFFER_SIZE="$K" OPTIX_OUTPUT_DIRECTORY="$OUTPUT" \
        "$ROOT/renderer/build_optix.sh"
    test -s "$OUTPUT/PYOPTIXFLARERENDERER.so"
    test -s "$OUTPUT/shaders.cu.ptx"
done

echo "Built K=16,32,64,128 OptiX backends."
