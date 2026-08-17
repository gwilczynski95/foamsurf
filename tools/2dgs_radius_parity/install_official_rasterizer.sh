#!/usr/bin/env bash
set -euo pipefail

RASTERIZER_REVISION="e0ed0207b3e0669960cfad70852200a4a5847f61"
BUILD_DIR="$(mktemp -d /tmp/foamsurf-2dgs-radius-XXXXXX)"
trap 'rm -rf "${BUILD_DIR}"' EXIT

git clone https://github.com/hbb1/diff-surfel-rasterization.git \
  "${BUILD_DIR}/diff-surfel-rasterization"
git -C "${BUILD_DIR}/diff-surfel-rasterization" checkout \
  "${RASTERIZER_REVISION}"
git -C "${BUILD_DIR}/diff-surfel-rasterization" submodule update \
  --init --recursive

# CUDA 12.4 exposes a missing include in the pinned upstream revision.
# This changes no rasterization logic; it only declares fixed-width integers.
HEADER="${BUILD_DIR}/diff-surfel-rasterization/cuda_rasterizer/rasterizer_impl.h"
if ! grep -q '^#include <cstdint>$' "${HEADER}"; then
  sed -i '/#pragma once/a #include <cstdint>' "${HEADER}"
fi

python -m pip install --no-build-isolation \
  "${BUILD_DIR}/diff-surfel-rasterization"

python - <<'PY'
import diff_surfel_rasterization
print("official diff-surfel-rasterization import: OK")
PY
