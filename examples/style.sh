#!/usr/bin/env bash
# Usage: examples/style.sh RUN DATA STYLE_IMAGE [OUT]  — your own style image; water and surface stylized, then combined
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/renderer:$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
OUT=${4:-renders/style}
python -m examples.style.stylize --run "$1" --scene "$2" --target voronoi --style-image "$3" --out "$OUT/water"
python -m examples.style.stylize --run "$1" --scene "$2" --target gaussians --style-image "$3" --out "$OUT/surface"
python -m examples.style.compose_style --gaussians "$OUT/surface" --medium "$OUT/water" --out "$OUT/both"
python render.py --run "$OUT/both" --scene "$2" --modes full --out "$OUT/render"
