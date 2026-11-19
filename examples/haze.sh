#!/usr/bin/env bash
# Usage: examples/haze.sh RUN DATA VIEW [OUT]  — rainbow haze patches over the seabed
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/renderer:$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m examples.edit --run "$1" --scene "$2" --view "$3" --edits haze --haze-strength 15 --haze-colour patches3d --patch-size 0.25 --medium-split 24 --out "${4:-renders/haze}"
