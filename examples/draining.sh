#!/usr/bin/env bash
# Usage: examples/draining.sh PACKAGE VIEW [OUT]  — PACKAGE has models/<scene>/ and data/<scene>/ (see docs/DRAINING.md); needs ffmpeg
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/renderer:$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m examples.draining.render_draining --package "$1" --scene IUI3-RedSea-Undistorted --view "$2" --width 960 --surface-dynamics --breaking-waves --impact-strength 3 --end-hold 11.916666666666666 --final-wave-scale 2 --final-rise 0.1 --rise-seconds 5 --post-rise-hold 10 --out "${3:-renders/draining}"
