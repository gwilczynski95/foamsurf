#!/usr/bin/env bash
# Usage: examples/edits.sh RUN DATA VIEW [OUT]  — surface deformations and water edits of a trained run
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/renderer:$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m examples.editing.edit --run "$1" --scene "$2" --view "$3" --edits bulge,bend,clear,murky,haze --out "${4:-renders/edits}"
