#!/usr/bin/env bash
# Usage: examples/insertion.sh RUN DATA [OUT]  — needs the demo assets in assets/ and scripts/install.sh --with-apps
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/renderer:$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m examples.insert --run "$1" --scene "$2" --assets assets --shot examples/shots/demo/underwater_objects.json --out "${3:-renders/insertion}" --still
