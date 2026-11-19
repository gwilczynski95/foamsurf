#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="$ROOT/.conda-env"
WITH_APPS=false
usage() {
    echo "Usage: scripts/install.sh [--prefix DIR] [--with-apps]"
    echo "Requires conda, a C++ compiler, NVIDIA GPU/driver, and OPTIX_ROOT (SDK 8.0)."
    echo "Creates a new environment; refuses to overwrite an existing directory."
}
while (($#)); do
    case "$1" in
        --prefix) PREFIX="${2:?--prefix needs a directory}"; shift 2 ;;
        --with-apps) WITH_APPS=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
if [[ -e "$PREFIX" ]]; then
    echo "Environment path already exists: $PREFIX. Select a new --prefix." >&2
    exit 1
fi
if [[ ! -f "${OPTIX_ROOT:-}/include/optix.h" ]]; then
    echo "Download NVIDIA OptiX SDK 8.0 and set OPTIX_ROOT before installing." >&2
    exit 1
fi
command -v conda >/dev/null || { echo 'Install conda first.' >&2; exit 1; }
cd "$ROOT"
conda env create --prefix "$PREFIX" --file environment.yml --yes
PREFIX="$(cd -- "$PREFIX" && pwd)"
export CONDA_ENV_PATH="$PREFIX" CONDA_PREFIX="$PREFIX"
export FOAMSURF_PYTHON="$PREFIX/bin/python"
export PATH="$PREFIX/bin:$PATH"
export CUDA_HOME="$PREFIX"
export TORCH_EXTENSIONS_DIR="$ROOT/.torch_extensions"
export PYTHONPATH="$ROOT/renderer:$ROOT/src:$ROOT:$ROOT/tests"
if "$WITH_APPS"; then
    "$FOAMSURF_PYTHON" -m pip install -r requirements-examples.txt
fi
source "$ROOT/scripts/build_env.sh"
./build.sh
"$FOAMSURF_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU is required for the render smoke test"'
"$FOAMSURF_PYTHON" -m pytest -q tests -m 'not slow'
# This renders a tiny Gaussian/Voronoi scene and checks RGB and gradients against
# an independent reference. It needs no dataset or downloaded checkpoint.
"$FOAMSURF_PYTHON" -m pytest -q renderer/tests/test_rgb_medium_renderer.py -k test_cuda_against_independent_reference_and_gradients
printf '\nInstallation verified. Use: conda activate %q\n' "$PREFIX"
printf 'From this repository, export PYTHONPATH=%q\n' "$PYTHONPATH"
