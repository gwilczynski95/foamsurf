#!/usr/bin/env bash
# Full build (requirements, topology extension, renderer) for the local RTX 3090
# workstation. build.sh itself defaults to the cluster's RTX 4090 (sm_89) setup;
# this sets the 3090 architecture (sm_86) and local paths, then runs it.
# Every variable can still be overridden from the environment.

set -Eeuo pipefail

REPOSITORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export CONDA_ENV_PATH="${CONDA_ENV_PATH:-/home/grzegos/.conda/envs/foamsurf}"
export FOAMSURF_OPTIX_DIRECTORY="${FOAMSURF_OPTIX_DIRECTORY:-/home/grzegos/projects/phd/foamsurf/drivers/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64}"
export FOAMSURF_DRIVER_LIBRARY_DIRECTORY="${FOAMSURF_DRIVER_LIBRARY_DIRECTORY:-/usr/lib/x86_64-linux-gnu}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec "$REPOSITORY/build.sh" "$@"
