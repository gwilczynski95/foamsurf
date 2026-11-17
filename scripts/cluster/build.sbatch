#!/usr/bin/env bash
# Builds the renderer (OptiX + CUDA, sm_89) and PPISP's extension on an RTX 4090 node, then runs the tests.
#   sbatch scripts/cluster/build.sbatch
#SBATCH --job-name=foamsurf_build
#SBATCH --output=/shared/results/gmtobias/repos/foamsurf/outputs/cluster_setup/%j_build.out
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --qos=batch
#SBATCH --time=01:30:00
#SBATCH --exclude=c11,c15,c22
set -euo pipefail
source /shared/results/gmtobias/repos/foamsurf/scripts/cluster/env.sh
cd "$REPOSITORY"
hostname; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
./build.sh
"$PYTHON" -c "from training.camera_isp import load_ppisp; load_ppisp(); print('ppisp: OK')"
FS_DATASETS=$REPOSITORY/foamsurf_seathru_2026-11-05/data "$PYTHON" -m pytest -q tests 2>&1 | tail -15 || true  # informative only; the build above must succeed
