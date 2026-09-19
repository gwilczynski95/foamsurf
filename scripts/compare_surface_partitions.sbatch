#!/usr/bin/env bash
#SBATCH --job-name=fs_partition
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/partition_%A_%a.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/partition_%A_%a.err
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --qos=batch
#SBATCH --exclude=c22,c11
#SBATCH --time=04:00:00
#SBATCH --array=0-3%2
set -Eeuo pipefail
ROOT=/shared/results/gmwilczn/foamsurf/foamsurf
FS_ENV=/shared/results/gmwilczn/conda/envs/foamsurf
DRIVER=/shared/results/gmwilczn/drivers/NVIDIA-Linux-x86_64-535.183.01
export PATH="$FS_ENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME="$FS_ENV"
export FOAMSURF_DRIVER_LIBRARY_DIRECTORY="$DRIVER"
export LD_LIBRARY_PATH="$DRIVER:$FS_ENV/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=8.9
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
cd "$ROOT"
python -m pytest -q tests/test_surface_partition.py
RUNS=(12 13 2 3)
RUN="${WARMUP_RUN:-$ROOT/outputs/warmup_organic/506329/run_${RUNS[$SLURM_ARRAY_TASK_ID]}}"
python scripts/compare_surface_partitions.py --run "$RUN" --iteration "${ITERATION:-5000}" --render --views 4
