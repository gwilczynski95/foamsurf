#!/usr/bin/env bash
#SBATCH --job-name=fs_colmap
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/colmap_%j.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/colmap_%j.err
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --qos=batch
#SBATCH --exclude=c22,c11
#SBATCH --time=24:00:00
set -Eeuo pipefail
ROOT=/shared/results/gmwilczn/foamsurf/foamsurf
FS_ENV=/shared/results/gmwilczn/conda/envs/foamsurf
DRIVER=/shared/results/gmwilczn/drivers/NVIDIA-Linux-x86_64-535.183.01
export PATH="$FS_ENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME="$FS_ENV"
export FOAMSURF_DRIVER_LIBRARY_DIRECTORY="$DRIVER"
export LD_LIBRARY_PATH="$DRIVER:$FS_ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=2
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TORCH_EXTENSIONS_DIR="$ROOT/outputs/colmap_test_build_${SLURM_JOB_ID}"
cd "$ROOT"
mkdir -p "$TORCH_EXTENSIONS_DIR"
: "${COLMAP_TEST_CERTIFICATE:?Set COLMAP_TEST_CERTIFICATE to the passed GPU test certificate}"
python scripts/colmap_warmup_gate.py check --certificate "$COLMAP_TEST_CERTIFICATE"
case "${TRAIN_MODE:-warmup}" in
  warmup)
    python scripts/train_colmap_warmup.py --output-dir "$ROOT/outputs/colmap_warmup/${SLURM_JOB_ID}" --iterations "${ITERATIONS:-5000}" ;;
  end_to_end)
    python scripts/train_end_to_end.py --config configs/end_to_end_iui3.yaml --output-dir "$ROOT/outputs/end_to_end_iui3/${SLURM_JOB_ID}" ;;
  *) echo "TRAIN_MODE must be warmup or end_to_end" >&2; exit 1 ;;
esac
