#!/usr/bin/env bash
#SBATCH --job-name=fs_gradaudit
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/gradaudit_%j.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/gradaudit_%j.err
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --qos=batch
#SBATCH --exclude=c22,c11
#SBATCH --time=03:00:00
# Renderer gradient audit (scripts/gradient_audit.py): the GPU test, the synthetic scene and the
# last checkpoints of the default runs (FS_AUDIT_SWEEP, default 543588), two test views per scene.
#   sbatch scripts/gradient_audit.sbatch
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
cd "$ROOT"
OUT="outputs/gradient_audit/${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"
nvidia-smi --query-gpu=name --format=csv,noheader
python -m pytest -q -s tests/test_gradient_audit.py -rs || echo "[gradaudit] test failed, continuing with the audits"
for scene in moderate dense; do
  python scripts/gradient_audit.py synthetic --scene "$scene" --output "$OUT/synthetic_$scene.json"
done
SWEEP="outputs/seathru/${FS_AUDIT_SWEEP:-543588}"
for scene in IUI3-RedSea-Undistorted Curasao-Undistorted JapaneseGradens-RedSea-Undistorted Panama-Undisorted; do
  for view in 0 1; do
    python scripts/gradient_audit.py checkpoint --run "$SWEEP/$scene" --view "$view" --output "$OUT/${scene}_view${view}.json"
  done
done
