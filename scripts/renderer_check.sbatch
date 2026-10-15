#!/usr/bin/env bash
#SBATCH --job-name=fs_rcheck
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/rcheck_%j.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/rcheck_%j.err
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --qos=batch
#SBATCH --exclude=c22,c11
#SBATCH --time=01:00:00
# RTX4090 baseline for docs/RENDERER_SPEEDUP.md: renderer unit tests, a reference
# snapshot compared with a second render of the same renderer (noise floor), and
# the benchmark of both handoff checkpoints. Results: outputs/renderer_baseline_4090/.
set -Eeuo pipefail
ROOT=/shared/results/gmwilczn/foamsurf/foamsurf
FS_ENV=/shared/results/gmwilczn/conda/envs/foamsurf
DRIVER=/shared/results/gmwilczn/drivers/NVIDIA-Linux-x86_64-535.183.01
DATA=/shared/sets/datasets/SeathruNeRF_dataset/IUI3-RedSea-Undistorted
export PATH="$FS_ENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME="$FS_ENV"
export FOAMSURF_DRIVER_LIBRARY_DIRECTORY="$DRIVER"
export LD_LIBRARY_PATH="$DRIVER:$FS_ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
cd "$ROOT"
OUT=outputs/renderer_baseline_4090
mkdir -p "$OUT"
nvidia-smi --query-gpu=name --format=csv,noheader
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
# Keep going after a test failure so the snapshot and benchmark still run.
status=0
python -m pytest -q -rs renderer/tests || status=1
DEFAULT=outputs/iui3_medium_fix/530646/B0_chroma100_bk
DENSE=outputs/iui3_medium_fix/532641/B0_chroma100_scov3e3_g700k_bk
python scripts/renderer_check.py capture --run "$DEFAULT" --dataset "$DATA" --reference "$OUT/default_reference.pt"
python scripts/renderer_check.py compare --run "$DEFAULT" --dataset "$DATA" --reference "$OUT/default_reference.pt" \
  | tee "$OUT/default_self_compare.txt"
python scripts/renderer_check.py bench --run "$DEFAULT" --dataset "$DATA" --json "$OUT/default_bench.json"
python scripts/renderer_check.py bench --run "$DENSE" --dataset "$DATA" --json "$OUT/g700k_bench.json"
exit $status
