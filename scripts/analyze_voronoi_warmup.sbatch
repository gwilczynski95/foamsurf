#!/usr/bin/env bash
#SBATCH --job-name=fs_vw_an
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/voronoi_warmup_analysis_%j.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/voronoi_warmup_analysis_%j.err
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --qos=batch
#SBATCH --exclude=c22,c11
#SBATCH --time=02:00:00

set -Eeuo pipefail

ROOT=/shared/results/gmwilczn/foamsurf/foamsurf
ENV=/shared/results/gmwilczn/conda/envs/foamsurf
DRIVER=/shared/results/gmwilczn/drivers/NVIDIA-Linux-x86_64-535.183.01
RUN=${RUN:-$ROOT/outputs/voronoi_prior_warmup/voronoi_prior_warmup_503195}

export PATH="$ENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME="$ENV"
export FOAMSURF_DRIVER_LIBRARY_DIRECTORY="$DRIVER"
export LD_LIBRARY_PATH="$DRIVER:$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "$ROOT/slurm_logs"
cd "$ROOT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"$ENV/bin/python" scripts/analyze_voronoi_warmup.py \
  --run "$RUN" \
  --iterations 500 1000 2000 3000 4000 5000 \
  --image-views 4

test -s "$RUN/analysis/analysis.json"
test -s "$RUN/analysis/training_dashboard.png"
test -s "$RUN/analysis/separation_dashboard.png"
test -s "$RUN/analysis/reconstruction_evolution_view00.png"
test -s "$RUN/analysis/checkpoint_separation_comparison_view00.png"
for ITERATION in 500 1000 2000 3000 4000 5000; do
  printf -v PADDED "%06d" "$ITERATION"
  test -s "$RUN/analysis/iteration_${PADDED}/cell_labels.pt"
  test -s "$RUN/analysis/iteration_${PADDED}/view_00_separation.png"
done
