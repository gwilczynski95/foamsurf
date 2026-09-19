#!/usr/bin/env bash
#SBATCH --job-name=fs_vwarm
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/voronoi_warmup_%j.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/voronoi_warmup_%j.err
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --qos=batch
#SBATCH --exclude=c22,c11
#SBATCH --time=24:00:00

set -Eeuo pipefail

ROOT=/shared/results/gmwilczn/foamsurf/foamsurf
ENV=/shared/results/gmwilczn/conda/envs/foamsurf
DRIVER=/shared/results/gmwilczn/drivers/NVIDIA-Linux-x86_64-535.183.01
RUN="voronoi_prior_warmup_${SLURM_JOB_ID}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/voronoi_prior_warmup}"
OUTPUT="$OUTPUT_ROOT/$RUN"

export PATH="$ENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME="$ENV"
export FOAMSURF_DRIVER_LIBRARY_DIRECTORY="$DRIVER"
export LD_LIBRARY_PATH="$DRIVER:$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "$ROOT/slurm_logs"
cd "$ROOT"

echo "FoamSurf native Voronoi warmup run=$RUN initial_cells=350000 adaptive=true"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$ENV/bin/python" -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

"$ENV/bin/python" scripts/run_voronoi_prior_warmup.py \
  --config configs/voronoi_prior_warmup.yaml \
  --output-dir "$OUTPUT" \
  --wandb-name "$RUN" \
  --wandb-mode "${WANDB_MODE:-online}"

test -f "$OUTPUT/voronoi_warmup_report.json"
for ITERATION in 0 500 1000 2000 3000 4000 5000; do
  printf -v PADDED "%08d" "$ITERATION"
  test -f "$OUTPUT/checkpoints/iter_${PADDED}.pt"
done
test -f "$OUTPUT/checkpoints/best.pt"
test -f "$OUTPUT/checkpoints/last.pt"
test -f "$OUTPUT/metrics/train.csv"
test -f "$OUTPUT/metrics/eval.csv"
test -f "$OUTPUT/summary.json"

"$ENV/bin/python" -c 'import json,sys; s=json.load(open(sys.argv[1])); assert s["status"] == "completed", s; assert int(s["final_iteration"]) == 5000, s; assert s["representation_mode"] == "voronoi_only", s; n=int(s["voronoi_count"]); assert 100000 <= n <= 500000, s; assert int(s["voronoi_adaptation_revision"]) > 0, s; assert int(s["total_voronoi_densified"]) + int(s["total_voronoi_pruned"]) > 0, s' "$OUTPUT/summary.json"

echo "Completed adaptive native Voronoi warmup: $OUTPUT"
