#!/usr/bin/env bash
#SBATCH --job-name=fs_sweep1
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/sweep1_%A_%a.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/sweep1_%A_%a.err
#SBATCH --array=0-19%10
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --qos=batch
#SBATCH --exclude=c22,c11
#SBATCH --time=24:00:00

set -Eeuo pipefail

PROJECT_DIRECTORY="/shared/results/gmwilczn/foamsurf/foamsurf"
CONDA_ENVIRONMENT="/shared/results/gmwilczn/conda/envs/foamsurf"
PYTHON="$CONDA_ENVIRONMENT/bin/python"
CONFIGURATION="$PROJECT_DIRECTORY/configs/b1_lego_layers.yaml"
DRIVER_DIRECTORY="/shared/results/gmwilczn/drivers/NVIDIA-Linux-x86_64-535.183.01"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-foamsurf_sweep1}"
SEED="${SEED:-42}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

LABELS=(
    "00_noreset"
    "01_freeze_only200"
    "02_reset001"
    "03_reset001_f200"
    "04_reset001_f500"
    "05_reset005"
    "06_reset005_f200"
    "07_reset010"
    "08_reset010_f200"
    "09_reset010_f500"
    "10_reset010_f200_keepadam"
    "11_official_white_reset"
    "12_loss_l1"
    "13_loss_l1dssim"
    "14_opacity_lr005"
    "15_opacity_cull005"
    "16_grad0005"
    "17_grad0010"
    "18_alpha_clone"
    "19_alpha_clone_split"
)

if (( TASK_ID < 0 || TASK_ID >= ${#LABELS[@]} )); then
    echo "Invalid sweep task id: $TASK_ID" >&2
    exit 2
fi
LABEL="${LABELS[$TASK_ID]}"
RUN_IDENTIFIER="${SLURM_ARRAY_JOB_ID:-manual}_${TASK_ID}"
OUTPUT_DIRECTORY="${OUTPUT_ROOT:-$PROJECT_DIRECTORY/outputs/foamsurf_sweep1}/${LABEL}_${RUN_IDENTIFIER}"
WANDB_NAME="sweep1-${LABEL}-${RUN_IDENTIFIER}"

export PATH="$CONDA_ENVIRONMENT/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME="$CONDA_ENVIRONMENT"
export FOAMSURF_DRIVER_LIBRARY_DIRECTORY="$DRIVER_DIRECTORY"
export LD_LIBRARY_PATH="$DRIVER_DIRECTORY:$CONDA_ENVIRONMENT/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

if [[ ! -x "$PYTHON" ]]; then
    echo "Python not found: $PYTHON" >&2
    exit 1
fi
if [[ ! -f "$PROJECT_DIRECTORY/renderer/foamsurf/_C.so" ]]; then
    echo "Renderer is not built. Run ./build.sh before sbatch." >&2
    exit 1
fi
if [[ ! -f "$PROJECT_DIRECTORY/renderer/foamsurf/_optix/PYOPTIXFLARERENDERER.so" ]]; then
    echo "OptiX renderer is not built. Run ./build.sh before sbatch." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIRECTORY" "$PROJECT_DIRECTORY/slurm_logs"
cd "$PROJECT_DIRECTORY"

echo "FoamSurf sweep1: 30k training / densification until 15k"
echo "array_job_id=${SLURM_ARRAY_JOB_ID:-manual}"
echo "array_task_id=$TASK_ID"
echo "node=${SLURMD_NODENAME:-unknown}"
echo "label=$LABEL"
echo "seed=$SEED"
echo "wandb_project=$WANDB_PROJECT"
echo "wandb_name=$WANDB_NAME"
echo "output=$OUTPUT_DIRECTORY"
echo "runaway_max_points=1000000"
nvidia-smi

"$PYTHON" scripts/run_sweep1.py \
    --config "$CONFIGURATION" \
    --index "$TASK_ID" \
    --output-dir "$OUTPUT_DIRECTORY" \
    --wandb-name "$WANDB_NAME" \
    --wandb-mode "$WANDB_MODE" \
    --wandb-project "$WANDB_PROJECT" \
    --seed "$SEED"

test -f "$OUTPUT_DIRECTORY/checkpoints/last.pt"
test -f "$OUTPUT_DIRECTORY/checkpoints/best.pt"
test -f "$OUTPUT_DIRECTORY/metrics/train.csv"
test -f "$OUTPUT_DIRECTORY/summary.json"
"$PYTHON" -c 'import json,sys; s=json.load(open(sys.argv[1])); assert s["status"] in {"completed", "runaway_population"}, s; assert int(s["gaussian_count"]) > 0, s; assert (s["status"] == "completed" and int(s["final_iteration"]) == 30000) or (s["status"] == "runaway_population" and int(s["gaussian_count"]) > 1000000), s' "$OUTPUT_DIRECTORY/summary.json"

echo "FoamSurf sweep1 task finished: status recorded in $OUTPUT_DIRECTORY/summary.json"
