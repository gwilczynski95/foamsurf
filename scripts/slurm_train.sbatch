#!/bin/bash
#SBATCH --job-name=foamsurf
#SBATCH --partition=rtx4090_batch
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%A_%a.out
# Train FoamSurf on the four SeaThru-NeRF scenes (one array task per scene):
#   sbatch --array=0-3 scripts/slurm_train.sbatch
#   FS_CONFIG=configs/garden_fog.yaml FS_DATA=/data/foggy_garden FS_SCENES="garden_easy garden_hard" \
#     sbatch --array=0-1 scripts/slurm_train.sbatch
# Settings (environment): FS_DATA (scene parent directory), FS_SCENES, FS_CONFIG, FS_SEED, FS_OUT (run parent
# directory), FS_ENV (conda environment), FS_WANDB (W&B project; logging off when empty).
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
ENV="${FS_ENV:-$CONDA_PREFIX}"
DATA="${FS_DATA:-/shared/sets/datasets/SeathruNeRF_dataset}"
read -r -a SCENES <<< "${FS_SCENES:-IUI3-RedSea-Undistorted Curasao-Undistorted JapaneseGradens-RedSea-Undistorted Panama-Undisorted}"
SCENE="${SCENES[${SLURM_ARRAY_TASK_ID:-0}]}"
SEED="${FS_SEED:-42}"
OUT="${FS_OUT:-$ROOT/outputs}/${SLURM_ARRAY_JOB_ID:-local}/${SCENE}_seed${SEED}"

export PATH="$ENV/bin:$PATH" CUDA_HOME="$ENV" LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR="$ROOT/outputs/.torch_extensions/${SLURM_JOB_ID:-local}"  # PPISP / SH kernels, per job
cd "$ROOT"
ARGS=(--scene "$DATA/$SCENE" --out "$OUT" --config "${FS_CONFIG:-configs/seathru.yaml}" --seed "$SEED")
[[ -n "${FS_WANDB:-}" ]] && ARGS+=(--wandb "$FS_WANDB")
[[ -d "$OUT" ]] && ARGS+=(--resume)
echo "scene=$SCENE seed=$SEED out=$OUT"
python train.py "${ARGS[@]}"
rm -rf "$TORCH_EXTENSIONS_DIR"
