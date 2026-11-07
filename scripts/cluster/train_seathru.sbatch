#!/usr/bin/env bash
# One SeaThru scene with configs/seathru.yaml, seed 42, plus extra overrides (EXTRA, space-separated KEY=VALUE).
#   sbatch --export=ALL,SCENE=Curasao-Undistorted,TAG=water_anchor,EXTRA="medium_fix.water_anchor_weight=0.03" \
#          scripts/cluster/train_seathru.sbatch
# Run: outputs/$TAG/$SCENE (train.py's layout), provenance.txt next to it.
#SBATCH --job-name=foamsurf_train
#SBATCH --output=/shared/results/gmtobias/repos/foamsurf/outputs/logs/%j_%x.out
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --qos=batch
#SBATCH --time=03:00:00
#SBATCH --exclude=c11,c15,c22
set -euo pipefail
source /shared/results/gmtobias/repos/foamsurf/scripts/cluster/env.sh
SCENE="${SCENE:?scene directory name under foamsurf_seathru_2026-11-05/data}"
TAG="${TAG:?output tag}"
DATA=$REPOSITORY/foamsurf_seathru_2026-11-05/data/$SCENE
OUT=$REPOSITORY/outputs/$TAG/$SCENE
mkdir -p "$OUT"
SETS=()
for item in ${EXTRA:-}; do SETS+=(--set "$item"); done
{
  date -Is; git -C "$REPOSITORY" rev-parse HEAD; git -C "$REPOSITORY" status --short 2>&1 | head || true
  hostname; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  echo "train.py --scene $DATA --out $OUT --seed 42 ${SETS[*]}"
} > "$OUT/provenance.txt"
cd "$REPOSITORY"
"$PYTHON" train.py --scene "$DATA" --out "$OUT" --seed 42 "${SETS[@]}" 2>&1 | tee "$OUT/train.log"
