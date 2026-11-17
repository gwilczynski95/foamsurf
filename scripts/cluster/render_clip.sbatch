#!/usr/bin/env bash
# One part of a ray-traced clip (vis/render_clip.py --part), frames to outputs/clips/$SHOT/frames.
#   sbatch --export=ALL,SHOT=panama_showcase,SCENE=Panama-Undisorted,RUNTAG=water_anchor,PART=0,PARTS=6 \
#          scripts/cluster/render_clip.sbatch
# Model: outputs/$RUNTAG/$SCENE (train_seathru.sbatch). Meshes: $FALCON/source/... (default $REPOSITORY/assets/FALCON).
# The video is assembled from all parts' frames afterwards (no ffmpeg on the nodes).
#SBATCH --job-name=fs_clip
#SBATCH --output=/shared/results/gmtobias/repos/foamsurf/outputs/logs/%j_%x.out
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --qos=batch
#SBATCH --time=02:00:00
#SBATCH --exclude=c11,c15,c22
set -euo pipefail
source /shared/results/gmtobias/repos/foamsurf/scripts/cluster/env.sh
SHOT="${SHOT:?shot name under vis/shots}"; SCENE="${SCENE:?scene directory}"; RUNTAG="${RUNTAG:-water_anchor}"
PART="${PART:-0}"; PARTS="${PARTS:-1}"
export FALCON="${FALCON:-$REPOSITORY/assets/FALCON}"
cd "$REPOSITORY"
echo "$(date -Is) $(git rev-parse HEAD) $(hostname) part $PART/$PARTS"
"$PYTHON" vis/render_clip.py --run "outputs/$RUNTAG/$SCENE" --scene "foamsurf_seathru_2026-11-05/data/$SCENE" \
    --shot "vis/shots/$SHOT.json" --out "outputs/clips/$SHOT" --part "$PART" "$PARTS"
