#!/usr/bin/env bash
#SBATCH --job-name=fs_medfix
#SBATCH --output=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/medfix_%A_%a.out
#SBATCH --error=/shared/results/gmwilczn/foamsurf/foamsurf/slurm_logs/medfix_%A_%a.err
#SBATCH --partition=rtx4090_batch
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --qos=batch
#SBATCH --exclude=c22,c11
#SBATCH --time=24:00:00
#SBATCH --array=0-23%7
# IUI3 sweeps over named variants of the B0 recipe (configs/iui3_5k.yaml with
# sigmoid RGB for Gaussians and cells). Each task trains, then audits best.pt and
# last.pt (scripts/medium_audit.py).
#
#   FS_VARIANT_SET=gauss sbatch --array=0-17%7 scripts/sweep_medium_fix.sbatch
#   FS_VARIANT_SET=far sbatch --array=0-23%7 scripts/sweep_medium_fix.sbatch
# Extra seeds for chosen variants (array size = variants x seeds):
#   FS_SEED_VARIANTS="B0_chroma100_bk" FS_SEEDS="44 45" sbatch --array=0-1%7 scripts/sweep_medium_fix.sbatch
# Resume a set (finished stages are skipped, the audits run again):
#   FS_RUN_ID=<job id> FS_RESUME=1 FS_VARIANT_SET=... sbatch --array=... scripts/sweep_medium_fix.sbatch
#
# Variant names: B0[_chroma<NNN>][_scov<NNN>|_scov<D>e<E>][_under<NNN>][_g<N>k][_far<D>s<SSS>][_bk][_seed<S>]
#   _chroma<NNN>  backscatter chroma prior weight NNN/100 (B0_chroma100_bk = configs/iui3_default.yaml)
#   _scov...      one-sided seabed coverage weight NNN/100 or D*10^-E
#   _under<NNN>   under-seabed medium penalty weight NNN/100
#   _g<N>k        densify threshold 2e-4 and at most N thousand Gaussians
#   _far<D>s<SSS> prune Gaussians larger than SSS/1000 farther than D from every training camera
#   _bk           black background
#   _seed<S>      seed S for warmup and joint
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

# Gaussian count: B0_bk, the default and the default + seabed coverage 3e-3,
# each at 500k / 700k / 1M Gaussians, two seeds.
GAUSS_VARIANTS=()
for base in B0 B0_chroma100 B0_chroma100_scov3e3; do
  for cap in 500 700 1000; do GAUSS_VARIANTS+=("${base}_g${cap}k_bk" "${base}_g${cap}k_bk_seed43"); done
done
# Pruning of large Gaussians far from the cameras (they fill open water), on the
# default + seabed coverage 3e-3 at up to 700k Gaussians: distance x scale grid,
# and seabed coverage 5e-3 / 1e-2 at (15, 0.05); two seeds. The unpruned control
# B0_chroma100_scov3e3_g700k_bk runs in the same set (same renderer and code).
FAR_VARIANTS=("B0_chroma100_scov3e3_g700k_bk")
for distance in 10 15 20; do
  for scale in 030 050 080; do
    FAR_VARIANTS+=("B0_chroma100_scov3e3_g700k_far${distance}s${scale}_bk")
  done
done
FAR_VARIANTS+=("B0_chroma100_scov5e3_g700k_far15s050_bk" "B0_chroma100_scov1e2_g700k_far15s050_bk")
SEEDED=()
for variant in "${FAR_VARIANTS[@]}"; do SEEDED+=("$variant" "${variant}_seed43"); done
FAR_VARIANTS=("${SEEDED[@]}")
case "${FS_VARIANT_SET:-gauss}" in
  gauss) VARIANTS=("${GAUSS_VARIANTS[@]}") ;;
  far) VARIANTS=("${FAR_VARIANTS[@]}") ;;
  *) echo "Unknown FS_VARIANT_SET: $FS_VARIANT_SET" >&2; exit 2 ;;
esac

# Prints one manifest override per line for a variant name.
variant_overrides() {
  local name=$1
  if [[ "$name" =~ ^(.+)_seed([0-9]+)$ ]]; then
    local seed=${BASH_REMATCH[2]}
    variant_overrides "${BASH_REMATCH[1]}" || return
    echo "variant.seed=$seed"
    echo "warmup.seed=$seed"
    echo "joint.seed=$seed"
    return 0
  fi
  if [[ "$name" =~ ^(.+)_bk$ ]]; then
    variant_overrides "${BASH_REMATCH[1]}" || return
    echo "warmup.background=[0.0, 0.0, 0.0]"
    echo "joint.background=[0.0, 0.0, 0.0]"
    return 0
  fi
  if [[ "$name" =~ ^(.+)_far([0-9]+)s([0-9]{3})$ ]]; then
    local distance=${BASH_REMATCH[2]} scale=${BASH_REMATCH[3]}
    variant_overrides "${BASH_REMATCH[1]}" || return
    echo "joint.gaussian_adaptation.far_prune_distance=${distance}.0"
    echo "joint.gaussian_adaptation.far_prune_scale=0.${scale}"
    return 0
  fi
  if [[ "$name" =~ ^(.+)_g([0-9]+)k$ ]]; then
    local cap=$(( BASH_REMATCH[2] * 1000 ))
    variant_overrides "${BASH_REMATCH[1]}" || return
    echo "joint.gaussian_adaptation.gradient_threshold=0.0002"
    echo "joint.gaussian_adaptation.max_points=$cap"
    return 0
  fi
  if [[ "$name" =~ ^(.+)_scov([1-9])e([0-9])$ ]]; then
    local digit=${BASH_REMATCH[2]} exponent=${BASH_REMATCH[3]}
    variant_overrides "${BASH_REMATCH[1]}" || return
    # YAML reads "1e-3" as a string; "1.0e-3" is a float.
    echo "medium_fix.surface_coverage_weight=${digit}.0e-${exponent}"
    return 0
  fi
  if [[ "$name" =~ ^(.+)_(chroma|scov|under)([0-9]{3})$ ]]; then
    local value=$(( 10#${BASH_REMATCH[3]} )) key=${BASH_REMATCH[2]}
    case "$key" in
      chroma) key=chroma_weight ;;
      scov) key=surface_coverage_weight ;;
      under) key=under_surface_weight ;;
    esac
    variant_overrides "${BASH_REMATCH[1]}" || return
    printf 'medium_fix.%s=%d.%02d\n' "$key" $(( value / 100 )) $(( value % 100 ))
    return 0
  fi
  if [[ "$name" == B0 ]]; then
    return 0
  fi
  echo "Unknown variant: $name" >&2
  return 2
}

TASK=${SLURM_ARRAY_TASK_ID:?run through sbatch}
if [[ -n "${FS_SEED_VARIANTS:-}" ]]; then
  read -r -a CHOSEN <<< "$FS_SEED_VARIANTS"
  read -r -a SEEDS <<< "${FS_SEEDS:?FS_SEEDS is required with FS_SEED_VARIANTS}"
  (( TASK < ${#CHOSEN[@]} * ${#SEEDS[@]} )) || { echo "Array index $TASK out of range" >&2; exit 2; }
  VARIANT="${CHOSEN[TASK / ${#SEEDS[@]}]}_seed${SEEDS[TASK % ${#SEEDS[@]}]}"
else
  (( TASK < ${#VARIANTS[@]} )) || { echo "Array index $TASK out of range" >&2; exit 2; }
  VARIANT=${VARIANTS[TASK]}
fi
variant_overrides "$VARIANT" > /dev/null  # fail loudly on an unknown name
OVERRIDES=()
mapfile -t OVERRIDES < <(variant_overrides "$VARIANT")

cd "$ROOT"
mkdir -p slurm_logs
export TORCH_EXTENSIONS_DIR="$ROOT/outputs/build_medfix_${SLURM_ARRAY_JOB_ID}_${TASK}"
mkdir -p "$TORCH_EXTENSIONS_DIR"
# Temporary files (W&B media) on shared storage: a node's local /tmp filled up in 533795_21.
export TMPDIR="$ROOT/outputs/tmp/${SLURM_ARRAY_JOB_ID}_${TASK}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
RUN_ID="${FS_RUN_ID:-${SLURM_ARRAY_JOB_ID}}"
OUTPUT="${FS_OUTPUT_ROOT:-outputs/iui3_medium_fix}/${RUN_ID}/${VARIANT}"

ARGS=(
  --config configs/iui3_5k.yaml
  --output-dir "$OUTPUT"
  --gaussian-rgb-activation sigmoid
  --medium-color-activation sigmoid
  --wandb-project "${FS_WANDB_PROJECT:-foamsurf_iui3_medium_color}"
  --run-suffix "$VARIANT"
)
for item in "${OVERRIDES[@]}"; do ARGS+=(--set "$item"); done
if [[ "${FS_RESUME:-0}" == 1 ]]; then ARGS+=(--resume); fi

echo "variant=$VARIANT output=$OUTPUT"
if (( ${#OVERRIDES[@]} )); then printf '  --set %s\n' "${OVERRIDES[@]}"; fi
if [[ "${FS_DRY_RUN:-0}" == 1 ]]; then
  python train.py "${ARGS[@]}" --dry-run > /dev/null
  echo "dry run ok"
  exit 0
fi
python train.py "${ARGS[@]}"
for checkpoint in best last; do
  python scripts/medium_audit.py --run "$OUTPUT" --checkpoint "$checkpoint"
done
