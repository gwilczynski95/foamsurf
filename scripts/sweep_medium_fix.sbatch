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
#SBATCH --array=0-10%7
# Global medium color / coverage-prior sweep on IUI3 (both sigmoid activations).
#
# Main sweep (11 runs, at most 7 at a time):
#   mkdir -p slurm_logs && sbatch scripts/sweep_medium_fix.sbatch
# Uniform (box) Voronoi initialization for B0 and the leader, 5k and 20k cells:
#   FS_VARIANT_SET=box sbatch --array=0-3%7 scripts/sweep_medium_fix.sbatch
# Stage A: Voronoi layout x color mode, all with a BLACK background (12 runs):
#   FS_VARIANT_SET=stageA sbatch --array=0-11%7 scripts/sweep_medium_fix.sbatch
# Slower medium updates in cells containing Gaussians, BLACK background (18 runs):
#   FS_VARIANT_SET=prox sbatch --array=0-17%7 scripts/sweep_medium_fix.sbatch
# Backscatter chroma prior and stronger cell-color smoothing for B0, BLACK background (11 runs):
#   smoke=$(sbatch --parsable scripts/smoke_tests.sbatch)
#   FS_VARIANT_SET=chroma sbatch --dependency=afterok:$smoke --array=0-10%7 scripts/sweep_medium_fix.sbatch
# Seabed coverage from the default B0_chroma100_bk, BLACK background (22 runs):
#   smoke=$(sbatch --parsable scripts/smoke_tests.sbatch)
#   FS_VARIANT_SET=seabed sbatch --dependency=afterok:$smoke --array=0-21%7 scripts/sweep_medium_fix.sbatch
# Smaller one-sided seabed coverage weights 1e-4, 1e-3, 3e-3 (6 runs):
#   FS_VARIANT_SET=seabed2 sbatch --array=0-5%7 scripts/sweep_medium_fix.sbatch
# Both stage A and prox in one array (30 runs):
#   FS_VARIANT_SET=stageA_prox sbatch --array=0-29%7 scripts/sweep_medium_fix.sbatch
# Extra seeds for chosen variants (array size = variants x seeds):
#   FS_SEED_VARIANTS="E1a_lr1_s01 E1b_w005_reset" FS_SEEDS="43 44" \
#     sbatch --array=0-3%7 scripts/sweep_medium_fix.sbatch
# Each task trains, then audits best.pt and last.pt (scripts/medium_audit.py).
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

VARIANTS=(
  B0
  E1a_lr1_s01 E1a_lr1_s1 E1a_lr03_s01 E1a_lr03_s1
  E1b_w001 E1b_w005 E1b_w02 E1b_w001_reset E1b_w005_reset E1b_w02_reset
)
BOX_VARIANTS=(B0_box5k B0_box20k E1a_lr1_s1_box5k E1a_lr1_s1_box20k)
# <base>[_in<P><r|n>][_prox<NN>][_ring]_bk[_seed<S>]: P% of the 5k sites as the box
# grid, the rest outside it (r = radial log shell, n = normal with box samples
# rejected); no _in part = the original normal layout; _prox<NN> = cells containing
# Gaussians keep 0.NN of each medium step (prox10 = 0.10) (_ring adds their neighbors); _bk = black background.
STAGEA_VARIANTS=()
PROX_VARIANTS=()
for base in B0 E1a_lr1_s1; do
  for layout in "" _in75r _in50r _in25r _in50n; do STAGEA_VARIANTS+=("${base}${layout}_bk"); done
  STAGEA_VARIANTS+=("${base}_bk_seed43")
  for layout in "" _in50r; do
    for prox in 50 30 10 00; do PROX_VARIANTS+=("${base}${layout}_prox${prox}_bk"); done
  done
  PROX_VARIANTS+=("${base}_in50r_prox10_ring_bk")
done
# _chroma<NNN>: backscatter chroma prior weight NNN/100; _csmooth<N>: cell color smoothing weight N.
CHROMA_VARIANTS=()
for setting in chroma001 chroma010 chroma100 csmooth10 csmooth100; do
  CHROMA_VARIANTS+=("B0_${setting}_bk" "B0_${setting}_bk_seed43")
done
CHROMA_VARIANTS+=(B0_in50r_chroma010_bk)
# On top of the default B0_chroma100_bk: _kappa<N> initial Gaussian edge sharpness N;
# _scov<NNN> one-sided seabed coverage weight NNN/100; _under<NNN> under-seabed medium
# penalty weight NNN/100; _scale0dens2 no Gaussian size penalty + densify threshold 2e-4.
SEABED_VARIANTS=()
for setting in kappa2 kappa4 scov001 scov010 scov100 scov200 under001 under010 under100 under200 scale0dens2; do
  SEABED_VARIANTS+=("B0_chroma100_${setting}_bk" "B0_chroma100_${setting}_bk_seed43")
done
# _scov<D>e<E>: one-sided seabed coverage weight D * 10^-E.
SEABED2_VARIANTS=()
for setting in scov1e4 scov1e3 scov3e3; do
  SEABED2_VARIANTS+=("B0_chroma100_${setting}_bk" "B0_chroma100_${setting}_bk_seed43")
done
ALL_VARIANTS=("${VARIANTS[@]}" "${BOX_VARIANTS[@]}" "${STAGEA_VARIANTS[@]}" "${PROX_VARIANTS[@]}"
              "${CHROMA_VARIANTS[@]}" "${SEABED_VARIANTS[@]}" "${SEABED2_VARIANTS[@]}")
case "${FS_VARIANT_SET:-main}" in
  main) ;;
  box) VARIANTS=("${BOX_VARIANTS[@]}") ;;
  stageA) VARIANTS=("${STAGEA_VARIANTS[@]}") ;;
  prox) VARIANTS=("${PROX_VARIANTS[@]}") ;;
  stageA_prox) VARIANTS=("${STAGEA_VARIANTS[@]}" "${PROX_VARIANTS[@]}") ;;
  chroma) VARIANTS=("${CHROMA_VARIANTS[@]}") ;;
  seabed) VARIANTS=("${SEABED_VARIANTS[@]}") ;;
  seabed2) VARIANTS=("${SEABED2_VARIANTS[@]}") ;;
  *) echo "Unknown FS_VARIANT_SET: $FS_VARIANT_SET" >&2; exit 2 ;;
esac

# Prints one override per line for a variant name.
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
  if [[ "$name" =~ ^(.+)_ring$ ]]; then
    variant_overrides "${BASH_REMATCH[1]}" || return
    echo "medium_fix.proximity_neighbors=true"
    return 0
  fi
  if [[ "$name" =~ ^(.+)_kappa([0-9]+)$ ]]; then
    local value=${BASH_REMATCH[2]}
    variant_overrides "${BASH_REMATCH[1]}" || return
    echo "joint.model.gaussian_initial_kappa=$value.0"
    return 0
  fi
  if [[ "$name" =~ ^(.+)_scov([1-9])e([0-9])$ ]]; then
    local digit=${BASH_REMATCH[2]} exponent=${BASH_REMATCH[3]}
    variant_overrides "${BASH_REMATCH[1]}" || return
    echo "medium_fix.surface_coverage_weight=${digit}.0e-${exponent}"
    return 0
  fi
  if [[ "$name" =~ ^(.+)_(scov|under)([0-9]{3})$ ]]; then
    local value=$(( 10#${BASH_REMATCH[3]} )) key=surface_coverage_weight
    [[ "${BASH_REMATCH[2]}" == under ]] && key=under_surface_weight
    variant_overrides "${BASH_REMATCH[1]}" || return
    printf 'medium_fix.%s=%d.%02d\n' "$key" $(( value / 100 )) $(( value % 100 ))
    return 0
  fi
  if [[ "$name" =~ ^(.+)_scale0dens2$ ]]; then
    variant_overrides "${BASH_REMATCH[1]}" || return
    echo "joint.optimization.gaussian_scale_regularization=0.0"
    echo "joint.gaussian_adaptation.gradient_threshold=0.0002"
    return 0
  fi
  if [[ "$name" =~ ^(.+)_chroma([0-9]{3})$ ]]; then
    local value=$(( 10#${BASH_REMATCH[2]} ))
    variant_overrides "${BASH_REMATCH[1]}" || return
    printf 'medium_fix.chroma_weight=%d.%02d\n' $(( value / 100 )) $(( value % 100 ))
    return 0
  fi
  if [[ "$name" =~ ^(.+)_csmooth([0-9]+)$ ]]; then
    local value=${BASH_REMATCH[2]}
    variant_overrides "${BASH_REMATCH[1]}" || return
    echo "rgb_smoothing.color=$value.0"
    return 0
  fi
  if [[ "$name" =~ ^(.+)_prox([0-9]{2})$ ]]; then
    local multiplier=0.${BASH_REMATCH[2]}
    variant_overrides "${BASH_REMATCH[1]}" || return
    echo "medium_fix.proximity_multiplier=$multiplier"
    return 0
  fi
  if [[ "$name" =~ ^(.+)_in([0-9]+)([rn])$ ]]; then
    local percent=${BASH_REMATCH[2]} outer=radial
    [[ "${BASH_REMATCH[3]}" == n ]] && outer=normal
    variant_overrides "${BASH_REMATCH[1]}" || return
    echo "profile.spatial_distribution=box_shell"
    echo "profile.spatial_inner_fraction=0.$percent"
    echo "profile.spatial_outer=$outer"
    return 0
  fi
  if [[ "$name" =~ ^(.+)_box([0-9]+)k$ ]]; then
    # Jittered near-cubic grid filling the COLMAP/camera box (+10% margin).
    local count=$(( BASH_REMATCH[2] * 1000 ))
    variant_overrides "${BASH_REMATCH[1]}" || return
    echo "profile.spatial_distribution=box"
    echo "profile.spatial_count=$count"
    echo "joint.voronoi_adaptation.max_points=$(( count > 5000 ? count : 5000 ))"
    return 0
  fi
  case "$name" in
    B0) return 0 ;;
    E1a_*|E1b_*) echo "medium_fix.color_mode=global" ;;
    *) echo "Unknown variant: $name" >&2; return 2 ;;
  esac
  case "$name" in
    E1a_lr03_*)
      echo "joint.optimization.medium_density_lr=0.003"
      echo "joint.optimization.medium_density_lr_final=0.0003"
      echo "joint.optimization.medium_sites_lr=1.5e-05"
      echo "joint.optimization.medium_sites_lr_final=1.5e-06" ;;
  esac
  case "$name" in
    E1a_*_s1)
      echo "rgb_smoothing.attenuation=1.0"
      echo "rgb_smoothing.backscatter=1.0" ;;
  esac
  if [[ "$name" == E1b_* ]]; then
    # WS-style culling while densifying; coverage prior on per-ray Gaussian alpha.
    echo "joint.gaussian_adaptation.min_opacity_densify=0.5"
    echo "medium_fix.coverage_start=1000"
    echo "medium_fix.coverage_ramp=2000"
    case "$name" in
      E1b_w001*) echo "medium_fix.coverage_weight=0.01" ;;
      E1b_w005*) echo "medium_fix.coverage_weight=0.05" ;;
      E1b_w02*) echo "medium_fix.coverage_weight=0.2" ;;
    esac
    if [[ "$name" == *_reset ]]; then
      echo "joint.gaussian_adaptation.opacity_reset_interval=3000"
      echo "joint.gaussian_adaptation.opacity_reset_value=0.5"
      echo "joint.gaussian_adaptation.opacity_reset_prune_grace=500"
      echo "joint.gaussian_adaptation.medium_freeze_after_reset=200"
    fi
  fi
}

TASK=${SLURM_ARRAY_TASK_ID:?run through sbatch}
if [[ -n "${FS_SEED_VARIANTS:-}" ]]; then
  read -r -a CHOSEN <<< "$FS_SEED_VARIANTS"
  read -r -a SEEDS <<< "${FS_SEEDS:?FS_SEEDS is required with FS_SEED_VARIANTS}"
  (( TASK < ${#CHOSEN[@]} * ${#SEEDS[@]} )) || { echo "Array index $TASK out of range" >&2; exit 2; }
  VARIANT=${CHOSEN[TASK / ${#SEEDS[@]}]}
  SEED=${SEEDS[TASK % ${#SEEDS[@]}]}
  RUN_NAME="${VARIANT}_seed${SEED}"
else
  (( TASK < ${#VARIANTS[@]} )) || { echo "Array index $TASK out of range" >&2; exit 2; }
  VARIANT=${VARIANTS[TASK]}
  SEED=""
  RUN_NAME=$VARIANT
fi

[[ " ${ALL_VARIANTS[*]} " == *" $VARIANT "* ]] || { echo "Unknown variant: $VARIANT" >&2; exit 2; }

cd "$ROOT"
mkdir -p slurm_logs
export TORCH_EXTENSIONS_DIR="$ROOT/outputs/build_medfix_${SLURM_ARRAY_JOB_ID}_${TASK}"
mkdir -p "$TORCH_EXTENSIONS_DIR"
RUN_ID="${FS_RUN_ID:-${SLURM_ARRAY_JOB_ID}}"
OUTPUT="${FS_OUTPUT_ROOT:-outputs/iui3_medium_fix}/${RUN_ID}/${RUN_NAME}"

ARGS=(
  --config configs/iui3_5k.yaml
  --output-dir "$OUTPUT"
  --gaussian-rgb-activation sigmoid
  --medium-color-activation sigmoid
  --wandb-project "${FS_WANDB_PROJECT:-foamsurf_iui3_medium_color}"
  --run-suffix "$RUN_NAME"
)
mapfile -t OVERRIDES < <(variant_overrides "$VARIANT")
if [[ -n "$SEED" ]]; then
  OVERRIDES+=("variant.seed=$SEED" "warmup.seed=$SEED" "joint.seed=$SEED")
fi
for item in "${OVERRIDES[@]}"; do ARGS+=(--set "$item"); done
if [[ "${FS_RESUME:-0}" == 1 ]]; then ARGS+=(--resume); fi

echo "variant=$VARIANT seed=${SEED:-default} output=$OUTPUT"
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
