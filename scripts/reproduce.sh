#!/usr/bin/env bash
# Paper training and evaluation, sequentially on one GPU.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/renderer:$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON=${FOAMSURF_PYTHON:-python}
original_args=("$@")
target=${1:-}
[[ -n "$target" ]] || { echo 'Usage: scripts/reproduce.sh seathru|ablation|garden|synthetic|applications|all [--seeds 42,43,44] [--scenes IUI3,lego] [--slurm] [--dry-run] [--check] [--applications insertion,edits,style,teaser]' >&2; exit 2; }
shift
case "$target" in
    seathru|ablation|garden|synthetic|applications|all) ;;
    *) echo "Unknown target: $target" >&2; exit 2 ;;
esac
seed_spec=''
scene_spec=''
out=${FS_OUTPUTS:-outputs/paper}
background=all
pipeline=hybrid
dry_run=0
slurm=0
check=0
applications=insertion,edits,style
while (($#)); do
    case "$1" in
        --seeds) seed_spec=$2; shift 2 ;;
        --scenes) scene_spec=$2; shift 2 ;;
        --out) out=$2; shift 2 ;;
        --background) background=$2; shift 2 ;;
        --pipeline) pipeline=$2; shift 2 ;;
        --applications) applications=$2; shift 2 ;;
        --check) check=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        --slurm) slurm=1; shift ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ "$background" =~ ^(black|white|all)$ ]] || { echo 'Invalid --background' >&2; exit 2; }
[[ "$pipeline" =~ ^(hybrid|voronoi|all)$ ]] || { echo 'Invalid --pipeline' >&2; exit 2; }
if [[ -n "$seed_spec" && ! "$seed_spec" =~ ^[0-9]+(,[0-9]+)*$ ]]; then echo 'Invalid --seeds' >&2; exit 2; fi
out=$(realpath -m "$out")
SEATHRU=${FS_DATASETS:-data/SeathruNeRF_dataset}
SYNTHETIC=${FS_SYNTHETIC:-data/synthetic}
FOGGY=${FS_FOGGY:-data/foggy_garden}
MASKS=${FS_MASKS:-assets/masks}

run() {
    printf '%q ' "$@"
    printf '\n'
    if (( ! dry_run )); then "$@"; fi
}

# Validate a complete training plan before the first expensive command.
# Dry runs stay usable before datasets have been downloaded.
if [[ "$target" != applications ]] && (( check || (! dry_run && ! slurm) )); then
    plan_args=()
    for arg in "${original_args[@]}"; do
        [[ "$arg" == --check || "$arg" == --dry-run || "$arg" == --slurm ]] || plan_args+=("$arg")
    done
    plan=$(bash "$ROOT/scripts/reproduce.sh" "${plan_args[@]}" --dry-run)
    printf '%s\n' "$plan" | "$PYTHON" scripts/preflight.py
    (( ! check )) || exit 0
fi
if [[ "$target" == applications ]] && (( ! slurm || check )); then
    app_args=(--applications "$applications" --scenes "$scene_spec" --out "$out")
    (( ! dry_run )) || app_args+=(--dry-run)
    (( ! check )) || app_args+=(--check)
    [[ -z "$seed_spec" ]] || { echo 'Applications use fixed pretrained models and style seed 0; omit --seeds.' >&2; exit 2; }
    exec "$PYTHON" scripts/reproduce_applications.py "${app_args[@]}"
fi

if [[ "$target" == applications ]] && (( slurm )); then
    [[ -z "$seed_spec" ]] || { echo 'Applications use fixed pretrained models and style seed 0; omit --seeds.' >&2; exit 2; }
    "$PYTHON" scripts/reproduce_applications.py --applications "$applications" --scenes "$scene_spec" --out "$out" --dry-run >/dev/null
fi

# Submit the same sequential workflow; arguments remain separate shell words.
if (( slurm )); then
    local_args=()
    for arg in "${original_args[@]}"; do
        [[ "$arg" == --slurm || "$arg" == --dry-run ]] || local_args+=("$arg")
    done
    run sbatch --chdir="$ROOT" --job-name="foamsurf-$target" "$ROOT/scripts/slurm/reproduce.sbatch" "${local_args[@]}"
    exit 0
fi

selected() {
    [[ -z "$scene_spec" || ",${scene_spec}," == *",$1,"* || ",${scene_spec}," == *",$2,"* ]]
}

seeds() { IFS=',' read -r -a SEEDS <<< "${seed_spec:-$1}"; }

render_metrics() {
    local run_dir=$1
    shift
    run "$PYTHON" render.py --run "$run_dir" --modes full --metrics "$@" --out "$run_dir/evaluation/render"
}

seathru() {
    local key scene seed run_dir
    seeds 42,43,44
    for key in IUI3 Curasao JapaneseGradens Panama; do
        case "$key" in
            IUI3) scene=IUI3-RedSea-Undistorted ;;
            Curasao) scene=Curasao-Undistorted ;;
            JapaneseGradens) scene=JapaneseGradens-RedSea-Undistorted ;;
            Panama) scene=Panama-Undisorted ;;
        esac
        selected "$key" "$scene" || continue
        for seed in "${SEEDS[@]}"; do
            run_dir="$out/seathru/full/$scene/seed_$seed"
            run "$PYTHON" train.py --config configs/seathru.yaml --scene "$SEATHRU/$scene" --seed "$seed" --out "$run_dir" --resume
            render_metrics "$run_dir" --fps 32
            run "$PYTHON" evaluate.py separation --run "$run_dir" --masks "$MASKS/$scene.npz" --out "$run_dir/evaluation/separation"
        done
    done
}

ablation() {
    local scene variant seed run_dir
    seeds 42,43
    for scene in IUI3-RedSea-Undistorted Curasao-Undistorted JapaneseGradens-RedSea-Undistorted Panama-Undisorted; do
        selected "${scene%%-*}" "$scene" || continue
        for variant in full nomedium homog nopriors nowlight noocean nbr0 noisp; do
            local config=configs/seathru.yaml
            [[ "$variant" == full ]] || config="configs/ablations/$variant.yaml"
            for seed in "${SEEDS[@]}"; do
                run_dir="$out/ablation/$variant/$scene/seed_$seed"
                run "$PYTHON" train.py --config "$config" --scene "$SEATHRU/$scene" --seed "$seed" --out "$run_dir" --resume
                render_metrics "$run_dir"
                run "$PYTHON" evaluate.py separation --run "$run_dir" --masks "$MASKS/$scene.npz" --out "$run_dir/evaluation/separation"
            done
        done
    done
}

garden() {
    local level seed run_dir
    seeds 42,43
    if [[ -n ${FS_FOG_DEPTHS:-} ]]; then
        run "$PYTHON" scripts/make_foggy_garden.py --depth-dir "$FS_FOG_DEPTHS" --clear-dir "${FS_GARDEN_CLEAR:?set FS_GARDEN_CLEAR}" --colmap-model "${FS_GARDEN_COLMAP:?set FS_GARDEN_COLMAP}" --output-dir "$FOGGY"
    fi
    for level in easy hard; do
        selected "$level" "garden_$level" || continue
        for seed in "${SEEDS[@]}"; do
            run_dir="$out/garden/full/garden_$level/seed_$seed"
            run "$PYTHON" train.py --config configs/garden_fog.yaml --scene "$FOGGY/garden_$level" --seed "$seed" --out "$run_dir" --resume
            render_metrics "$run_dir"
            run "$PYTHON" evaluate.py restoration --run "$run_dir" --out "$run_dir/evaluation/restoration"
        done
    done
}

synthetic() {
    local scene kind bg seed run_dir rgb object_layer
    seeds 42,43,44
    for scene in lego ficus; do
        selected "$scene" "$scene" || continue
        object_layer=lego_lego
        [[ "$scene" != ficus ]] || object_layer=ficus_object
        for kind in hybrid voronoi; do
            [[ "$pipeline" == all || "$pipeline" == "$kind" ]] || continue
            for bg in black white; do
                [[ "$background" == all || "$background" == "$bg" ]] || continue
                rgb='[0,0,0]'
                [[ "$bg" == black ]] || rgb='[1,1,1]'
                for seed in "${SEEDS[@]}"; do
                    [[ "$bg" == black || "$seed" == 42 ]] || continue
                    run_dir="$out/synthetic/${kind}_${bg}/$scene/seed_$seed"
                    run "$PYTHON" train.py --config "configs/synthetic_$kind.yaml" --scene "$SYNTHETIC/$scene/${scene}_both" --seed "$seed" --out "$run_dir" --resume \
                        --set "both.background=$rgb" --set "both.dataset.gaussian_ground_truth_path=$SYNTHETIC/$scene/$object_layer" \
                        --set "both.dataset.medium_ground_truth_path=$SYNTHETIC/$scene/${scene}_clouds"
                    render_metrics "$run_dir"
                    run "$PYTHON" evaluate.py layers --run "$run_dir" --out "$run_dir/evaluation/layers"
                    run "$PYTHON" evaluate.py transmittance --run "$run_dir" --out "$run_dir/evaluation/transmittance"
                done
            done
        done
    done
}

case "$target" in
    seathru|ablation|garden|synthetic) "$target" ;;
    all) seathru; ablation; garden; synthetic ;;
    *) echo "Unknown target: $target" >&2; exit 2 ;;
esac
run "$PYTHON" scripts/summarize.py "$target" --root "$out"
