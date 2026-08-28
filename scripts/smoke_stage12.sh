#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIRECTORY="/shared/results/gmwilczn/foamsurf/foamsurf"
CONDA_ENVIRONMENT="/shared/results/gmwilczn/conda/envs/foamsurf"
PYTHON="$CONDA_ENVIRONMENT/bin/python"
DRIVER_DIRECTORY="/shared/results/gmwilczn/drivers/NVIDIA-Linux-x86_64-535.183.01"
MODE="${1:-all}"
WANDB_MODE="${WANDB_MODE:-online}"
DRY_RUN="${DRY_RUN:-0}"
RUN_ID="${HOSTNAME:-local}_$(date +%Y%m%d_%H%M%S)_$$"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIRECTORY/outputs/stage12_baselines_local/$RUN_ID}"

case "$MODE" in
    all)
        PHASES=(gaussian voronoi)
        ;;
    gaussian|voronoi)
        PHASES=("$MODE")
        ;;
    *)
        echo "Usage: $0 [all|gaussian|voronoi]" >&2
        exit 2
        ;;
esac

case "$WANDB_MODE" in
    online|offline|disabled)
        ;;
    *)
        echo "WANDB_MODE must be online, offline, or disabled" >&2
        exit 2
        ;;
esac

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
    echo "Renderer is not built: renderer/foamsurf/_C.so" >&2
    echo "Run ./build.sh first." >&2
    exit 1
fi
if [[ ! -f "$PROJECT_DIRECTORY/renderer/foamsurf/_optix/PYOPTIXFLARERENDERER.so" ]]; then
    echo "OptiX renderer is not built." >&2
    echo "Run ./build.sh first." >&2
    exit 1
fi

printf 'FoamSurf Stage 1/2 full standalone baselines\n'
printf 'host=%s\n' "${HOSTNAME:-unknown}"
printf 'mode=%s\n' "$MODE"
printf 'iterations=%s\n' "30000 per representation"
printf 'wandb_mode=%s\n' "$WANDB_MODE"
printf 'output_root=%s\n' "$OUTPUT_ROOT"

if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$OUTPUT_ROOT"
    cd "$PROJECT_DIRECTORY"
    nvidia-smi
fi

for phase in "${PHASES[@]}"; do
    output_directory="$OUTPUT_ROOT/$phase"
    if [[ -e "$output_directory" ]]; then
        echo "Output already exists: $output_directory" >&2
        exit 1
    fi

    if [[ "$phase" == "gaussian" ]]; then
        configuration="$PROJECT_DIRECTORY/configs/stage1_gaussian_only.yaml"
        wandb_name="stage1-gaussian-baseline-$RUN_ID"
        wandb_project="foamsurf_stage1_gaussian"
    else
        configuration="$PROJECT_DIRECTORY/configs/stage2_voronoi_only.yaml"
        wandb_name="stage2-voronoi-baseline-$RUN_ID"
        wandb_project="foamsurf_stage2_voronoi"
    fi

    printf '\nFull 30k %s-only baseline\n' "$phase"
    printf 'config=%s\n' "$configuration"
    printf 'wandb_project=%s\n' "$wandb_project"
    printf 'wandb_name=%s\n' "$wandb_name"
    printf 'output=%s\n' "$output_directory"

    command=(
        "$PYTHON" train_b0.py
        --config "$configuration"
        --output-dir "$output_directory"
        --wandb-mode "$WANDB_MODE"
        --wandb-name "$wandb_name"
    )
    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'command:'
        printf ' %q' "${command[@]}"
        printf '\n'
        continue
    fi

    "${command[@]}"

    test -f "$output_directory/checkpoints/last.pt"
    test -f "$output_directory/checkpoints/best.pt"
    test -f "$output_directory/metrics/train.csv"
    test -f "$output_directory/metrics/eval.csv"
    test -f "$output_directory/summary.json"

    "$PYTHON" -c \
        'import json, sys; summary=json.load(open(sys.argv[1])); assert summary["status"] == "completed", summary; assert summary["final_iteration"] == 30000, summary' \
        "$output_directory/summary.json"

    printf '%s-only full baseline: OK (%s)\n' "$phase" "$output_directory"
done

if [[ "$DRY_RUN" == "1" ]]; then
    printf '\nDry run completed; no training was started.\n'
else
    printf '\nAll requested Stage 1/2 full baselines completed.\n'
    printf 'Results: %s\n' "$OUTPUT_ROOT"
fi
