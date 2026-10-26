#!/usr/bin/env bash
# Run tasks of scripts/train_seathru.sbatch one after another on a machine without Slurm
# (e.g. the RTX 3090 workstation). Same variant sets and suffixes as on the cluster.
#
#   FS_DATASETS=/path/to/SeathruNeRF_dataset FS_VARIANT_SET=diag FS_SLIM=0 scripts/run_local.sh 0 15
#
# Paths default to the 3090 workstation; override any of them from the environment:
#   FS_DATASETS (required: the SeathruNeRF_dataset folder), FS_ENV (conda env), FS_DRIVER (driver libraries),
#   FS_RUN_ID (output folder under outputs/seathru, default: local_<date>), TORCH_CUDA_ARCH_LIST.
# A failed task is logged and the next one starts. Logs: slurm_logs/seathru_<run id>_<task>.{out,err}.
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
FIRST=${1:?first task index}
LAST=${2:-$FIRST}
export FS_ROOT="$ROOT"
export FS_ENV="${FS_ENV:-/home/grzegos/.conda/envs/foamsurf}"
export FS_DATASETS="${FS_DATASETS:?set FS_DATASETS to the SeathruNeRF_dataset folder}"
export FS_DRIVER="${FS_DRIVER:-/usr/lib/x86_64-linux-gnu}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
RUN_ID="${FS_RUN_ID:-local_$(date +%Y%m%d_%H%M)}"
export FS_RUN_ID="$RUN_ID"

[[ -x "$FS_ENV/bin/python" ]] || { echo "No python in FS_ENV=$FS_ENV" >&2; exit 2; }
[[ -d "$FS_DATASETS" ]] || { echo "No dataset folder FS_DATASETS=$FS_DATASETS" >&2; exit 2; }
mkdir -p "$ROOT/slurm_logs"
failed=()
for (( task = FIRST; task <= LAST; task++ )); do
  log="$ROOT/slurm_logs/seathru_${RUN_ID}_${task}"
  echo "[run_local] task $task -> ${log}.out"
  if ! SLURM_ARRAY_TASK_ID=$task SLURM_ARRAY_JOB_ID=$RUN_ID \
       bash "$ROOT/scripts/train_seathru.sbatch" > "${log}.out" 2> "${log}.err"; then
    echo "[run_local] task $task FAILED (see ${log}.err)" >&2
    failed+=("$task")
  fi
done
echo "[run_local] done: outputs/seathru/$RUN_ID; failed tasks: ${failed[*]:-none}"
(( ${#failed[@]} == 0 ))
