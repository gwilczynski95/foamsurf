# Running a sweep on the RTX 3090 workstation

`scripts/run_local.sh` runs tasks of `scripts/train_seathru.sbatch` one after another, without Slurm.
It uses the same variant sets, suffixes and output layout as the cluster. Its defaults fit the
workstation:

- conda env: `/home/grzegos/.conda/envs/foamsurf`
- driver libraries: `/usr/lib/x86_64-linux-gnu`
- `TORCH_CUDA_ARCH_LIST=8.6`

The native extensions are compiled for the 3090 on the first task.

## 1. Get the code

```bash
cd ~/foamsurf            # the workstation clone
git fetch origin
git checkout dev/medium_fix
git pull
```

## 2. Quick check (about 1 min)

```bash
conda activate foamsurf
python -m pytest -q tests/test_oceansplat.py -k suffixes
```

## 3. Start the sweep

Use a run id that does not clash with cluster jobs. The example below is the `overlap` set: cluster
job 550635, `_clamp05_minit_rel_nooverlap`, 8 tasks.

```bash
export FS_DATASETS=/path/to/SeathruNeRF_dataset
export FS_VARIANT_SET=overlap
export FS_RUN_ID=550635_3090
nohup bash -c 'scripts/run_local.sh 6 7 && scripts/run_local.sh 2 5 && scripts/run_local.sh 0 1' \
  > run_3090.log 2>&1 &
```

Task numbering in every set: scene-major, then seed.

| task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| scene | IUI3 | IUI3 | Curaçao | Curaçao | JG | JG | Panama | Panama |
| seed | 42 | 43 | 42 | 43 | 42 | 43 | 42 | 43 |

The order above runs Panama, then Curaçao and JG, then IUI3, so the most informative scenes finish
first. One task takes roughly 2–3 h on the 3090, so the 8 tasks take about a day.

`run_local.sh` logs a failed task and starts the next one. Every `&&` link after the failure is
skipped, though, so check `run_3090.log`.

## 4. Watch it

```bash
tail -f run_3090.log
tail -f slurm_logs/seathru_550635_3090_<task>.out
tail -1 outputs/seathru/550635_3090/<run name>/joint/metrics/train.csv   # iteration, ...
```

A run is finished when `outputs/seathru/550635_3090/<run name>/joint_done.json` exists. Its test
metrics are in `joint/metrics/eval.csv`.

## 5. Copy the results to the cluster

```bash
rsync -av outputs/seathru/550635_3090/ \
  gmwilczn@<cluster>:/shared/results/gmwilczn/foamsurf/foamsurf/outputs/seathru/550635_3090/
```

The cluster report compares them with the control of the same sweep generation: `_clamp05_minit_rel`
in job 550540. Run-to-run differences between the 3090 and the 4090s are within seed noise.

## Notes

- After training, `train_seathru.sbatch` runs `scripts/medium_audit.py`, which needs the SeaThru
  SAM masks. If they are missing on the workstation the task is reported as failed. Training and
  evaluation are still complete, so check for `joint_done.json`.
- Once the workstation has started a set, cancel the same set on the cluster (`scancel <job>`), so
  the GPUs are not spent twice.
