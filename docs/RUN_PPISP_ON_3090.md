# Running the PPISP sweep (`isp` set) on the RTX 3090 workstation

Instructions for whoever drives the workstation (a person or a coding agent). The general setup
(conda env, `scripts/run_local.sh`, task numbering, copying results) is in `docs/RUN_ON_3090.md`;
this file only adds what is specific to PPISP.

## What is being run

Variant `_clamp05_minit_rs_rel_oceane_isp`: the current best configuration (`_rs_rel_oceane`,
cluster job 550540) plus PPISP (Deutsch et al., CVPR 2026), a learned camera model applied to the
rendered image:

- per training photo: an exposure offset and a colour correction;
- per camera: vignetting and a response curve.

Test views are rendered with neutral exposure and colour; nothing is fitted on test photos. The
code is `training/camera_isp.py`, switched on by `joint.optimization.camera_isp=true` (the `_isp`
suffix). The cluster job for the same set is 551590; use run id `551590_3090`.

The variant has no anti-aliasing: `ray_jitter=false` and `gaussian_mip_filter=0.0` (the defaults;
only the `_jit` / `_mip` suffixes switch them on). The anti-aliasing sweep (cluster job 551393) has
not finished yet, so this run measures PPISP alone against `_rs_rel_oceane`. Do not add `_jit` or
`_mip`.

## 1. Get the code

```bash
cd ~/foamsurf            # the workstation clone
git fetch origin
git checkout dev/medium_fix
git pull
```

## 2. PPISP installation

Nothing to install with pip. PPISP v1.0.0 is vendored in `third_party/ppisp/` (Apache-2.0, see
`third_party/ppisp/VENDORED.md`). Its CUDA extension `ppisp_cuda` is compiled by
`torch.utils.cpp_extension.load` the first time a run creates the camera model. The build takes a few
minutes and lands in `~/.cache/torch_extensions/<py>_<cu>/ppisp_cuda`.

Do not `pip install ppisp`. The trainer puts `third_party/ppisp` first on `sys.path`, but a second
installed copy only causes confusion.

The JIT build needs `nvcc` and `ninja` in the conda env (both are already used for the other
extensions):

```bash
conda activate foamsurf
export CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST=8.6
which nvcc ninja          # both must resolve inside the env; otherwise: pip install ninja
```

Build the extension and check it before starting the sweep (about 5 min including the build):

```bash
python -m pytest -q tests/test_camera_isp.py
```

All 6 tests must pass. The last one is a short real training run with PPISP, OceanSplat terms,
resume and evaluation.

If the build fails:

- `nvcc not found` or `CUDA_HOME` errors: export `CUDA_HOME` as above.
- `Ninja is required`: run `pip install ninja` in the env.
- An unsupported arch / `no kernel image`: set `TORCH_CUDA_ARCH_LIST=8.6`, delete
  `~/.cache/torch_extensions/*/ppisp_cuda` and rebuild.
- A half-finished build after an interrupted run (a `lock` file in that folder): delete the folder and
  rebuild.

## 3. Start the sweep

The task numbering is the same as in every set (scene-major, then seed):

| task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| scene | IUI3 | IUI3 | Curaçao | Curaçao | JG | JG | Panama | Panama |
| seed | 42 | 43 | 42 | 43 | 42 | 43 | 42 | 43 |

Curaçao s42 runs first, then IUI3 s42: these are the scenes where PPISP gained most when fitted after
training (+0.48 and +0.43 dB). `;` instead of `&&` keeps going after a failed task.

```bash
export FS_DATASETS=/path/to/SeathruNeRF_dataset
export FS_VARIANT_SET=isp
export FS_RUN_ID=551590_3090
nohup bash -c 'scripts/run_local.sh 2 2; scripts/run_local.sh 0 0; scripts/run_local.sh 3 3;
               scripts/run_local.sh 1 1; scripts/run_local.sh 4 7' > run_3090_isp.log 2>&1 &
```

One task takes roughly 70–95 min on the 3090: `_rs_rel_oceane` took 45–60 min on a 4090, and the
3090 is about 1.5× slower. All 8 tasks take about 10–12 h. Peak memory is about 9 GB.

## 4. Check that PPISP is really on

After the first task starts:

```bash
grep -n camera_isp outputs/seathru/551590_3090/Curasao-Undistorted_clamp05_minit_rs_rel_oceane_isp/joint/config.yaml
# expected: camera_isp: true
```

When the first checkpoint exists, `joint/checkpoints/last.pt` contains a `camera_isp` entry.

## 5. Watch it and copy the results

```bash
tail -f run_3090_isp.log
tail -1 outputs/seathru/551590_3090/<run name>/joint/metrics/eval.csv
```

A run is finished when `outputs/seathru/551590_3090/<run name>/joint_done.json` exists. Its test PSNR
is `eval/primary_psnr` in the last row of `joint/metrics/eval.csv`.

The reference, `_clamp05_minit_rs_rel_oceane`, mean of 2 seeds:

| | IUI3 | Curaçao | JG | Panama | mean |
|---|---|---|---|---|---|
| PSNR | 29.40 | 31.26 | 23.73 | 30.98 | 28.84 |

Per seed: Curaçao 30.75 / 31.78, IUI3 29.55 / 29.24. On Curaçao s42 the view to look at is
MTN_1288 (20–24 dB in earlier runs, a colour/exposure offset).

Copy the results to the cluster as in `docs/RUN_ON_3090.md`, into
`/shared/results/gmwilczn/foamsurf/foamsurf/outputs/seathru/551590_3090/`.

## Notes

- Cluster job 551590 (the same set) stays queued. Cancelling it is the user's decision; do not run
  `scancel` on your own.
- A failure of the post-training `scripts/medium_audit.py` (missing SAM masks) does not affect
  training or evaluation. Check for `joint_done.json`.
