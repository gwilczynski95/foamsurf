# Speeding up the FoamSurf RGB renderer — handoff

You are working on a fresh clone on a machine with an **RTX 3090** (compute
capability 8.6). Nothing outside this repository exists there except what is
listed in §1; every path below is relative to the repository root unless it is
one of the variables `$ENV`, `$OPTIX`, `$DRIVER`, `$DATA`.

**Goal.** Make `NativeHybridRenderer.render` faster for the RGB medium used by
every current experiment, **without changing what it computes**: images,
the `direct`/`backscatter` outputs and gradients must match the current
renderer within the tolerances of §5, and a full training must reach the same
quality (§6).

Read first: `renderer/README.md` (scalar renderer contract: Gaussian event,
Voronoi traversal, compositing, backward) and `renderer/foamsurf/rgb_medium.py`.
Order of work: §1 build → **Step 0 baseline** → §3 changes → §5 and §6
acceptance → §7 report.

## 1. Machine setup

Required (the build scripts check versions and stop otherwise):

| item | version |
|---|---|
| Python | 3.11 (conda env, called `$ENV` below) |
| PyTorch | 2.6.0+cu124 |
| `nvcc` inside `$ENV/bin` | CUDA 12.4 (`conda install -c nvidia cuda-toolkit=12.4`) |
| OptiX SDK headers | 8.0.0 (ABI 87), directory `$OPTIX` with `include/optix.h` |
| NVIDIA driver | provides `libcuda.so.1` and `libnvoptix.so.1` in `$DRIVER` (usually `/usr/lib/x86_64-linux-gnu`) |

```bash
export CONDA_ENV_PATH=$ENV
export FOAMSURF_OPTIX_DIRECTORY=$OPTIX
export FOAMSURF_DRIVER_LIBRARY_DIRECTORY=$DRIVER
export TORCH_CUDA_ARCH_LIST=8.6      # also sets the OptiX build to sm_86
export CUDA_VISIBLE_DEVICES=0        # build.sh defaults to 1
./build.sh                           # requirements, topology extension, renderer
ls renderer/foamsurf/_C*.so renderer/foamsurf/_optix_k64/PYOPTIXFLARERENDERER.so
```

Only the 64-hit OptiX backend (`_optix_k64`) is built and used; never use
`K=16`. After editing anything in `renderer/native/` or the vendored OptiX code,
rebuild with `renderer/build.sh` (same variables). Build products (`*.so`,
`_optix_k64/`, `build/`) are git-ignored and must not be committed.

**Data to copy from the cluster** (the user does this):

- `$DATA` — the IUI3 dataset (`IUI3-RedSea-Undistorted`, 132 MB; COLMAP model in
  `sparse/0` or a flat `sparse/`).
- `renderer_handoff_2026-10-15.tar` (185 MB), unpacked to `handoff/` in Step 0. It contains two
  trained runs, each with `pipeline_config.json`, `joint/config.yaml`,
  `joint/checkpoints/best.pt` (+ sidecars), `joint/metrics/eval.csv`,
  `joint/summary.json`:
  - `runs/B0_chroma100_bk` — the default configuration
    (`configs/iui3_default.yaml`), ~287k Gaussians, 5000 Voronoi cells;
  - `runs/B0_chroma100_scov3e3_g700k_bk` — ~673k Gaussians (the speed case
    we care about next).

The run directories store the cluster's dataset path; every tool below takes
`--dataset $DATA` (or `--set ...source_path=$DATA`) to override it.

## Step 0 — baseline on the 3090, before touching the renderer

Do this in the first session, on the unmodified checkout (`dev/medium_fix`),
right after the build of §1. It checks that the build and the tools work on
this GPU and records the "before" state that every later comparison uses.

```bash
DATA=<path to IUI3-RedSea-Undistorted>
mkdir -p handoff && tar -xf renderer_handoff_2026-10-15.tar -C handoff   # RUNS=handoff
OUT=outputs/renderer_baseline_3090 && mkdir -p $OUT
python -m pytest -q -rs renderer/tests 2>&1 | tee $OUT/renderer_tests.txt
python -m pytest -q -rs tests 2>&1 | tee $OUT/tests.txt
for run in B0_chroma100_bk B0_chroma100_scov3e3_g700k_bk; do
  R=handoff/runs/$run
  python scripts/renderer_check.py capture --run $R --dataset $DATA --reference $OUT/$run.pt
  python scripts/renderer_check.py compare --run $R --dataset $DATA --reference $OUT/$run.pt \
    | tee $OUT/${run}_self_compare.txt
  python scripts/renderer_check.py bench --run $R --dataset $DATA --json $OUT/${run}_bench.json \
    | tee $OUT/${run}_bench.txt
done
```

- **`$OUT/*.pt` are the reference renders and gradients of the current
  renderer. Never overwrite or regenerate them after renderer code has
  changed** — every later `compare` is against these files. They are large and
  stay out of git.
- The self-compare (old renderer vs itself) must print `PASS`; its numbers are
  the noise floor (forward should be exact, gradients differ only by atomic
  accumulation order). If it fails, the tolerances or the tool are wrong — stop
  and report, do not start on the renderer.
- `renderer/tests/test_rgb_medium_renderer.py` and `scripts/renderer_check.py`
  have not yet run on a GPU before this step. If either fails for a reason
  unrelated to the renderer (an API mismatch, a path, memory), fix the test or
  the tool in a separate commit and say so; do not change renderer code to make
  them pass.
- Report to the user: the two pytest summaries, both `*_self_compare.txt`
  and both `*_bench.txt`. The `bench` output replaces the 4090 profile numbers of
  §2 as the baseline for this GPU.

## 2. What the renderer does now (RGB medium)

Per `render(rays, gaussians, voronoi=RGBVoronoiScene, background)` call
(`renderer/foamsurf/cuda.py`):

1. **Start cell.** If `start_points` is not given: `torch.unique` over all ray
   origins (one per camera, but the unique runs over every ray), brute-force
   nearest site per unique origin, broadcast back.
2. **OptiX hit generation** (`self._optix.Render`): depth-sorted Gaussian hits,
   buffers `hit_depths`/`hit_indices` of shape `[64, R]` (more hits are
   truncated).
3. **Six launches of the scalar fused kernel** (`compose_cuda` in
   `rgb_medium.py` calls `_HybridCudaFunction.apply` six times), for each
   channel `c`:
   - *direct pass*: densities = `attenuation[:, c]`, medium colors = 0,
     Gaussian colors and background as given → takes `rgba[:, c]` as
     `direct_c` and `1 - rgba[:, 3]` as the direct transmittance;
   - *backscatter pass*: densities = `backscatter[:, c]`, medium colors as
     given, Gaussian colors = 0, background = 0 → `rgba[:, c]` is
     `backscatter_c`.

   Each pass computes three channels and keeps one, traverses the same Voronoi
   cells and evaluates the same Gaussians. Backward is six backward launches.
4. Python stacks the results: `image = direct + backscatter`,
   `alpha = 1 - mean_c(direct transmittance)`, `num_intersections` = max over
   passes; `NativeHybridRender` also returns `direct`, `backscatter` and
   `direct_transmittance`.

Reference recurrence per ray (`rgb_medium.reference_segments`; segment `i` of
length `l_i` lies in cell `k(i)` and ends at Gaussian event `i` with alpha
`a_i`, color `g_i`; per channel):

```text
backscatter += Tg * Tb * c_k * (1 - exp(-sb_k * l_i))
Td *= exp(-sa_k * l_i);  Tb *= exp(-sb_k * l_i)
direct      += Tg * Td * a_i * g_i
Tg *= (1 - a_i)
end: direct += Tg * Td * background
```

**Early termination is per pass.** The scalar kernel
(`renderer/native/hybrid_cuda.cu`, `hybrid_forward_kernel`) stops when its own
transmittance `T <= transmittance_threshold`, checked after every Gaussian event
and before every Voronoi step (also stops at `far` and after
`max_intersections` Voronoi steps). The direct pass for channel `c` uses
`T = Tg * Td_c`, the backscatter pass `T = Tg * Tb_c`. Thresholds: training
`1e-4`, evaluation `1e-2` (`renderer.*_transmittance_threshold`). A fused kernel
must stop accumulating each of the six quantities exactly when *its* transmittance
crosses the threshold (and keep traversing while any is active), otherwise the
evaluation images change measurably.

**Other callers in training** (`training/colmap_joint_trainer.py`,
`training/trainer.py`): the priors differentiate `hybrid.backscatter` and
`hybrid.direct` directly, and call `render_gaussians` (Gaussian-only, black
background) on up to 131072 rays; while Gaussians densify (iterations <
15000) every training step also runs `collect_diagnostics=True` (an extra
forward-only diagnostics launch). Training uses full images (1383×917 ≈ 1.27M
rays per step).

**Measured on an RTX 4090** (default run, one 1383×917 test view, evaluation
threshold; numbers from an earlier profile, re-measure with §4):
whole `render` ≈ 20.4 ms (≈ 49 FPS, `eval/hybrid_fps` in `eval.csv`); OptiX
hits ≈ 1.6 ms; one scalar pass ≈ 2.1 ms, six ≈ 12.6 ms; the remaining ≈ 6 ms is
start-cell lookup, allocations and Python. WaterSplatting renders the same scene
at ≈ 174 FPS on that GPU.

## 3. Suggested work, in order

1. **Fused RGB kernel (forward and backward).** One thread per ray, one
   Voronoi traversal and one evaluation per Gaussian event, accumulating the
   three-channel `direct`, `backscatter`, `Td`, `Tb` and the scalar `Tg` with
   per-quantity termination as above. Backward: the same analytic reversal as
   the scalar kernel, summing what the six passes contribute today (gradients
   for sites, attenuation, backscatter, medium colors, Gaussian means,
   log-scales, quaternions, colors, opacity logits, kappa). Expected gain:
   most of the ~10 ms of the six passes.
   Keep `compose_cuda` reachable (e.g. an environment variable or constructor
   flag selecting the six-pass path) so both can be compared in one build.
2. **Start-cell lookup without `torch.unique` over every ray.** All rays of an
   evaluation view share one origin; the training sampler also knows its
   camera. Compute the start cell once per camera (or accept `start_points`
   from the caller). Results must be identical (same nearest site).
3. **Allocation/Python overhead.** Ray normalization copies, `zeros_like`
   per pass, stacking, the `NativeHybridRender` reshapes.
4. **Only then** consider removing the `[64, R]` hit buffer (single OptiX
   traversal). That changes results on rays with more than 64 Gaussian hits;
   §5 reports the fraction of such rays (`K full`) — quantify it before
   proposing it, and keep it a separate, optional step.

Do not change: the public API (`renderer/foamsurf/__init__.py`, signatures of
`render`, `render_gaussians`, `render_voronoi`, the fields of
`NativeHybridRender`), the scalar medium path (used by the synthetic
pipeline), `collect_diagnostics` semantics (on/off must not change RGB or
gradients), or anything outside `renderer/` except to pass `start_points`.

## 4. Tools

```bash
# Unit tests: PyTorch oracle vs CUDA, values and gradients, incl. the RGB medium
python -m pytest -q -rs renderer/tests          # run from the repo root; test_rgb_medium_renderer.py is the RGB oracle test
python -m pytest -q tests/                      # training code; includes short GPU trainings

# Real-checkpoint snapshot, comparison, benchmark (scripts/renderer_check.py)
python scripts/renderer_check.py capture --run handoff/runs/B0_chroma100_bk --dataset $DATA --reference <file>.pt
python scripts/renderer_check.py compare --run handoff/runs/B0_chroma100_bk --dataset $DATA --reference <file>.pt
python scripts/renderer_check.py bench   --run handoff/runs/B0_chroma100_bk --dataset $DATA --json <file>.json
```

`capture` stores, for the 4 test views (`MTN_5894/5903/5911/5928`), the hybrid
image, backscatter, alpha, Gaussian-only alpha and Gaussian hit counts, plus the
gradients of every model parameter for one training image under
`L1(image) + 0.1·mean(backscatter) + 0.1·mean(Gaussian-only alpha)`.
`compare` prints per-view/per-quantity differences and per-parameter relative
L2 / cosine, and exits 1 on failure. `bench` prints per-view times split into
start-cell lookup, OptiX hits and composition, Gaussian-only and medium-only
renders, and a training forward / forward+backward with peak memory.

## 5. Acceptance — equivalence

The references are the Step 0 files. After the change (rebuilt with
`renderer/build.sh`), for both runs:

```bash
python scripts/renderer_check.py compare --run handoff/runs/$run --dataset $DATA \
  --reference outputs/renderer_baseline_3090/$run.pt
python scripts/renderer_check.py bench --run handoff/runs/$run --dataset $DATA --json outputs/<new>/${run}_bench.json
```

`compare` must print `PASS`:

- images (hybrid, backscatter, alpha, Gaussian alpha): mean |Δ| ≤ 1e-5 and
  |Δ| > 1e-3 on at most 0.01% of values, per view;
- gradients: relative L2 ≤ 1e-3 for every parameter tensor (atomic accumulation
  makes them non-bitwise even old vs old);
- plus `python -m pytest renderer/tests tests` all green on the 3090.

If a tolerance is exceeded, find the reason (termination, ordering, a missing
term) instead of loosening it; if a difference is intended (e.g. no 64-hit
cap), report it with the numbers and leave the tolerance unchanged.

## 6. Acceptance — training

Train the default configuration with the old renderer (once, as the 3090
baseline) and with the new one; same command, different `--output-dir`:

```bash
python train.py --config configs/iui3_default.yaml --output-dir outputs/<name> \
  --set warmup.dataset.source_path=$DATA --set joint.dataset.source_path=$DATA \
  --set joint.wandb.mode=disabled
python scripts/medium_audit.py --run outputs/<name> --checkpoint best   # disentanglement audit
```

Add `--set variant.seed=43 --set warmup.seed=43 --set joint.seed=43` for a
second seed. Each run evaluates every 1000 iterations; compare
`outputs/<name>/joint/metrics/eval.csv` (`eval/hybrid_psnr`, `eval/hybrid_fps`)
and `joint/summary.json` (`best_psnr`, `training_seconds`,
`gaussian_count`). Training is not bitwise reproducible (atomics), seed-to-seed
spread of best test PSNR is about ±0.4 dB, so the new renderer passes when its
best PSNR is within that band of the old-renderer run(s) on the 3090 and the
audit numbers (`audit/best.json`: `bottom_alpha_gt_0.99`,
`open_water_alpha_gt_0.5`, `clean_near/delta_psnr_bottom`) are similar.

Reference on the RTX 4090 (30k iterations, from the handoff runs):

| run | seed | best test PSNR (iter.) | final PSNR | eval FPS | Gaussians | training time |
|---|---|---|---|---|---|---|
| default `B0_chroma100_bk` | 42 | 28.98 (19k) | 28.73 | 46.9 | 287k | 2.0 h |
| default `B0_chroma100_bk` | 43 | 28.77 (28k) | 28.55 | 47.2 | 279k | 2.0 h |
| `B0_chroma100_scov3e3_g700k_bk` | 42 | 28.73 (19k) | 28.32 | 47.7 | 673k | 2.3 h |
| `B0_chroma100_scov3e3_g700k_bk` | 43 | 29.19 (28k) | 29.02 | 49.2 | 671k | 2.3 h |

Expect the 3090 to be slower in absolute terms; compare old vs new on the same
GPU.

## 7. What to report back

- `compare` output for both references (PASS/FAIL table);
- `bench` before/after JSON for both runs (per-view breakdown, training
  forward/backward, memory);
- the training comparison (best/final PSNR, FPS, time, audit numbers);
- what changed in `renderer/` and anything that deliberately differs.
