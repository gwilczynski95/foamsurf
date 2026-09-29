# Three-way RGB disentanglement: overlap, Gaussian opacity, graph smoothness

Status: implementation prepared; 21 CPU tests passed. Required RTX4090 test job
**513474** was submitted. GPU tests are NOT yet certified at the time this note
was written. No new training array was submitted. Experiments must wait for the
test gate and subsequent user decision.

Repository: `/shared/results/gmwilczn/foamsurf/foamsurf`.
Conda environment: `/shared/results/gmwilczn/conda/envs/foamsurf`.
Dataset: `/shared/sets/datasets/SeathruNeRF_dataset/IUI3-RedSea-Undistorted`.

## Isolation and prior push

The preceding implementation was committed and pushed to
`origin/feature/voronoi-rgb-medium`:

- `7b48798`: RGB attenuation/backscatter model, renderer, tests, paired sweep.
- `c47667d`: isolated no-warmup graph-smoothing sweep.
- `ad25a37`: CPU ray heatmaps and profiles.

All new source, configuration, tests and sbatches in this task are confined to
`experiments/rgb_disentangle/`. Nothing is changed under the original gate's
hashed source directories, nor under `experiments/rgb_smooth/`.
The pending array 513211 continues to use the existing implementation and its
certificate `outputs/rgb_smooth_tests/513209/passed.json`, verified after edits.
This extension selects its trainer only within its own process. Its new test
gate fingerprints base source, smoothing, this directory, and the CPU diagnostic
used as a reference. Do not edit these sources after certification/queueing.

## What is implemented

- `losses.py`: validated settings, independent ramps, conditional opacity loss,
  differentiable suspect_overlap with detached Gaussian profile.
- `renderer.py`: opt-in ray-profile renderer, native CUDA Gaussian-only opacity,
  native OptiX hits evaluated with the native Gaussian alpha formula, sampled
  nearest-cell medium integration in PyTorch, diagnostics and timings.
- `trainer.py`: instance-local capture of the current hybrid training rays,
  addition of regularizers before the existing backward, metrics, checkpoint
  settings validation. Reuses the previously implemented graph smoothing.
- `train.py`: zero-warmup initialization and ordinary adaptive joint training,
  strict manifest/sidecar resume, eight combinations with strength/seed variants.

There is NO medium-only training phase. The initialization helper still writes
an iteration-zero checkpoint under `warmup/` but performs zero optimization.
The RGB renderer and original reconstruction loss remain unchanged. We do not
replace the entire CUDA compositor to add these regularizers; the opt-in renderer
path adds a small native Gaussian render and differentiable sampled profiles.

## Loss definitions and gradient routing

Let N be the uniformly sampled pixel count, S the number of equal depth bins,
dt their width, and L=S*dt. Default N=128, S=1024. Rays are sampled with replacement
from the current training image using an iteration/seed-local RNG that does not
consume the camera sampler's or adaptation's RNG. This estimates the pixel mean,
not an unnormalized resolution-dependent sum. Repeated sampled pixels are valid.

### Overlap: literal mean of the diagnostic score

For each ray, activate bRGB and B exactly as in joint training. For bin j:

    Tb_j = exp(-sum_{k<j} b_k*dt)
    TG_j = exp(-sum_{k<j} gaussian_tau_k - 0.5*gaussian_tau_j)
    q_j = mean_RGB[TG_j * Tb_j * B_j * (1-exp(-b_j*dt))]

Define:

    Q = sum(q)
    C = 1 - (sum(q)^2 / sum(q^2/dt)) / L
    strength = Q/(Q+0.05)

Empty profiles have C=0. Let Gaussian-only front-to-back weights yield total
opacity A, weighted mean depth mu and depth variance var. The neighborhood
half-width h=0.02L and surface confidence is:

    confidence = A * exp(-var/h^2)
    near_fraction = sum(q_j for |t_j-mu|<=h) / Q
    suspect_overlap = C * strength * near_fraction * confidence
    L_overlap = mean_pixels(suspect_overlap)

ALL Gaussian-derived terms, hit assignments and cell assignments are detached.
Gradients go to the Voronoi backscatter coefficients bRGB and source color B.
There is intentionally no gradient from this loss to Gaussian parameters, to
Voronoi site positions, or to attenuation aRGB (a does not enter the backscatter
equation). Reconstruction still learns all parameters; graph smoothing learns
a,b,B. Not differentiating site assignments also prevents direct site-position
movement as a route for decreasing this auxiliary loss in the current step.

This is the proposed literal heatmap loss, not a substituted local-excess loss.
It can decrease through removing local emission, spreading it, moving its mass
elsewhere via optical changes, or changing upstream b. Do NOT interpret a black
map alone as success. A loss reduction does not guarantee semantic separation.

### Gaussian opacity

Use native Gaussian-only opacity over the renderer's full near/far interval,
with no medium and black background:

    mask_p = stop_gradient(A_G(p) > 0.1)
    L_G = sum(mask_p * (1-A_G(p))) / max(sum(mask_p),1)

This averages opacity VALUES, not the Boolean predicate. Empty selection gives
zero with zero gradients. The gradient reaches only Gaussian geometry/opacity/
shape through the native renderer, not Voronoi. Gaussian color is not needed to
make opacity high. The mask fraction is logged because the changing hard mask
can remove pixels from the penalty. This loss does not force blank-water pixels
to acquire Gaussians and does not replace per-primitive binarization.

### Graph smoothness

Reuse `experiments/rgb_smooth/loss.py` unmodified. Mean Huber differences over
undirected neighboring cells, all edge weights one:

- log(aRGB+1e-6), Huber delta 0.1;
- log(bRGB+1e-6), Huber delta 0.1;
- activated B, Huber delta 0.05.

Graph edges refresh on topology revision/pointer/count changes. No gradient
through graph construction. Existing Gaussian binarization/pruning stays intact.

## Sampling, geometry and limitations

Overlap uses L=min(renderer far,8*training camera extent), starting at zero,
matching the final CPU diagnostic setup (~49.32 scene units for IUI3). These are
scene units, not meters. Gaussian opacity loss instead uses native near/far.
Current production near=0. The thresholds are explicit, configurable heuristics.

Nearest-cell assignment is exact at each bin midpoint using a fresh CPU cKDTree.
It is rebuilt every overlap step because sites can move even without a topology
revision. The bins do NOT resolve exact Voronoi boundary lengths. Profiles and
their gradients are differentiable with respect to sampled optical attributes.
GPU tensor gathering/compositing handles backprop; CPU lookup is an overhead
measured by `conflict/seconds`. No stale site cache or ray-by-all-sites GPU matrix
is used. Gradient and forward tests explicitly cover this approximation.

Gaussian events come from native OptiX support (octagon, unlike the exploratory
CPU helper's circular support). Gaussian alpha uses the same quaternion/scale/
kappa transforms as CUDA. Events within a bin are accumulated through optical
thickness and midpoint TG. This differs from exact event ordering inside a bin;
doubling depth samples should be considered when interpreting fine structure.
Hit-buffer capacity remains the original configured K (64 in production), not
a new K experiment. `conflict/hit_full_fraction` exposes potential truncation.
The regularizer does not claim GPU-native equality with CPU heatmap pixels.

## Candidate experiment matrix — prepared, NOT launched

All runs: no warmup, 30k joint steps, 25k initial Voronoi cells, spatial mix 25%
COLMAP and extent 2, adaptive Gaussian/Voronoi populations. W&B project:
`foamsurf_iui3_rgb_conflict`; group `rgb-conflict-ARRAY_JOB_ID`.

Three seeds (42,43,44), one control per seed. Each of the seven nonempty
combinations has three strength multipliers (0.3,1,3) and three seeds: **66 runs**.

| IDs | Combination |
|---|---|
| 0–2 | control |
| 3–11 | overlap |
| 12–20 | Gaussian opacity |
| 21–29 | smoothness |
| 30–38 | overlap + opacity |
| 39–47 | overlap + smoothness |
| 48–56 | opacity + smoothness |
| 57–65 | all three |

Within each nonempty group, multiplier is outer and seed is inner order.
At multiplier 1: overlap weight .1, opacity weight .01; smoothing weights
(a,b,B)=(.01,.01,.1). Multiplier scales all enabled losses together; this is NOT
an exhaustive independent-weight grid. Weights are exploratory, not calibrated.

Overlap/opacity: zero through step1000, linear increase, full strength at2000.
Smoothness: linear ramp from start, full strength at250. All remain on afterward.
Start/ramp parameters can be set separately in YAML. This first matrix varies
combinations and strength, NOT all onset schedules. Additional schedules should
follow validated gradients and initial ablation results.

Array concurrency: seven. Separate existing arrays retain their own limits.

## Tests and launch discipline

```bash
cd /shared/results/gmwilczn/foamsurf/foamsurf
sbatch experiments/rgb_disentangle/test.sbatch
```

Already submitted once as **513474**. Logs:
`slurm_logs/rgb_disentangle_test_513474.{out,err}`.
Certificate/aggregate JUnit:
`outputs/rgb_disentangle_tests/513474/passed.{json,xml}`.

29 new tests: 21 CPU + 6 native renderer GPU + 2 end-to-end GPU, followed by the
preceding 133-test suite. Total: **162 required tests**, no skip-based pass.
The gate checks GPU model, test commands, fingerprints and failure/skip counts.
CUDA/OptiX suites run in separate processes to avoid incompatible K ABIs.

Coverage includes NumPy heatmap agreement, float64 gradcheck, zero/large/empty
fields, hard-mask boundary, all gradient routes, native opacity agreement,
additional-forward safety for reconstruction backward, actual trainer loss
injection, zero warmup, adaptive topology, finite updates, strict sidecars,
resume and full-resolution production ray/sample budgets.

Only AFTER a passing gate and user decision:

```bash
export RGB_CONFLICT_TEST_CERTIFICATE="$PWD/outputs/rgb_disentangle_tests/513474/passed.json"
sbatch --export=ALL experiments/rgb_disentangle/train.sbatch
```

No dependency-linked training job has been submitted. Checkpoints require both
`.smooth.json` and `.conflict.json` settings sidecars; keep them with `.pt` files.
Resume rejects missing or different settings rather than silently losing losses.

## How to analyze

Compare best/final held-out PSNR/SSIM at matched steps and seeds. Inspect actual
backscatter and direct-with-background in all views, not only Voronoi-only.
Track every component separately: concentration, strength, near_fraction,
near_mass, Gaussian confidence, backscatter mean, conditional opacity mask
fraction, raw/weighted losses, native hit fullness, population and wall time.

Success: surface texture disappears from backscatter, reappears in direct,
combined reconstruction remains good, and Gaussians do not spread into water.
Failure modes: mask shrinks, medium emission merely shifts away from the surface,
backscatter goes dark without compensation, smoothness spreads medium farther,
or added compute yields no improvement. Always compare combined score against
Gaussian surface confidence alone; overlap is not an independent solid label.
