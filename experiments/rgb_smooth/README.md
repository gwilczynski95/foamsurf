# RGB medium smoothing, no warmup

Repository: `/shared/results/gmwilczn/foamsurf/foamsurf`.
Environment: `/shared/results/gmwilczn/conda/envs/foamsurf` (`foamsurf`).
Data: `/shared/sets/datasets/SeathruNeRF_dataset/IUI3-RedSea-Undistorted`.

## Isolation from the running experiment

The running sweep 512923 still has queued tasks. Its gate hashes the original
training/model/renderer/config/scripts/tests directories. ALL new implementation,
tests, configuration and entrypoints are under `experiments/rgb_smooth/`, outside
that fingerprint. No original source or configuration is edited. The original
certificate `outputs/rgb_medium_tests/local_512889/passed.json` remains valid.

The new entrypoint selects a trainer subclass only in its own worker process.
It uses the existing CUDA renderer and joint pipeline; the old entrypoint never
imports this extension. This avoids changing both live workers and queued jobs.
The new gate fingerprints BOTH original source AND this extension. Editing this
extension does invalidate its own new certificate, as intended.

## What changes

Every run starts joint training with Gaussians and RGB Voronoi immediately.
There is **zero medium-only optimization**. The initialization infrastructure
still exports an iteration-zero checkpoint in the `warmup/` directory; this is
not pretraining. Its serialized constructor config says one iteration because
the generic trainer requires a positive budget, but the actual worker explicitly
executes zero steps. Tests assert checkpoint iteration 0 and no warmup train CSV.

Initial Voronoi: 25k sites, 25% sampled around COLMAP, extent 2. Gaussians: all
21,907 original COLMAP points. Both representations are adaptive, with the
previous binarization/pruning settings; joint duration 30k. No SH, no new medium
physics, no opacity reset or additional density-cap mechanism is introduced.

## Loss

For the set E of undirected current Voronoi adjacency edges, all edge weights
are one. Define Huber as $H_\delta(x)=x^2/2$ for $|x|\leq\delta$, otherwise
$\delta(|x|-\delta/2)$. Average over edges AND RGB components:

$$L_a=\frac{1}{3|E|}\sum_{(i,j)\in E}\sum_q
H_{0.1}(\log(a_{iq}+\epsilon)-\log(a_{jq}+\epsilon)),$$

$$L_b=\frac{1}{3|E|}\sum_{(i,j)\in E}\sum_q
H_{0.1}(\log(b_{iq}+\epsilon)-\log(b_{jq}+\epsilon)),$$

$$L_B=\frac{1}{3|E|}\sum_{(i,j)\in E}\sum_q H_{0.05}(B_{iq}-B_{jq}),$$

$$L_{smooth}(t)=r(t)(\lambda_a L_a+\lambda_b L_b+\lambda_B L_B),\quad
r(t)=\min(1,t/250),\quad\epsilon=10^{-6}.$$

These are activated physical coefficients and DC-activated colors, NOT raw
parameters. There is no physical-distance weighting. The graph is discrete;
the loss differentiates optical parameters, not site positions through edge
selection. The cached edge list is refreshed on topology revision/pointer/count
changes. Empty graphs give differentiable zero. Logs record all three raw losses
even when their weights are zero (disabled terms are measured without autograd).

The smoothing total is added to the existing disentanglement total BEFORE the
normal trainer backward pass; existing Gaussian binarization is retained.
Its weights are initial experimental choices, not calibrated universal values.
Huber here is NOT SmoothL1; the factor delta matters when comparing weights.

## Fifteen runs, one new W&B project

| IDs | Variant | lambda a | lambda b | lambda B | Seeds |
|---|---|---:|---:|---:|---|
| 0–2 | control | 0 | 0 | 0 | 42,43,44 |
| 3–5 | color_moderate | 0 | 0 | .1 | 42,43,44 |
| 6–8 | color_strong | 0 | 0 | 1 | 42,43,44 |
| 9–11 | all_moderate | .01 | .01 | .1 | 42,43,44 |
| 12–14 | all_strong | .1 | .1 | 1 | 42,43,44 |

W&B project: **foamsurf_iui3_rgb_smooth**, group `rgb-smooth-ARRAY_JOB_ID`.
Outputs: `outputs/rgb_smooth/ARRAY_JOB_ID/run_INDEX/`.
Logs: `slurm_logs/rgb_smooth_ARRAY_JOB_ID_INDEX.{out,err}`.
Array 0–14%7: at most seven concurrent RTX4090 jobs IN THIS ARRAY. The separate
old array retains its own limit; submitting both can exceed seven jobs total.
Use `--dependency=afterany:512923` when submitting the new array if it should
start only after the old sweep finishes. Do not modify or cancel old tasks.

Run test gate, then a dependent sweep:

```bash
cd /shared/results/gmwilczn/foamsurf/foamsurf
test_job=$(sbatch --parsable experiments/rgb_smooth/test.sbatch)
test_job=${test_job%%;*}
export RGB_SMOOTH_TEST_CERTIFICATE="$PWD/outputs/rgb_smooth_tests/$test_job/passed.json"
sbatch --dependency=afterok:$test_job --export=ALL experiments/rgb_smooth/train.sbatch
```

Alternatively set the variable to an already-passed local certificate and submit
the training sbatch directly. The old RGB certificate alone cannot authorize this
extension's sweep. Keep quoted certificate paths on a single line.

Manual entrypoint (GPU required):

```bash
python experiments/rgb_smooth/train.py --variant 12 \
  --output-dir outputs/rgb_smooth/manual/run_12
```

`--dry-run` emits the manifest; `--resume` requires the same manifest/output.
Checkpoints have an accompanying `.smooth.json` settings file. Keep it with the
checkpoint: a missing/mismatched sidecar rejects resume instead of silently
dropping or changing smoothing. Ordinary checkpoints alone do not encode this
extension's loss settings, so use this entrypoint to resume these experiments.

## Testing and analysis

Tests cover the analytic Huber formula, float64 gradcheck, constants/empty graph,
zero weights, ramp, edge orientation/repetition, color-only gradients, invalid
settings and all 15 manifests. GPU smoke checks loss contribution in the actual
training loop, graph refresh after population changes, no warmup optimization,
sidecar/optimizer resume, finite updates, full-resolution reconstruction and
truncated rays. The full gate also reruns the preceding 122-test RGB/legacy suite.

Compare best AND final held-out PSNR/SSIM, matched iterations and three seeds.
Inspect all four held-out views, especially MTN_5903. The primary separation
diagnostic is actual `*_backscatter.png`: does seabed/coral structure disappear
without losing its reconstruction through the Gaussian direct component?
Voronoi-only is a counterfactual without Gaussian occlusion and is not sufficient.

Track `rgb_smooth/attenuation`, `/backscatter`, `/color`, `/weighted`, `/ramp`,
`/edge_count`, optical histograms, populations and direct/backscatter means.
Color-only smoothing can move texture into b instead of removing it; all-field
smoothing tests this compensation. A lower smoothness loss alone is not success:
an over-smoothed scene or an almost black backscatter image can also lower it.
No semantic ground truth is available. Smooth fields can still model broad
surface structure; this regularizer is a bias, not a mathematical guarantee.
