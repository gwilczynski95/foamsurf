# Spatial RGB attenuation/backscatter medium (first implementation)

Branch: `feature/voronoi-rgb-medium`. Repository:
`/shared/results/gmwilczn/foamsurf/foamsurf`.
Environment: `/shared/results/gmwilczn/conda/envs/foamsurf`.
Dataset: `/shared/sets/datasets/SeathruNeRF_dataset/IUI3-RedSea-Undistorted`.

## Scope

An opt-in `model.medium_type: rgb_backscatter` adds independent RGB direct
attenuation and effective RGB backscatter. Default `scalar` retains the previous
image model and native kernels. This is NOT an SH experiment and is NOT a full
multiple-scattering or inverse-lighting solver. Existing Gaussian geometry,
appearance and opacity parametrizations are unchanged.

The production implementation uses SIX differentiable native CUDA compositing
passes with one shared OptiX hit buffer and one shared Voronoi start lookup.
It is a correctness-first renderer backend, not a new fused single-pass CUDA
kernel. Additional traversal cost is real and must be measured. No Python loop
over rays or cells is used in production. A separate PyTorch recurrence is the
independent mathematical oracle for tests.

## Relation to WaterSplatting

Reference: https://arxiv.org/html/2408.08206v1 (equations 6, 15–18).
Local implementation: `../water-splatting/water_splatting/water_splatting.py`.
WaterSplatting predicts attenuation RGB, backscatter RGB and medium color from
SH-encoded world ray direction using a shared MLP, once per pixel. These
parameters remain constant along that ray. Here parameters are attached to
spatial Voronoi cells instead; no directional MLP is used.

For a homogeneous ray ending at one opaque surface:

$$I=J\odot\exp(-\mathbf aD)+\mathbf B\odot[1-\exp(-\mathbf bD)].$$

The effective direct attenuation includes absorption AND out-scattering; do not
interpret it as pure physical absorption. Independent effective backscatter
coefficients do not by themselves define a full energy-conserving transport law.

## Cell parameters and activation

Each cell has position plus nine optical values:

- $\mathbf a_i\in\mathbb R_+^3$: direct attenuation;
- $\mathbf b_i\in\mathbb R_+^3$: effective backscatter coefficient;
- $\mathbf B_i\in\mathbb R_+^3$: local source color.

For checkpoint and population compatibility, six unconstrained coefficients are
packed into `raw_density[N,6]`: first attenuation RGB, then backscatter RGB.
This is NOT six physical densities nor an extra independent scalar density.
`raw_color[N,3]` keeps the existing COLMAP DC activation:

$$\mathbf B_i=\max(0,\tfrac12+C_0\mathbf u_i),\qquad C_0=0.28209479177387814.$$

Consequently B can exceed 1, as in the existing COLMAP branch. Sigmoid color
remains available for generic model construction; checkpoint metadata records
the activation. Do not silently clamp physical color during handoff.

Joint coefficients use $\operatorname{softplus}(u)$. Warmup uses the existing
$\operatorname{softplus}_{10}(u)$; canonical checkpoint coefficients are exported
by inverse beta-1 softplus. Both sets of raw parameters are retained for exact
warmup resume. `medium_type` and coefficient width are checked when restoring.

For a controlled comparison, spatial positions, scalar random initialization
and random-number consumption match the baseline. Each initial scalar raw
coefficient is copied into the six channels; they are independent trainable
parameters thereafter. This is spectrally neutral but spatially random, NOT
spatially uniform initialization. It deliberately isolates model flexibility
from an additional initialization change. No trained scalar warmup is converted
for the primary RGB experiment: the RGB model trains from its own initialization.

## Piecewise integration

Let $x(t)=o+t d$, with unit direction and path length in native COLMAP units.
Integrate only on $[near,far]$ (not from zero if near is positive). For segment i,
length $\ell_i$, define accumulated transmittances at its start:

$$\mathbf T_{D,i}=\exp[-\sum_{j<i}\mathbf a_j\ell_j],\qquad
\mathbf T_{B,i}=\exp[-\sum_{j<i}\mathbf b_j\ell_j].$$

Let $T_G$ be the scalar product of $(1-\alpha)$ for Gaussian hits already passed.
The segment contribution and updates are:

$$\Delta I_{B,i}=T_G\mathbf T_{B,i}\odot\mathbf B_i
\odot[1-\exp(-\mathbf b_i\ell_i)],$$

$$\mathbf T_D\leftarrow\mathbf T_D\odot\exp(-\mathbf a_i\ell_i),\qquad
\mathbf T_B\leftarrow\mathbf T_B\odot\exp(-\mathbf b_i\ell_i).$$

At Gaussian k, with effective ray opacity $\alpha_k$ and color $C_k$:

$$\Delta I_{G,k}=T_G\alpha_k\mathbf T_D\odot C_k,\qquad
T_G\leftarrow T_G(1-\alpha_k).$$

Split segments at BOTH cell boundaries and Gaussian intersections, including
Gaussian hits strictly inside a cell. An opaque Gaussian blocks everything
behind it, including backscatter. The finite-far boundary condition is:

$$I=\sum_k\Delta I_{G,k}+\sum_i\Delta I_{B,i}
+T_G(far)\mathbf T_D(far)\odot C_{bg}.$$

No infinite-water tail is added. In IUI3 near=0, far=1000, background is white.
The coefficients have units inverse COLMAP length; without metric scale they
must not be reported as inverse meters.

## Six-pass decomposition and gradients

For each channel q, direct uses the scalar native compositor with sigma=a[q],
medium color zero, original Gaussian colors and original background. Backscatter
uses sigma=b[q], original medium colors, zero Gaussian colors and zero background.
Both passes retain Gaussian opacity and geometry. Select output channel q and
sum the two. This is exactly the piecewise equation above with threshold zero
and sufficient traversal/hit capacity. Native backward derivatives through cell
boundaries, Gaussian depth/shape/opacity and all optical attributes accumulate
through autograd. Cameras, discrete topology and hit ordering are not optimized.

For tied coefficients $a_i=b_i=\sigma_i(1,1,1)$ and $B_i=c_i$, image formation
reduces to the legacy renderer. For identical cells the artificial subdivision
does not affect the mathematical result. Nonzero transmittance thresholds and
finite intersection budgets are approximations, as in the original renderer.
Each branch stops separately; one branch never prematurely terminates the other.

## Alpha, quantiles, adaptation and diagnostics

There is no unique scalar opacity for the new RGB effective model. The API's
`alpha` is explicitly $1-\operatorname{mean}_q[T_G T_{D,q}]$ and `transmittance`
is its complement. It is NOT object silhouette alpha. The warmup opacity
objective uses this scalar; real IUI3 images have all-ones loader alpha.

Existing depth-quantile regularization uses the auxiliary field
$\bar a_i=\operatorname{mean}_q a_{i,q}$. It is a regularizer on mean direct
optical depth, NOT exact quantiles of a mixture of RGB transmissions and NOT a
backscatter-depth estimate. This extension keeps the previously requested
quantile prior without pretending that six coefficients form one density.

Population contribution/visibility heuristics use
$\sigma_{proxy,i}=\operatorname{mean}_q\max(a_{i,q},b_{i,q})$ and the ACTUAL RGB
reconstruction error. It covers attenuation-only and backscatter-only cells.
These statistics are marked `max_rgb_coefficient_proxy`; reported evaluation
keys are `proxy_responsibility/...`. Their conservation check applies only to
that auxiliary scalar composition, not to the image's physical components.
Traversal diagnostics also flag any RGB branch reaching max_intersections.

The shared population machinery copies/reorders all six coefficients and three
colors together, including Adam moments. Warmup low-contribution reset resets
all six coefficients. Site counts and Gaussian schedules remain adaptive.
All six coefficients share the existing density LR; this is a first setting,
not a claim that the old LR is optimal. Separate-rate tuning is a future ablation.

Scalar TV, density-cap, optical-sparsity and opaque-if-used regularizers are
rejected in the RGB configuration until explicitly redefined/calibrated.
Gaussian opacity binarization/pruning remain enabled. The paired baseline also
has those medium regularizers off. SH is deferred to the next controlled study.

## Saved diagnostics

Each selected evaluation view additionally saves:

- `*_direct_with_background.png`: attenuated Gaussian contribution PLUS finite-far background;
- `*_backscatter.png`: actual backscatter with Gaussian occlusion;
- `*_direct_transmittance_rgb.png`: RGB direct transmission including Gaussian occlusion;
- `*_gaussian_only.png`: counterfactual scene without water;
- `*_voronoi_only.png`: counterfactual scene without Gaussian occlusion;
- `*_proxy_responsibility.png`: auxiliary adaptation proxy, NOT an RGB decomposition.

Before PNG clipping, direct_with_background + backscatter equals the hybrid RGB.
Histograms include attenuation and backscatter separately. Existing scalar
`medium_density` logs mean attenuation only; raw warmup quantiles pool six values.

## Test plan and implementation

`tests/rgb_medium/test_renderer.py`: independent geometric event construction
and PyTorch recurrence versus CUDA forward and every optical/Gaussian/site
gradient; float64 finite-difference gradcheck of recurrence; homogeneous analytic
solution including zero/tiny/large coefficients; tied-field legacy equivalence;
opaque occlusion. `test_model.py`: parameter shapes, metadata, reorder/restore.
`test_pipeline.py`: zero/positive prefix, exact handoff, fresh optimizers,
population growth/reset, finite updates, joint resume and production-resolution
IUI3 smoke. `test_config.py`: all twelve paired settings.

`scripts/rgb_medium_gate.py` runs the new groups plus existing COLMAP/end-to-end
and native renderer regressions in isolated processes. Required GPU tests cannot
be skipped; certificate hashes source/config/scripts/tests. Production smoke is
isolated to avoid lingering native OptiX contexts from previous tests.

## Twelve paired experiments

Config `configs/colmap_rgb_medium.yaml`: initialization 25k, COLMAP fraction .25,
extent 2; warmup 250/500, scalar/RGB medium, seeds 42/43/44; 30k joint steps.

| IDs | Warmup | Model | Seeds |
|---|---:|---|---|
| 0–2 | 250 | scalar | 42,43,44 |
| 3–5 | 250 | RGB | 42,43,44 |
| 6–8 | 500 | scalar | 42,43,44 |
| 9–11 | 500 | RGB | 42,43,44 |

Seven concurrent RTX4090 jobs; W&B project `foamsurf_iui3`, group
`rgb-medium-ARRAY_JOB_ID`, output `outputs/rgb_medium/ARRAY_JOB_ID/run_INDEX`.
The extra passes increase compute; budgets must be compared in both iterations
and wall time. No full sweep is automatically submitted by test preparation.

```bash
cd /shared/results/gmwilczn/foamsurf/foamsurf
test_job=$(sbatch --parsable scripts/test_rgb_medium.sbatch)
test_job=${test_job%%;*}
export RGB_MEDIUM_TEST_CERTIFICATE="$PWD/outputs/rgb_medium_tests/$test_job/passed.json"
sbatch --dependency=afterok:$test_job --export=ALL scripts/train_rgb_medium.sbatch
```

After a valid local RTX4090 certificate is available, set the variable to that
certificate and submit the training script directly. Do not change source or
config between certification and queued-task startup. One command outside SLURM:

```bash
python scripts/train_colmap_joint.py --config configs/colmap_rgb_medium.yaml \
  --variant 3 --output-dir outputs/rgb_medium/manual/run_3
```

Analyze best AND final PSNR/SSIM, every held-out view, seed variability and time/
memory. Inspect actual backscatter for seabed structure, not just Voronoi-only.
Use same-batch pre/post adaptation logs to distinguish topology disturbances
from generalization. Additional off-cycle full-view pre/post evaluation is NOT
implemented in this version; periodic eval still aligns with adaptation.
An early PSNR peak is not evidence of successful separation or identifiable
physical coefficients. Medium-only prefix cannot uniquely identify attenuation,
backscatter and source color from RGB alone; inspect coefficient drift.
