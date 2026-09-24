# IUI3: Stage A — spatial Voronoi + COLMAP Gaussians

## Scope

This is the FIRST experiment family, not stages B/C. No partition, semantic
labels, RadFoam checkpoint, surface-site removal or density reset at handoff.
The historical LEGO end-to-end path remains independent.

Repository: `/shared/results/gmwilczn/foamsurf/foamsurf`.
Environment: `/shared/results/gmwilczn/conda/envs/foamsurf`.
Dataset: `/shared/sets/datasets/SeathruNeRF_dataset/IUI3-RedSea-Undistorted`.
Same native coordinates, PINHOLE intrinsics, every-eighth test split (25/4),
white background, near=0, far=1000, K=64. No separate layer ground truth.

## Variants

0 = existing RadFoam-like PCD control: floor(.9*21907)=19716 sampled+jittered
COLMAP sites plus 5000 outer sites, N(0,10^2). Raw density U(0,1) for inner,
-0.5 for outer, beta=10. Outer provenance is retained during warmup but NOT
treated as a permanent guard/protection in joint.

1–18 enumerate count [25000,50000,100000], COLMAP fraction [0,.1,.25],
extent [1,2], in that order (extent varies fastest). Counts include ALL sites:

| IDs | Count | Fractions (each moderate/wide) |
|---|---:|---|
| 1–6 | 25000 | 0, .1, .25 |
| 7–12 | 50000 | 0, .1, .25 |
| 13–18 | 100000 | 0, .1, .25 |

For spatial variants, let lo=min(COLMAP coordinate p1, train camera minimum),
hi=max(COLMAP p99, train camera maximum). Center=(lo+hi)/2;
axiswise standard deviation=max((hi-lo)/2,1e-3)*extent. Sample a Gaussian,
NOT an outer shell: points cover camera–surface space AND far surroundings.
No hard clipping to these robust bounds. The COLMAP part samples with replacement
and isotropic jitter std=max(0.001*norm(hi-lo),1e-5). Fraction zero means NO sites
sampled at COLMAP points, but COLMAP still determines the spatial scale.
All spatial sites are ordinary learnable cells: raw density U(0,1), beta=10,
raw DC=0 -> RGB=.5, guard mask false. This differs from the control's low-density
outer population, so the control comparison is a whole-initialization comparison,
not a pure positions-only ablation. Within the 18 spatial variants the density
law is identical. Initialization is seeded (42); geometries are saved as PNG/JSON.

## Training

- 1000 Voronoi-only warmup steps, native existing SmoothL1/opacity/quantile
  objectives and RadFoam-like LR/progressive resolution. Real-image alpha is
  the loader's all-ones target, not a supplied object mask.
- Warmup population adaptation starts at 500; organic target 2097152 and
  horizon 11000 remain common across variants. Short warmup does NOT attempt
  to reach that target. Log the actual handoff count for every run.
- Transfer ALL remaining sites/density/colors; zero removed. Spawn all 21907
  Gaussians at original COLMAP positions/colors, ordinary model initialization
  for orientation/scale. No Gaussian optimizer steps occur during warmup.
- Joint starts at local iteration zero, fresh optimizers, 30000 steps. Preserve
  physical sigma via canonical beta=1 density; preserve color EXACTLY using
  opt-in DC activation instead of clipping to sigmoid's range. This changes
  joint color parameterization ONLY in this new path; old checkpoints default
  to sigmoid. Numerical tests cover DC >1, checkpoint reload and transfer.
- Both populations adaptive. Gaussian clone/split until 15000, opacity pruning
  through 30000, interval 250. Binary weight .001 starts 1000, cosine ramp 3000.
  Pruning ramps from .005 to .10 over the same interval; at most 10% removed
  per opacity-prune event, at least 1000 Gaussians retained. No opacity resets.
- Gaussian max population 500k. Voronoi split fraction .05/prune fraction .02,
  every 500 steps from 1000 until the end (controller uses exclusive upper
  bound), min 10k, max 500k. Limits are shared, not proportional to initial N.
  Reaching capacity does not freeze the model. Oversized handoff errors instead
  of truncating. Bounds expand to existing sites; moved sites are not clamped
  back to a LEGO box. Ordinary existing joint losses/renderer otherwise apply.
- TV, density cap, optical sparsity, opaque-if-used OFF in Stage A. These are
  existing configurable mechanisms for subsequent experiments, not removed.
- Existing joint appearance is RGB (no new SH ablation).

## Launch

From the repository root (create slurm_logs before sbatch if absent):

```bash
cd /shared/results/gmwilczn/foamsurf/foamsurf
test_job=$(sbatch --parsable scripts/test_colmap_joint.sbatch)
test_job=${test_job%%;*}
export COLMAP_JOINT_TEST_CERTIFICATE="$PWD/outputs/colmap_joint_tests/$test_job/passed.json"
sbatch --dependency=afterok:$test_job --export=ALL scripts/train_colmap_joint.sbatch
```

Default array 0–18, maximum 3 concurrent jobs, one RTX4090 each, 64G RAM,
24h per task. To run ONLY the control initially, add `--array=0` to the last
command. `--array=1` runs the 25k / zero-COLMAP / moderate spatial variant.
The training job verifies a source-fingerprinted RTX4090 test certificate;
changing code/config afterward requires rerunning tests. No unchecked bypass.

W&B online project `foamsurf_iui3`, group `colmap-initialization-ARRAY_JOB_ID`,
separate stage IDs (warmup/joint). Existing login must be available on the node.
Outputs: `outputs/colmap_joint/ARRAY_JOB_ID/run_INDEX/`.
Logs: `slurm_logs/colmap_joint_ARRAY_JOB_ID_INDEX.{out,err}`.

Manual single-command pipeline in the foamsurf environment:

```bash
python scripts/train_colmap_joint.py --config configs/colmap_joint_stage_a.yaml \
  --variant 1 --output-dir outputs/colmap_joint/manual/run_1
```

Add `--resume` to restart the SAME output/config. `--dry-run` prints the resolved
manifest without running training. Warmup=0 is supported by the pipeline, but
not part of this first array. No launching of B/C sweeps is included.

## Analysis protocol

Compare best AND final held-out PSNR/SSIM/(available LPIPS), not only best PSNR.
Inspect hybrid/Gaussian-only/Voronoi-only, responsibility and transmittance for
all four test cameras. Separate component renders are counterfactuals, not
additive RGB layers. No GT separation IoU/accuracy exists here.

Look for surface detail moving to Gaussians, water contribution retained in
Voronoi, absence of dense background sheets, and stable population growth.
Track initial and densified Gaussians separately via `colmap_joint.birth_steps`
in checkpoints. `joint/colmap_diagnostics/` records their count, opacity and
distance quantiles to the original COLMAP cloud at each evaluation.
Distance is not a semantic score: a Gaussian can absorb water effects in color
while remaining on the surface. Growth away from sparse COLMAP can also fill
real missing surfaces. Inspect these together with renders.

Evaluation/checkpoints every 500 joint steps, plus evaluation at iteration 0.
Use warmup initialization/final and joint initialization/500/1000/5000/15000/30000
to examine transitions. Existing train/eval metrics and population event logs
record adaptation, losses and rendering diagnostics. Birth state, camera stack,
Python/NumPy RNG and medium adaptation accumulators are saved for resume.

Choose two initializations using reconstruction AND representation behavior;
then evaluate warmup lengths. Do not pick a winner from counts/PSNR alone.

## Implementation and tests

`training/colmap_initialization.py`: spatial sampler.
`training/colmap_joint.py`: stage-A manifest, orchestrator, exact transfer.
`training/colmap_joint_trainer.py`: opt-in lineage/diagnostics/resume.
`configs/colmap_joint_stage_a.yaml`: settings; inherited renderer/LRs in base files.
`tests/colmap_joint/`: CPU and real IUI3 GPU tests including full resolution.
`scripts/colmap_joint_gate.py`: tests + existing end-to-end/warmup regressions.
Production smoke runs in a separate process because native OptiX contexts can
outlive PyTorch object cleanup. Never combine it with a parent holding a GPU trainer.
