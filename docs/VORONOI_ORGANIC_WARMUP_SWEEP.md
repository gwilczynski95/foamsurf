# Organic Voronoi warmup: implementation, tests and experiment protocol

## Scope and evidence

This is the initialization stage of FoamSurf, trained on the combined scene.
It produces a canonical Voronoi checkpoint for later surface/medium
classification and Gaussian spawning. It does not spawn Gaussians or change
the joint training algorithm, renderer, population controller, or defaults.
All new behavior is opt-in via WarmupProfile and WarmupTrainer.

Repository: /shared/results/gmwilczn/foamsurf/foamsurf

Environment: /shared/results/gmwilczn/conda/envs/foamsurf

Training data: /shared/sets/datasets/foamsurf/lego/lego_both

Evaluation-only layers:
/shared/sets/datasets/foamsurf/lego/lego_lego and
/shared/sets/datasets/foamsurf/lego/lego_clouds.
These layers never enter training, pruning, error maps or parent selection.

Reference RadFoam run (read-only):
/shared/results/gmwilczn/foamsurf/radfoam/output/lego_both_prior_487402

Previous native sweep:
/shared/results/gmwilczn/foamsurf/foamsurf/outputs/warmup_radfoam_like/504916

The previous reference had ~355k starting sites, +2% population every 500
steps and almost no pruning. RadFoam had 131072 sites INCLUDING outer sites,
fell to ~83k at its first adaptation, and grew to ~371k at 4k. This coincided
with a reconstruction/separation improvement. It is a hypothesis, not proof,
that population dynamics caused that improvement. SH, sampling, color
parameterization and classification also differ.

Correction to the earlier interpretation: an empty predicted MEDIUM render
proves failure of the chosen partition; it does not alone prove that the
field cannot be separated. The old RadFoam classifier used contribution>0,
while the native analyzer applies Otsu to log contribution before TV Otsu.
Neither similar thresholds across models nor similar site counts prove
equivalent fields. Keep oracle metrics and distributions in the comparison.

## New full variant

6000 optimization steps; SmoothL1 RGB + alpha MSE (weight 1) + quantile width.
Quantile weight reaches 1e-4 at 10k; its horizon is NOT compressed to 6k.
The existing 18k/20k LR horizons and 2k density LR warmup are retained.
Density uses softplus(beta=10), scale=1; initial inner raw=0,
outer raw=-0.5. Appearance remains sigmoid RGB (no SH).

126072 inner sites are uniform IID in [-1.5,1.5]^3. 5000 outer sites are
Gaussian with standard deviation 10, centered at the scene bbox center.
All positions, densities and colors are trainable, including outer sites.
Outer provenance is tracked across pruning, children and permutations,
but does not protect them from population updates in the organic policy.
The existing analyzer still excludes outer provenance from semantic labels.

Every training step draws 1000000 pixels uniformly with replacement across
ALL train views. The same flattened indices select ray, RGB and alpha.
No full-image SSIM is computed on mixed rays. Progressive resolution remains
4/2/1 at zero-based steps 0/150/500; the mixed batch remains 1M rays at every
resolution. Loss means normalize by the actual sampled ray count.
The existing single-camera option remains available for the paired ablation.

Adaptation starts at completed iteration 2000. Each event:

1. Collect a fresh map over every train camera at the CURRENT training
   resolution. Take one random offset of a factor-2 pixel lattice per view.
2. Backpropagate sum(pixel-wise mean-channel absolute RGB error), with no
   alpha or quantile loss and no optimizer update. Accumulate the complete
   position gradient within a view before taking its norm, then sum norms
   across views. This is RadFoam's geometry-gradient error signal, not a
   pixel residual assigned directly to cells.
3. Sum contribution within each view, then take the per-cell MAX across
   views. Error-map rendering is chunked at 65536 rays; reductions preserve
   the per-view objective. Gradients are cleared before each view and after
   collection.
4. Prune cells with contribution <=0.01, no neighbor with contribution>0.01,
   and farthest-neighbor radius <0.1 scene units. No patience/minimum-100k
   rule from the joint controller is used.
5. Reset cells with contribution<0.001 to physical sigma
   softplus_beta10(-1)=4.5398899e-6 (raw=-1 for the reference activation).
   Like RadFoam, existing Adam moments are retained during this density reset.
6. Sample floor(0.15*N_before) parents without replacement, with weights
   position-gradient-error * farthest-neighbor-radius. Zero-score candidates
   cannot be selected. A zero/insufficient support set reduces the count.
7. Put children at parent + .25*(farthest_neighbor-parent), plus a random
   unit-direction perturbation of magnitude .1 times that displacement.
   Children inherit parent density AFTER reset and parent color. Child Adam
   moments start at zero; surviving moments are preserved.
8. Rebuild triangulation transactionally. If construction exhausts retries,
   retain the previous population AND undo density reset. Report failure;
   do not silently count it as successful densification.
9. Schedule the next event after:
   max(100, int(.15 * N_after * (11000-2000)/(2097152-131072))).
   The denominator uses the actual initial count in the 350k ablation.

Target population is 2097152 total sites. Events stop when the current count
reaches 90% of that budget or the configured adaptation window ends; a hard
cap prevents overshoot. The 6k screening stops before the 11k growth horizon.
Counts are data-dependent; this does not force the RadFoam count trajectory.

Between events, the RadFoam-style topology refresh period grows 1,3,5,...,101;
successful topology/population changes keep renderer geometry synchronized.
The period resets to 1 at population events and is stored in checkpoints.
The legacy fixed-period policy remains an option.

## Deliberate differences from literal RadFoam reproduction

- No SH; sigmoid RGB instead of clamped SH DC plus directional terms.
- Existing native renderer/near/far/cutoffs remain unchanged.
- LR is applied BEFORE the step; RadFoam applies its schedule afterward and
  therefore has an initial full-LR step. This pre-existing warmup convention
  is retained to isolate population/sampling changes.
- Correct CSR neighbor sums include the first neighbor of each cell.
  The local RadFoam code indexes an unpadded cumulative sum; that off-by-one
  is not reproduced.
- Transactional retries, insufficient-sampling-support handling, a hard
  capacity limit and protection against fewer than four survivors are
  explicit safety differences.
- Native analysis uses contribution-Otsu then TV-Otsu, excludes outer sites
  and outer-inner edges. Its results are comparable across this sweep, but
  are not a literal reimplementation of the earlier RadFoam classifier.

## Sixteen experiments

IDs here belong to the ORGANIC suite, not the earlier 50-run sweep.
All runs last 6000 iterations; all use opacity supervision except control 0.

| IDs | Variant | Seeds | Purpose |
|---|---|---|---|
| 0 | Previous run 1: beta1, SmoothL1, old settings | 0 | Promising separation control |
| 1 | Previous run 7: old reference R | 0 | Failed-separation control |
| 2,3 | Full organic variant | 0,42 | Main hypothesis |
| 4,5 | Full organic, single-camera full-image batches | 0,42 | Multi-view sampling and ray-budget package |
| 6,7 | Full organic, original FoamSurf population controller | 0,42 | Population algorithm package |
| 8,9 | Full organic, no low-contribution density reset | 0,42 | Reset contribution |
| 10,11 | Full organic, stratified initialization | 0,42 | Initial spatial distribution |
| 12,13 | Full organic, quantile weight 0 | 0,42 | Quantile contribution |
| 14,15 | Full organic, start 350k inner +5k outer | 0,42 | Initial population size |

The original-population ablation uses +2% every500 starting at2k, the shared
score/pruning rules and protected outer population. This intentionally tests
the whole population policy, not one isolated constant. The single-camera
ablation uses fewer rays, especially at low resolution; comparisons should
include wall time and rays processed, not only iteration number.

## Checkpoints, resume and handoff

Milestones: 0,500,1000,1999,2000,2500,3000,4000,5000,6000.
Also retain preadapt_*.pt immediately before population events.
Pre-adaptation snapshots are diagnostics only, not supported resume points.

Canonical model density is inverse-softplus(beta1) of the learned sigma;
the standard FoamSurf Trainer/analysis/handoff reads the same physical field.
The warmup payload retains original density parameters, profile, RNG state,
sampler stack, topology schedule and optimizer state. Organic controller state
retains next adaptation iteration and cumulative density resets. Resume with
WarmupTrainer and the same profile, never a joint optimizer resume.

No automatic Gaussian initialization or continuation is submitted by this suite.

## Tests and launching

All runtime tests are prepared for RTX4090 through SLURM. This implementation
was checked statically and configurations were validated on the current node;
GPU test success is still required. The test job runs the existing warmup,
sampler, native renderer and initialization tests plus the new tests:

- mixed ray/RGB/alpha alignment and RNG replay;
- strict pruning thresholds and correct CSR neighbor protection;
- weighted sampling without replacement, zero weights and finite checks;
- child displacement/jitter and adaptive timing;
- manifest isolation, seeds and total initial population;
- actual mixed-camera training, adaptation and checkpoint/resume;
- full fit/checkpoint/logging smoke with 131072 sites and 1000000 rays per
  step, including quantile backward (smaller source images reduce data residency);
- stable CSV columns before and after population updates;
- canonical checkpoint compatibility and unchanged Gaussian parameters;
- chunked error-map equivalence, gradient clearing and no parameter updates;
- failed triangulation rollback including density reset;
- outer-parent lineage, pruning, inherited color/reset density, preserved
  survivor Adam moments and zero child moments.

The content-addressed certificate includes the new code and tests. Existing
certificates are stale after these edits. There is no bypass.

From the repository:

```bash
cd /shared/results/gmwilczn/foamsurf/foamsurf
TEST_JOB_ID=$(sbatch --parsable scripts/test_warmup_organic.sbatch)
TEST_JOB_ID=${TEST_JOB_ID%%;*}
echo "$TEST_JOB_ID"
```

Then queue the dependent experiment (same shell):

```bash
sbatch --dependency=afterok:${TEST_JOB_ID} \
  --export=ALL,WARMUP_TEST_CERTIFICATE="$PWD/outputs/warmup_tests/${TEST_JOB_ID}/passed.json" \
  scripts/sweep_warmup_organic.sbatch
```

Array 0-15, at most four concurrent RTX4090 jobs; 64GB host RAM, 8 CPUs,
12h per task. Each task trains then analyzes its milestone checkpoints.
Do not modify fingerprinted code/config/tests while the dependent jobs run.

Outputs: outputs/warmup_organic/ARRAY_JOB_ID/run_ID/

Each run also writes metrics/population_events.jsonl: every adaptation event,
including iteration, pre/post counts, densified/pruned/reset counts, pre/post
PSNR, next scheduled event and error-map time. This captures events that do
not coincide with the periodic CSV logging interval.

Logs: slurm_logs/organic_ARRAY_JOB_ID_TASK_ID.{out,err}

Tests: slurm_logs/organic_test_JOB_ID.{out,err}

Certificate: outputs/warmup_tests/TEST_JOB_ID/passed.json

CPU aggregation after/while jobs finish:

```bash
/shared/results/gmwilczn/conda/envs/foamsurf/bin/python \
  scripts/summarize_warmup_radfoam_like.py \
  --root outputs/warmup_organic/ARRAY_JOB_ID --count 16
```

## Analysis protocol

First confirm exit status, final_iteration=6000, finite objectives, adaptation
failure count, renderer truncated-ray fraction, time and peak VRAM. A populated
output directory or attractive RGB image does not imply a successful prior.

Compare both seeds of 2/3 against controls at 2k,4k,5k and6k. Inspect:

1. Full RGB quality and oracle separation quality. If oracle succeeds but
   checkpoint-only labels fail, investigate classifier support/thresholds.
2. Medium PSNR/SSIM versus lego_clouds, surface PSNR/SSIM versus lego_lego,
   surface alpha on cloud-only rays and medium alpha on LEGO-only rays.
3. Cloud contribution mislabeled surface, contribution-weighted confusion,
   class counts AND actual class render contribution. Empty medium is failure
   even if leakage is zero.
4. Inner/outer counts, total densified/pruned/reset, per-event reconstruction
   drop, and recovery between adaptations. Inspect preadapt checkpoints if
   post-event quality drops. Do not equate a post-event checkpoint with the
   fully recovered field.
5. Sigma/local-TV/contribution distributions and threshold drift. Check whether
   near-zero unused cells dominate the histogram; never infer physical density
   of clouds from a global Otsu threshold alone.
6. Full/surface/medium panels and fixed-density occupancy in all four saved
   views, including the persistent compact cloud. Occupancy is a debug render
   at artificial density, not the trained medium's opacity.
7. Compare each ablation pair with its matching full seed. Record which
   mechanism improves separation and which only improves full-scene PSNR.

The existing analyzer also produces oracle panels, quantile maps and objective
dashboards. Its handoff labels remain in ORIGINAL checkpoint site order.
Select checkpoints using reconstruction, separation and stability together;
best.pt is selected by full PSNR and is not automatically the best prior.

No universal requirement to hit exactly RadFoam's 37dB layer result is assumed.
If full organic dynamics improve reconstruction but oracle separation is poor,
the learned field still differs. If oracle separation is good and Otsu poor,
revisit the classifier before rejecting the warmup. If both fail, next tests
should isolate appearance/DC parameterization, optimizer timing and renderer
forward/backward parity on identical fields.

## Implementation map

- training/warmup_profile.py: opt-in controls and validation.
- training/warmup_sampling.py: uniform mixed-camera pixel batches.
- training/warmup_population_math.py: CPU-testable policy math.
- training/warmup_population.py: population policy, rollback and schedule state.
- training/warmup_trainer.py: warmup integration, fresh maps, initialization,
  outer provenance, checkpoints and topology schedule.
- training/warmup_organic_sweep.py: explicit 16-run manifest.
- scripts/run_warmup_radfoam_like.py --suite organic: verified launch.
- scripts/test_warmup_organic.sbatch and scripts/sweep_warmup_organic.sbatch.
- tests/warmup_radfoam_like/test_organic*.py: new CPU/GPU tests in the gate.
