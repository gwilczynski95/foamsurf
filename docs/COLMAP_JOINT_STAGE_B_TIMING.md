# IUI3 Stage B: joint start versus Voronoi-only warmup

## Objective and scope

Determine whether medium pretraining helps or entrenches the undesired solid
geometry in Voronoi. This is a timing experiment with the EXISTING scalar-density
medium, not the proposed RGB attenuation/backscatter/SH model. No surface partition,
cell removal, density reset, Gaussian-only prefix, or freeze is introduced.

Repository: `/shared/results/gmwilczn/foamsurf/foamsurf`.
Conda environment: `/shared/results/gmwilczn/conda/envs/foamsurf` (`foamsurf`).
Dataset: `/shared/sets/datasets/SeathruNeRF_dataset/IUI3-RedSea-Undistorted`.
Same native coordinates, 25 training / 4 held-out cameras, renderer K64,
white background and 30,000 joint steps as Stage A job 512192.

## Fourteen experiments

Run IDs 0–6: 25,000 initial Voronoi sites, COLMAP fraction 0, extent 2
(Stage A run 2 geometry).

Run IDs 7–13: 25,000 initial Voronoi sites, COLMAP fraction .25, extent 2
(Stage A run 6 geometry; best completed Stage A final PSNR at analysis time).

Within each group, warmup lengths are respectively:
`0, 100, 250, 500, 1000, 2000, 4000` iterations.
Thus controls B4 and B11 reproduce A2 and A6 numerical settings at seed 42.
They are reproducibility controls, NOT additional independent random seeds.

At warmup 0: initialize Voronoi, perform zero optimization steps, export the
iteration-zero checkpoint, create all 21,907 COLMAP Gaussians and train both.
Constructing/exporting a warmup trainer is infrastructure, not pretraining.

At positive warmup: optimize Voronoi alone for exactly the requested length.
Gaussians do not participate in this rendering or receive optimization steps.
At handoff, preserve every Voronoi site and its physical density/color; create
Gaussians from original COLMAP xyz/RGB, initialize fresh optimizers and reset
the JOINT clock to zero. Joint then runs its full 30k steps without freezes.

Warmup retains existing SmoothL1, opacity and quantile losses, LR schedules,
resolution transitions at 150/500, and organic adaptation from 500. Schedules
are NOT rescaled to the selected prefix duration. Different lengths therefore
also differ in maturity, resolution reached and population size. This measures
the practical prefix recipe, not an isolated fixed-population optimization effect.
Total training steps = warmup length + 30,000, so compute is not matched.

All Stage A joint settings stay identical: both populations adaptive, Voronoi
cap 500k, Gaussian densification to 15k, opacity pruning through 30k with .10
target, binarization .001 with 1000–4000 cosine ramp. TV/cap/sparsity remain off.
Oversized warmup handoff raises an error instead of truncating cells.

## Run: tests first, then dependent sweep

Run from the repository root:

```bash
cd /shared/results/gmwilczn/foamsurf/foamsurf
test_job=$(sbatch --parsable scripts/test_colmap_joint.sbatch)
test_job=${test_job%%;*}
export COLMAP_JOINT_TEST_CERTIFICATE="$PWD/outputs/colmap_joint_tests/$test_job/passed.json"
sbatch --dependency=afterok:$test_job --export=ALL scripts/train_colmap_timing.sbatch
```

The test job requires RTX4090 and certifies the current source fingerprint.
Old Stage A certificates become stale after this change. A failed test prevents
the dependent sweep from running. Do not edit source/config until queued tasks
have passed the certificate check. Array 0–13, at most SEVEN concurrent tasks,
one RTX4090 each, 64GB RAM, 8 CPUs, 24h maximum. No jobs submitted by preparation.

W&B: online project `foamsurf_iui3`, group `colmap-timing-ARRAY_JOB_ID`;
separate `ct-ARRAY_JOB_ID-TASK_ID-warmup` and `...-joint` runs.
Outputs: `outputs/colmap_timing/ARRAY_JOB_ID/run_INDEX/`.
Logs: `slurm_logs/colmap_timing_ARRAY_JOB_ID_INDEX.{out,err}`.
The one-command local pipeline is:

```bash
python scripts/train_colmap_joint.py --config configs/colmap_joint_stage_b.yaml \
  --variant 0 --output-dir outputs/colmap_timing/manual/run_0
```

Use `--dry-run` to inspect or `--resume` with identical config/output to resume.

## Analysis protocol

1. Verify completion, test certificate, no truncated rays/failed adaptations,
   exact handoff count retention and correct prefix/joint clocks. Check controls
   B4 vs A2 and B11 vs A6 before interpreting differences.
2. Compare best checkpoint AND final PSNR/SSIM across all four held-out views.
   Also compare joint iterations 0,500,1000,5000,15000,30000. Record prefix length,
   wall time, handoff population and final populations. Report best-step selection
   on these four views as validation, not independent test-set improvement.
3. Inspect hybrid, Gaussian-only and Voronoi-only renders for EVERY held-out view.
   Does seabed/coral structure disappear from Voronoi while water remains?
   Are Gaussians coherent surfaces rather than sparse residual fragments?
   MTN_5903 previously deteriorated most; do not average away that failure.
4. Track medium weight fraction, Gaussian opacity, initial/born lineage and
   distance-to-COLMAP diagnostics. Weight fraction and distance are NOT semantic
   ground truth; component renders are counterfactuals, not additive layers.
5. Examine training-vs-validation trajectories, and Voronoi same-batch pre/post
   adaptation PSNR deltas. Eval/adaptation both occur every 500 joint steps:
   immediate post-adaptation evaluations can be pessimistic. No new off-cycle
   evaluation is added, to retain Stage A comparability.
6. Prefer improved final/generalization behavior AND better separation; do not
   declare success from an early PSNR spike or lower medium contribution alone.
   If short/no warmup helps, repeat promising settings with independent seeds
   before expanding. If all converge to the same failure, stop tuning timing
   and proceed to the separately planned medium model experiment.

## Tests

CPU contracts enumerate all 14 manifests, compare every joint numerical setting
against Stage A, check 1000-step controls and invalid prefix lengths. Existing
GPU pipeline tests now cover both Stage A and timing configurations with zero
and positive prefixes, exact transfer, fresh optimizers, iteration zero,
adaptation, finite updates and resume. Existing production/legacy GPU regression
groups remain in the same test gate. GPU tests are submitted through sbatch.

Preparation validation: Python AST and shell syntax checked; the isolated pure
variant generator checked all 19 legacy and 14 timing variants, identical joint
overrides and rejection of invalid lengths. Local pytest collection on the CPU
node fails because the existing topology extension initializes CUDA at import.
Thus the full contract tests have NOT passed locally; run the RTX4090 gate above.
