# Native FoamSurf end-to-end baseline

Repo: `/shared/results/gmwilczn/foamsurf/foamsurf`.
Environment: `/shared/results/gmwilczn/conda/envs/foamsurf` (conda `foamsurf`).
GPU: RTX4090 through SLURM `rtx4090_batch`.

## One entry point

```bash
cd /shared/results/gmwilczn/foamsurf/foamsurf
python scripts/train_end_to_end.py --config configs/end_to_end.yaml --output-dir outputs/end_to_end/my_run
```

Python API:

```python
from training.end_to_end import EndToEndConfig, EndToEndTrainer
EndToEndTrainer(EndToEndConfig.load("configs/end_to_end.yaml")).fit()
```

One `fit()` executes the three internal stages serially, with separate worker
processes to release GPU memory between stages. The user never manually selects
a checkpoint, extracts features, classifies cells or launches the hybrid stage.
No RadFoam model is read. Shared PCA and boundary-neighbor helper functions are
reused from the prior initializer; no RadFoam density/color conversion is applied.

## Stage 1: native Voronoi warmup

Default 5000 iterations, settings of organic run 12 (seed 0):

- 126072 inner sites + 5000 trainable outer sites initially; adaptive population.
- Uniform mixed-camera batches, 1 million rays per step.
- SmoothL1 RGB, opacity supervision weight 1, depth-quantile weight 0.
- Softplus beta=10; cosine LR schedules and density LR warmup.
- Progressive resolutions 4/2/1 at steps 0/150/500.
- Organic population pruning/reset/splitting begins at 2000, target capacity
  2097152, growth horizon 11000. The trained final count is data-dependent.

This baseline still uses available alpha supervision. That choice has not been
validated as an equivalent baseline for RGB photographs without alpha.
Separate LEGO/cloud ground-truth paths are optional evaluation diagnostics;
neither classification nor handoff reads separated layers.

## Stage 2: partition and initialization

Read the FINAL warmup `last.pt`, not the checkpoint selected by highest PSNR.
Native checkpoints contain canonical beta=1 inverse-softplus parameters for the
same physical sigma. Warmup resume metadata retains beta=10 raw parameters;
joint initialization uses canonical model tensors and fresh optimizers.

Accumulate compositing contribution over all TRAIN views in 8192-ray chunks.
Compute local TV using inner-inner Delaunay edges. Apply the selected classifier
without GT. Supported methods:

- `weighted_valley_w1_s2` (default): contribution-weighted histogram, smoothing 2 bins.
- `otsu_w1`: contribution-weighted Otsu.
- `gmm_2`: two-component sigma/TV mixture.

The other eleven sweep presets and GMM 3/auto are removed from active code.
Existing experiment outputs and their historical documentation remain intact.
Classification failure or empty SURFACE stops the pipeline; no silent fallback.

Binary classes are SURFACE and its entire complement. Outer-provenance and
zero-contribution cells remain in the complement. At handoff:

1. Sample up to 100000 unique SURFACE sites uniformly, with deterministic seed.
   If fewer exist, use all of them. Positions and RGB initialize the Gaussians;
   orientations are local PCA frames, scales use the existing nearest-neighbor
   initializer and opacity uses the joint baseline value 0.1.
2. Delete every SURFACE site from Voronoi (also those not sampled as Gaussians).
3. Mark retained sites adjacent to any removed site in the source graph.
   Set their physical density to `1e-8` (finite raw parameter, trainable).
4. Preserve sites, colors and canonical densities of all other retained cells.
   EMPTY and MEDIUM are not treated separately. Keep ALL retained sites,
   including outer provenance. They are trainable with no guard freeze.
5. Rebuild the retained population's topology through the standard Trainer.

Masks and Gaussian source indices are exported in original checkpoint order.
Geometry is checked against the source checkpoint before initialization.
The standard model schema requires at least 32 retained medium cells; a smaller
result is rejected rather than silently filled or resampled.

## Stage 3: ordinary hybrid training

Default 30000 new iterations using the accepted prior-initialized run30 joint
configuration. Gaussian densification ends at joint iteration 15000. No warmup
iterations are counted in that schedule. New Adam optimizers have empty state.
Voronoi population adaptation is disabled: retained cells are never reduced to
the historical 8k limit. Their positions, density and appearance still optimize.

The only Trainer extension is an optional `initial_model` argument, incompatible
with `resume`; the existing training loop and regularizers are unchanged.

## Configuration

- `configs/end_to_end.yaml`: public wrapper, full warmup profile, partition,
  Gaussian count, PCA neighbors, reset density and common overrides.
- `configs/end_to_end_warmup.yaml`: full warmup ExperimentConfig, including
  iterations, initial population, adaptation window and diagnostics.
- `configs/end_to_end_joint.yaml`: full ordinary hybrid ExperimentConfig.

All paths to stage configs are relative to the wrapper YAML. `common` overrides
apply to both stages, then each stage's `overrides` mapping applies. For example:

```yaml
common:
  seed: 42
  dataset:
    source_path: /path/to/combined_scene
    gaussian_ground_truth_path: null
    medium_ground_truth_path: null
partition:
  method: otsu_w1
  batch_size: 8192
  image_views: 4
```

When changing warmup duration, also keep milestone iterations, adaptation window
and progressive schedule consistent. Defaults reproduce the accepted 5k recipe.
The initial `joint.model.medium_points=32` is a schema placeholder: it is replaced
by the exact retained count BEFORE model construction. It never truncates cells.

## Tests and launch

CPU checks cover configuration isolation, checkpoint-order permutation, boundary
reset sets, deterministic sampling, preservation of physical density/color,
all retained parameters trainable, orchestration failure/resume and classifier
regressions. The test sbatch additionally runs the existing warmup GPU regressions
and a real multi-process native warmup -> partition -> hybrid smoke test, including
actual renders and an idempotent resume. The smoke uses small images/populations,
four steps per training stage and Otsu to avoid requiring a semantically bimodal
field after four toy steps; all three classifiers are tested on synthetic fields.
This is correctness testing, not a claim that a production training has completed.

```bash
cd /shared/results/gmwilczn/foamsurf/foamsurf
TEST_JOB_ID=$(sbatch --parsable scripts/test_end_to_end.sbatch)
TEST_JOB_ID=${TEST_JOB_ID%%;*}
sbatch --dependency=afterok:${TEST_JOB_ID} \
  --export=ALL,END_TO_END_TEST_CERTIFICATE="$PWD/outputs/end_to_end_tests/${TEST_JOB_ID}/passed.json" \
  scripts/train_end_to_end.sbatch
```

Training verifies an RTX4090 success certificate matching the entire code/config
fingerprint AND containing the end-to-end tests. Skipped tests or a warmup-only
certificate cannot authorize it. Do not edit fingerprinted source between the
test and training jobs. No jobs were submitted during this implementation turn.

Logs: `slurm_logs/e2e_test_JOB.out/.err`, `slurm_logs/end_to_end_JOB.out/.err`.
Default training outputs: `outputs/end_to_end/JOB/`.

## Outputs and resume

- `pipeline_config.json`: resolved configuration of all stages.
- `warmup/`: learning curves, evaluation images and canonical checkpoints.
- `partition/histograms.png`, `decision_boundary.png`, `view_XX_partition.png`:
  automatic threshold and full/SURFACE/NON-SURFACE renders before handoff.
- `partition/handoff.pt`: original-order masks, sampled sites and provenance.
- `handoff_report.json`: removed/retained/reset counts and classifier settings.
- `transition/view_XX_initialization.png`: hybrid/Gaussian/Voronoi immediately
  after initialization; compare to partition images to see the handoff cost.
- `joint/checkpoints/initialization.pt`: iteration zero with fresh optimizers.
- `joint/`: normal hybrid training checkpoints, curves and layer diagnostics.
- `*_done.json`: atomic stage completion markers; `summary.json`: final outcome.

```bash
python scripts/train_end_to_end.py --config configs/end_to_end.yaml \
 --output-dir outputs/end_to_end/JOB --resume
```

Resume requires the same resolved configuration. Completed stages are skipped;
unfinished training resumes from its last checkpoint. A crash between final
checkpoint and completion marker can be recovered without retraining the stage.
An interruption before the FIRST checkpoint may leave a nonempty stage directory;
it deliberately errors rather than deleting files. Use a new output directory
in that case. Joint resume follows existing Trainer semantics and is not claimed
to be bitwise identical across an interruption.

For sbatch resume set `OUTPUT_DIR` to the existing directory and `RESUME=1`,
and provide a valid test certificate as above. The wrapper configuration must
match the original job. No checkpoint data or historical sweep outputs are deleted.

## Baseline status recorded before the adaptive-hybrid experiments

RTX4090 test job 508219 passed the end-to-end integration and renderer regression
gate after correcting the four-step smoke test's inherited pruning schedule.
The online training launcher changes only W&B logging relative to this tested
configuration; warmup and joint optimization settings are identical.

During the running hybrid experiment, the user's visual review reports that
clouds remain in Voronoi, Gaussians do not take them over, and Voronoi does not
take over LEGO. This records an observed intermediate result, not a claim of
completed training or a newly measured quantitative score.

This baseline keeps the post-handoff Voronoi population fixed while optimizing
its positions, density and color. Adaptive Voronoi population during hybrid
training and additional Gaussian opacity constraints are future experiments,
not part of this baseline commit.
