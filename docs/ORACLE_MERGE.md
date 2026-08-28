# Frozen oracle checkpoint merge

This evaluation imports the Gaussian branch from the qualified
`gk_458910_17` checkpoint and the medium branch from the qualified
`vref_462246_28` checkpoint. It performs no training, creates no optimizer,
and executes no Gaussian or Voronoi adaptation event.

The source renderer contract is used first to reproduce each standalone
checkpoint metric. The merged hybrid, Gaussian-only, and Voronoi-only renders
then use every renderer setting from
`configs/baseline_best_2d_gaussians.yaml`, including K=64, `far=1000`, and the
validation transmittance threshold of 0.01.

## Submit

The canonical qualified checkpoints are the job defaults:

```bash
cd /shared/results/gmwilczn/foamsurf/foamsurf
sbatch scripts/evaluate_oracle_merge.sbatch
```

Alternative checkpoints or output settings can be supplied without editing
the job:

```bash
GAUSSIAN_CHECKPOINT=/path/to/gaussian/best.pt \
VORONOI_CHECKPOINT=/path/to/voronoi/best.pt \
OUTPUT_DIR=/path/to/output \
WANDB_MODE=offline \
sbatch scripts/evaluate_oracle_merge.sbatch
```

`MAX_EVAL_VIEWS` defaults to all test views and `IMAGE_VIEWS` defaults to 4.
`REPRODUCTION_TOLERANCE` defaults to 0.0005 dB. Reproduction enforcement is
enabled for checkpoint files named `best.pt`; another filename is evaluated
and reported but has no reliable stored metric for a strict comparison.

## Outputs

The output directory contains:

- `config.yaml`: resolved frozen hybrid evaluation config;
- `oracle_manifest.json`: source paths, SHA-256 hashes, compatibility checks,
  branch reproduction results, and the renderer contract;
- `metrics/eval.csv`: aggregate hybrid and isolated-branch metrics;
- `evaluation/iteration_000000/`: per-view metrics and diagnostic renders;
- `summary.json`: final metrics with `optimizer_steps=0` and
  `adaptation_events=0`.

The output is explicitly labelled as oracle-only and must not be used to
initialize a clean joint training run.
