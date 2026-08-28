# Massive Gaussian sweep for lego_lego

The sweep contains 372 unique 30k-iteration configurations. Every Gaussian
configuration starts from the working FlaRe control that reached 25.309 dB.
There is deliberately no `max_points` or `runaway_max_points` cap.

## Families and Slurm indices

| Indices | Count | Family | Main variables |
|---|---:|---|---|
| 0-47 | 48 | control | seeds, initial count/opacity, KNN scale init, reset-only, scale regularization |
| 48-191 | 144 | graphdeco | screen-gradient threshold, interval, reset, opacity prune, clone/split boundary |
| 192-263 | 72 | scale_shape | scale regularization, min/max clamp, scale learning rate |
| 264-371 | 108 | flare_movement | movement threshold, interval, opacity/effective-scale prune, reset |

`graphdeco` is the 3DGS-style accumulated screen-gradient clone/split/prune
policy also present in the current FlaRe source. `flare_movement` is the
post-Adam-step implementation from the locally working `flare_new`: it keeps
the pre-step and post-step locations of sufficiently mobile Gaussians and maps
the Adam moments to the new topology.

Opacity reset is tested both with and without topology changes. The fixed
reset-only cases use `policy: fixed`, so they cannot accidentally densify.

Scale behavior is covered in several independent ways:

- regularization weights from 0 through 0.02;
- minimum scale clamps from 1e-5 through 5e-4;
- maximum scale fractions from 0.025 through 0.10;
- scale learning rates 0.02 and 0.08;
- initial KNN scale estimates using 1, 2, 3, 4, 8, or 16 neighbors;
- `percent_dense` values 0.005 and 0.02 in the GraphDECO grid.

## Launch

All 372 jobs, with at most eight running simultaneously:

```bash
cd /shared/results/gmwilczn/foamsurf/foamsurf
sbatch --export=ALL,WANDB_MODE=online scripts/sweep_stage1_gaussians.sbatch
```

A safer operational approach is to submit families separately. This still
runs every configuration but makes failures and queue usage easier to inspect:

```bash
sbatch --array=0-47%8 --export=ALL,WANDB_MODE=online scripts/sweep_stage1_gaussians.sbatch
sbatch --array=48-191%8 --export=ALL,WANDB_MODE=online scripts/sweep_stage1_gaussians.sbatch
sbatch --array=192-263%8 --export=ALL,WANDB_MODE=online scripts/sweep_stage1_gaussians.sbatch
sbatch --array=264-371%8 --export=ALL,WANDB_MODE=online scripts/sweep_stage1_gaussians.sbatch
```

The script excludes `c11` and `c22` and performs the same CUDA,
`foamsurf_topology`, renderer, and Python environment bootstrap as the
working control job. Because the population is uncapped by design, aggressive
configurations can exhaust GPU memory; such failures identify unstable regions
of the sweep rather than silently changing the algorithm.

## Rank results

```bash
/shared/results/gmwilczn/conda/envs/foamsurf/bin/python \
  scripts/summarize_gaussian_sweep.py \
  --csv outputs/stage1_gaussian_sweep/ranking.csv \
  --top 40
```

Every run stores its resolved config, family, and label. The summary script
ignores unfinished directories, ranks completed runs by best PSNR, and reports
the final Gaussian count. A failed individual task can be rerun with its exact
index, for example:

```bash
sbatch --array=271 --export=ALL,WANDB_MODE=online scripts/sweep_stage1_gaussians.sbatch
```
