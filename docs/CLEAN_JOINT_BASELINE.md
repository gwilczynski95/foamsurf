# Stage 3 clean joint baseline

This is the first non-oracle hybrid training run. Both representations are
initialized from scratch and trained only against `lego_both`. The aligned
`lego_lego` and `lego_clouds` images are loaded on CPU for test diagnostics and
have no path to the sampler or training loss.

The configuration combines the qualified Gaussian and Voronoi recipes without
adding a disentanglement loss. Renderer settings come exactly from
`configs/baseline_best_2d_gaussians.yaml`.

```bash
cd /shared/results/gmwilczn/foamsurf/foamsurf
sbatch scripts/train_clean_joint.sbatch
```

The default run uses seed 0, 30,000 iterations, full 800x800 images, K=64,
100,000 initial adaptive Gaussians, and 2,048 adaptive Voronoi sites with an
8,192-site limit. Useful submission overrides are `SEED`, `ITERATIONS`,
`EVAL_INTERVAL`, `OUTPUT_DIR`, `WANDB_MODE`, and `WANDB_NAME`.

This run must not use either standalone checkpoint. Its purpose is to expose
the natural joint-training failure mode before selecting scheduling changes or
any separation regularizer.
