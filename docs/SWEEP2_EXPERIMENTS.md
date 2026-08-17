# FoamSurf sweep 2: foreground sharpness from the sweep1/05 baseline

Sweep 2 uses `sweep1/05_reset005` as its common baseline: opacity is reset
to `0.05` every 3,000 steps, medium is never frozen, images are sampled
without replacement, and every task trains on full 800x800 images for 30,000
iterations. Gaussian densification runs from step 500 until step 15,000.

The W&B project is `foamsurf_sweep2`. Local artifacts are written below
`outputs/foamsurf_sweep2`. Each task retains resolved configuration, CSV
metrics, layer diagnostics, best/last checkpoints and a summary. Population
growth beyond one million Gaussians is saved and reported as
`runaway_population` before another renderer rebuild can cause an OOM.

| ID | Label | Question |
|---:|---|---|
| 0 | `00_control_mse` | Repeat the `05_reset005` MSE control. |
| 1 | `01_loss_l1` | Does L1 improve foreground detail? |
| 2 | `02_loss_l1dssim` | Does 0.8 L1 + 0.2 DSSIM improve edges and structure? |
| 3 | `03_preserve_split` | Can opacity-preserving split sharpen the surface while regular cloning maintains population? |
| 4 | `04_preserve_clone` | Isolate opacity preservation during cloning. |
| 5 | `05_preserve_clone_split` | Preserve opacity during both topology operations. |
| 6 | `06_l1dssim_preserve_split` | Combine the most promising loss and split behavior. |
| 7 | `07_reset0075_mse` | Test the reset midpoint between 0.05 and 0.10. |
| 8 | `08_reset0075_l1dssim` | Combine the midpoint reset and sharpness loss. |
| 9 | `09_reset010_l1dssim` | Combine the sweep1/07 reset with the sharpness loss. |

Submit all tasks with:

```bash
sbatch scripts/sweep2_30k.sbatch
```

Limit concurrency if necessary:

```bash
sbatch --array=0-9%4 scripts/sweep2_30k.sbatch
```

Use another seed without changing code:

```bash
SEED=1337 sbatch scripts/sweep2_30k.sbatch
```

Set `WANDB_MODE=offline` to retain W&B runs locally without network access.
