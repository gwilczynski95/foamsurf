# FoamSurf sweep 1: opacity, stability and Gaussian population

All tasks train for 30,000 iterations on full 800x800 images, sample cameras
without replacement, densify from iteration 500 until 15,000, and use seed 42.
There is no optimization population cap. A task that exceeds 1,000,000
Gaussians writes `last.pt` and `summary.json` with status
`runaway_population`, then exits without attempting another OptiX rebuild.

The W&B project is `foamsurf_sweep1`. Every output contains the resolved
`config.yaml`, CSV metrics, best/last checkpoints and layer diagnostics.

| ID | Label | Question |
|---:|---|---|
| 0 | `00_noreset` | Reproduce unrestricted no-reset runaway as a control. |
| 1 | `01_freeze_only200` | Isolate medium freezing without an opacity reset. |
| 2 | `02_reset001` | Reference reset to 0.01 without medium freezing. |
| 3 | `03_reset001_f200` | Reset 0.01 plus 200 frozen-medium steps. |
| 4 | `04_reset001_f500` | Reset 0.01 plus 500 frozen-medium steps. |
| 5 | `05_reset005` | Softer reset to 0.05 without freezing. |
| 6 | `06_reset005_f200` | Reset 0.05 plus 200 frozen-medium steps. |
| 7 | `07_reset010` | Soft reset to 0.10 without freezing. |
| 8 | `08_reset010_f200` | Reset 0.10 plus 200 frozen-medium steps. |
| 9 | `09_reset010_f500` | Reset 0.10 plus 500 frozen-medium steps. |
| 10 | `10_reset010_f200_keepadam` | Clamp opacity but retain its Adam moments. |
| 11 | `11_official_white_reset` | 3DGS white-background reset at 500 and every 3000 steps. |
| 12 | `12_loss_l1` | Isolate L1 instead of MSE. |
| 13 | `13_loss_l1dssim` | Test the 3DGS/2DGS 0.8 L1 + 0.2 DSSIM objective. |
| 14 | `14_opacity_lr005` | Isolate the 2DGS opacity LR of 0.05. |
| 15 | `15_opacity_cull005` | Isolate the 2DGS opacity cull threshold of 0.05. |
| 16 | `16_grad0005` | Raise the densification gradient threshold to 0.0005. |
| 17 | `17_grad0010` | Raise the densification gradient threshold to 0.0010. |
| 18 | `18_alpha_clone` | Preserve combined opacity across retained-parent cloning. |
| 19 | `19_alpha_clone_split` | Preserve combined opacity across both clone and split. |

Run the complete array with:

```bash
sbatch scripts/sweep1_30k.sbatch
```

Limit concurrent jobs by overriding the array at submission time, for example:

```bash
sbatch --array=0-19%4 scripts/sweep1_30k.sbatch
```
