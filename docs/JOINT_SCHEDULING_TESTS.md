# Clean-joint rerun and scheduling tests

The exact baseline rerun keeps seed 0, K=64, the full-image L1+0.2 DSSIM
objective, initialization, learning rates, adaptation settings, and 30k
iterations from `configs/baseline_clean_joint.yaml`. Its only behavioral change
is the transactional `replace_population` repair.

Submit it with:

```bash
sbatch scripts/rerun_clean_joint_fixed.sbatch
```

The scheduling array contains four single-variable tests, all still at K=64:

| Task | Label | Only scheduling change |
|---:|---|---|
| 0 | `voronoi_warmup_2k` | Do not step or adapt Gaussians through iteration 2000 |
| 1 | `voronoi_offset_100` | Adapt Voronoi at 1100, 1600, ... instead of coincident 500-step events |
| 2 | `gaussian_adapt_until_10k` | End Gaussian population adaptation at 10k instead of 15k |
| 3 | `freeze_voronoi_at_8k` | Stop Voronoi statistics and population replacement after reaching 8192 sites |

Submit all four with:

```bash
sbatch scripts/sweep_joint_scheduling.sbatch
```

Run one task alone with, for example:

```bash
sbatch --array=1 scripts/sweep_joint_scheduling.sbatch
```

These are diagnostic training runs, not a combinatorial sweep. No alternative
hit-buffer K is built or selected.
