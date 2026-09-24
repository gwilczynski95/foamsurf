# Stage A — validation and ready-to-submit commands

Validated on RTX4090 in the foamsurf environment: **95 tests passed, zero
failures/errors/skips**. CPU contracts, lineage, all 19 configurations,
new warmup/joint/resume path, exact rendered transfer with RGB >1, 25k and
100k full-resolution smoke, and existing COLMAP/LEGO pipeline regressions.
Full-resolution smoke checks zero truncated evaluation rays. These are smoke
and regression tests, not a completed 30k experiment or a quality guarantee.

Certificate:
`outputs/colmap_joint_tests/local_stage_a_verified/passed.json`

Log:
`outputs/colmap_joint_gate_verified.log`

The certificate covers the current source fingerprint and is checked by the
training sbatch. Earlier development-test outputs may contain failures; use
the **local_stage_a_verified** result. Source/config changes invalidate it.

## Submit the first experiment family (19 runs, max 3 concurrent)

```bash
cd /shared/results/gmwilczn/foamsurf/foamsurf
export COLMAP_JOINT_TEST_CERTIFICATE="$PWD/outputs/colmap_joint_tests/local_stage_a_verified/passed.json"
sbatch scripts/train_colmap_joint.sbatch
```

For one spatial pilot instead: `sbatch --array=1 scripts/train_colmap_joint.sbatch`.
Index 1 is 25k sites, no COLMAP-seeded medium sites, moderate extent.
Index 0 is the RadFoam-like initialization control. Gaussians always start
at all original COLMAP points. Full protocol: `COLMAP_JOINT_STAGE_A.md`.

No sweep has been submitted by the assistant. No commits or pushes were made.
W&B is online in `foamsurf_iui3`; node credentials must already be configured.

## Revalidate after source changes

```bash
test_job=$(sbatch --parsable scripts/test_colmap_joint.sbatch)
test_job=${test_job%%;*}
export COLMAP_JOINT_TEST_CERTIFICATE="$PWD/outputs/colmap_joint_tests/$test_job/passed.json"
sbatch --dependency=afterok:$test_job --export=ALL scripts/train_colmap_joint.sbatch
```

## Storage caution

At handoff, the shared filesystem had approximately 791 GB free and was 98%
full. Check free space before launching the complete array. Checkpoints every
500 steps plus all-view component renders can consume substantial storage;
no automatic deletion of old experiments is performed.
