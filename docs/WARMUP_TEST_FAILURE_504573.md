# Regression gate after job 504573

Job 504573 finished with 50 passed / 3 failed. The CUDA compilation defect
and missing reference-package search path from earlier jobs were resolved.

## Changes limited to tests

- RadFoam comparison now executes in a fresh Python interpreter, with RadFoam
  imported before FoamSurf or any render kernels. Its import configures the CUDA
  malloc heap; importing it after earlier GPU tests raised invalid argument.
  Exit errors, crashes and timeouts fail the parent test; nothing is skipped.
  Forward and density-backward tolerances remain unchanged.
- Alpha independence uses the real warmup training step on one fixed model and
  topology. Only optimizer.step is intercepted to observe gradients without an
  update. RNG and camera sampling are restored. Runs alpha=0, alpha=0 (numerical
  control), alpha=1 compare targets/rays exactly and losses/all medium gradients
  at the existing tolerances. Parameters must remain exactly unchanged.
  This deliberately does not compare Adam updates across rebuilt triangulations.
- A separate iteration-zero checkpoint test checks physical field, camera/ray/RGB
  sampling, quantile RNG and optimizer epsilon. The existing multistep resume,
  adaptation and canonical handoff test is retained unchanged.
- CUDA_LAUNCH_BLOCKING=1 in the test sbatch localizes asynchronous CUDA failures.
  It is not added to experiment scripts.

The first-step parameter discrepancy in the old alpha test is not conclusively
diagnosed: small geometric-gradient differences amplified by Adam remain a
hypothesis, not a proven explanation. No production optimizer settings or
tolerances have been changed to conceal it.

## Run

Environment: /shared/results/gmwilczn/conda/envs/foamsurf.
GPU: RTX4090 allocated by Slurm. Do not execute GPU tests on the current node.

```bash
cd /shared/results/gmwilczn/foamsurf/foamsurf
sbatch scripts/test_warmup_radfoam_like.sbatch
```

Main log: slurm_logs/warmup_test_JOB_ID.out.
Reference subprocess log: outputs/warmup_tests/JOB_ID/radfoam_reference.log.
Certificate and JUnit: outputs/warmup_tests/JOB_ID/passed.json and passed.xml.
Only a passed certificate for the current source fingerprint permits the sweep.
No job is submitted automatically. Static validation does not certify GPU tests.
