#!/usr/bin/env bash
# Submit only when ready: array runs ONLY after the RTX4090 test gate succeeds.
set -Eeuo pipefail
ROOT=/shared/results/gmwilczn/foamsurf/foamsurf
cd "$ROOT"
mkdir -p slurm_logs
surface_test_job=$(sbatch --parsable experiments/rgb_surface/test.sbatch)
surface_test_job=${surface_test_job%%;*}
surface_certificate="$ROOT/outputs/rgb_surface_tests/$surface_test_job/passed.json"
surface_array_job=$(sbatch --parsable --dependency="afterok:$surface_test_job" --kill-on-invalid-dep=yes \
  --export="ALL,RGB_SURFACE_TEST_CERTIFICATE=$surface_certificate" experiments/rgb_surface/train.sbatch)
printf 'RTX4090 tests: %s\nDependent sweep: %s\nCertificate: %s\n' "$surface_test_job" "$surface_array_job" "$surface_certificate"
