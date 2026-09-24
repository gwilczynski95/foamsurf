# RGB medium validation — 2026-09-24

Branch `feature/voronoi-rgb-medium`, interactive allocation 512889, c21,
assigned physical GPU 2, NVIDIA RTX4090, PyTorch 2.6.0+cu124, conda foamsurf.
Only the assigned GPU was used. No full experiment sweep was submitted.

Final gate: **122 passed, 0 failures, 0 errors, 0 skipped**.

| Group | Passed |
|---|---:|
| RGB math/model/config | 10 |
| RGB warmup/joint/resume | 2 |
| RGB production-resolution smoke | 1 |
| Legacy native renderer/diagnostics | 6 |
| Shared contracts | 81 |
| Joint-adaptation contracts | 10 |
| Previous COLMAP GPU tests | 5 |
| Previous production smoke | 2 |
| Previous COLMAP pipeline | 1 |
| Previous checkpoint restore | 1 |
| Previous million-ray growth | 1 |
| Previous end-to-end pipeline | 2 |

Certificate: `outputs/rgb_medium_tests/local_512889/passed.json`.
Combined JUnit: same directory, `passed.xml`.
Log: `outputs/rgb_medium_gate_512889.log`.
Source fingerprint:
`8b93ac2297230225fc415926b71501ea6aa2050ceecf5732f7880b290c1dd56d`.
Verified by `rgb_medium_gate.require` after completion.

The early integration checks found and fixed a per-cell mask applied to flattened
six-channel coefficients, and duplicate proxy metric prefixes. The final gate
uses the corrected source. These development failures are not GPU OOMs.

Full-resolution smoke covers approximately 1.27M pixels, starting with 25k cells
and all 21,907 COLMAP Gaussians. It does NOT prove memory bounds for every later
500k-cell/500k-Gaussian configuration or quality after 30k iterations. One earlier
successful production smoke recorded about 5.27 GiB PyTorch peak allocated and
0.30s for its last joint step; this excludes non-PyTorch native allocations and
is not a matched speed comparison with the old backend.

Current limitations are explicit in `VORONOI_RGB_MEDIUM.md`: six CUDA traversals
rather than one fused kernel; scalar proxy for population diagnostics and mean
attenuation for depth-quantile regularization; no SH; scalar medium losses blocked;
no additional full-view off-cycle adaptation evaluation. The model is ready for
the paired experiment, not claimed to improve PSNR or separation yet.

Launch using the already verified local certificate (no need to repeat tests
unless code/config changes):

```bash
cd /shared/results/gmwilczn/foamsurf/foamsurf
export RGB_MEDIUM_TEST_CERTIFICATE="$PWD/outputs/rgb_medium_tests/local_512889/passed.json"
sbatch --export=ALL scripts/train_rgb_medium.sbatch
```

Twelve variants, maximum seven concurrent. W&B `foamsurf_iui3`, group
`rgb-medium-ARRAY_JOB_ID`. Outputs `outputs/rgb_medium/ARRAY_JOB_ID/run_INDEX`.
The mathematical specification, parameter mapping and test/retest commands are
in `VORONOI_RGB_MEDIUM.md`.
