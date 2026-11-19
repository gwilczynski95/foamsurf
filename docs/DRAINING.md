# Draining the medium

The draining application clips the learned medium along each ray while preserving the surface Gaussians. Optional animated water surfaces add reflection, refraction, foam and spray. These are appearance edits, not a validated fluid simulation. Checkpoints are not modified.

Rebuild the native renderer with `./build.sh` after updating to this code. Draining requires the fused RGB backend, a CUDA device and `ffmpeg` on PATH. The implementation is adapted from the contributor revision `acdcd9a84bac39bc0d4b1440f1643b1f70745649`.

## Paper animation

Extract the draining package with this layout:

```text
PACKAGE/
  models/IUI3-RedSea-Undistorted/joint/{config.yaml,checkpoints/last.pt}
  data/IUI3-RedSea-Undistorted/
```

From the repository root:

```bash
python scripts/render_draining.py \
  --package PACKAGE --scene IUI3-RedSea-Undistorted --view MTN_5911 --width 960 \
  --surface-dynamics --breaking-waves --impact-strength 3 \
  --end-hold 11.916666666666666 --final-wave-scale 2 \
  --final-rise 0.1 --rise-seconds 5 --post-rise-hold 10 \
  --out outputs/draining-5911
```

Use `--view MTN_5903 --out outputs/draining-5903` for the other paper view. Add `--dry-run` to validate input paths and print all three commands without using a GPU. Use separate output directories for the two views.

The pipeline runs two 240-frame calibration passes and then a final 814-frame animation at 24 fps. Results are written under `OUT/IUI3-RedSea-Undistorted/{calibration,smooth,animation}/`. The final directory contains frames, `draining.mp4`, `settings.json` and `frame_stats.json`. Surface dynamics requires consecutive frames and cannot use `--preview`.

For a single stage, use `python edit.py drain --help`. The application modules live in `apps/draining/`; the native renderer accepts optional per-ray water intervals only for inference with fused RGB.

## Verification status

The contributor's 25 CPU tests pass after integration; the complete release test suite on CPU reports 180 passed and 57 skipped. The native renderer builds for sm_89 with CUDA 12.4 and imports successfully without a GPU. CUDA regression tests cover analytic water lengths, fully wet equivalence and preservation of Gaussians when dry. They pass on an RTX 4090, and the paper command above runs end to end on this code, writing all 814 frames, `draining.mp4`, `settings.json` and `frame_stats.json`. Late frames differ slightly from the contributor's renders (less than 0.5% of pixels); the animations are not bit-identical.
