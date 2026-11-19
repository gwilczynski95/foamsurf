# Draining the medium

The draining application clips the learned medium along each ray while preserving the surface Gaussians. Optional animated water surfaces add reflection, refraction, foam and spray. These are appearance edits, not a validated fluid simulation. Checkpoints are not modified.

Rebuild the native renderer with `./build.sh` after updating to this code. Draining requires the fused RGB backend, a CUDA device and `ffmpeg` on PATH.

## Example animation

Put a trained IUI3 run (`train.py`) and its data in this layout:

```text
PACKAGE/
  models/IUI3-RedSea-Undistorted/joint/{config.yaml,checkpoints/last.pt}
  data/IUI3-RedSea-Undistorted/
```

From the repository root:

```bash
python -m examples.draining.render_draining \
  --package PACKAGE --scene IUI3-RedSea-Undistorted --view MTN_5911 --width 960 \
  --surface-dynamics --breaking-waves --impact-strength 3 \
  --end-hold 11.916666666666666 --final-wave-scale 2 \
  --final-rise 0.1 --rise-seconds 5 --post-rise-hold 10 \
  --out outputs/draining-5911
```

Any other view works, e.g. `--view MTN_5903 --out outputs/draining-5903`. Add `--dry-run` to validate input paths and print all three commands without using a GPU. Use separate output directories for the two views.

The pipeline runs two 240-frame calibration passes and then a final 814-frame animation at 24 fps. Results are written under `OUT/IUI3-RedSea-Undistorted/{calibration,smooth,animation}/`. The final directory contains frames, `draining.mp4`, `settings.json` and `frame_stats.json`. Surface dynamics requires consecutive frames and cannot use `--preview`.

For a single stage, use `python -m examples.editing.edit drain --help`. The application modules live in `examples/draining/`; the native renderer accepts optional per-ray water intervals only for inference with fused RGB.
