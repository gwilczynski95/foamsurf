# Run 3, step 30000: first ray heatmaps

Final output directory:
`outputs/ray_heatmaps/512923_run3_30k_full/`.
Start with `overview.png`, then `MTN_5903_panel.png` and
`MTN_5911_panel.png`. There are four per-view panels, individual maps, raw NPZ
arrays, ray profiles and probe-location images. See README.md for definitions.

## Visual reading

The intrinsic b*B source concentration alone is broad and high across much of
the image, including parts of the water. It is NOT a usable solid mask as-is.

The combined score is much more localized to seabed/coral regions, with open
water mostly dark in all four views. This is promising as a diagnostic, but
partly built into the score: it is multiplied by Gaussian surface confidence.
Therefore the pictures do NOT independently prove correct solid detection or
that every highlighted medium cell should be removed. A future ablation should
compare this score against Gaussian confidence alone, and test multi-view
consistency and true local optical excess.

Black pixels can also reflect missing/uncertain Gaussian support. This method
cannot find all incorrectly represented solids where no useful Gaussians exist.
The last-column display saturates at 0.25 consistently across all views; raw
scores remain [0,1] and are available in NPZ and the individual heatmaps.

## Validation

Four CPU tests pass. No training, checkpoint or existing sweep files changed.
The original RGB certificate was checked after implementation and remains valid.
The pending smoothing gate only adds `experiments/rgb_smooth/` to the original
fingerprint, so this sibling directory does not affect its certificate.

On the every-eighth-pixel grid, changing 1024 to 2048 depth samples changes the
combined score by the following absolute amounts:

| Camera | Mean | Maximum |
|---|---:|---:|
| MTN_5894 | 0.00282 | 0.04864 |
| MTN_5903 | 0.00127 | 0.02530 |
| MTN_5911 | 0.00302 | 0.04954 |
| MTN_5928 | 0.00237 | 0.09033 |

This is adequate for an exploratory visualization, not a pixelwise guarantee.
Spatial sampling is 173x115, not full resolution. Range is 49.3246 scene units,
not the renderer's full far=1000 interval; neighborhood half-width is 0.9865.
No GPU-native equality claim is made: medium integration is sampled, Gaussian
events are binned, and Gaussian support is circular rather than octagonal.

The preliminary directory `512923_run3_30k` used only 12.33 units and omitted
important distant surfaces. It is retained as exploratory output, but all final
links and conclusions refer to the `_full` directory. Here `_full` means the
expanded diagnostic range, NOT full image resolution or full renderer range.
