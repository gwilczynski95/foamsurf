# Reproducing the applications figure

The four paper panels use the original derived checkpoints, not fresh style training. Shot configurations are adapted from contributor revision `be9738c` and preserve the supplied cameras, materials, lights and sampling settings. The release renderer includes sun caustics, light shafts, lamp beam controls and world-size object placement used by these panels.

## Inputs

Extract the editing and applications packages, then set:

```bash
export FS_EDIT_PACKAGE=/path/to/foamsurf_paper_editing_2026-11-15
export FS_FIGURE_PACKAGE=/path/to/foamsurf_fig_applications_2026-11-18
export FS_ASSETS=/path/to/assets
```

The editing package is distributed in multiple ZIP parts. Extract them into the same parent directory: part06 contains `iui3_both` and `curasao_both`, part07 contains `bulge`, and part01 contains the scene data. A ZIP downloaded from a shared folder may itself contain these ZIP parts.

| Panel | Model relative to its package | Data | Camera |
| --- | --- | --- | --- |
| `s1b_iui3_style` | editing: `models/style/iui3_both` | editing: `data/IUI3-RedSea-Undistorted` | MTN_5903 |
| `s2c_iui3_edit` | editing: `models/edits/bulge` | editing: `data/IUI3-RedSea-Undistorted` | MTN_5911 |
| `s3b_curacao_style` | editing: `models/style/curasao_both` | editing: `data/Curasao-Undistorted` | MTN_1288 |
| `s4k_jg_haze` | applications: `models/jg_haze/model` | applications: `models/jg_haze/data/JapaneseGradens-RedSea-Undistorted` | MTN_1098 |

Keep the supplied JG dataset with its haze model: its COLMAP export differs from the editing package. Place the Fiat manifest and its 18 PLY parts under `$FS_ASSETS/models/fiat_bicycle/`. In the contributor package these files are under `assets/falcon/source/fiat_bicycle/`.

The boat mesh is not included in the public repository or public asset package. Obtain model #5130630, [2 Submarines by CreaConcept on CGTrader](https://www.cgtrader.com/free-3d-print-models/miniatures/vehicles/submarine-a14e7e0a-717e-4d97-a271-05a16f6d76ef), and export the `submarine-01.stl` file locally:

```bash
python scripts/export_stl.py submarine-01.stl "$FS_ASSETS/models/submarine_01" \
  --hull_size 0.02 --stripes '[[-22,-15],[-6,1],[10,17]]'
```

Expected source STL SHA256: `3453298fb5a2d36fe75e76a3e296e028455b04e67e470653a0ff2d7486065b9b`.

## Render and compare

From the repository root, using the environment installed with application dependencies:

```bash
bash scripts/reproduce.sh applications --applications figure --check
bash scripts/reproduce.sh applications --applications figure --out outputs/paper
python scripts/compare_application_panels.py \
  --renders outputs/paper/applications/figure \
  --reference "$FS_FIGURE_PACKAGE/reference" \
  --out outputs/paper/applications/figure/comparison.json
```

`--check` validates all selected input files before GPU work. `--dry-run` prints commands without requiring downloaded data. `--scenes IUI3` selects the two IUI3 panels. `--slurm` uses the existing scheduler wrapper. The `figure` group is explicit and does not retrain the models.

Each panel produces `still.png`, `still_noshadow.png` and `still_reference.png`. The reference pass shows the trained scene without inserted objects or sun. Comparison requires the native image resolution and passes for identical images or PSNR greater than 50 dB; reports also contain MAE and maximum absolute error in 8-bit pixel units. A failure is reported with a nonzero exit status. It must be investigated rather than hidden by relaxing the threshold.

For a fast composition check, append `--preview` to an individual printed `insert.py --still` command. This halves the resolution and reduces sampling; preview images are not suitable for the reference comparison.

## Verification status

Input preflight passes on the received packages. CPU tests cover sun refraction, energy conservation in caustic filtering, world-size placement, scene selection and comparison failure handling. Full GPU render/reference verification is pending; integration alone does not establish reproduction of the paper images.
