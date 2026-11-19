# Release data and model packages

The download links in the README are placeholders until the packages are published.
The following is the required archive layout, not a claim that every archive is already available.
Unpack at the repository root, or set the corresponding environment variable to the directory
shown below. Existing local directories can be linked into this layout.

| Variable | Default | Contents |
|---|---|---|
| `FS_DATASETS` | `data/SeathruNeRF_dataset` | The four undistorted scene directories listed in the README |
| `FS_SYNTHETIC` | `data/synthetic` | `lego/` and `ficus/`, with both, object and clouds layers |
| `FS_FOGGY` | `data/foggy_garden` | `garden_easy/` and `garden_hard/` |
| `FS_MASKS` | `assets/masks` | `<undistorted-scene-name>.npz` for each SeaThru scene |
| `FS_MODELS` | `models` | The pretrained runs listed below |
| `FS_ASSETS` | `assets` | Insertion references, mesh assets and style images |
| `FS_OUTPUTS` | `outputs/paper` | Destination for reproduction outputs |

## Pretrained models

```text
models/
  base/
    iui3/joint/{config.yaml,checkpoints/last.pt}
    curasao/joint/{config.yaml,checkpoints/last.pt}
    jg/joint/{config.yaml,checkpoints/last.pt}
  synthetic/
    lego/joint/{config.yaml,checkpoints/last.pt}
    ficus/joint/{config.yaml,checkpoints/last.pt}
```

Application experiments use the final trained FoamSurf base models, not a separate set
of newly trained application baselines. These final runs and a base-model archive have
already been located in the project workspace; obtaining them from coauthors again is
not a release dependency. Public packaging and download links remain to be prepared.

Package complete run directories, including resolved configurations and available pipeline
manifests. Preserve the camera order and scene normalization of the source dataset.
The synthetic insertion benchmarks require the paper's **hybrid black-background** checkpoints,
without PPISP or water-light correction. A fresh synthetic training run can be linked as
`models/synthetic/lego` or `models/synthetic/ficus`; fresh SeaThru runs can likewise be linked
under `models/base/`. They reproduce the experiment, but are not the exact paper checkpoints.

The application runner supplies `--scene` explicitly, so saved absolute dataset paths do not
need to be edited. Edited/stylized models remain derived from the same base run. Composing
surface and water styles from unrelated reconstructions is not supported by this workflow.

## Assets

```text
assets/
  masks/<scene>.npz
  insertion/{lego,ficus}/
    insert.json
    transforms_test.json
    gt_rgba/<view>.png
    masks/<view>.png
  models/<object>/
    manifest.json
    ... mesh parts and textures referenced by the manifest ...
  styles/
    starry_night.jpg
    fire.jpg
    scream.jpg
    red_canna.jpg
```

Only the optional teaser demo needs the object directories `treasure_chest`, `antique_ceramic_vase_01`,
`dutch_ship_large_01`, `submarine_01`, `portable_searchlight`, and `barramundi`.
Keep manifest-relative mesh and texture paths intact. Benchmark masks must retain their
alpha channel. Do not rename the benchmark views or regenerate their cameras independently.

The style images must be the exact reference files for pixel-level figure comparison.
A similarly named image is not an equivalent reference. The runner does not download these
images, datasets, checkpoints or meshes automatically. CLIP may download its model weights
on first use; prepare its cache in advance for offline jobs.

Before publishing archives, include a file listing source, author, license and attribution
for each external asset, plus SHA-256 checksums and the code revision/configuration of each
checkpoint. Image and mesh redistribution rights remain to be established; see
[THIRD_PARTY.md](THIRD_PARTY.md). Do not include environment folders, compiler caches or
private dataset paths in public manifests.

## Output and validation

Training results use `TARGET/VARIANT/SCENE/seed_N/evaluation/` under `FS_OUTPUTS`;
`TARGET_summary.csv` reports final-checkpoint means and sample standard deviations.
Applications write `applications/insertion/SCENE/metrics.json`, edit PNGs under
`applications/edits/iui3/`, style runs under `applications/style/`, and the current teaser
under `applications/teaser/iui3/`. `applications/execution.json` records commands,
completion count, status and insertion metrics for that invocation. Qualitative images are
not assigned synthetic quality scores. Use a fresh `--out` for each application invocation;
the insertion benchmark refuses to overwrite a nonempty output directory.

`--check` checks selected filesystem inputs on CPU, without training, rendering, downloading
or submitting a job. It checks model/config presence, COLMAP model files, Blender camera
metadata and referenced images, masks, style images and insertion assets. It does not load
checkpoints or prove camera/checkpoint compatibility, CUDA availability, available VRAM or
numerical equivalence. These still require the corresponding runtime tests.

`all` retains its meaning of all **training** targets. Applications are explicit because they
require separate pretrained models and assets. Missing inputs fail the complete selected
plan before its first computation. Slurm jobs perform this check on the compute node; use
`--check` before `--slurm` to check the inputs locally as well.

Reproducing the paper teaser is outside the release scope. The default application
selection is `insertion,edits,style`; `--applications teaser` is an optional demo and
its assets are not required by the default selection.

## Applications figure packages

The explicit `figure` application group uses the original composed style models, bulge edit and JG haze model. See [APPLICATION_PANELS.md](APPLICATION_PANELS.md) for the separate editing/applications package layouts, `FS_EDIT_PACKAGE` and `FS_FIGURE_PACKAGE`, and local boat export instructions. The boat mesh must be excluded from public download packages.
