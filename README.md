# FoamSurf

FoamSurf represents underwater scenes with two components rendered by a shared CUDA/OptiX ray tracer:

- **2D Gaussian surfels** represent surfaces such as the seabed and coral, with spherical-harmonic appearance.
- **Voronoi cells** represent water or fog, with per-cell attenuation and backscatter.

The method includes a PPISP camera model, OceanSplat losses, and surface–medium priors.
The reported SeaThru-NeRF mean over test views and three seeds is **30.75 dB**
(IUI3 31.78, Curacao 34.33, Japanese Gardens 24.33, Panama 32.57).

## Installation

Linux, an NVIDIA GPU, the installed NVIDIA driver, conda, Git and a C++ compiler are required.
The tested stack is Python 3.11, CUDA 12.4, PyTorch 2.6 and OptiX 8.0; dependency versions
are pinned in `environment.yml` and `requirements.txt`.

Download and unpack **NVIDIA OptiX SDK 8.0** manually from NVIDIA (account and license
acceptance required), then run:

```bash
git submodule update --init --recursive
export OPTIX_ROOT=/path/to/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64
scripts/install.sh --with-apps
conda activate "$PWD/.conda-env"
export PYTHONPATH="$PWD/renderer:$PWD/src:$PWD"
export TORCH_EXTENSIONS_DIR="$PWD/.torch_extensions"
```

`--with-apps` adds Warp for insertion and CLIP for style transfer. Omit it for training
and evaluation only. `--prefix DIR` chooses a different new environment; the installer
refuses to overwrite an existing directory. It builds the topology, CUDA renderer and
both OptiX hit buffers (K=64 for training, K=32 for paper FPS), runs the test suite and
renders a small synthetic scene against an independent reference. Dataset-dependent
tests are skipped when their datasets are absent.

For an existing compatible environment, activate it and run `./build.sh`. Builds stay
inside this checkout. `CONDA_ENV_PATH` and `FOAMSURF_PYTHON` can select the build environment
explicitly. The GPU architecture is detected automatically; override `TORCH_CUDA_ARCH_LIST`
if needed (8.9 for RTX 4090, 8.6 for RTX 3090). With multiple PyTorch architecture targets,
also set one OptiX target, e.g. `CUDA_ARCH=89`.

`FOAMSURF_DRIVER_LIBRARY_DIRECTORY` selects the directory containing the installed
`libcuda.so.1` and `libnvoptix.so.1` (default `/usr/lib/x86_64-linux-gnu`). At runtime, this
directory must precede CUDA stub libraries in `LD_LIBRARY_PATH`. PPISP and SH color kernels
compile on first use; `TORCH_EXTENSIONS_DIR` controls their cache. OpenCV is pinned to
4.11.0.86 to retain the tested NumPy 1.x environment.

## Data

### SeaThru-NeRF

Download the [original SeaThru-NeRF dataset](https://drive.google.com/uc?export=download&id=1RzojBFvBWjUUhuJb95xJPSNP3nJwZWaT)
from the [authors' project page](https://sea-thru-nerf.github.io/) and unpack it locally.
Use the white-balanced images and undistort them with **COLMAP** before training, following
[WaterSplatting's data preparation](https://github.com/water-splatting/water-splatting#data-preprocessing).
For example, for IUI3:

```bash
export RAW_SEATHRU=/path/to/SeathruNeRF_dataset
export FS_DATASETS="$PWD/data/SeathruNeRF_dataset"
mkdir -p "$FS_DATASETS/IUI3-RedSea-Undistorted"
colmap image_undistorter \
  --image_path "$RAW_SEATHRU/IUI3-RedSea/images_wb" \
  --input_path "$RAW_SEATHRU/IUI3-RedSea/colmap/sparse/0" \
  --output_path "$FS_DATASETS/IUI3-RedSea-Undistorted" \
  --output_type COLMAP
```

Use the actual white-balanced image directory in the downloaded archive (`images_wb`
or `Images_wb`, depending on its layout). Repeat for all four scenes, using the matching
COLMAP reconstruction and the following **output** directory names:
`IUI3-RedSea-Undistorted`, `Curasao-Undistorted`,
`JapaneseGradens-RedSea-Undistorted`, and `Panama-Undisorted`.
Preserve these spellings: the reproduction runner and mask filenames use them.
COLMAP writes undistorted images to `images/` and a camera model under `sparse/`;
the loader accepts both `sparse/` and `sparse/0/`, with binary or text model files.
See the [COLMAP command-line documentation](https://colmap.github.io/cli.html).
Every eighth image is held out for testing. Exact paper comparisons require the same
camera reconstruction and image set as the paper runs; rerunning SfM can change them.

### Synthetic scenes

> **TODO (link):** download the synthetic dataset from <LINK> and unpack it into data/synthetic/.

Set `FS_SYNTHETIC` if it is stored elsewhere. The required layout is:

```text
data/synthetic/
  lego/{lego_both,lego_lego,lego_clouds}/
  ficus/{ficus_both,ficus_object,ficus_clouds}/
```

Each layer contains Blender camera JSON files and RGBA images. The combined layers
need `transforms_train.json` and `transforms_test.json`; isolated object/cloud layers
need test cameras and images for evaluation. Keep the camera poses and names aligned
across layers. Do not rename `ficus_object` to `ficus_ficus`.

### Foggy Garden, masks and models

Foggy Garden contains `garden_easy/` and `garden_hard/`. Generate it from clear Garden
images, camera-z depths and a COLMAP model with `scripts/make_foggy_garden.py --help`,
or use the prepared package. Setting `FS_FOG_DEPTHS`, `FS_GARDEN_CLEAR` and
`FS_GARDEN_COLMAP` makes the `garden` target generate it before training.

Separation evaluation needs one `<scene>.npz` mask file per SeaThru scene under
`assets/masks/` (or `FS_MASKS`). Masks are not used in training;
`scripts/sam_water_mask.py` can regenerate them with Segment Anything, but the provided
paper masks are required for an exact comparison.

> **TODO (link):** download the prepared Foggy Garden dataset and evaluation masks from <LINK>.

> **TODO (link):** download the trained paper models from <LINK> and unpack them into models/.

See [the package layout and validation instructions](docs/RELEASE_DATA.md) for models,
assets, environment variables and output paths.

## Training

```bash
python train.py --scene /data/SeathruNeRF_dataset/Curasao-Undistorted --out outputs/curacao
python train.py --scene /data/foggy_garden/garden_easy --out outputs/garden_easy --config configs/garden_fog.yaml
```

- `--seed N` selects the random seed (default: 42).
- `--wandb PROJECT` enables Weights & Biases logging (disabled by default).
- `--set key.path=value` overrides a configuration field.
- `--resume` resumes an interrupted run; `--dry-run` prints the resolved configuration without training.

SeaThru training uses 20,000 iterations, with test metrics every 500 iterations and
rendered images every 5,000 iterations and at the end. The reported result always comes
from the final iteration. On an RTX 4090, the original runs took approximately 25–35 minutes
of training; evaluation and image export add to the total runtime.
To submit all four scenes to Slurm, activate the environment, set the dataset variables
below, and run `scripts/reproduce.sh seathru --slurm`.

Run directory:

| Path | Contents |
|---|---|
| `joint/checkpoints/last.pt` | Model, optimizers, camera response and water light |
| `joint/metrics/train.csv`, `eval.csv` | Training losses and statistics; test metrics at each evaluation |
| `joint/evaluation/iteration_*/` | Test renders: full image, Gaussians, medium, opacity and error |
| `joint/summary.json` | Completion status, runtime and model size |
| `joint/config.yaml`, `pipeline_config.json` | Resolved stage and pipeline configurations |

## Rendering

```bash
python render.py --run outputs/curacao  # Test views, all rendering modes
python render.py --run outputs/curacao --views train --modes full,gaussians
python render.py --run outputs/curacao --export-cameras cameras.json
python render.py --run outputs/curacao --poses custom_poses.json --modes gaussians,voronoi --npy
python render.py --run models/curacao --scene /data/Curasao-Undistorted
python render.py --run outputs/curacao --modes full --metrics --fps 32
```

Rendering needs the original dataset's cameras and images for camera-response and
water-light handling. If the dataset has moved, supply its new location with `--scene`.
The final checkpoint is loaded. `--metrics` reports PSNR, SSIM and LPIPS-VGG from float
images; `--fps 32` measures rendering speed with the paper's K=32 hit buffer.

Modes (`--modes`): PNG files are written to `OUT/<mode>/<view>.png`; `--npy` also exports float32 arrays.

| Mode | Output |
|---|---|
| `full` | Full evaluation image: Gaussians and medium, water light and camera response |
| `raw` | Full image before camera response |
| `gaussians` | Gaussians alone on a black background |
| `alpha` | Accumulated Gaussian opacity |
| `voronoi` | Medium alone, without Gaussians |
| `water` | Water contribution before the surface in the full render |
| `depth` | Surface depth: normalized in PNG, metric in NPY |

Pose files contain a list of cameras in the dataset's world coordinate system, using the
same fields as `--export-cameras`:
`{"name", "origin", "right", "down", "forward", "width", "height", "fx", "fy", "cx", "cy"}`.

## Configurations

- `configs/seathru.yaml`: SeaThru-NeRF paper configuration.
- `configs/garden_fog.yaml`: foggy Garden configuration, extending the SeaThru configuration.
- `configs/synthetic_hybrid.yaml` and `configs/synthetic_voronoi.yaml`: synthetic pipelines.
- `configs/ablations/*.yaml`: the seven paper ablations.
- `configs/base.yaml`: configuration defaults.

`tests/test_configs.py` compares resolved configurations with saved paper-run manifests
in `tests/data/`. Update those reference files only when deliberately changing the method.

## Code map

| Path | Responsibility |
|---|---|
| `renderer/` | CUDA/OptiX ray tracing, compositing and gradients; see [renderer/README.md](renderer/README.md) |
| `renderer/foamsurf/cuda.py` | `NativeHybridRenderer`: hybrid, Gaussian-only and medium-only rendering |
| `src/`, `torch_bindings/` | CUDA Delaunay triangulation |
| `foamsurf_model/gaussians.py` | Surfel parameters, SH appearance and Mip filtering |
| `foamsurf_model/medium.py`, `topology.py` | Medium coefficients, cells and adjacency |
| `foamsurf_model/gaussian_adaptation.py` | Gaussian densification, pruning and opacity resets |
| `foamsurf_data/` | COLMAP/Blender loading and ray generation |
| `training/trainer.py` | Training loop, image loss, camera model and adaptation |
| `training/colmap_joint_trainer.py` | Surface–medium priors |
| `training/colmap_joint.py`, `warmup_trainer.py` | Initialization and warmup |
| `training/water_*.py`, `camera_isp.py`, `oceansplat.py` | Water light, camera response and OceanSplat losses |
| `evaluation/` | Shared rendering and evaluation metrics |
| `train.py`, `render.py`, `evaluate.py` | Training, rendering and evaluation entry points |
| `insert.py`, `edit.py`, `stylize.py`, `apps/` | Insertion, scene editing and style transfer |

## Reproducing the paper

`scripts/reproduce.sh` runs training and final-checkpoint evaluation sequentially on one GPU, and resumes interrupted runs. Set the dataset and mask directories before running it:

```bash
export FS_DATASETS=/path/to/SeathruNeRF_dataset
export FS_SYNTHETIC=/path/to/synthetic
export FS_FOGGY=/path/to/foggy_garden
export FS_MASKS=/path/to/masks
scripts/reproduce.sh seathru --scenes IUI3 --seeds 42
scripts/reproduce.sh synthetic --scenes lego --background black --seeds 42
```

The commands use the complete paper schedules. `--dry-run` prints the commands, `--slurm` submits the same sequential workflow through `scripts/slurm/reproduce.sbatch`, `--out DIR` selects the output root, and `--pipeline voronoi|hybrid|all` selects the synthetic pipeline (default: hybrid). `--check` validates selected input files without GPU work or job submission; actual runs check their inputs before computation. The other training targets are `ablation`, `garden`, and `all`. `scripts/summarize.py TARGET --root DIR` aggregates the evaluation JSON files using the sample standard deviation across seeds; scene averages are computed per seed first.

The training targets cover the following paper results. Commands report FoamSurf results;
baseline implementations and results are not recomputed by this repository.

| Paper result | Command or output | Reference / expected output |
|---|---|---|
| `tab:seathru`, `tab:suppl_per_scene`, `tab:suppl_separation` | `scripts/reproduce.sh seathru` | PSNR/SSIM/LPIPS, FPS and separation; SeaThru mean PSNR approximately 30.75 dB |
| `fig:nvs`, `fig:suppl_reconstruction` | SeaThru runs, then `render.py --run RUN --modes full` | Test-view RGB images |
| `fig:decomposition`, `fig:suppl_decomposition` | `render.py --run RUN --modes full,gaussians,water,alpha` | Surface/medium decomposition and surface opacity |
| `tab:ablation` | `scripts/reproduce.sh ablation` | Full model and seven ablations, four scenes, seeds 42 and 43 |
| `tab:garden`, `fig:garden`, `fig:suppl_garden` | `scripts/reproduce.sh garden` | Foggy-image metrics and clear-image restoration metrics for easy/hard fog |
| `tab:synthetic`, `tab:suppl_synthetic`, `fig:synthetic`, `fig:suppl_synthetic` | `scripts/reproduce.sh synthetic --pipeline all` | Hybrid and Voronoi-first results, isolated layers and medium transmittance |
| `tab:insertion`, `tab:suppl_insertion`, `fig:suppl_insertion_lego`, `fig:suppl_insertion_ficus` | `scripts/reproduce.sh applications --applications insertion` | FoamSurf insertion renders and Blender-reference metrics |
| `fig:suppl_edit_surface`, `fig:suppl_edit_surface_layers`, `fig:suppl_edit_water`, `fig:suppl_edit_water_layers` | `scripts/reproduce.sh applications --applications edits` | Edited full images and separate layers |
| `fig:suppl_style` | `scripts/reproduce.sh applications --applications style` | Stylized final model and preview images |
| `tab:supp_weights`, `tab:supp_learning_rates` | `train.py --config configs/seathru.yaml --dry-run` | Resolved loss weights, learning rates and schedules |
| `fig:teaser`, `fig:applications`, `fig:draining` | `scripts/reproduce.sh applications --applications teaser` renders the available insertion shot only | Complete composite figure presets and draining are pending |
| `fig:overview` | Method schematic | Explanatory diagram, not an evaluation output |

Release training controls on RTX 4090 (seed 42, final checkpoint):

| Control | Release PSNR | Original run PSNR |
|---|---:|---:|
| IUI3 | 31.7766 dB | 31.8010 dB |
| Lego, hybrid, black background | 34.2084 dB | 34.2231 dB |
| Garden, easy fog | 34.0741 dB | 34.0829 dB |

These are individual control runs, not a replacement for the multi-seed paper tables.
Full Ficus retraining remains deferred; its dataset loading and resolved training command
have been checked on CPU. Style presets are implemented but have not been rerun as a
complete GPU workflow on this release.

The control runs use IUI3 seed 42 (31.80 ± 0.15 dB) and hybrid Lego black seed 42
(34.22 ± 0.30 dB). These are tolerances for fresh training; loading an existing paper
checkpoint is tested separately at a tighter 0.001 dB PSNR tolerance. Synthetic black
backgrounds use seeds 42/43/44; white backgrounds use seed 42. The hybrid synthetic pipeline
uses a 5,000-step warmup and 20,000 joint steps; the Voronoi-first pipeline uses
5,000 warmup steps followed by 30,000 joint steps. Runtime depends on scene size and
evaluation frequency; final release timing measurements are still being collected.

The application workflow uses pretrained models and separate assets:

```bash
export FS_MODELS=/path/to/models
export FS_ASSETS=/path/to/assets
scripts/reproduce.sh applications --dry-run
scripts/reproduce.sh applications --applications insertion --scenes lego --check
scripts/reproduce.sh applications --applications insertion --scenes lego
scripts/reproduce.sh applications --applications edits,style --scenes IUI3
```

Available application groups are `insertion` (Lego/Ficus), `edits` (nine IUI3 edits,
including Curaçao water transplant), `style` (IUI3/Curaçao/Japanese Gardens), and `teaser`
(the existing IUI3 insertion shot). Without `--applications`, all four are selected.
Style runs use the paper's 5,000 steps and seed 0, then compose the surface/water models
and render them. Original style runs took approximately 18–32 minutes each on RTX 3090;
this is not a new release benchmark. Application inputs and outputs are specified in
[RELEASE_DATA.md](docs/RELEASE_DATA.md). Use a fresh output directory for each application
invocation. `all` runs training targets only; it does not implicitly run applications.

The draining/bounded-medium-removal code for `fig:draining` and the complete
`fig:applications`/`fig:teaser` presets are still pending. The `teaser` group renders
only the available insertion shot, not every panel of the paper teaser.

## Application entry points

- `insert.py --run RUN --shot apps/raytrace/shots/ruins/teaser.json --assets assets --out renders/teaser` renders inserted objects.
- `insert.py --benchmark --run RUN --gt ASSETS/insertion/lego --out renders/lego_insertion` evaluates FoamSurf against the packaged Blender references.
- `edit.py --run RUN --view VIEW --edits bulge,bend,haze,clear --out renders/edits` edits the surface or medium; haze accepts `--haze-strength` and `--haze-rise`.
- `stylize.py` optimizes surface or medium style; `python -m apps.style.compose_style` combines their saved models.

> **TODO (link):** download the asset package from <LINK> and unpack it into assets/

The teaser requires the `treasure_chest`, `antique_ceramic_vase_01`, `dutch_ship_large_01`, `submarine_01`, `portable_searchlight`, and `barramundi` model directories. Insertion benchmarks require the corresponding `insert.json`, `transforms_test.json`, `gt_rgba/`, and `masks/`. The benchmark computes FoamSurf results; the baseline numbers are those reported in the paper.

## Method, citation and licenses

See [the implementation guide](docs/METHOD.md), [research references](docs/references.bib),
and [third-party attribution and distribution status](docs/THIRD_PARTY.md).
The public FoamSurf BibTeX entry will be added when the author list and paper URL are
public. The root Apache-2.0 license does not replace third-party terms. The FlaRe-base
license/notice and per-asset redistribution information remain prerequisites for public
packaging; they are not silently covered by the root license.
