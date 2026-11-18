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

- **SeaThru-NeRF:** use the undistorted scenes, each containing `images/` and `sparse/0/`.
  Every eighth image is held out for testing. Scene directory names are
  `IUI3-RedSea-Undistorted`, `Curasao-Undistorted`, `JapaneseGradens-RedSea-Undistorted`,
  and `Panama-Undisorted` (preserve these dataset spellings).
- **Foggy Garden:** `garden_easy/` and `garden_hard/`, generated from clear Garden images
  and camera-z depths by `scripts/make_foggy_garden.py`.
- **Synthetic scenes:** `lego/` and `ficus/`. Each contains `<scene>_both/` and
  `<scene>_clouds/`; the object-only directory is `lego_lego/` for Lego and
  `ficus_object/` for Ficus. These directories contain Blender camera JSON files and RGBA images.
- **Separation masks:** one `<scene>.npz` file per SeaThru scene. These masks are used for
  evaluation, not training. `scripts/sam_water_mask.py` can regenerate them with Segment Anything.

> **TODO (link):** download the prepared datasets and evaluation masks from <LINK>.

> **TODO (link):** download the trained paper models from <LINK> and unpack them into models/.

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

The commands use the complete paper schedules. `--dry-run` prints the commands, `--slurm` submits the same sequential workflow through `scripts/slurm/reproduce.sbatch`, `--out DIR` selects the output root, and `--pipeline voronoi|hybrid|all` selects the synthetic pipeline (default: hybrid). The other training targets are `ablation`, `garden`, and `all`. `scripts/summarize.py TARGET --root DIR` aggregates the evaluation JSON files using the sample standard deviation across seeds; scene averages are computed per seed first.

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
| `tab:insertion`, `tab:suppl_insertion`, `fig:suppl_insertion_lego`, `fig:suppl_insertion_ficus` | `insert.py --benchmark --run RUN --gt ASSETS/insertion/SCENE --out OUT` | FoamSurf insertion renders and Blender-reference metrics |
| `fig:suppl_edit_surface`, `fig:suppl_edit_surface_layers`, `fig:suppl_edit_water`, `fig:suppl_edit_water_layers` | `edit.py --run RUN --view VIEW --edits EDITS --out OUT` | Edited full images and separate layers |
| `fig:suppl_style` | `stylize.py --run RUN --target gaussians --style-image IMAGE --out OUT` (or `--target voronoi`) | Stylized final model and preview images |
| `tab:supp_weights`, `tab:supp_learning_rates` | `train.py --config configs/seathru.yaml --dry-run` | Resolved loss weights, learning rates and schedules |
| `fig:overview` | Method schematic | Explanatory diagram, not an evaluation output |

The control runs use IUI3 seed 42 (31.80 ± 0.15 dB) and hybrid Lego black seed 42
(34.22 ± 0.30 dB). These are tolerances for fresh training; loading an existing paper
checkpoint is tested separately at a tighter 0.001 dB PSNR tolerance. Synthetic black
backgrounds use seeds 42/43/44; white backgrounds use seed 42. The hybrid synthetic pipeline
uses a 5,000-step warmup and 20,000 joint steps; the Voronoi-first pipeline uses
5,000 warmup steps followed by 30,000 joint steps. Runtime depends on scene size and
evaluation frequency; final release timing measurements are still being collected.

Release integration is still in progress: the `applications` reproduction target,
exact shot/style presets for all figures, and the draining/bounded-medium-removal code
for `fig:draining`, `fig:applications` and the complete `fig:teaser` are pending.
The current teaser shot can be rendered through `insert.py`, as shown below.

## Application entry points

- `insert.py --run RUN --shot apps/raytrace/shots/ruins/teaser.json --assets assets --out renders/teaser` renders inserted objects.
- `insert.py --benchmark --run RUN --gt ASSETS/insertion/lego --out renders/lego_insertion` evaluates FoamSurf against the packaged Blender references.
- `edit.py --run RUN --view VIEW --edits bulge,bend,haze,clear --out renders/edits` edits the surface or medium; haze accepts `--haze-strength` and `--haze-rise`.
- `stylize.py` optimizes surface or medium style; `python -m apps.style.compose_style` combines their saved models.

> **TODO (link):** download the asset package from <LINK> and unpack it into assets/

The teaser requires the `treasure_chest`, `antique_ceramic_vase_01`, `dutch_ship_large_01`, `submarine_01`, `portable_searchlight`, and `barramundi` model directories. Insertion benchmarks require the corresponding `insert.json`, `transforms_test.json`, `gt_rgba/`, and `masks/`. The benchmark computes FoamSurf results; the baseline numbers are those reported in the paper.
