# FoamSurf

Scena podwodna jako hybryda dwóch reprezentacji renderowanych jednym ray tracerem (CUDA + OptiX):

- **Gaussy 2D (surfele)** — powierzchnia sceny (dno, rafa), kolor z harmonik sferycznych;
- **komórki Voronoi** — ośrodek (woda albo mgła): tłumienie i rozpraszanie wsteczne w każdej komórce.

Do tego model kamery PPISP (ekspozycja i kolor każdego zdjęcia), straty OceanSplat i priory, które trzymają
Gaussy poza otwartą wodą. Wynik na SeaThru-NeRF (widoki testowe, 3 seedy): **30.75 dB** średnio
(IUI3 31.78, Curaçao 34.33, Japanese Gardens 24.33, Panama 32.57).

## Instalacja

Potrzebne: GPU NVIDIA z compute capability ≥ 8.0 (np. RTX 3090 / 4090 / A100), sterownik ≥ 550, Linux.

1. Kod i submoduł Eigen:
   ```bash
   git clone <adres repo> foamsurf && cd foamsurf
   git checkout slim
   git submodule update --init
   ```
2. Środowisko conda z Pythonem 3.11, CUDA 12.4 (`nvcc`) i PyTorch 2.6:
   ```bash
   conda create -n foamsurf python=3.11 -y && conda activate foamsurf
   conda install -c nvidia cuda-toolkit=12.4 -y
   pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
   ```
3. OptiX SDK 8.0.0: pobrać ze strony NVIDIA (wymaga konta), rozpakować i wskazać katalog:
   ```bash
   export OPTIX_DIRECTORY=/sciezka/do/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64
   ```
4. Architektura GPU, dla której kompilujemy (8.9 = RTX 4090, 8.6 = RTX 3090, 8.0 = A100):
   ```bash
   export TORCH_CUDA_ARCH_LIST=8.9
   ```
5. Kompilacja (renderer, OptiX, topologia Delaunay; na końcu sprawdza importy, wypisuje `FoamSurf B0 build: OK`):
   ```bash
   CONDA_ENV_PATH=$CONDA_PREFIX ./build.sh
   ```
   Jeśli biblioteki sterownika (`libcuda.so`, `libnvoptix.so`) nie są w `/usr/lib/x86_64-linux-gnu`, ustaw
   `FOAMSURF_DRIVER_LIBRARY_DIRECTORY`. Model kamery (PPISP) i kernel kolorów SH kompilują się same przy pierwszym
   użyciu (`TORCH_EXTENSIONS_DIR` ustala, gdzie trafiają).
6. Test, że wszystko działa (bez GPU część testów jest pomijana; z GPU i danymi SeaThru jest też krótki trening):
   ```bash
   python -m pytest -q tests
   ```

## Dane

- **SeaThru-NeRF** — każda scena to katalog COLMAP: `images/`, `sparse/0/`. Widoki testowe: co 8. zdjęcie.
  U nas: `/shared/sets/datasets/SeathruNeRF_dataset/{IUI3-RedSea-Undistorted, Curasao-Undistorted,
  JapaneseGradens-RedSea-Undistorted, Panama-Undisorted}`.
- **Garden z mgłą** (syntetyczna mgła WaterSplatting): `/shared/sets/datasets/foamsurf/foggy_garden/{garden_easy, garden_hard}`.

## Trening

```bash
python train.py --scene /data/SeathruNeRF_dataset/Curasao-Undistorted --out outputs/curacao
python train.py --scene /data/foggy_garden/garden_easy --out outputs/garden_easy --config configs/garden_fog.yaml
```

- `--seed N` (domyślnie 42), `--wandb PROJEKT` (logowanie do Weights & Biases; domyślnie wyłączone),
  `--set klucz.ścieżka=wartość` (zmiana dowolnego ustawienia), `--resume` (dokończenie przerwanego runu),
  `--dry-run` (wypisuje pełną konfigurację i kończy).
- 20 tys. iteracji: 25–35 min treningu na RTX 4090, z ewaluacjami ok. 75 min. Metryki widoków testowych co 500
  iteracji (`eval.csv`), rendery co 5000 i na końcu; w artykule raportujemy wynik z ostatniej iteracji.
- Slurm, wszystkie 4 sceny SeaThru: `sbatch --array=0-3 scripts/slurm_train.sbatch` (ustawienia w nagłówku pliku).

Katalog runu:

| ścieżka | zawartość |
|---|---|
| `joint/checkpoints/last.pt` | model (Gaussy, komórki), optymalizatory, model kamery, światło wody |
| `joint/metrics/train.csv`, `eval.csv` | straty i statystyki co krok, metryki testowe co ewaluację |
| `joint/evaluation/iteration_*/` | rendery testowe: pełny obraz, same Gaussy, sam ośrodek, krycie, błąd |
| `joint/summary.json` | status, czas, liczba Gaussów |
| `joint/config.yaml`, `pipeline_config.json` | pełna konfiguracja runu |

## Renderowanie

```bash
python render.py --run outputs/curacao                                   # widoki testowe, wszystkie tryby
python render.py --run outputs/curacao --views train --modes full,gaussians
python render.py --run outputs/curacao --export-cameras kamery.json      # kamery zbioru danych jako szablon póz
python render.py --run outputs/curacao --poses moje_pozy.json --modes gaussians,voronoi --npy
python render.py --run modele/curacao --scene /dane/Curasao-Undistorted   # run wytrenowany na innym komputerze
```

`render.py` potrzebuje zbioru danych runu (kamery i zdjęcia treningowe: model kamery i światło wody dopasowują się do
nich przy wczytaniu). Ścieżka jest zapisana w runie; jeśli dane leżą gdzie indziej, podaj `--scene`.

Tryby (`--modes`, PNG w `OUT/<tryb>/<widok>.png`, z `--npy` także tablice float32):

| tryb | co to jest |
|---|---|
| `full` | obraz jak w ewaluacji: Gaussy + woda, światło wody, model kamery |
| `raw` | to samo przed modelem kamery |
| `gaussians` | same Gaussy na czarnym tle — scena bez wody |
| `alpha` | skumulowane krycie Gaussów |
| `voronoi` | sam ośrodek (komórki Voronoi), bez Gaussów |
| `water` | światło wody przed powierzchnią w pełnym renderze |
| `depth` | głębokość powierzchni (w PNG znormalizowana, w `--npy` metryczna) |

Plik póz: lista kamer w układzie świata zbioru danych (COLMAP), format jak w `--export-cameras`:
`{"name", "origin", "right", "down", "forward", "width", "height", "fx", "fy", "cx", "cy"}`.

## Konfiguracje

- `configs/seathru.yaml` — metoda na SeaThru-NeRF. Każde ustawienie z komentarzem; liczby to wartości, które
  faktycznie działają.
- `configs/garden_fog.yaml` — `seathru.yaml` plus jawna lista różnic dla Garden z mgłą.
- `configs/base.yaml` — pełny szablon wszystkich ustawień (wartości domyślne).

`tests/test_configs.py` sprawdza, że obie konfiguracje dają dokładnie ustawienia naszych raportowanych runów
(`tests/data/*.json`): po celowej zmianie metody trzeba zaktualizować te pliki.

## Mapa kodu

| katalog / plik | co tam jest |
|---|---|
| `renderer/` | ray tracer CUDA/OptiX: Gaussy (OptiX), komórki Voronoi (CUDA), składanie, gradienty. Opis: [renderer/README.md](renderer/README.md) |
| `renderer/foamsurf/hybrid.py` | `NativeHybridRenderer`: `render(...)`, `render_gaussians(...)`, `render_voronoi(...)` |
| `src/`, `torch_bindings/` | triangulacja Delaunay komórek Voronoi (CUDA) |
| `foamsurf_model/gaussians.py` | surfele: parametry, kolor SH (×1.5 sigmoid), filtr Mip, `render_scene(origin)` |
| `foamsurf_model/medium.py`, `topology.py` | komórki Voronoi: tłumienie, rozpraszanie, kolor; sąsiedztwo |
| `foamsurf_model/gaussian_adaptation.py` | zagęszczanie, usuwanie i reset krycia Gaussów |
| `foamsurf_data/` | wczytywanie COLMAP i Blender, generowanie promieni |
| `training/trainer.py` | pętla treningu (strata obrazu, OceanSplat, model kamery, zagęszczanie) |
| `training/colmap_joint_trainer.py` | priory między Gaussami a wodą (chroma, pokrycie dna, czysta woda, woda przy kamerach, nakładanie się) |
| `training/colmap_joint.py`, `warmup_trainer.py` | inicjalizacja: punkty COLMAP → Gaussy, 5000 komórek, dopasowanie wody |
| `training/water_*.py`, `camera_isp.py`, `oceansplat.py` | światło wody, model kamery, straty OceanSplat |
| `evaluation/evaluator.py` | metryki i rendery testowe |
| `train.py`, `render.py` | punkty wejścia |
| `render.py --fps 32` | pomiar szybkości renderu wytrenowanego runu |

## Paper training checks

`scripts/reproduce.sh` runs training and final-checkpoint evaluation sequentially on one GPU, and resumes interrupted runs. Set the dataset and mask directories before running it:

```bash
export FS_DATASETS=/path/to/SeathruNeRF_dataset
export FS_SYNTHETIC=/path/to/synthetic
export FS_FOGGY=/path/to/foggy_garden
export FS_MASKS=/path/to/masks
scripts/reproduce.sh seathru --scenes IUI3 --seeds 42
scripts/reproduce.sh synthetic --scenes lego --background black --seeds 42
```

The commands use the complete paper schedules. `--dry-run` prints the commands, `--out DIR` selects the output root, and `--pipeline voronoi|hybrid|all` selects the synthetic pipeline (default: hybrid). The other training targets are `ablation`, `garden`, and `all`. `scripts/summarize.py TARGET --root DIR` aggregates the evaluation JSON files using the sample standard deviation across seeds; scene averages are computed per seed first.

## Application entry points

- `insert.py --run RUN --shot apps/raytrace/shots/ruins/teaser.json --assets assets --out renders/teaser` renders inserted objects.
- `insert.py --benchmark --run RUN --gt ASSETS/insertion/lego --out renders/lego_insertion` evaluates FoamSurf against the packaged Blender references.
- `edit.py --run RUN --view VIEW --edits bulge,bend,haze,clear --out renders/edits` edits the surface or medium; haze accepts `--haze-strength` and `--haze-rise`.
- `stylize.py` optimizes surface or medium style; `python -m apps.style.compose_style` combines their saved models.

> **TODO (link):** download the asset package from <LINK> and unpack it into assets/

The teaser requires the `treasure_chest`, `antique_vase`, `dutch_ship`, `submarine01`, `portable_searchlight`, and `barramundi` model directories. Insertion benchmarks require the corresponding `insert.json`, `transforms_test.json`, `gt_rgba/`, and `masks/`. The benchmark computes FoamSurf results; the baseline numbers are those reported in the paper.
