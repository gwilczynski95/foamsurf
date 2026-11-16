# FoamSurf

Scena podwodna jako hybryda dwóch reprezentacji renderowanych jednym ray tracerem (CUDA + OptiX):

- **Gaussy 2D (surfele)** — powierzchnia sceny (dno, rafa), kolor z harmonik sferycznych;
- **komórki Voronoi** — ośrodek (woda albo mgła): tłumienie i rozpraszanie wsteczne w każdej komórce.

Do tego model kamery PPISP (ekspozycja i kolor każdego zdjęcia), straty OceanSplat i priory, które trzymają
Gaussy poza otwartą wodą. Wynik na SeaThru-NeRF (widoki testowe, 3 seedy): **30.74 dB** średnio
(IUI3 31.87, Curaçao 34.29, Japanese Gardens 24.31, Panama 32.51).

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
| `joint/checkpoints/last.pt`, `best.pt` | model (Gaussy, komórki), optymalizatory, model kamery, światło wody |
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

## Transfer stylu

Stylizacja wytrenowanego runu obrazem (albo opisem tekstowym), **osobno dla każdej reprezentacji**: Gaussów
(powierzchnia: dno, rafa) albo komórek Voronoi (woda). Strata jak w FlaRe / CLIPGaussian (`style_transfer.py`):
kierunkowa CLIP (zmiana renderu względem renderu przed stylizacją ma iść w kierunku styl − „a Photo”), CLIP na 64
wycinkach 128 px z losową perspektywą, treść VGG19 (conv4_2, conv5_2). 5000 kroków, w każdym jedna losowa kamera
zbioru danych.

Dodatkowe pakiety (tylko do stylizacji; wagi CLIP ViT-B/32 i VGG19 pobierają się przy pierwszym uruchomieniu):

```bash
pip install torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install ftfy regex git+https://github.com/openai/CLIP.git
```

Styl to dowolny obraz (`--style-image`, JPG/PNG) albo tekst (`--style-prompt "a painting by Van Gogh"`):

```bash
# powierzchnia (Gaussy), np. ogień
python stylize.py --run outputs/iui3 --target gaussians --style-image style/fire.jpg --out outputs/iui3_fire
# woda (komórki Voronoi), np. Gwiaździsta noc
python stylize.py --run outputs/iui3 --target voronoi --style-image style/starry_night.jpg --out outputs/iui3_starry
# obie naraz: Gaussy z pierwszego runu, woda z drugiego (oba stylizowane z tego samego runu)
python compose_style.py --gaussians outputs/iui3_fire --medium outputs/iui3_starry --out outputs/iui3_fire_starry
python render.py --run outputs/iui3_fire_starry --views all --modes full,gaussians,voronoi
```

Wynik każdego kroku to zwykły katalog runu (`joint/config.yaml`, `joint/checkpoints/last.pt`), który wczytują
`render.py` i sam `stylize.py` (np. drugi styl na już stylizowanym runie). W `OUT/style/` są
`losses.csv`, `style_config.json` i podglądy (`previews/<widok>_{full,<cel>}_<krok>.png`). Jak przy `render.py`,
`--scene` podaje zbiór danych, jeśli nie leży w ścieżce zapisanej w runie.

| `--target` | strata liczona na | trenowane | stałe |
|---|---|---|---|
| `gaussians` | same Gaussy na czarnym tle (tryb `gaussians`) | kolor (DC SH), geometria z małym lr | woda; kontur rafy (strata tła), kolor ≤ 1 |
| `voronoi` | sam ośrodek (tryb `voronoi`) | kolor komórek, rozpraszanie wsteczne, położenie komórek | tłumienie światła bezpośredniego, Gaussy |

Ustawienia domyślne to nasz baseline (IUI3: `fire` na Gaussach, `starry_night` na wodzie), ok. 18–19 min na cel na
RTX 3090, ok. 13 GB pamięci GPU:

- **Gaussy**: krok koloru `--color-lr 0.015·C0` (1.5× krok CLIPGaussian; większy daje mocniejszy styl i więcej
  szumu barw), kolor każdego kanału ≤ 1 (`--color-cap`), gradient koloru uśredniany z 8 najbliższymi Gaussami
  (`--color-grad-smooth`). Bez tych dwóch w jasnych miejscach zostaje mozaika różowych i turkusowych Gaussów;
  `--no-geometry` trzyma geometrię, `--sh-rest train` uczy też wyższe pasma SH.
- **Woda**: każda komórka dzielona na 27 (`--medium-split`, 5000 → 135 tys.), bo 5000 komórek uśrednia wzór wzdłuż
  promieni; kolor (`--medium-color-lr 0.04·C0`), rozpraszanie wsteczne (`--medium-backscatter-lr 1e-2`, jasne
  plamy) i położenie komórek (`--medium-sites-lr 1e-3`, triangulacja co `--topology-interval 25` kroków).
- Wspólne: `--steps`, `--views all|train`, wagi strat `--lambda-dir 5 --lambda-patch 90 --lambda-content 0.8`
  (`--lambda-bg`: 1000 dla Gaussów, 0 dla wody), `--save-every` (podglądy), `--seed`.

Wskazówki:

- **Jasny styl na wodzie** (np. ogień): strata widzi samą wodę, bez rafy. Z trenowanym rozpraszaniem woda przed
  rafą robi się nieprzezroczysta i zasłania dno. `--medium-backscatter-lr 0` trzyma jej nieprzezroczystość z
  rekonstrukcji, a styl zmienia tylko kolor i rozkład komórek.
- **Obraz a prompt:** obraz z czarnym tłem (`fire.jpg`) zwykle ściemnia wodę. Prompt tekstowy (`"bright orange fire
  flames"`) daje kolor bez tej czerni.
- **Pamięć:** zdjęcia 1794×1188 (Curaçao, Panama) przy `--target gaussians` zajmują prawie całą kartę 24 GB. Na
  współdzielonym GPU OptiX może nie dostać pamięci („Caught an unknown exception”).

## Edycje

`edit_scene.py` edytuje wytrenowany (albo stylizowany) run **bez ponownego trenowania**, osobno powierzchnię albo
wodę:

- każda edycja zaczyna od wczytanego modelu;
- wynik renderuje się z jednej kamery;
- z `--save` edycja jest zapisywana jako zwykły run dla `render.py`, z parametrami edycji w `edit.json`.

```bash
python edit_scene.py --run outputs/iui3 --view MTN_5911 --edits bulge,relief,bend,clear,murky,haze,plume \
    --out outputs/iui3_edits --save outputs/iui3_edited
python edit_scene.py --run outputs/iui3 --view MTN_5911 --edits transplant --transplant-from outputs/curasao \
    --out outputs/iui3_edits --save outputs/iui3_edited
python render.py --run outputs/iui3_edited/bulge --views test --modes full,gaussians,voronoi
```

Edycje są zdefiniowane w układzie dopasowanym do rafy widocznej z `--view`:

- normalna płaszczyzny najmniejszych kwadratów przez widoczne punkty powierzchni;
- dwie osie poziome;
- skala `L`: zasięg widocznej rafy.

| edycja | co robi |
|---|---|
| `waves`, `ripple`, `twist` | wydmy, koncentryczna fala, wir wokół normalnej |
| `bulge`, `relief`, `bend` | wzgórze pod rafą, rzeźba terenu ×2.5, dno zwinięte w rynnę |
| `clone` | kopia fragmentu rafy spod jednego piksela w miejsce pod drugim (`--clone`) |
| `clear`, `murky` | tłumienie i rozpraszanie wody ×0.25 / ×2.5 |
| `turquoise`, `green`, `redtide` | inny typ wody: odcień koloru (z zachowaną jasnością komórek) i widmo tłumienia |
| `transplant` | statystyki wody innego runu (`--transplant-from`), przestrzenny rozkład zostaje |
| `haze` | gęsta mgła przy dnie (komórki dzielone ×8) |
| `plume` | pióropusz osadu nad punktem `--plume-at` (150 tys. komórek dodanych lokalnie) |

Deformacje:

- środki Gaussów przechodzą przez odwzorowanie;
- osie styczne i skale idą przez jego jakobian (różnice centralne + SVD), więc kowariancja surfela to `J Σ Jᵀ`;
- kolor, krycie i woda zostają bez zmian.

Edycje wody zmieniają tylko współczynniki i kolor komórek. Wzory i stałe są w docstringu `edit_scene.py`.
Deformacja odsłania miejsca, których kamery nie widziały (zbocza, tył skał); tam nie ma Gaussów i prześwituje woda.

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
| `stylize.py`, `compose_style.py` | transfer stylu Gaussów albo wody; złożenie dwóch stylizacji w jeden run |
| `style_transfer.py` | strata stylu FlaRe / CLIPGaussian (CLIP kierunkowa i na wycinkach, treść VGG19) |
| `edit_scene.py` | edycje bez treningu: deformacje Gaussów (z jakobianem), zmiany wody (współczynniki, kolor, lokalne komórki) |
| `scripts/fps_bench.py` | pomiar szybkości renderu wytrenowanego runu |
