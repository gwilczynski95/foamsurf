# Walkthrough repozytorium FoamSurf B0

Ten dokument jest mapą kodu dla osoby, która chce rozwijać FoamSurf. Opisuje
aktywną ścieżkę B0. Zaadaptowany komponent Delaunay jest izolowany za API
`foamsurf_topology`; repo nie zawiera legacy aplikacji treningowej RadFoam.

## Model mentalny

Scena ma dwie uczone reprezentacje:

1. `SurfaceGaussianModel` opisuje powierzchnie orientowanymi splatami 2D.
2. `VoronoiMediumModel` opisuje medium stałą gęstością i barwą wewnątrz
   każdej komórki Voronoi.

Jedna paczka rayów przechodzi przez obie reprezentacje. Renderer układa odcinki
medium i trafienia splatów według głębokości, daje jeden obraz `RGB_hybrid`, a
jedyny loss B0 to MSE względem obrazu referencyjnego. To samo `backward()`
dostarcza gradienty do obu modeli. Oddzielne optimizery są kwestią różnych
learning rate'ów i stanu Adam, a nie dwóch niezależnych lossów.

## Najważniejsze katalogi

### Punkt wejścia i konfiguracja

- `train_b0.py` parsuje CLI, czyta YAML, nakłada cztery bezpieczne override'y
  i wywołuje `Trainer.fit()`.
- `configs/b0_lego.yaml` jest pełnym, wersjonowanym eksperymentem B0.
- `training/config.py` zawiera zagnieżdżone dataclassy i walidację. Nie
  odczytujemy luźnych kluczy konfiguracyjnych w różnych miejscach kodu.

Dodając opcję, najpierw dodaj ją do odpowiedniego dataclassa, potem do YAML i
testu walidacji. Nie przekazuj nowych globali do trenera.

### Dane: `foamsurf_data/`

- `factory.py` rozpoznaje Blender/NeRF Synthetic albo COLMAP i zwraca wspólny
  obiekt datasetu.
- `blender.py` czyta `transforms_train.json` i `transforms_test.json`,
  obrazy RGBA oraz pozy kamer.
- `colmap.py` obsługuje układ danych używany przez 3DGS/Mip-NeRF 360,
  normalizację dużych scen, point cloud i split `llffhold`.
- `rays.py` jest jedynym miejscem generowania rayów z intrinsics i
  camera-to-world. Zarówno trening, jak walidacja korzystają z tego kodu.

Wspólny kontrakt kamery udostępnia `origin`, `rays()`,
`ground_truth(background)`, nazwę oraz rozdzielczość. Dzięki temu model nie
ma warunków typu „jeśli Blender, to...”.

### Model: `foamsurf_model/`

`gaussians.py` zawiera `SurfaceGaussianModel`:

- pozycje startują z point cloudu datasetu albo próbkowania zastępczego;
- skale startowe wynikają z odległości do najbliższych sąsiadów;
- orientacja, opacity i kappa mają parametryzację zgodną z rendererem;
- appearance może być bezpośrednim RGB lub SH do stopnia 3;
- `render_scene(camera_origin)` przygotowuje `foamsurf.GaussianScene`,
  w tym liczy kierunkowy kolor SH przed wejściem do kernela.

`medium.py` zawiera `VoronoiMediumModel`:

- bbox bierze odporne percentyle surface point cloudu i pozycje kamer albo
  jawne `medium_bbox_min/max`;
- wnętrze jest próbkowane stratified-jittered w całym volume;
- opcjonalne osiem guard sites znajduje się przy powiększonych rogach bboxa;
- uczone są `sites`, `raw_density` i `raw_color`;
- renderer widzi `softplus(raw_density)` i `sigmoid(raw_color)`.

To celowo inna inicjalizacja niż surface-oriented RadFoam: Voronoi ma na B0
pokrywać medium, a nie przyklejać się do powierzchni.

`scene.py` jest małym kontenerem obu modeli i centralnym punktem ich
inicjalizacji/checkpoint metadata.

`topology.py` jest właścicielem dyskretnej topologii:

- tworzy Delaunay triangulation z aktualnych sites;
- przechowuje CSR `adjacency/offsets` wymagane przez renderer;
- podczas permutacji punktów permutuje razem parametry i momenty Adam;
- okresowo przebudowuje connectivity, gdy sites się poruszają;
- synchronizuje nową topologię z rendererem.

B0 nie zmienia liczby sites. Przyszły split/prune/merge ma wejść właśnie do
`VoronoiTopologyController`, a nie rozrastać pętlę treningową.

### Renderer: `renderer/`

To oddzielny projekt z własnym `setup.py`, buildem, testami i publicznym
pakietem `foamsurf`. Pythonowy model przekazuje mu wyłącznie typy sceny i
tensory. Renderer nie zna klas treningowych.

Najważniejsze punkty API:

- `set_gaussian_geometry(...)` synchronizuje OptiX poza hot pathem;
- `set_voronoi_geometry(...)` synchronizuje sites i CSR po zmianie topologii;
- `render(...)` daje hybrid;
- `render_gaussians(...)` i `render_voronoi(...)` dają izolowane warianty.

Szczegółowy layout, równania compositingu i backward są w
[renderer/README.md](../renderer/README.md). Przed modyfikacją CUDA trzeba
przeczytać ten dokument w całości.

### Trening: `training/`

- `sampler.py` losuje jedną kamerę, potem piksele/raye z tej kamery. Jedna
  camera origin na batch umożliwia policzenie SH raz dla splatów, bez
  per-ray/per-Gaussian macierzy kolorów.
- `losses.py` definiuje kontrakt B0: dokładnie RGB MSE i raportowany PSNR.
- `optimizer.py` buduje osobny Adam dla Gaussianów i medium z nazwanymi
  param groups. Nazwy są ważne, bo topology controller używa ich przy
  bezpiecznej podmianie parametrów.
- `trainer.py` orkiestruje kroki, ale nie zawiera matematyki renderera,
  inicjalizacji ani topology internals.
- `experiment_logging.py` zapisuje stabilne schematy CSV i W&B.
- `checkpoints.py` atomowo zapisuje `last.pt`/`best.pt`.

### Walidacja: `evaluation/`

`Evaluator` bierze każdą kamerę testową i dokładnie te same raye renderuje
jako:

1. hybrid,
2. Gaussian-only,
3. Voronoi-only.

Dla każdego trybu liczy PSNR, SSIM, alpha i pełny czas renderu/FPS. Zapisuje
`per_view.csv`, osobne obrazy oraz panel:

```text
target | hybrid | absolute error | Gaussian-only | Voronoi-only | transmittance
```

`metrics/eval.csv` zawiera średnie po widokach. W&B dostaje te same liczby,
panele i histogramy parametrów. Rozdzielenie komponentów w walidacji jest
diagnostyką, nie osobnym sygnałem uczącym.

## Jeden krok treningowy, po kolei

`Trainer._train_step()` wykonuje następującą sekwencję:

1. sampler wybiera kamerę i losowe piksele;
2. model liczy kamerowo-zależne kolory Gaussianów i materializuje obie sceny;
3. ray origins/directions są składane do `[R,6]`;
4. oba optimizery zerują gradienty;
5. jeden `renderer.render(..., gaussians=..., voronoi=...)` tworzy hybrid RGB;
6. `b0_reconstruction_loss` liczy wyłącznie RGB MSE;
7. jeden `loss.total.backward()` uruchamia ręczny CUDA backward;
8. diagnostyka mierzy normy gradientów obu gałęzi;
9. oba optimizery wykonują `step()`;
10. co `topology_update_interval` kroków przebudowywane jest Delaunay i
    synchronizowany renderer;
11. po zmianie geometrii Gaussianów wykonywane jest jawne
    `set_gaussian_geometry()`;
12. czas kroku trafia do CSV/W&B.

Kolejność punktów 9–11 jest istotna: renderer następnego kroku musi widzieć
geometrię po optimizerze. `set_geometry` nie jest ukryte wewnątrz
`render()`.

## Checkpointy i resume

`checkpoints/last.pt` jest nadpisywany atomowo na zadanym interwale i po
każdej walidacji. `checkpoints/best.pt` jest nadpisywany tylko wtedy, gdy
średni validation **hybrid PSNR** przekracza poprzedni rekord.

Checkpoint zawiera:

- oba `state_dict` modelu;
- oba pełne stany Adam;
- konfigurację i metadata modeli;
- iterację, skumulowany training time i best PSNR;
- stan RNG CPU i wszystkich widocznych GPU;
- revision topologii.

Po resume model jest odtwarzany przed topologią, następnie momenty Adam i
ewentualna permutacja topologii zostają podłączone do właściwych nowych
`Parameter`. `--iterations` oznacza docelową końcową iterację.

## CSV i W&B

`metrics/train.csv` ma jeden stabilny header i między innymi:

- loss, RGB MSE, train PSNR;
- czas kroku i skumulowany czas;
- normę gradientu Gaussianów i medium;
- opacity/scale Gaussianów;
- density/color medium;
- liczności reprezentacji i revision topologii.

`metrics/eval.csv` ma PSNR/SSIM/alpha/czas/FPS dla trzech trybów.
`evaluation/iteration_*/per_view.csv` pozwala znaleźć kamerę odstającą od
średniej. Ten lokalny zapis działa niezależnie od sieci i W&B.

W&B ma tryby `online`, `offline`, `disabled`. Logowane są te same scalars
co do CSV, panele walidacyjne i histogramy. CSV pozostaje źródłem prawdy do
skryptów/wykresów, więc awaria W&B nie odbiera wyników eksperymentu.

## Co uruchomić podczas developmentu

Po zmianie czystego Pythona:

```bash
python -m pytest -q tests/b0
```

Po zmianie treningu, checkpointu lub integracji:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/smoke_test_b0.py
```

Po zmianie renderera:

```bash
cd renderer
CUDA_VISIBLE_DEVICES=1 python -m pytest -q
```

Smoke test tworzy malutki dataset Blender w katalogu tymczasowym, wykonuje
trening, pełną walidację, sprawdza obrazy/CSV/checkpointy, następnie wznawia run
i sprawdza ciągłość iteracji.

## Gdzie dodawać nowy kod

- Nowy loss lub regularizer: nowy moduł w `training/`, a nie w rendererze.
- Nowe statystyki potrzebne do densyfikacji: wynik renderera/types + collector
  w modelu/topology controller.
- Split/prune/merge medium: `foamsurf_model/topology.py` wraz z migracją stanu
  optimizera.
- Zmiana reprezentacji parametru: odpowiedni model i jego
  `render_scene()`; renderer powinien dostawać wartości fizyczne.
- Nowa metryka/obraz: `evaluation/` i `ExperimentLogger`.
- Nowy format datasetu: adapter w `foamsurf_data/` zwracający ten sam kontrakt
  kamery.

Nową logikę modelu dodawaj do modułów FoamSurf opisanych powyżej; kod
`src/` jest izolowanym backendem topologii, a nie miejscem na logikę treningu.
