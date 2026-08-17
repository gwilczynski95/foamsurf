# B1: bazowa polityka Gaussianów zgodna z 2DGS

Ten profil zastępuje wcześniejsze eksperymentalne warianty B1. Jego celem
jest przeniesienie sprawdzonej polityki optymalizacji i adaptacji punktów z
oficjalnego 2D Gaussian Splatting do hybrydowego treningu FoamSurf, bez
dodawania własnych heurystyk stabilizujących.

Referencją jest `hbb1/2d-gaussian-splatting` w commitcie
`335ad612f2e783a4e57b9cbc4d1e167bd599fc98` oraz rasterizer w commitcie
`e0ed0207b3e0669960cfad70852200a4a5847f61`.

## Co dokładnie odtwarzamy

- trening na całym obrazie;
- losowanie kamer bez zwracania w obrębie cyklu;
- loss `L1 + 0.2 * (1 - SSIM)`;
- początkowe opacity `0.1`;
- opacity learning rate `0.05`;
- progressive SH, jeden stopień co 1000 iteracji;
- densyfikację od iteracji 500 do 15000, co 100 iteracji;
- próg średniego gradientu screen-space `0.0002`;
- clone dla małych i split na dwoje dla dużych Gaussianów;
- próg clone/split równy `0.01 * cameras_extent`;
- pruning opacity poniżej `0.05`;
- pruning world-space powyżej `0.1 * cameras_extent` po iteracji 3000;
- clamp opacity do maksymalnie `0.01` w iteracji 500 dla białego tła i potem
  co 3000 iteracji, razem z wyzerowaniem stanu Adam
  dla opacity.

Nie ma limitu 200k ani top-k. Densyfikacja kończy się zgodnie z harmonogramem,
a pruning działa dalej podczas eventów adaptacji. Limit miliona punktów jest
wyłącznie bezpiecznikiem przed OOM i kończy błędny run zamiast wpływać na
zdrową trajektorię.

## Visibility i promień projekcji

Oficjalny 2DGS akumuluje gradient dla splatów, których rasterizer zwrócił
`radii > 0`. FoamSurf robi teraz to samo semantycznie: widoczność pochodzi z
analitycznie policzonego promienia projekcji, a nie z liczby trafień przez
wylosowane promienie. Jest to szczególnie ważne przy pełnym obrazie.

Promień jest liczony z Jacobianu projekcji dwóch lokalnych osi planarnego
Gaussiana, z filtrem kowariancji i supportem `ceil(3 sigma)`. Osobny test
porównuje go z przypiętym rasterizerem CUDA 2DGS z tolerancją jednego piksela.

## Dlaczego screen-space pruning jest wyłączony

Kod treningu 2DGS przekazuje próg `20 px` po iteracji 3000, lecz przypięta
implementacja `densification_postfix()` zeruje `max_radii2D` podczas clone i
split, zanim wykonywany jest końcowy pruning. W praktyce ta gałąź nie usuwa
punktów w tych eventach. Profil FoamSurf jawnie ustawia
`screen_pruning_enabled: false`, aby odtworzyć faktyczne zachowanie kodu, a
nie samą intencję widoczną w wywołaniu. Mechanizm pozostaje dostępny wyłącznie
do jawnych ablation i testów.

## Co pozostaje specyficzne dla FoamSurf

To nadal trening hybrydowy: Gaussy powierzchniowe i medium Voronoi są
renderowane razem i uczone z tego samego obrazu `lego_both`. `lego_lego` i
`lego_clouds` służą wyłącznie do ewaluacji. Medium, parametr kappa oraz
hybrydowy renderer nie występują w 2DGS.

Nie włączamy normal consistency ani distortion loss z publikacji 2DGS. B1 ma
izolować wpływ populacji Gaussianów i standardowego reconstruction loss;
regularyzacje powierzchni należą do osobnego etapu i ablation.

## Uruchomienie

Pełny run 30k w osobnym projekcie W&B:

```bash
cd /shared/results/gmwilczn/foamsurf/foamsurf
sbatch scripts/train_b1_2dgs_baseline.sbatch
```

Krótki run można jawnie nadpisać bez zmiany pliku:

```bash
ITERATIONS=10000 sbatch scripts/train_b1_2dgs_baseline.sbatch
```

Domyślny projekt W&B to `foamsurf_2dgs_baseline`. Można go zmienić przez
`WANDB_PROJECT`, podobnie `WANDB_MODE`, `SEED` i `OUTPUT_ROOT`.

## Wyniki i diagnostyka

Run zachowuje pełny kontrakt FoamSurf: CSV metryk, obrazy hybrid/Gaussian/
Voronoi, diagnostykę `lego_lego` i `lego_clouds`, checkpoint `last.pt`,
model `best.pt` wybierany po PSNR oraz podsumowanie końcowe. W&B dostaje te
same metryki i obrazy.

Najważniejsze sygnały adaptacji to liczba Gaussianów, kandydaci clone/split,
liczba usunięta przez opacity/world-scale, kwantyle gradientu, reset opacity,
aktywny stopień SH, LR pozycji i pamięć GPU.

## Testy

```bash
python -m pytest -q tests/b1/test_twodgs_profile.py
python -m pytest -q tests/b1/test_gaussian_adaptation.py
bash -n scripts/train_b1_2dgs_baseline.sbatch
```

Pierwszy zestaw blokuje dokładne wartości profilu i kontrakt joba. Drugi
sprawdza statystyki, clone/split/prune, mapowanie parametrów i momentów Adam,
reset opacity oraz jawny wariant screen-space pruning. Starsze sweepy
ustawiają ten wariant explicite, dzięki czemu pozostają odtwarzalne.
