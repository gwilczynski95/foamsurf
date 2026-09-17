# Aktualny baseline disentanglementu: RadFoam prior -> FoamSurf

## Status

Ten dokument jest kanonicznym opisem aktualnego baseline'u disentanglementu
FoamSurf. Wynik referencyjny pochodzi z ukończonego joba Slurm `502753`:

`/shared/results/gmwilczn/foamsurf/foamsurf/outputs/radfoam_prior/radfoam_prior_run30_502753/`

To pierwszy eksperyment, w którym jednocześnie:

- pełny render hybrydowy zachowuje wysoką jakość;
- Gaussy rekonstruują prawie wyłącznie ciało stałe (`lego_lego`);
- Voronoi rekonstruuje prawie wyłącznie medium (`lego_clouds`);
- chmura pozostająca w Gaussach we wcześniejszych sweepach praktycznie znikła.

Każdy następny eksperyment dotyczący disentanglementu należy porównywać z tym
baseline'em. Wcześniejsze runy 30/41/61 ze sweepów eviction/handoff nie są już
głównym punktem odniesienia.

## Repozytorium, środowisko i sprzęt

- repozytorium: `/shared/results/gmwilczn/foamsurf/foamsurf`;
- Conda env: `/shared/results/gmwilczn/conda/envs/foamsurf`;
- nazwa env: `foamsurf`;
- partycja Slurm: `rtx4090_batch`;
- wymagany zasób: jedna karta RTX 4090;
- config: `configs/radfoam_prior_run30.yaml`;
- entrypoint: `scripts/run_radfoam_prior_training.py`;
- launcher: `scripts/train_radfoam_prior_run30.sbatch`.

Pełny trening wymaga GPU i renderera OptiX. Nie należy uruchamiać go na login
node ani na nodzie bez GPU. Launcher sprawdza CUDA i, jeśli potrzeba, buduje
backend K=64 pod blokadą `flock`.

## Idea eksperymentu

Zwykły trening hybrydowy pozwalał Gaussom zbyt szybko przejmować zarówno
obiekt, jak i chmury. Lossy, pruning i okresowe freezy poprawiały sytuację, ale
nie usuwały uporczywej chmury z reprezentacji Gaussowskiej.

Tutaj wcześniej wytrenowany RadFoam jest przestrzennym priorem. Jego komórki
zostały sklasyfikowane jako:

- `SURFACE`: regiony ciała stałego;
- `MEDIUM`: regiony medium/chmur;
- `EMPTY`: przestrzeń pusta.

Handoff do FoamSurf przebiega następująco:

1. Usuwamy z Voronoi wszystkie komórki `SURFACE`.
2. W pozycjach `SURFACE` inicjalizujemy Gaussy.
3. Zachowujemy wszystkie komórki `MEDIUM` i `EMPTY` jako Voronoi.
4. Zerujemy density pustych i topologicznie naruszonych komórek.
5. Wszystkie zachowane parametry Voronoi pozostają trenowalne.
6. Uruchamiamy normalny trening FoamSurf od iteracji zero, ze świeżymi
   optimizerami i polityką Gaussian eviction z wcześniejszego runu 30.

RadFoam nie jest kontynuowany ani trenowany w tym jobie. Dostarcza geometrię,
density, kolor i podział przestrzeni używane do inicjalizacji FoamSurf.

## Dane i prerekwizyty

### Dane FoamSurf

- target treningowy: `/shared/sets/datasets/foamsurf/lego/lego_both`;
- GT ciała stałego: `/shared/sets/datasets/foamsurf/lego/lego_lego`;
- GT medium: `/shared/sets/datasets/foamsurf/lego/lego_clouds`.

`lego_lego` i `lego_clouds` służą do diagnostyki disentanglementu. Nie są
osobnymi targetami treningowymi.

### Gotowy checkpoint i etykiety RadFoam

Checkpoint po 4000 iteracjach:

`/shared/results/gmwilczn/foamsurf/radfoam/output/lego_both_prior_487402/checkpoints/iter_00004000.pt`

Podział checkpoint-only metodą `local_contrast_otsu`:

`/shared/results/gmwilczn/foamsurf/radfoam/output/lego_both_prior_487402/separation/iter_00004000/checkpoint_only/local_contrast_otsu/cell_labels.pt`

Są to wymagane wejścia; launcher nie tworzy ich ponownie. Kontrola:

```bash
test -s /shared/results/gmwilczn/foamsurf/radfoam/output/lego_both_prior_487402/checkpoints/iter_00004000.pt
test -s /shared/results/gmwilczn/foamsurf/radfoam/output/lego_both_prior_487402/separation/iter_00004000/checkpoint_only/local_contrast_otsu/cell_labels.pt
```

Sposób przygotowania priora jest opisany w
`docs/RADFOAM_CHECKPOINT_SEPARATION_HANDOFF.md`.

## Dokładna inicjalizacja

Checkpoint zawiera 371004 komórki:

| Klasa | Liczba | Działanie |
| --- | ---: | --- |
| `SURFACE` | 211125 | usunięcie z Voronoi; źródło Gaussów |
| `MEDIUM` | 104490 | pozostaje w Voronoi |
| `EMPTY` | 55389 | pozostaje topologicznie, density resetowane |
| `AMBIGUOUS` | 0 | obecność tej klasy powodowałaby błąd |

### Gaussy

Z 211125 pozycji `SURFACE` losujemy bez zwracania 100000 pozycji z
`seed: 0`. Każda wybrana pozycja tworzy jeden Gauss:

- mean: pozycja site'u RadFoam;
- kolor: DC SH RadFoam -> stały RGB -> logit FoamSurf;
- opacity: `0.10`;
- skala: inicjalizacja FoamSurf na podstawie trzech sąsiadów;
- orientacja: PCA ośmiu najbliższych site'ów `SURFACE`, gdzie najmniejszy
  wektor własny jest normalną dysku;
- kappa: `1.01`;
- brak dodatkowego jitteru pozycji.

### Zachowany Voronoi

Pozostaje dokładnie 159879 komórek `MEDIUM + EMPTY`. Zachowana komórka jest
naruszona, gdy jej pierwotna gwiazda Delaunaya zawierała usuniętego sąsiada
`SURFACE`. Takich komórek jest 68363.

Density jest resetowane do `1e-8` dla wszystkich 55389 komórek `EMPTY` oraz
wszystkich naruszonych komórek. Unia zawiera 84964 komórki. `1e-8` jest
renderowo równoważne zeru, ale zachowuje ścieżkę gradientu przy parametryzacji
`softplus`. Pozostałe komórki `MEDIUM` zachowują density i kolor RadFoam.

Po usunięciu `SURFACE` topologię przebudowuje standardowy
`VoronoiTopologyController`. Wszystkie zachowane site'y, density i kolory są
trenowalne, również komórki naruszone i wyzerowane.

Translacja density:
`softplus(raw, beta=10)` RadFoam -> inverse-softplus FoamSurf z `beta=1`.
Maksymalny błąd numeryczny w runie referencyjnym wyniósł `9.54e-7`.
Bounding box pochodzi z configu RadFoam i wynosi `[-1.5, 1.5]^3`.

## Polityka populacji i optymalizacji

### Voronoi

`voronoi_adaptation.enabled: false` jest elementem baseline'u:

- brak densyfikacji i pruningu komórek;
- brak redukcji do dawnego limitu około 8k;
- stała populacja 159879 przez cały trening.

Nie oznacza to zamrożenia Voronoi. Site'y, density i kolory uczą się, a
topologia jest aktualizowana co 25 iteracji.

### Gaussy

Używana jest polityka Gaussian eviction z wcześniejszego runu 30:

- densyfikacja: iteracje 500--15000, co 250;
- gradient threshold: `0.0004`;
- opacity pruning: iteracje 3000--15000, co 250;
- minimalne opacity: `0.20` od pierwszego zdarzenia, bez rampy;
- brak opacity reset;
- brak limitu końcowej liczby Gaussów;
- `percent_dense: 0.02`;
- `gaussian_scale_regularization: 0.001`.

Nie ma medium-only warmupu, freezu Gaussów ani opacity-binarization loss.
Separacja wynika z inicjalizacji przestrzennej i późniejszego eviction/pruningu.

## Jak uruchomić baseline

### Zalecane uruchomienie przez Slurm

```bash
cd /shared/results/gmwilczn/foamsurf/foamsurf
sbatch scripts/train_radfoam_prior_run30.sbatch
```

Launcher zamawia jedną RTX 4090, 48 GB RAM, 8 CPU i limit 24 godzin. Katalog
wynikowy jest automatycznie tworzony jako:

`outputs/radfoam_prior/radfoam_prior_run30_<SLURM_JOB_ID>/`

Logi:

- `slurm_logs/radfoam_prior_<SLURM_JOB_ID>.out`;
- `slurm_logs/radfoam_prior_<SLURM_JOB_ID>.err`.

Monitorowanie:

```bash
squeue -j <SLURM_JOB_ID>
tail -f slurm_logs/radfoam_prior_<SLURM_JOB_ID>.out
tail -f slurm_logs/radfoam_prior_<SLURM_JOB_ID>.err
```

W&B domyślnie działa online w projekcie `foamsurf_radfoam_prior`. Tryb offline:

```bash
WANDB_MODE=offline sbatch scripts/train_radfoam_prior_run30.sbatch
```

Opcjonalna zmiana tylko katalogu nadrzędnego:

```bash
OUTPUT_ROOT=/wybrana/sciezka sbatch scripts/train_radfoam_prior_run30.sbatch
```

Do wiernej reprodukcji nie należy edytować configu ani seed. Zmiana
checkpointu, klasyfikatora, liczby komórek/Gaussów, pruning threshold lub
harmonogramu oznacza nowy eksperyment.

### Bezpośrednio na przydzielonym nodzie GPU

```bash
cd /shared/results/gmwilczn/foamsurf/foamsurf
/shared/results/gmwilczn/conda/envs/foamsurf/bin/python \
  scripts/run_radfoam_prior_training.py \
  --config configs/radfoam_prior_run30.yaml \
  --output-dir outputs/radfoam_prior/manual_reproduction \
  --wandb-name radfoam_prior_manual_reproduction \
  --wandb-mode offline
```

Entrypoint obsługuje też `--prepare-only`, `--iterations` i
`--eval-interval`, lecz run z tymi opcjami nie jest pełną reprodukcją.
`--prepare-only` wykonuje inicjalizację, ewaluację iteracji zero oraz zapis
`checkpoints/initialization.pt`, po czym kończy działanie.

## Artefakty runu

Najważniejsze pliki:

- `config.yaml`: zmaterializowany config;
- `radfoam_initialization_report.json`: pełny raport importu;
- `checkpoints/initialization.pt`: stan po handoffie;
- `checkpoints/best.pt`: checkpoint o najlepszym głównym PSNR;
- `checkpoints/last.pt`: stan po 30000 iteracjach;
- `summary.json`: status, czas i statystyki populacji;
- `metrics/train.csv` i `metrics/eval.csv`;
- `evaluation/iteration_XXXXXX/`: rendery diagnostyczne;
- `wandb/`: lokalne dane W&B.

Dla czterech kamer zapisywane są m.in. `*_panel.png`, `*_hybrid.png`,
`*_gaussian_only.png`, `*_voronoi_only.png`, `*_responsibility.png`,
`*_lego_lego.png`, `*_lego_clouds.png`, `*_error.png`,
`*_transmittance.png` i `*_layer_masks.png`.

## Wynik referencyjny: job 502753

Run ukończył 30000 iteracji. Właściwy trening trwał 2279 s (około 38 minut);
pełny wall time wraz z ewaluacją i obrazami był większy. Peak GPU allocation:
około 8698 MiB.

| Metryka | Iteracja 0 | Iteracja 30000 |
| --- | ---: | ---: |
| hybrid PSNR | 18.023 dB | **34.824 dB** |
| `lego_lego/psnr` | 15.011 dB | **35.231 dB** |
| `lego_clouds/psnr` | 30.574 dB | **36.039 dB** |
| Gaussian alpha na cloud-only | 0.43457 | **0.00933** |
| medium alpha na Lego-only | 0.01110 | **0.00737** |
| dominacja medium na cloud-only | 8.72% | **99.71%** |
| dominacja Gaussów na Lego-only | ~100% | **99.45%** |
| liczba Gaussów | 100000 | 105185 |
| liczba Voronoi | 159879 | 159879 |

Gaussian alpha w regionach cloud-only spadła około 46.6 razy, czyli o 97.85%.
Medium nie przejęło przy tym ciała stałego.

### Przełomowy moment treningu

Pierwszy mocny pruning nastąpił przy iteracji 3250:

- usunięto 78568 Gaussów w jednym zdarzeniu;
- populacja spadła z 120678 po iteracji 3000 do 44471;
- przy iteracji 4000 pozostało 40318 Gaussów;
- `lego_clouds/psnr`: 29.365 dB przy 3000 -> 37.117 dB przy 4000;
- medium dominance na cloud-only: 68.72% -> 90.40%;
- Gaussian alpha na cloud-only: 0.1834 -> 0.0835.

Gaussy następnie odrosły przez densyfikację, ale głównie w regionach ciała
stałego. Przy iteracji 15000, kiedy kończy się densyfikacja i pruning, było ich
105185. Ta liczba pozostała stała do końca, a separacja nadal się poprawiała.

### Porównanie z wcześniejszym eviction runem 30

Stary run osiągał na końcu `lego_lego/psnr = 24.032 dB` oraz
`lego_clouds/psnr = 25.846 dB`. Nowy baseline osiąga 35.231 dB i 36.039 dB:
poprawę około 11.20 dB dla ciała stałego i 10.19 dB dla medium. Hybrid PSNR
jest również wyższy od wcześniejszego najlepszego wyniku około 33.99 dB.

## Jak analizować reprodukcję

### 1. Integralność inicjalizacji

W `radfoam_initialization_report.json`:

- `source_cell_count == 371004`;
- `removed_surface_count == 211125`;
- `retained_voronoi_count == 159879`;
- `gaussian_count == 100000`;
- `affected_retained_count == 68363`;
- `reset_union_count == 84964`;
- `voronoi_population_adaptation == false`.

### 2. Populacje

`model/voronoi_count` musi stale wynosić 159879. W `summary.json` oczekujemy
`total_voronoi_densified == 0`, `total_voronoi_pruned == 0` i
`total_voronoi_failed_events == 0`. Duży pruning Gaussów około 3250 jest
oczekiwany, a nie jest awarią.

### 3. Właściwe metryki

Najważniejsze kolumny `metrics/eval.csv`:

- `eval/hybrid_psnr`;
- `lego_lego/psnr` i `lego_clouds/psnr`;
- `entanglement/gaussian_alpha_on_cloud_only`;
- `entanglement/medium_alpha_on_lego_only`;
- `entanglement/medium_dominance_on_cloud_only`;
- `entanglement/gaussian_dominance_on_lego_only`;
- `model/gaussian_count` i `model/voronoi_count`.

Uwaga: `eval/gaussian_only_psnr` i `eval/voronoi_only_psnr` porównują
pojedynczą reprezentację z pełnym targetem hybrydowym. Ich wartości około 19 dB
i 13 dB nie oznaczają słabej separacji. Warstwy oceniamy przez
`lego_lego/psnr`, `lego_clouds/psnr` i metryki entanglementu.

### 4. Obrazy

Należy obejrzeć iteracje 0, 3000, 4000, 15000 i 30000 dla wszystkich czterech
kamer. W poprawnym finale:

- `gaussian_only` pokazuje ostre LEGO bez chmur;
- `voronoi_only` pokazuje chmury bez rekonstrukcji obiektu;
- uporczywa chmura nie występuje w Gaussach;
- `hybrid` nie ma dziur po pruningu;
- podział jest spójny między kamerami.

### 5. Kryterium kwalifikacji

Reprodukcja powinna:

- ukończyć 30000 iteracji bez błędów adaptacji;
- zachować 159879 komórek Voronoi;
- uzyskać hybrid PSNR zbliżony do 34.82 dB;
- uzyskać około 35--36 dB na obu warstwach;
- mieć dominację medium na cloud-only i Gaussów na Lego-only bliską 1;
- wizualnie nie zawierać chmury w `gaussian_only`.

Sama wysoka jakość hybrydy nie wystarcza. Jeśli Gaussy ponownie rekonstruują
chmury, eksperyment nie odtwarza najważniejszej własności baseline'u.

## Ograniczenia

- Wynik referencyjny pochodzi obecnie z jednego seeda i jednej sceny.
- Używa silnego priora RadFoam trenowanego na tej samej scenie.
- 159879 komórek pozostaje przez cały trening; reprezentacja nie jest jeszcze
  zoptymalizowana pod względem rozmiaru.
- W `voronoi_only` pozostają pojedyncze drobne ślady koloru obiektu, mimo
  bardzo niskiego medium alpha na Lego-only.
- Powtarzalność, inne checkpointy/sceny i redukcję populacji trzeba badać
  osobno, z kontrolami.

## Kanoniczne odwołania

- config: `configs/radfoam_prior_run30.yaml`;
- launcher: `scripts/train_radfoam_prior_run30.sbatch`;
- entrypoint: `scripts/run_radfoam_prior_training.py`;
- import: `training/radfoam_import.py`;
- handoff: `training/radfoam_handoff.py`;
- przygotowanie checkpointu/labels:
  `docs/RADFOAM_CHECKPOINT_SEPARATION_HANDOFF.md`;
- output: `outputs/radfoam_prior/radfoam_prior_run30_502753/`;
- stdout/stderr: `slurm_logs/radfoam_prior_502753.out` i
  `slurm_logs/radfoam_prior_502753.err`.
