# Native FoamSurf Voronoi prior warmup

## Cel

Ten eksperyment jest pierwszym krokiem wycofania zewnętrznego prioru RadFoam.
FoamSurf ma sam nauczyć się gęstej reprezentacji Voronoia całej sceny, zanim
z checkpointu wyodrębnimy regiony `SURFACE`, `MEDIUM` i `EMPTY`, usuniemy
komórki powierzchniowe i zainicjalizujemy w ich miejscu Gaussy.

Ten etap **nie wykonuje jeszcze klasyfikacji, Otsu, usuwania komórek ani
spawnowania Gaussów**. Wytwarza wyłącznie natywny, analizowalny prior
Voronoia oraz serię checkpointów.

## Co dokładnie jest trenowane

- Dane treningowe: pełna scena
  `/shared/sets/datasets/foamsurf/lego/lego_both`.
- Tryb: `representation_mode: voronoi_only`.
- Cel: `dataset.train_target: both`, czyli pełny obraz, a nie samo
  `lego_clouds`.
- Gaussy: jeden techniczny placeholder wymagany przez wspólny model; jest
  zamrożony, nie jest renderowany i nie dostaje kroku optymalizatora.
- Voronoi: pozycje site'ów, density i RGB są uczone.
- Funkcja celu: `L1 + 0.2 * DSSIM`.
- Tło: czarne.
- Batch: pełny obraz, kamery bez zwracania.
- Długość: 5000 iteracji.
- Seed: 0.
- Bounding box: `[-1.5, 1.5]^3`, bez guard sites.
- Renderer: `near=0.1`, `far=1000`, próg treningowy `1e-4`, próg ewaluacji
  `0.01`, maksymalnie 1024 przecięcia.

Ścieżki `lego_lego` i `lego_clouds` są podane jako layer ground truth tylko
do diagnostyki. Nie zmieniają pełnoscenowego celu treningowego.

## Adaptacyjna liczba komórek

`350000` oznacza populację początkową, a nie wymuszony wynik końcowy.
Kontroler adaptacji działa od iteracji 500 do 5000 co 500 kroków:

- dzieli komórki o dużym błędzie i istotnym udziale w renderze;
- usuwa komórki długotrwale nieaktywne, małe i pozbawione aktywnego
  sąsiedztwa;
- po każdej zmianie przebudowuje triangulację i zachowuje stan optymalizatora;
- może zejść do bezpiecznika 100000 i wzrosnąć do budżetu 500000.

Granice 100k/500k są zabezpieczeniami, nie docelową liczbą komórek. To
sygnały rekonstrukcyjne sceny decydują o zmianach wewnątrz tego zakresu.
Trajektoria jest zapisywana w `metrics/train.csv` w kolumnach:

- `model/voronoi_count`;
- `voronoi_adaptation/densified` i `pruned`;
- `voronoi_adaptation/total_densified` i `total_pruned`;
- `voronoi_adaptation/active_fraction` i `observed_fraction`;
- `voronoi_adaptation/pre_psnr`, `post_psnr` i `psnr_delta`;
- `voronoi_adaptation/triangulation_failed`.

Dzięki temu dla innej sceny nie dziedziczymy arbitralnej liczby 371004 z
RadFoam. Nadal należy traktować górny budżet jako parametr sprzętowy; jeżeli
inna scena stale dobija do 500k, jest to informacja, że trzeba zwiększyć
budżet lub zmienić początkową rozdzielczość, a nie dowód zbieżności.

## Pliki implementacji

- Konfiguracja: `configs/voronoi_prior_warmup.yaml`.
- Runner: `scripts/run_voronoi_prior_warmup.py`.
- SBatch: `scripts/train_voronoi_prior_warmup.sbatch`.
- Kontrakt konfiguracji: `training/config.py`, sekcja
  `voronoi_warmup`.
- Checkpointy milestone: `training/trainer.py`, metoda
  `save_voronoi_warmup_milestone`.
- Testy kontraktowe: `tests/b3/test_voronoi_prior_warmup.py`.

## Środowisko i uruchomienie

Repozytorium:

```text
/shared/results/gmwilczn/foamsurf/foamsurf
```

Conda env używany przez job:

```text
/shared/results/gmwilczn/conda/envs/foamsurf
```

Job wymaga RTX 4090 i uruchamia się przez:

```bash
cd /shared/results/gmwilczn/foamsurf/foamsurf
sbatch scripts/train_voronoi_prior_warmup.sbatch
```

Opcjonalnie można wyłączyć wysyłanie do W&B bez zmiany pliku:

```bash
WANDB_MODE=offline sbatch scripts/train_voronoi_prior_warmup.sbatch
```

Logi Slurma trafiają do:

```text
slurm_logs/voronoi_warmup_<JOB_ID>.out
slurm_logs/voronoi_warmup_<JOB_ID>.err
```

Wynik trafia domyślnie do:

```text
outputs/voronoi_prior_warmup/voronoi_prior_warmup_<JOB_ID>/
```

## Artefakty

Runner zapisuje niezmienne checkpointy:

```text
checkpoints/iter_00000000.pt
checkpoints/iter_00000500.pt
checkpoints/iter_00001000.pt
checkpoints/iter_00002000.pt
checkpoints/iter_00003000.pt
checkpoints/iter_00004000.pt
checkpoints/iter_00005000.pt
```

Dodatkowo powstają `best.pt`, `last.pt`, `metrics/train.csv`,
`metrics/eval.csv`, rendery ewaluacyjne, `summary.json` i
`voronoi_warmup_report.json`. SBatch po treningu sprawdza kompletność
artefaktów, zakończenie 5000 iteracji, zakres liczebności i to, czy co
najmniej jedna rzeczywista adaptacja populacji została wykonana.

## Jak analizować wynik

Najpierw sprawdzamy poprawność procesu, dopiero potem jakość separacji.

1. `summary.json` musi mieć `status=completed`, końcową iterację 5000 oraz
   dodatnią rewizję adaptacji.
2. W `metrics/train.csv` oglądamy liczbę komórek i skoki PSNR na zdarzeniach
   adaptacji. Powtarzające się `triangulation_failed=1`, duże ujemne
   `psnr_delta` albo natychmiastowe dobicie do limitu są błędem lub źle
   ustawioną adaptacją.
3. Porównujemy rendery i metryki checkpointów 0/500/1000/2000/3000/4000/5000.
   Interesuje nas najwcześniejszy checkpoint, w którym można stabilnie
   odróżnić bryłę LEGO od chmury — nie automatycznie checkpoint o najwyższym
   PSNR.
4. Dla każdego checkpointu w następnym etapie uruchomimy ten sam pipeline
   diagnostyczny co dla RadFoam: klasy `SURFACE`, `MEDIUM`, `EMPTY` oraz
   rendery pełne, klasowe i occupancy-style.
5. Sukces oznacza, że region LEGO tworzy spójną klasę powierzchniową, chmura
   pozostaje w klasie medium, a empty nie niesie zauważalnego obrazu.
6. Dopiero po wyborze checkpointu wykonamy handoff: usunięcie `SURFACE`,
   wyzerowanie density naruszonych sąsiadów, sampling Gaussów w usuniętym
   volume i trening hybrydowy.

Sam wysoki PSNR warmupu nie jest kryterium sukcesu: Voronoi celowo próbuje
odtworzyć także ciało stałe. W tym eksperymencie pytamy, czy jego nauczona
geometria i parametry tworzą wystarczający sygnał przestrzenny do późniejszej
separacji.
