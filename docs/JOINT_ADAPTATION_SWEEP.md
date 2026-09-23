# Adaptacja Voronoi po natywnym handoffie — 7 eksperymentów

## Cel i kontrola

Sprawdzamy, czy po udanym rozdzieleniu LEGO/chmur możemy adaptować populację
Voronoi bez utraty separacji. Drugą osią jest binaryzacja opacity Gaussianów
wraz z przedłużeniem ich pruningu. To nie jest nowy sweep warmupu ani metod podziału.

Kontrola to istniejący run **508222**, katalog `outputs/end_to_end/508222/joint`.
Nie uruchamiamy go ponownie i nie modyfikujemy jego plików ani kodu zwykłego Trainer.
W kontroli Voronoi **uczy parametry**: stała jest liczba komórek, a nie ich pozycje,
gęstości czy kolory. „Fixed” poniżej również oznacza stałą populację, nie zamrożenie.

Każdy nowy run ładuje dokładnie ten sam checkpoint po natywnym warmupie 5k
i handoffie `weighted_valley_w1_s2`:

`/shared/results/gmwilczn/foamsurf/foamsurf/outputs/end_to_end/508222/joint/checkpoints/initialization.pt`

To iteracja **0 etapu hybrydowego**, 100 000 Gaussianów i 297 912 pozostałych
komórek. Nie powtarzamy warmupu, klasyfikacji ani losowania Gaussianów. Nie jest
to wznowienie wytrenowanego modelu. Loader odrzuca iterację różną od 0 i niepuste
stany optymalizatorów. Konfiguracja bazowa pochodzi z samego checkpointu.

## Macierz

| ID | Populacja Voronoi | Gaussy |
|---|---|---|
| 0 | stała | binaryzacja + przedłużony pruning |
| 1 | tylko pruning | jak w kontroli |
| 2 | tylko pruning | binaryzacja + przedłużony pruning |
| 3 | pruning + densyfikacja od początku | jak w kontroli |
| 4 | pruning + densyfikacja od początku | binaryzacja + przedłużony pruning |
| 5 | pruning + densyfikacja po 2k | jak w kontroli |
| 6 | pruning + densyfikacja po 2k | binaryzacja + przedłużony pruning |

„Od początku” oznacza zbieranie statystyk od iteracji 1 i pierwszy event w 500.
Dla opóźnienia: statystyki od 2001, pierwszy event w 2500. LR medium i uczenie
jego parametrów działają od iteracji 1 także w wariantach opóźnionych.

Wszystkie runy mają 30k iteracji hybrydowych, seed 0, te same dane, renderer K64,
LR i ich harmonogramy, rozdzielczość, metryki i ewaluację co 1000 iteracji.
Nie dodajemy freeze, LR boost, opaque-if-used, graph TV ani density cap.
Wspólna inicjalizacja i seed nie gwarantują bitowej identyczności z kontrolą:
odtwarzanie RNG przez istniejący Trainer oraz operacje CUDA mogą zmieniać trajektorię.

## Parametry adaptacji

Konfiguracja: `configs/joint_adaptation_sweep.yaml`.

- Eventy Voronoi co 500 iteracji, ostatni w 29 500 (koniec zakresu jest wyłączny).
- Pruning do 2% aktualnej populacji na event, wyłącznie po spełnieniu warunków
  istniejącego kontrolera: niska aktywność, brak aktywnych sąsiadów, mała komórka,
  2 kolejne eventy niskiej aktywności, z zachowaniem guard mask.
- Warianty densyfikujące dodają do 2% na event, kierując się błędem renderingu,
  wkładem, gradientami i geometrią komórek. To limit, a nie gwarancja wzrostu.
- Minimum: `ceil(0.5 * N0)` = 148 956. Maksimum z densyfikacją:
  `floor(1.5 * N0)` = 446 868. W pruning-only maksimum N0.
- Contribution threshold 1e-6, min visibility 1, radius fraction 0.02,
  child jitter fraction 0.025; bez zatrzymania pruningu po osiągnięciu limitu.
- Densyfikacja zachowuje istniejącą politykę ograniczania nowych dzieci do boxa
  sceny. Specjalny adapter rozszerza granice **tylko na czas replace_population**,
  by retry triangulacji nie ściągał zachowanych outer sites do boxa sceny.
  Pierwotne granice wracają także przy wyjątku. Retry nadal może lekko perturbować
  pozycje; nie gwarantujemy ich dokładnej niezmienności podczas naprawy triangulacji.

Binaryzacja: waga docelowa 0.001, start 1000, rampa cosine długości 3000,
pełna siła od 4000 do końca. Jest to zachęta do opacity 0/1, nie twardy clamp.
Próg pruningu Gaussianów pozostaje **0.20**, od 3000 co 250; w binarnych wariantach
do 30k zamiast 15k. Ostatni event w 29 750, gdyż zakres jest wyłączny.
Densyfikacja Gaussianów nadal kończy się w 15k. Dwie zmiany Gaussianów testujemy
jako pakiet — ten sweep sam nie rozdziela ich indywidualnych efektów.

## Kod i izolacja

- `training/joint_adaptation_sweep.py`: warianty, kontrola inicjalizacji,
  okno zbierania statystyk, zabezpieczenie outer sites.
- `scripts/train_joint_adaptation.py`: zapis manifestu i konfiguracji, wczytanie
  inicjalizacji, instalacja kontrolera sweepu, standardowy `Trainer.fit()`.
- `tests/joint_adaptation/test_contract.py`: CPU — warianty, roundtrip YAML,
  brak mutacji bazy, odrzucanie złych checkpointów, obsługa granic i wyjątków.
- `tests/joint_adaptation/test_gpu.py` + `gpu_worker.py`: rzeczywisty mały natywny
  warmup/handoff, potem wszystkie 7 ścieżek po 6 iteracji w izolowanych procesach.
  Sprawdzane są aktualizacje optymalizatorów, skończoność parametrów, wzrost
  populacji w wariantach densyfikujących oraz brak nieudanych eventów triangulacji.
- Regresja `tests/b2/test_topology_replace_population.py` biegnie oddzielnie od
  GPU, ponieważ używa atrapy rozszerzenia topologii.

Smoke sprawdza funkcjonalność, **nie** gwarantuje braku OOM przy 446k komórek
i docelowej liczbie Gaussianów. Nie zmieniamy oryginalnego pipeline end-to-end.

## Uruchomienie: test, następnie zależny sweep

Repo: `/shared/results/gmwilczn/foamsurf/foamsurf`.
Env: `/shared/results/gmwilczn/conda/envs/foamsurf`.
Skrypty ustawiają środowisko same. GPU: `rtx4090_batch`, 1 GPU/run, 64 GB RAM,
8 CPU, maksymalnie 3 runy jednocześnie. W&B online: **foamsurf_end_to_end**,
grupa `joint-adaptation-<ARRAY_JOB_ID>`; potrzebne istniejące uwierzytelnienie W&B.

```bash
cd /shared/results/gmwilczn/foamsurf/foamsurf
test_job=$(sbatch --parsable scripts/test_joint_adaptation.sbatch)
test_job=${test_job%%;*}
sbatch --dependency=afterok:${test_job} \
  --export=ALL,JOINT_ADAPTATION_TEST_CERTIFICATE=$PWD/outputs/joint_adaptation_tests/${test_job}/passed.json \
  scripts/train_joint_adaptation.sbatch
```

Sweep dodatkowo sprawdza certyfikat z RTX4090, zestaw testów i fingerprint kodu.
Nie zmieniaj źródeł/configów między testem a startem sweepu. Zmiana wymaga nowego
testu. Testy GPU nie zostały uruchomione lokalnie podczas przygotowania.
Launcher nie nadpisuje niepustych wyników i nie implementuje automatycznego resume.

## Wyniki i analiza

Wyniki: `outputs/joint_adaptation/<ARRAY_JOB_ID>/run_<0..6>/`.
Manifest `sweep_manifest.json` zawiera ścieżkę i SHA256 źródłowego checkpointu,
ID wariantu i ustawienia; `source_config.yaml` i `config.yaml` pokazują różnice.
Metryki: `metrics/train.csv`, `metrics/eval.csv`; checkpointy i standardowe
wizualizacje ewaluacji znajdują się w katalogu runu. Logi Slurm:
`slurm_logs/joint_adaptation_<ARRAY_JOB_ID>_<TASK_ID>.{out,err}`.

1. Najpierw sprawdzić kompletność: finalna iteracja 30k, NaN/OOM/runaway,
   nieudane przebudowy, przebieg liczby komórek i Gaussianów, czas i pamięć.
2. Porównywać te same widoki i iteracje: 1k, 3k, 5k, 10k, 15k, 20k, 30k.
   Porównanie z kontrolą 508222 dotyczy iteracji hybrydowych, bez doliczania 5k warmupu.
3. Oglądać razem render pełny, Gaussian-only oraz medium-only — zwłaszcza
   problematyczną chmurę i kontury LEGO. Ładny render sumy nie dowodzi separacji.
4. Użyć istniejących metryk warstw względem `lego_lego` i `lego_clouds`, wraz
   z PSNR/SSIM/LPIPS pełnego obrazu, jeśli dana metryka jest raportowana.
   Same udziały energii reprezentacji nie wystarczają do oceny semantycznej separacji.
5. Sprawdzić skoki PSNR przed/po eventach Voronoi, liczbę split/prune i retry,
   opacity Gaussianów oraz liczbę usuniętych Gaussianów po 15k.
   Jeśli pruning nie usuwa nic, wariant nie dowodzi skuteczności pruningu — sprawdzić
   observed/active fraction i kryterium rozmiaru komórki.
6. Porównania par: 1 vs 2, 3 vs 4, 5 vs 6 mierzą wpływ pakietu Gaussianów.
   3 vs 5 oraz 4 vs 6 mierzą opóźnienie adaptacji. 0 vs kontrola sprawdza pakiet
   Gaussianów bez adaptacji medium. 1/3/5 vs kontrola oceniają samą adaptację.
7. Sukces: zachowana lub poprawiona separacja, brak powrotu chmur do Gaussianów
   i LEGO do medium, brak dziur po pruningu, dobra jakość obrazu przy rozsądnym koszcie.
   Pojedynczy seed to screening; najlepsze warianty wymagają potem replikacji.
