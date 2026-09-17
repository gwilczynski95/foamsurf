# Medium handoff sweep: warmup, Gaussian freeze and medium LR boost

## Cel

Ten eksperyment jest drugim krokiem disentanglementu po sweepie Gaussian eviction.
Punktem wyjścia są trzy wybrane runy poprzedniej fazy: `30`, `41` i `61`.
W każdym z nich pruning pomaga usunąć część Gaussów z chmur, ale medium nie zawsze
zdąża przejąć zwolnione miejsce. Testujemy dwa sposoby stworzenia medium przewagi
optymalizacyjnej:

1. dodatkowy warmup przed właściwym treningiem, podczas którego obraz `lego_both`
   widzi wyłącznie Voronoi;
2. czasowe zamrożenie Gaussów po zakończeniu densyfikacji, opcjonalnie połączone
   ze zwiększeniem LR density i color medium.

Nie testujemy nowego `K`: wszystkie runy używają kwalifikowanego renderera `K=64`.
Główna faza zawsze ma dokładnie 30 000 iteracji i seed 0.

## Trzy bazy

- `b30`: poprzedni indeks 30, hard pruning `0.20` od 3k do 15k, bez opacity
  binarization;
- `b41`: indeks 41, próg `0.01 -> 0.15`, start 1k, rampa 5k, pruning do 30k;
- `b61`: indeks 61, hard pruning `0.10` od 3k do 30k oraz opacity
  binarization `1e-4`, start 5k, cosine ramp 3k, utrzymana do końca.

Bazowe override'y są importowane bezpośrednio z manifestu poprzedniego sweepu,
więc nie ma ręcznie przepisanej, potencjalnie rozbieżnej konfiguracji.

## Dokładna semantyka implementacji

### Prefix warmup

Warmup jest **dodatkowy**. Przykład `prefix=3000` wykonuje 3000 kroków prefiksu,
a następnie pełne kroki główne `1..30000`. Harmonogram densyfikacji, pruningu,
binaryzacji i LR fazy głównej nie jest przesuwany ani skracany.

W `medium_only`:

- targetem nadal jest wyłącznie `lego_both`;
- renderowane i optymalizowane jest tylko medium;
- Gaussiany istnieją, ale nie są renderowane ani aktualizowane;
- liczba komórek jest stała: nie działa population densification/pruning Voronoi;
- site'y mogą się poruszać, a connectivity jest aktualizowane normalnie co 25 kroków;
- po prefiksie learned sites/density/color zostają zachowane.

Domyślnie stan Adama medium jest zerowany przed główną fazą (`reset`). Osobny blok
`retain` sprawdza, czy zachowanie momentów optymalizatora pomaga lub destabilizuje.
Po każdym prefiksie zerowane są statystyki kontrolerów adaptacji oraz ich lokalne
revision counters.

Przed prefiksem zapisywany jest dokładny stan Python/NumPy/Torch/CUDA RNG po
inicjalizacji modelu. Przed główną iteracją 1 jest on odtwarzany i tworzony jest
świeży sampler. Dzięki temu kolejność kamer oraz losowość adaptacji w fazie głównej
są porównywalne z kontrolą bez prefiksu. Nie jest to tylko ponowne ustawienie seeda.

`joint` prefix jest kontrolą kosztu obliczeniowego: obie reprezentacje uczą się
przez dodatkowe 1k lub 3k kroków, ale bez population adaptation. Przed fazą główną
zerowane są optymalizatory obu reprezentacji. Porównanie `medium_only` z `joint`
odpowiada, czy efekt wynika z head startu medium, czy po prostu z większej liczby
kroków.

Dla runów z prefiksem powstają:

- `metrics/warmup.csv` z lokalną osią `prefix_iteration`;
- ewaluacja pełnej reprezentacji na głównej iteracji `0`;
- `checkpoints/prefix.pt`, z którego można wznowić już główną fazę bez ponownego
  wykonywania warmupu.

### Gaussian freeze

Freeze `[start, start + duration)` jest domknięty z lewej strony. Dla startu 15000
i duration 1000 zamrożone są kroki `15000..15999`, a krok 16000 znów aktualizuje
Gaussy.

Podczas freeze:

- Gaussy nadal są renderowane i wpływają na residual;
- Gaussian optimizer nie wykonuje `step()`;
- nie wolno densyfikować Gaussów;
- zaplanowany opacity pruning nadal może usuwać Gaussy. To jest celowe, szczególnie
  dla baz 41 i 61, gdzie pruning trwa do 30k;
- medium jest uczone normalnie;
- mnożniki LR, jeśli skonfigurowane, działają tylko wewnątrz okna freeze i są
  nakładane na normalny LR wynikający z harmonogramu danej głównej iteracji;
- LR site'ów pozostaje `x1` we wszystkich głównych testach. Wzmacniamy density i/lub
  color, aby nie prowokować awarii topologii przez agresywny ruch site'ów.

Ewaluacja jest wymuszana w `start-1`, `start`, `end-1` i `end` niezależnie od
regularnego interwału 1000. Pozwala to zmierzyć stan tuż przed freeze, po pierwszym
kroku, po ostatnim kroku i po odmrożeniu.

## Naprawa odporności Voronoi

Poprzedni sweep miał 18 przerwanych runów: 13 `ambiguous triangulation` i 5
`divergent growth iterations`. Nie był to OOM. Wszystkie wyjątki wychodziły z
`replace_population`.

Nowa implementacja:

1. nadal buduje kandydacką triangulację przed mutacją modelu i Adama;
2. najpierw perturbuję deterministycznie nowe dzieci;
3. po kilku nieudanych próbach perturbuję mikroskopijnie wszystkie nie-guard
   site'y, bo degeneracja może istnieć wśród survivors;
4. guard points nigdy nie są przesuwane;
5. jeśli 12 prób nie wystarczy, całe zdarzenie population replacement jest
   atomowo pomijane zamiast przerwać trening;
6. po pominięciu zerowane są nagromadzone statystyki i low-activity streak, aby
   następny event nie odtwarzał natychmiast tego samego wadliwego planu.

Pominięcia są widoczne w:

- `voronoi_adaptation/skipped`;
- `voronoi_adaptation/triangulation_failed`;
- `voronoi_adaptation/triangulation_retries`;
- `voronoi_adaptation/total_failed_events`;
- `summary.json: total_voronoi_failed_events`.

Pominięte zdarzenie nie zmienia parametrów, optimizer state, populacji, topologii
ani revision counter. Run z dużą liczbą pominięć nie powinien być traktowany jako
w pełni porównywalny bez dodatkowej inspekcji.

## Macierz 150 runów

| Indeksy | N | Rodzina | Zmieniane czynniki |
|---|---:|---|---|
| 0-2 | 3 | `base_control` | dokładne bazy 30, 41, 61 |
| 3-20 | 18 | `medium_prefix_reset` | 3 bazy x prefix `{250,500,1000,2000,3000,5000}`, reset Adam |
| 21-29 | 9 | `medium_prefix_retain` | 3 bazy x prefix `{1000,2000,3000}`, retain Adam |
| 30-65 | 36 | `freeze_only` | 3 bazy x start `{15000,17500,20000}` x duration `{500,1000,2000,4000}` |
| 66-119 | 54 | `freeze_density_color_boost` | 3 bazy x start `{15000,20000}` x duration `{1000,2000,4000}` x boost `{2,5,10}` |
| 120-125 | 6 | `component_boost` | 3 bazy x `{density-only,color-only}`, start 15k, duration 2k, boost x5 |
| 126-143 | 18 | `prefix_freeze_boost` | 3 bazy x prefix `{1k,2k,3k}` x freeze duration `{1k,2k}`; start 15k, density+color x5 |
| 144-149 | 6 | `joint_prefix_control` | 3 bazy x joint prefix `{1k,3k}`, reset obu optimizerów |

Pełny, indeksowany manifest jest w
`training/medium_handoff_sweeps.py`. Każdy ukończony run zapisuje także
`sweep_spec.json`.

## Hipotezy

- Krótki medium-only prefix powinien zwiększyć początkową zdolność medium do
  generowania niskoczęstotliwościowego składnika, ale zbyt długi prefix zacznie
  kodować bryłę Lego i może utrudnić późniejszą separację.
- Freeze po 15k powinien działać lepiej niż przed końcem densyfikacji, bo nie
  pozwala Gaussom natychmiast odtworzyć usuniętych elementów chmur.
- Sam freeze może nie wystarczyć, jeżeli malejący LR medium jest zbyt mały;
  density+color boost powinien zwiększyć przejęcie chmur.
- `x10` może dać krótkotrwałą poprawę medium, ale grozi overshootem, wzrostem
  mętności w bryle i pogorszeniem hybrid PSNR.
- Kompozycja ma sens tylko wtedy, gdy efekty warmupu i późnego freeze są co
  najmniej częściowo niezależne.

## Jak analizować wyniki

### 1. Kompletność i stabilność

Najpierw policzyć runy z `summary.json` i `status=completed`; oczekujemy 150.
Sprawdzić stderr pod kątem CUDA OOM, błędów renderera i tracebacków. Osobno
zestawić `total_voronoi_failed_events`. Pominięcie pojedynczego eventu jest
recoverable; wiele pominięć może zmienić efektywny harmonogram populacji.

Należy porównywać także `training_seconds` i `prefix_training_seconds`. Runy z
warmupem mają intencjonalnie więcej kroków; dlatego nie wolno przypisywać efektu
head startowi bez kontroli `joint_prefix_control`.

### 2. Najpierw porównania sparowane w obrębie tej samej bazy

Nie mieszać baz 30/41/61 w pierwszym rankingu. Dla każdej bazy:

1. warmup-reset przeciw `base_control`, a potem krzywa dawka-efekt długości;
2. retain przeciw reset dla tych samych `{1k,2k,3k}`;
3. medium-only 1k/3k przeciw joint-prefix 1k/3k;
4. każdy `freeze_only` przeciw base control;
5. każdy boost przeciw dokładnie pasującemu `freeze_only` o tym samym start i
   duration — to izoluje mnożnik LR;
6. density-only i color-only przeciw freeze-only oraz density+color x5 dla
   `start=15k,duration=2k`;
7. kompozycje przeciw odpowiadającemu warmupowi oraz odpowiadającemu freeze+x5.

Dopiero po tych kontrastach sprawdzić, czy wniosek replikuje się na co najmniej
dwóch z trzech baz.

### 3. Metryki trajektorii, nie tylko finalny PSNR

Dla prefiksu analizować `metrics/warmup.csv`, ewaluację iteration 0 i zmianę od
0 do 1k/3k/5k głównej fazy. Szukamy, czy Gaussy przejmują scenę wolniej oraz czy
medium-only PSNR nie zapada się natychmiast po ich włączeniu.

Dla freeze używać wymuszonych punktów `start-1`, `start`, `end-1`, `end`, a także
3k i 5k kroków po `end`. Najważniejsze są:

- `eval/primary_psnr` i SSIM — jako ograniczenie jakości;
- `eval/gaussian_only_psnr` oraz `eval/voronoi_only_psnr`;
- metryki warstw `lego_lego/*`, `lego_clouds/*`, `entanglement/*` i `regions/*`;
- Gaussian count, rozkład opacity i udział opacity `<0.1`, `>0.5`, `>0.9`;
- density p50/p95/p99 i udział density `>1`;
- contribution/alpha/weight obu reprezentacji w masce chmur i w masce bryły;
- faktyczne LR z `optimization/medium_{sites,density,color}_lr`;
- `gaussian/handoff_frozen`, aby zweryfikować granice harmonogramu.

Pożądany sygnał to wzrost udziału/PSNR medium na chmurach podczas freeze, który
utrzymuje się po odmrożeniu, bez wzrostu medium na bryle. Sam wzrost globalnej
gęstości lub Voronoi-only PSNR nie jest dowodem disentanglementu.

### 4. Kryterium wyboru

Kandydat musi równocześnie:

1. zachować akceptowalny hybrid PSNR/SSIM względem własnej bazy;
2. zmniejszyć Gaussian contribution/alpha w chmurach;
3. zwiększyć medium contribution i jakość warstwy chmur;
4. nie zwiększyć medium contribution w bryle Lego ani nie rozmyć krawędzi i
   tekstur w obrazach;
5. zachować efekt kilka tysięcy kroków po odmrożeniu;
6. nie opierać wyniku na licznych pominięciach adaptacji Voronoi.

Finalnie należy obejrzeć te same validation views dla bazy i kandydata w punktach
granicznych oraz 30k: hybrid, Gaussian-only, Voronoi-only, warstwy, alpha i
responsibility heatmaps. Ranking liczbowy służy do shortlisty, nie zastępuje
inspekcji przestrzennej chmur.

## Uruchomienie

```bash
cd /shared/results/gmwilczn/foamsurf/foamsurf
sbatch scripts/sweep_medium_handoff.sbatch
```

Domyślny output: `outputs/medium_handoff/med_handoff_<JOBID>_<INDEX>`.
Domyślnie uruchamia się maksymalnie 8 zadań równolegle na `rtx4090_batch`, po
jednym GPU na task. Można wyłączyć W&B lub zmienić katalog wyników:

```bash
WANDB_MODE=disabled OUTPUT_ROOT=/inna/sciezka \
  sbatch scripts/sweep_medium_handoff.sbatch
```

Kontrola pojedynczego wariantu bez tworzenia trenera/GPU:

```bash
/shared/results/gmwilczn/conda/envs/foamsurf/bin/python \
  scripts/run_medium_handoff_sweep.py \
  --config configs/baseline_clean_joint.yaml --index 66 \
  --output-dir /tmp/unused --wandb-name inspect --wandb-mode disabled \
  --print-spec
```
