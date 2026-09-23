# Wyniki sweepu adaptacji 508307 — 2026-09-23

Plan, konfiguracje i uruchamianie: [JOINT_ADAPTATION_SWEEP.md](JOINT_ADAPTATION_SWEEP.md).

## Status i ograniczenia

Kontrola: `outputs/end_to_end/508222/joint`.
Sweep: `outputs/joint_adaptation/508307/run_<ID>`.
Wyniki pochodzą z `metrics/eval.csv`, `metrics/train.csv`, `summary.json`,
paneli `evaluation/iteration_030000/*_lego_{lego,clouds}.png` oraz logów
`slurm_logs/joint_adaptation_508307_<ID>.{out,err}`.
Wyniki i checkpointy pozostają lokalne; nie są częścią repozytorium.

Runy 0–2 ukończyły 30k iteracji. Runy 3–6 NIE są zweryfikowanymi wariantami
pełnoskalowego treningu: padły przy pierwszej densyfikacji Voronoi, odpowiednio
w iteracji 500 (3/4) i 2500 (5/6). Błąd pozostaje nierozwiązany:

```
RuntimeError: CUDA call at .../src/delaunay/sorted_map.cuh:99 failed:
an illegal memory access was encountered
```

Stos: `VoronoiAdaptationController.apply_plan` → `replace_population` →
`foamsurf_topology.Triangulation(candidate_sites)`. Nie jest to komunikat OOM;
dokładna przyczyna wymaga osobnej diagnostyki. Mały smoke test nie pokrył tego
przypadku produkcyjnego. Nie należy interpretować runów 3–6 jako negatywnego
wyniku jakościowego densyfikacji ani traktować ich jako gotowych do ponowienia.

## PSNR i separacja

Wszystkie ukończone warianty osiągają najlepszy średni PSNR ewaluacyjny w 30k.

| Run | Hybrid PSNR | Gaussy vs LEGO PSNR | Medium vs chmury PSNR | Medium alpha na LEGO-only | Gaussian alpha na cloud-only |
|---|---:|---:|---:|---:|---:|
| Kontrola 508222 | 34.879 | 36.420 | 36.737 | 0.619% | 0.0278% |
| 0: stała populacja + binaryzacja/pruning Gaussianów | 34.537 | 36.026 | 37.366 | 0.521% | 0.0301% |
| 1: pruning Voronoi | 34.874 | 36.394 | 34.075 | 1.959% | 0.0254% |
| 2: pruning Voronoi + binaryzacja/pruning Gaussianów | 34.536 | 36.009 | 34.913 | 1.206% | 0.0264% |

Alpha oznacza średnią opacity niewłaściwej reprezentacji w regionie wyznaczonym
z masek GT, nie procent błędnie sklasyfikowanych prymitywów. Są to wskaźniki
projekcyjne, nie pełna ocena semantycznego podziału 3D. Regiony overlap nie wchodzą
do masek LEGO-only/cloud-only. Definicje: `evaluation/layer_metrics.py`.

Wizualnie w medium-only runów 1/2 pojawiają się żółte i jasne fragmenty LEGO,
wyraźniejsze niż w kontroli i runie 0 (sprawdzono m.in. widoki r_000 i r_002).
Pruning Voronoi zachowuje jakość renderu sumarycznego, ale pogarsza separację.
Run 1 redukuje populację z 297912 do 189545 komórek (około 36%); run 2 do 189634.

Run 0 ma najczystsze medium spośród nowych wariantów, przy stracie 0.343 dB
hybrid PSNR względem kontroli. Nie dominuje kontroli we wszystkich kryteriach:
kontrola lepiej rekonstruuje LEGO i ma minimalnie mniejszą alpha Gaussianów na
chmurach. Foreground PSNR chmur jest niemal identyczny: 30.958 dB w kontroli
i 30.954 dB w runie 0. Poprawa pełnokadrowej metryki warstwy medium jest więc
spójna przede wszystkim z mniejszym zanieczyszczeniem poza chmurami.

## Efekt binaryzacji

W kontroli końcowa średnia opacity Gaussianów wynosi 0.741, a 46.0% Gaussianów
ma opacity > 0.9. W runie 0: odpowiednio 0.975 i 93.4%. Po 15k dodatkowy pruning
usunął 3593 Gaussy w runie 0 oraz 3470 w runie 2. To potwierdza działanie pakietu,
ale nie rozdziela wpływu samej binaryzacji i wydłużonego pruningu.

Separacja medium pogarsza się w czasie: w runie 1 alpha medium na LEGO-only
rośnie z 0.145% w 1k do 1.679% w 15k i 1.959% w 30k. W kontroli kończy na 0.619%,
a w runie 0 na 0.521%. Hipoteza: usuwanie nieaktywnych komórek zmienia granice
pozostałych i pozwala medium wejść w obszary solid. Nie jest to jeszcze dowód
przyczyny — wymaga osobnego sprawdzenia geometrii i kolejnych eventów pruningu.

## Punkty odniesienia na następny etap

- Kontrola 508222: najlepszy PSNR pełnej sceny i warstwy solid.
- Run 0: kandydat przy priorytecie czystego medium i wysokiej opacity Gaussianów.
- Nie zastępować kontroli runem 1 wyłącznie na podstawie podobnego hybrid PSNR.
- Nie wyciągać wniosków o jakości densyfikacji przed naprawą błędu runów 3–6.
- Wyniki dotyczą jednej sceny i jednego seeda; różnice nie mają jeszcze oceny
  powtarzalności na innych inicjalizacjach/scenach.

Ta dokumentacja zapisuje stan eksperymentu. Nie wprowadza napraw treningu ani
nie zmienia domyślnego pipeline'u end-to-end.
