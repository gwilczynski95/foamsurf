# B0.1: diagnostyka odpowiedzialności

B0.1 nie zmienia funkcji celu ani parametrów modelu B0. Trening nadal używa
wyłącznie jednego hybrid RGB MSE i jednego backwardu. Nowy kod obserwuje, jak
Gaussy i medium dzielą między siebie rekonstrukcję, żeby entanglement był
mierzalny przed densyfikacją albo regularyzacją.

## Uruchomienie na obecnym Lego

~~~bash
cd /shared/results/gmwilczn/foamsurf/foamsurf
sbatch scripts/train_b01_lego.sbatch
~~~

Config to configs/b01_lego.yaml. Diagnostykę kontroluje:

~~~yaml
diagnostics:
  enabled: true
  save_per_primitive: true
  contribution_epsilon: 1.0e-6
~~~

Domyślna wartość w kodzie to enabled=false, dlatego stary config B0 zachowuje
dotychczasową ścieżkę i koszt. B0.1 włącza ją jawnie.

## Co mierzy renderer

Opcjonalny kernel jeszcze raz przechodzi przez ten sam depth-ordered strumień
zdarzeń. Nie zmienia RGB ani autogradu i nie alokuje macierzy
rays x primitives. Zwraca:

- [R,8]: suma wag Gaussianów, suma wag medium, długość drogi w medium,
  długość ważona wejściową transmitancją, liczba Gaussian events, liczba
  segmentów medium, końcowa transmitancja i flaga limitu traversalu;
- [G,4]: contribution, visibility/event count, suma alpha i maksymalny błąd
  raya dotykającego prymitywu;
- [V,5]: contribution, visibility/segment count, path length, path length
  ważony transmitancją i maksymalny błąd raya przechodzącego przez komórkę.

Contribution jest sumą faktycznej wagi compositingu T_before * alpha.
Dla pojedynczego raya kontrolujemy:

~~~text
gaussian_weight + medium_weight + final_transmittance = 1
~~~

Maksymalny błąd per prymityw jest średnim absolutnym błędem RGB danego raya.
Redukcje per-prymityw są atomowe. Statystyki są diagnostyczne i odłączone od
grafu autograd.

Bazowy eval/hybrid_fps jest mierzony na renderze bez instrumentacji.
eval/diagnostics_fps mierzy osobny pełny przebieg z instrumentacją, dzięki
czemu FPS B0 i B0.1 pozostają porównywalne.

## Artefakty

Każda evaluation/iteration_XXXXXX zawiera:

- per_view.csv z PSNR/SSIM, gainem hybrid nad obiema gałęziami, flagami
  przegranych widoków, percentylami błędu i odpowiedzialnością per-ray;
- gaussian_primitives.csv i medium_primitives.csv;
- standardowy panel porównawczy;
- obraz responsibility:
  Gaussian contribution | medium contribution | medium fraction;
- osobny absolute-error i transmittance render.

metrics/eval.csv zawiera średnie, medianę/p05/min/max hybrid PSNR, odsetek
widoków przegrywanych przez hybrid, aktywne frakcje prymitywów, całkowite
contribution, koncentrację top-1, maksymalne błędy i kontrolę zachowania masy.
metrics/train.csv zawiera percentyle opacity/scale/density oraz percentyle,
maksimum i aktywną frakcję gradient magnitude per prymityw.

W&B dostaje te same skalary, comparison/responsibility panels oraz histogramy
contribution, visibility, path length, max ray error i gradient magnitude.

## Interpretacja na zwykłym Lego

Ten dataset nie ma ground-truth podziału surface/medium. B0.1 może precyzyjnie
pokazać, że obie gałęzie odpowiadają za ten sam obszar, ale nie rozstrzyga,
która gałąź semantycznie ma rację. Obecne Lego służy do regresji względem B0 i
znalezienia failure views.

Dataset z osobnymi, współrejestrowanymi warstwami hybrid/surface/medium
umożliwi później decomposition metrics. Warstwy pomocnicze pozostaną tylko
walidacją; bazowy trening nadal nie będzie ich używał.

## Kryteria regresji

~~~bash
CUDA_VISIBLE_DEVICES=0 python -m pytest -q   renderer/tests/test_b01_diagnostics.py
~~~

Testy sprawdzają:

1. identyczne RGB, alpha i wszystkie gradienty przy statistics on/off;
2. wyłącznie zredukowane bufory [R,8], [G,4], [V,5];
3. zachowanie sumy wag compositingu;
4. brak dodatkowego launchu i outputu, gdy diagnostyka jest wyłączona;
5. obecność pól responsibility/gradient w CSV i W&B;
6. komplet artefaktów w end-to-end train/resume smoke teście.
