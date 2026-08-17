# B1: densyfikacja i pruning Gaussianów

> **Dokument historyczny.** Opisuje politykę używaną we wcześniejszych
> sweepach i nie jest już domyślną konfiguracją B1. Aktualny profil,
> odtwarzający efektywne zachowanie przypiętego oficjalnego 2DGS, opisuje
> [B1_2DGS_BASELINE.md](B1_2DGS_BASELINE.md).

B1 portuje cykl adaptacji z referencyjnego GraphDECO 3D Gaussian Splatting.
Nie używa ray error, contribution ani autorskiego rankingu do podejmowania
decyzji. Jedynym odstępstwem budżetowym jest bramka 200k opisana poniżej.

Źródła referencyjne:

- `scene/gaussian_model.py` w `graphdeco-inria/gaussian-splatting`;
- `train.py` w `graphdeco-inria/gaussian-splatting`;
- `getNerfppNorm` w `scene/dataset_readers.py`.

## Cameras extent

Ta sama definicja działa dla Blender/NeRF Synthetic i COLMAP:

1. obliczamy średnią pozycji wszystkich kamer treningowych;
2. znajdujemy maksymalną odległość kamery od tej średniej;
3. `cameras_extent = 1.1 * max_distance`.

BBox medium nie uczestniczy w densyfikacji ani pruningu Gaussianów.

## Statystyki zbierane co krok

Do końca okna densyfikacji akumulujemy, wyłącznie dla widocznych Gaussianów:

- normę gradientu pozycji projekcji w screen space;
- liczbę obserwacji (`gradient_denom`);
- największy promień projekcji w pikselach (`max_radii2d`).

`SurfaceGaussianModel.training_scene()` dodaje zerowy dwuwymiarowy offset w
znormalizowanych współrzędnych viewportu `[-1, 1]`. Jednostkowe przesunięcie
w osi X odpowiada `depth * tan(fov_x / 2)` w world-space; analogicznie dla Y.
To odtwarza mnożnik `width / 2`, `height / 2` z backwardu rasteryzera
GraphDECO. Forward geometrii pozostaje identyczny, a osobny gradient nie
zawiera gradientu pozycji pochodzącego od SH.

Promień ekranowy jest liczony z Jacobianu projekcji dwóch obróconych osi
planarnego splata, z clampem perspektywy, filtrem `+0.3` w kowariancji
i `ceil(3 sigma)` jak w rasterizerze GraphDECO. Próg `20 px` nie korzysta
z większego supportu przecięcia ray tracera FoamSurf; są to dwa różne pojęcia.

## Optymalizacja Gaussianów jak w 3DGS

B1 inicjalizuje opacity wartością `0.1` i używa Adam `eps=1e-15`. Grupy
Gaussianów mają referencyjne wartości GraphDECO:

- SH DC: `0.0025`; SH rest: `0.0025 / 20 = 0.000125`;
- opacity: `0.025`; scale: `0.005`; rotation: `0.001`;
- position: od `0.00016 * cameras_extent` do
  `0.0000016 * cameras_extent`, log-liniowo przez 30000 iteracji.

Trening SH zaczyna się od degree 0 i zwiększa degree co 1000 iteracji. Kappa
pozostaje parametrem specyficznym dla FoamSurf z osobnym LR, ponieważ 3DGS nie
ma jego odpowiednika. Loss pozostaje bazowym hybrid RGB MSE FoamSurf, a opacity
reset nadal jest wyłączony zgodnie z zakresem B1.

## Clone, split i pruning

Co 100 iteracji, po iteracji 500 i przed iteracją 15000:

1. średni gradient to `gradient_sum / gradient_denom`; brak obserwacji daje 0;
2. gradient co najmniej `0.0002` kwalifikuje Gaussian do densyfikacji;
3. skala `<= 0.01 * cameras_extent` oznacza clone;
4. większa skala oznacza split na dwoje;
5. rodzic splitu jest usuwany, dzieci dostają offset w lokalnej płaszczyźnie
   i skale podzielone przez `1.6`;
6. jedna maska pruningowa usuwa opacity `< 0.005`;
7. po iteracji 3000 usuwa również `max_radii2d > 20 px` oraz world scale
   `> 0.1 * cameras_extent`.

Kolejność eventu jest taka sama jak w GraphDECO: najpierw clone, potem split,
a dopiero potem pruning starych rodziców. Użyteczny Gaussian przekraczający
próg screen-space dostaje więc replacement przed usunięciem: mały dostaje
clone, a duży dwoje pomniejszonych dzieci. World-oversized rodzic jest
usuwany bez dzieci tylko wtedy, gdy dzieci po podziale nadal przekraczałyby
world threshold. Nie ma minimalnej liczby punktów ani ochrony przez
contribution.

## Audyt zgodności z 3DGS

Zgodne elementy to: camera extent, normalized-viewport gradient, harmonogram
i grupy LR, progressive SH, opacity `0.1`, kryteria clone/split, dzielnik
skali `1.6`, kolejność clone -> split -> prune, progi opacity/screen/world,
okno adaptacji oraz przenoszenie momentów Adam.

Celowe różnice wynikające z FoamSurf:

- Gaussian jest planarny i ma kappa, więc split losuje offset tylko w jego
  lokalnej płaszczyźnie;
- loss to hybrid RGB MSE optymalizujący razem Gaussy i Voronoi, nie 3DGS
  `L1 + 0.2 DSSIM`;
- trening używa minibatcha rayów, a 3DGS rasteryzacji pełnego obrazu;
- bramka 200k nie występuje w referencji;
- opacity reset pozostaje świadomie wyłączony w B1;
- FoamSurf nie optymalizuje per-camera exposure i nie używa random background.

Otwarte różnice wymagające osobnych testów lub ablation:

- widoczność FoamSurf oznacza rzeczywiste trafienie przez wylosowany ray;
  3DGS otrzymuje widoczność wszystkich Gaussianów z pełnego rasteryzera;
- promień jest odtwarzany analitycznie, ponieważ ray tracer nie zwraca
  rasterizerowego `radii`; jego zgodność musi pozostać testowana;
- synthetic camera jest losowana z powtórzeniami, podczas gdy 3DGS przechodzi
  losowo przez każdą kamerę raz na cykl;
- FoamSurf inicjalizuje orientacje planarnego splata losowo, 3DGS inicjalizuje
  quaternion identity; bezpośrednia zmiana spłaszczyłaby wszystkie 2D splaty
  do jednej płaszczyzny i wymaga ablation;
- FoamSurf używa najbliższego sąsiada do skali początkowej, a `distCUDA2` 3DGS
  agreguje lokalne sąsiedztwo. To również wymaga kontrolowanego ablation;
- próg gradientu `0.0002` ma te same jednostki geometryczne, ale inny loss
  i Monte Carlo ray sampling mogą zmieniać jego statystyczną skalę.

Diagnostyka raportuje więc kwantyle promienia, surowych kandydatów gradientu,
clone/split, przecięcia gradientu z trzema maskami pruningowymi oraz liczbę
screen-oversized rodziców, które dostały replacement.

## Bramka 200k

Jeżeli liczność na początku eventu wynosi co najmniej 200k, clone i split są
pomijane, ale pełny pruning nadal działa. Jeżeli pruning obniży liczność,
densyfikacja może wrócić przy kolejnym evencie. Event rozpoczęty poniżej 200k
adaptuje wszystkich kandydatów, więc może zakończyć się nieco powyżej limitu.
Nie ma top-k ani limitu liczby nowych punktów na event.

## Optimizer, checkpointy i logi

Ocalałe wiersze zachowują momenty Adam. Nowe clone/split dzieci dostają zerowe
momenty. Statystyki adaptacji, kwantyle gradientu i promienia, liczba kandydatów
clone/split, przecięcia gradient-prune, replacementy oversized, osobne
przyczyny pruningu, bieżący position LR, stopień SH, maksymalny promień
2D, `cameras_extent`, bramka budżetu, liczność i pamięć GPU są zapisywane do
CSV, W&B i checkpointu.

## Uruchomienie

```bash
cd /shared/results/gmwilczn/foamsurf/foamsurf
sbatch scripts/train_b1_lego_layers.sbatch
```

Job wykonuje najpierw natywny smoke test train/resume, a następnie 35k
iteracji na `lego_both`. `lego_lego` oraz `lego_clouds` pozostają ground truth
wyłącznie dla diagnostyki entanglementu.

## Testy

```bash
python -m pytest -q tests/b1/test_gaussian_adaptation.py
python -m pytest -q \
  renderer/tests/test_b1_gaussian_adaptation.py
python scripts/smoke_test_b1.py
```

Testy obejmują cameras extent, jednostki normalized viewport, forward-noop i
izolację gradientu screen-space, referencyjne grupy LR, position schedule, progresję SH,
promień w pikselach, clone/split, replacement oversized przed pruningiem, bramkę 200k,
mapowanie momentów Adam, render no-op oraz checkpoint/resume.
