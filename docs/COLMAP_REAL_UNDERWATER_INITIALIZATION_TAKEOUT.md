# Takeout: inicjalizacja FoamSurf z COLMAP dla realnej sceny podwodnej IUI3

## Cel i kontekst

Docelowym celem FoamSurf jest rozdzielenie dwóch składników sceny:

- **Gaussy 2D** mają reprezentować twarde, stabilne powierzchnie;
- **komórki Voronoi** mają reprezentować medium, w tym przypadku efekty
  wody takie jak absorpcja, backscatter, zamglenie i przestrzennie zmienny
  kolor medium.

Jednym z głównych problemów treningu hybrydowego jest to, że Gaussy uczą się
szybciej niż Voronoi. Jeśli obie reprezentacje wystartują równocześnie,
Gaussy mogą przejąć część sygnału należącego do medium, zanim pole Voronoi
zdąży się nauczyć sensownego wyjaśnienia obrazu.

Rozważana hipoteza jest prostsza od budowania osobnego klasyfikatora
`SURFACE/MEDIUM/EMPTY`:

> Sparse COLMAP może sam dostarczyć wystarczająco czysty prior powierzchniowy.
> Gaussy można zainicjalizować wyłącznie w punktach COLMAP, natomiast Voronoi
> pozostawić jako reprezentację całej objętości i medium.

## Scena i dozwolone dane

Analizowana realna scena podwodna znajduje się w:

```text
/shared/sets/datasets/SeathruNeRF_dataset/IUI3-RedSea-Undistorted
```

Model sparse COLMAP znajduje się w:

```text
/shared/sets/datasets/SeathruNeRF_dataset/IUI3-RedSea-Undistorted/sparse/0
```

Obrazy wejściowe znajdują się w:

```text
/shared/sets/datasets/SeathruNeRF_dataset/IUI3-RedSea-Undistorted/images
```

W tym eksperymencie wolno korzystać wyłącznie z realnych, pełnych obrazów
tej sceny oraz z modelu COLMAP utworzonego z tych obrazów. Nie ma i nie należy
zakładać dostępu do osobnych warstw surface/medium, masek medium ani innej
uprzywilejowanej ground truth.

Środowisko:

```text
/shared/results/gmwilczn/conda/envs/foamsurf
```

## Diagnostyka istniejącego modelu COLMAP

Model zawiera:

- 29 zarejestrowanych obrazów;
- 1 wspólną kamerę `PINHOLE` o rozdzielczości `1383 x 917`;
- 21 907 punktów 3D;
- średnią długość tracku około `8.08` obserwacji;
- medianę długości tracku równą `5`;
- średni reprojection error około `0.557 px`;
- medianę reprojection error około `0.306 px`.

Plik PLY i projekcje punktów na sześć losowych kamer zostały zapisane w:

```text
/shared/sets/datasets/SeathruNeRF_dataset/diagnosis/IUI3-RedSea-Undistorted
```

Najważniejsze artefakty:

```text
points3D.ply
comparison_contact_sheet.png
comparison_MTN_5902.png
comparison_MTN_5904.png
comparison_MTN_5906.png
comparison_MTN_5908.png
comparison_MTN_5931.png
comparison_MTN_5933.png
summary.json
```

Każdy panel ma postać:

```text
pełny obraz GT | render samych punktów COLMAP
```

Render korzysta bezpośrednio z estymowanych przez COLMAP pozy i intrinsics,
kolorów punktów oraz z-bufora. Poprawność projekcji zweryfikowano na 176 920
obserwacjach 2D zapisanych w modelu: mediana błędu projekcji wyniosła około
`0.37 px`, a średnia około `0.71 px`.

## Najważniejsza obserwacja

Po wizualnej analizie PLY i paneli dla IUI3 stwierdziliśmy, że sparse COLMAP
w rozsądnym stopniu odtwarza twarde powierzchnie sceny. Nie widać znaczącej
populacji punktów zawieszonej swobodnie w wodzie ani gęstej pseudo-geometrii
odpowiadającej samemu medium.

Jest to inny przypadek niż syntetyczne LEGO z wyraźnymi, strukturalnymi
chmurami. Tam chmury zawierały dostatecznie stabilne i kontrastowe struktury,
aby COLMAP utworzył na nich punkty. W realnej scenie IUI3 efekty wody są
najwyraźniej zbyt rozproszone, gładkie albo zależne od widoku, aby sparse
matching i triangulacja traktowały je jak stabilną powierzchnię.

Obecna interpretacja jest następująca:

```text
stabilne powierzchnie wielowidokowe -> punkty sparse COLMAP -> Gaussy
woda / backscatter / haze           -> brak punktów COLMAP  -> Voronoi
```

To nie jest matematyczna gwarancja dla każdej sceny podwodnej. Jest to jednak
mocna obserwacja empiryczna dla konkretnej sceny IUI3 i uzasadnia wykonanie
prostego eksperymentu bez dodatkowego priora RadFoam ani klasyfikacji komórek.

## Proponowana inicjalizacja FoamSurf

### Gaussy

Gaussy należy zainicjalizować bezpośrednio z punktów:

```text
/shared/sets/datasets/SeathruNeRF_dataset/IUI3-RedSea-Undistorted/sparse/0/points3D.bin
```

Punkty zawierają pozycje, kolory, reprojection error i tracki. Na pierwszy
baseline nie należy wykonywać semantycznego filtrowania ani używać ręcznych
masek. Opcjonalne późniejsze filtry mogą opierać się wyłącznie na danych
COLMAP, np. długości tracku, reprojection error, kącie triangulacji,
lokalnej gęstości i lokalnej planarity.

### Voronoi

Pole Voronoi powinno zostać zainicjalizowane w objętości sceny w standardowy
sposób FoamSurf. Ma ono nauczyć się efektów medium z tych samych pełnych
obrazów treningowych.

Nie należy używać osobnego targetu medium. Zarówno warmup, jak i późniejszy
trening hybrydowy mają korzystać wyłącznie z pełnego RGB sceny IUI3.

## Proponowany harmonogram treningu

Nie zaleca się włączania obu reprezentacji od pierwszej iteracji. Gaussy
zwykle znacznie szybciej redukują reconstruction loss i mogą przejąć wodę,
zanim Voronoi zdąży zbudować sensowne pole medium.

Proponowany harmonogram:

### Etap 1: krótki warmup Voronoi

- renderowane i optymalizowane jest wyłącznie Voronoi;
- Gaussy są już utworzone w punktach COLMAP, ale pozostają zamrożone i nie są
  uwzględniane w renderze albo są w pełni nieaktywne;
- targetem jest pełny obraz RGB;
- celem nie jest idealna rekonstrukcja całej sceny przez Voronoi, tylko
  nauczenie pola początkowego medium przed pojawieniem się szybszej gałęzi;
- długość warmupu powinna być traktowana jako parametr eksperymentu, np.
  `500`, `1000` i `2000` iteracji.

### Etap 2: włączenie Gaussianów

- Gaussy COLMAP zostają odblokowane i włączone do wspólnego renderu;
- Voronoi zachowuje stan wyuczony w warmupie;
- obie reprezentacje są dalej optymalizowane wspólnym reconstruction lossem;
- początkowo warto rozważyć opóźnienie adaptacji Gaussianów względem samego
  odblokowania parametrów, aby natychmiastowy clone/split nie rozszerzył
  populacji w stronę medium.

Przykładowa sekwencja:

```text
iteracje 0-1000:
    Voronoi-only warmup na pełnych obrazach
    Gaussy COLMAP zamrożone

iteracje 1001-2000:
    trening hybrydowy
    Gaussy aktywne, ale bez clone/split

iteracje 2001+:
    pełny trening hybrydowy
    kontrolowana adaptacja Gaussianów i ewentualna adaptacja Voronoi
```

Jest to propozycja do sprawdzenia, nie ustalony optimum.

## Minimalna macierz eksperymentalna

Najbardziej informacyjne będą co najmniej trzy porównywalne runy:

1. **Gaussian-only COLMAP**
   - Gaussy startują z `points3D.bin`;
   - brak Voronoi;
   - określa jakość standardowej reprezentacji powierzchniowej i pokazuje,
     które efekty wodne Gaussy próbują wbudować w geometrię lub kolor.

2. **FoamSurf bez warmupu**
   - Gaussy z COLMAP i Voronoi aktywne od iteracji 1;
   - pokazuje, czy dobra inicjalizacja sama wystarcza mimo różnicy szybkości
     uczenia obu reprezentacji.

3. **FoamSurf z warmupem Voronoi**
   - identyczna inicjalizacja jak w runie 2;
   - najpierw Voronoi-only, potem trening hybrydowy;
   - jest obecnie najbardziej obiecującym wariantem.

Opcjonalnie należy przetestować kilka długości warmupu, zachowując wszystkie
pozostałe parametry bez zmian.

## Wymagana diagnostyka

Ponieważ nie ma rozdzielonych warstw ground truth, nie można bezpośrednio
policzyć surface/medium leakage. Ocena musi opierać się na pełnych obrazach i
wewnętrznej diagnostyce reprezentacji.

Należy zapisywać:

- rendery `hybrid`, `gaussian_only` i `voronoi_only` z tych samych kamer;
- transmittance i responsibility obu gałęzi;
- PSNR, SSIM i LPIPS pełnego renderu;
- liczbę Gaussianów i zdarzenia clone/split/prune;
- liczbę komórek Voronoi i zdarzenia split/prune;
- density, contribution i path length Voronoi;
- opacity, scale i contribution Gaussianów;
- odległość Gaussianów od początkowego point clouda COLMAP;
- osobno statystyki Gaussianów początkowych oraz utworzonych przez adaptację;
- checkpoint przed włączeniem Gaussianów i checkpointy krótko po handoffie.

Szczególnie ważne jest sprawdzenie, czy po odblokowaniu Gaussianów:

- Gaussian-only render pozostaje związany z ostrymi powierzchniami;
- Voronoi-only render przejmuje gładkie, niskoczęstotliwościowe efekty wody;
- nowe Gaussy nie zaczynają wypełniać wolnej objętości;
- jakość hybrydowa przewyższa Gaussian-only bez gwałtownego wzrostu liczby
  Gaussianów.

## Kryterium decyzji

Hipotezę można uznać za obiecującą, jeśli inicjalizacja COLMAP połączona z
krótkim warmupem Voronoi daje jednocześnie:

- lepszą rekonstrukcję pełnych obrazów niż Gaussian-only;
- powierzchniowy i zwarty Gaussian-only render;
- zauważalny, przestrzennie gładki wkład Voronoi;
- brak masowego odpływu nowych Gaussianów od początkowej geometrii COLMAP;
- stabilny trening bez potrzeby używania masek lub rozdzielonej ground truth.

Jeżeli Gaussy mimo tej inicjalizacji zaczną przejmować medium, dopiero wtedy
warto dodać kolejne mechanizmy, np. opóźnioną densyfikację, ograniczenie ruchu
względem początkowego point clouda albo geometryczne filtrowanie nowych
Gaussianów. Pierwszy eksperyment powinien jednak pozostać możliwie prosty,
aby sprawdzić, ile daje sama inicjalizacja COLMAP i kolejność włączania gałęzi.

## Aktualny wniosek roboczy

Dla realnej sceny `IUI3-RedSea-Undistorted` sparse COLMAP wygląda jak użyteczny
i stosunkowo czysty prior powierzchniowy. Najbardziej naturalny następny krok
to zainicjalizować Gaussy bezpośrednio z tego modelu, najpierw wykonać krótki
warmup pełnoscenowego Voronoi, a następnie włączyć Gaussy i kontynuować trening
hybrydowy. Pozwala to przetestować FoamSurf bez zewnętrznego RadFoam i bez
jakiejkolwiek warstwowej ground truth.
