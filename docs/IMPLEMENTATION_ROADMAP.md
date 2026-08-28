# FoamSurf: roadmapa badań i implementacji

Ten dokument definiuje kolejność rozwoju FoamSurf oraz kryteria przejścia
między etapami. Najważniejsza zasada brzmi: najpierw kwalifikujemy każdą
reprezentację osobno, następnie badamy ich interakcję, dopiero potem
rozwiązujemy entanglement i poprawiamy końcową jakość.

Nie należy równocześnie zmieniać reprezentacji, adaptacji topologii i lossu
disentanglement. Taki eksperyment nie pozwala ustalić przyczyny poprawy lub
regresji.

Dokument koncepcyjny pozostaje w
[entanglement_proposition.md](../entanglement_proposition.md).

## Protokół danych

Aligned Lego zawiera trzy zgodne zestawy obrazów:

- `lego_lego`: sama powierzchnia Lego;
- `lego_clouds`: samo medium;
- `lego_both`: kompozycja powierzchni i medium.

Ich rola zależy od etapu:

| etap | supervision treningowe | rola pozostałych warstw |
|---|---|---|
| 1. Gaussian-only | `lego_lego` | kwalifikacja powierzchni |
| 2. Voronoi-only | `lego_clouds` | kwalifikacja medium |
| 3. Integracja | wyłącznie `lego_both` | `lego_lego` i `lego_clouds` tylko do ewaluacji |
| 4. Disentanglement | wyłącznie `lego_both` | `lego_lego` i `lego_clouds` tylko do ewaluacji |
| 5. Dofitowanie | wyłącznie `lego_both` | `lego_lego` i `lego_clouds` tylko do ewaluacji |

W etapach 1 i 2 warstwy są świadomie używane jako targets do kwalifikacji
zdolności reprezentacji. Od etapu 3 nie wolno używać ich w lossie, wybierać
nimi pikseli ani przekazywać modelowi masek warstw.

Każdy etap zachowuje osobny train/test split kamer. Wynik na widokach
treningowych nie jest kryterium ukończenia.

Do eksperymentów hybrydowych nie przenosimy wag wytrenowanych na
`lego_lego` lub `lego_clouds`. Etap 3 zaczyna się od świeżej inicjalizacji.
Przenosimy wyłącznie sprawdzoną architekturę, algorytmy adaptacji i ustalone
hiperparametry. Transfer layer-supervised weights byłby oracle initialization
i fałszowałby ocenę disentanglementu.

## Istniejąca infrastruktura: B0 i B0.1

Baseline zawiera:

- planar 2D Gaussians i sparse volumetric Voronoi medium;
- wspólny depth-ordered renderer z gradientami obu reprezentacji;
- tryby hybrid, Gaussian-only i Voronoi-only;
- CSV, W&B, obrazy diagnostyczne oraz last/best checkpoints;
- aligned evaluation dla `lego_lego` i `lego_clouds`;
- testy renderera, gradientów i checkpoint/resume.

B0.1 dostarcza diagnostykę odpowiedzialności bez zmiany obrazu i lossu.
Szczegóły opisuje [B01_DIAGNOSTICS.md](B01_DIAGNOSTICS.md). Raportowane są
między innymi contribution, visibility, transmittance, overlap, responsibility
oraz layer-specific leakage.

Kod adaptacji Gaussianów i profil odpowiadający efektywnemu zachowaniu
przypiętego oficjalnego 2DGS są zaimplementowane. Szczegóły opisuje
[B1_2DGS_BASELINE.md](B1_2DGS_BASELINE.md). Nie oznacza to jeszcze
kwalifikacji reprezentacji: dotychczasowy run był hybrydowy, więc wynik
Gaussianów był zaburzony przez medium.

## Etap 1: kwalifikacja Gaussian-only

**Status implementacji:** tryb, target, izolacja gradientów, testy i sweep są gotowe; kwalifikacja jakościowa czeka na wyniki 30k.

### Cel

Udowodnić, że powierzchniowa reprezentacja FoamSurf samodzielnie tworzy ostre,
stabilne i wielowidokowo spójne Lego. Na tym etapie medium nie może wpływać na
forward, loss, gradient ani optymalizator.

### Eksperyment

- trening i walidacja na odpowiednich splitach `lego_lego`;
- wyłącznie Gaussian renderer;
- pełny obraz i kamery losowane bez zwracania;
- inicjalizacja, SH, learning rates, densyfikacja, pruning i opacity schedule
  kontrolowane względem oficjalnego 2DGS;
- pełny run 30k oraz co najmniej trzy seedy dla finalnej konfiguracji;
- bez regularyzacji wykorzystujących wiedzę o medium.

Porównanie z oficjalnym 2DGS powinno używać tych samych kamer, obrazów,
rozdzielczości, białego tła i możliwie zgodnej inicjalizacji. Różnice
renderera muszą być jawnie opisane.

### Diagnostyka

Raportujemy:

- PSNR, SSIM, LPIPS i foreground PSNR;
- obrazy oraz cropy wysokich częstotliwości;
- liczbę Gaussianów i zdarzenia clone/split/prune;
- opacity, projected radii, world scales i stopień SH;
- skoki metryk wokół resetów oraz eventów topologii;
- czas treningu, FPS i peak GPU memory.

### Kryterium ukończenia

- jakość testowych widoków jest zbliżona do ustalonego baseline'u 2DGS;
- detale nie znikają po opacity resetach ani pruningu;
- run 30k przechodzi bez NaN/OOM i trwałego załamania jakości;
- wyniki są stabilne między seedami;
- projected-radius i gradient regression tests pozostają zgodne;
- checkpoint/resume zachowuje trajektorię w oczekiwanej tolerancji.

Jeżeli ten etap nie przechodzi, nie modyfikujemy medium ani
disentanglementu. Naprawiamy wyłącznie surface model, renderer lub jego
optymalizację.

## Etap 2: kwalifikacja adaptacyjnego Voronoi-only

**Status implementacji:** pure forward, adaptacyjny split/prune, migracja Adam, checkpoint/resume, testy i sweep 2k–32k są gotowe; etap nie jest zaliczony bez analizy runów 30k.

### Cel

Udowodnić, że Voronoi samodzielnie reprezentuje miękkie, półprzezroczyste
medium. Adaptacyjna densyfikacja i pruning są obowiązkową częścią tego etapu,
a nie późniejszym dodatkiem.

### Eksperyment

- trening i walidacja na odpowiednich splitach `lego_clouds`;
- wyłącznie Voronoi renderer;
- brak Gaussian events, parametrów i gradientów;
- baseline ze stałą liczbą sites jako kontrola;
- wariant adaptacyjny z pełnym testem 30k i kilkoma seedami;
- porównania różnych budżetów punktów przy tym samym rendererze i lossie.

Nie kopiujemy RadFoamowego depth quantization ani kryterium twardej
powierzchni. Voronoi ma modelować objętościową absorpcję/emisję.

### Statystyki adaptacji komórki

Dla każdej komórki zbieramy w stabilnym oknie lub EMA:

- visibility i liczbę ray segments;
- długość drogi ważoną wejściową transmittance;
- contribution `T * alpha_medium`;
- korelację contribution z residualem RGB;
- gradient density, color i site position;
- stabilny proxy rozmiaru/objętości komórki;
- wiek komórki i czas od ostatniej mutacji.

Pojedynczy batch nie może wywoływać zmiany topologii.

### Densyfikacja

Densyfikujemy komórkę, gdy jest dostatecznie obserwowana oraz ma utrzymujący
się błąd, gradient lub niewystarczającą rozdzielczość przestrzenną. Nowy site
powstaje wewnątrz volume:

- w lokalizacji ważonej błędnymi odcinkami rayów;
- albo wzdłuż osi największej przestrzennej niepewności komórki;
- z minimalnym odstępem od istniejących sites.

Density i color dzieci inicjalizujemy tak, aby pierwszy render po operacji
możliwie mało zmieniał funkcję medium.

### Pruning i merge

Sama niska density nie jest wystarczającym kryterium usunięcia. Rzadka
komórka może poprawnie reprezentować delikatny fragment chmury albo być
potrzebna topologicznie.

Kandydat powinien przez odpowiednio długi czas mieć małą visibility lub
contribution, mały wpływ na residual i bezpieczne sąsiedztwo. Preferujemy
merge z podobnym sąsiadem, jeśli lepiej zachowuje funkcję. Guard sites nigdy
nie są usuwane.

### Kontrakt implementacyjny

`VoronoiTopologyController` ma trzy jawne fazy:

```text
collect_statistics(render_aux, residual)
propose_changes(iteration, budget) -> TopologyPlan
apply_plan(model, optimizer, plan)
```

Plan jest logowalny przed mutacją. Aplikacja planu przenosi parametry i stan
Adam, zachowuje guard mask, wykonuje pełny rebuild Delaunay i synchronizuje
renderer.

### Kryterium ukończenia

- Voronoi-only poprawnie odtwarza testowe widoki `lego_clouds`;
- wariant adaptacyjny mierzalnie przewyższa fixed-site baseline przy
  porównywalnym budżecie;
- adaptacja nie tworzy twardych powierzchni ani artefaktów komórkowych;
- neutralny split/merge zmienia render poniżej ustalonej tolerancji;
- testy obejmują optimizer-state migration, guard sites i losowe sekwencje
  mutacji topologii;
- run 30k jest stabilny, mieści się w budżecie i działa po resume;
- CSV/W&B raportują każdą przyczynę split/prune/merge.

Jeżeli etap nie przechodzi, nie rozwiązujemy jeszcze entanglementu. Najpierw
poprawiamy możliwości samego medium.

## Etap 3: integracja obu reprezentacji

### Cel

Połączyć dwie zakwalifikowane reprezentacje i jednoznacznie zmierzyć ich
naturalną interakcję bez mechanizmu disentanglementu.

### Protokół

- świeża inicjalizacja obu reprezentacji;
- trening wyłącznie na `lego_both`;
- `lego_lego` i `lego_clouds` wyłącznie w ewaluacji;
- adaptacyjne Gaussy i adaptacyjne Voronoi;
- brak layer-aware lossów, masek i oracle initialization;
- identyczny split kamer, renderer i budżet dla wszystkich ablation.

Minimalna macierz:

1. obie reprezentacje uczone od początku;
2. Gaussy uczone, medium zamrożone;
3. medium uczone, Gaussy zamrożone;
4. Gaussian warm-up, potem wspólny trening;
5. Voronoi warm-up, potem wspólny trening.

Warianty zamrożone służą do audytu gradientów i nie są kandydatami finalnej
metody.

### Kryterium ukończenia

- renderer zachowuje standalone forward i gradient obu reprezentacji;
- wiemy, kiedy oraz jak jedna gałąź przejmuje sygnał drugiej;
- znamy wpływ wspólnego treningu na population, opacity, density, scale i
  contribution;
- co najmniej trzy seedy pokazują, czy interakcja jest stabilna;
- otrzymujemy jawny, odtwarzalny baseline entanglementu dla etapu 4.

Etap 3 nie musi mieć dobrego rozdzielenia warstw. Jego wynikiem jest poprawnie
zmierzony problem, a nie jego rozwiązanie.

## Etap 4: disentanglement

### Cel

Sprawić, aby Gaussy odpowiadały za powierzchnię, a Voronoi za medium, bez
korzystania z rozdzielonych warstw podczas treningu.

### Zasady

- supervision pozostaje wyłącznie `lego_both`;
- `lego_lego` i `lego_clouds` są wyłącznie ewaluacją i narzędziem rozwoju;
- mechanizmy wprowadzamy pojedynczo;
- każdy składnik ma osobny config, log i wagę zero odtwarzającą etap 3;
- nie maskujemy pikseli oracle maskami;
- nie przenosimy layer-supervised weights.

Kandydaci z
[entanglement_proposition.md](../entanglement_proposition.md) obejmują między
innymi priory reprezentacji, regularyzację odpowiedzialności, schedule
aktywacji oraz ograniczenie nakładania contribution. Kolejność wdrażania
wynika z danych z etapu 3, a nie z jednoczesnego sweepu wielu nowych lossów.

### Metryki sukcesu

- spadek `medium_alpha_on_lego_only` i
  `medium_weight_on_lego_only`;
- spadek `gaussian_alpha_on_cloud_only` i
  `gaussian_weight_on_cloud_only`;
- wzrost Gaussian dominance na `lego_lego`;
- wzrost medium dominance na `lego_clouds`;
- spadek overlap i leakage na background;
- zachowanie lub poprawa hybrid PSNR/SSIM/LPIPS.

### Kryterium ukończenia

- poprawa rozdzielenia występuje na testowych kamerach i kilku seedach;
- hybrid quality nie jest kupowana przez wygaszenie jednej reprezentacji;
- obie reprezentacje zachowują jakość udowodnioną w etapach 1 i 2;
- ablation pokazuje wpływ każdego aktywnego składnika;
- wynik przenosi się na co najmniej jedną dodatkową scenę.

## Etap 5: dofitowanie reprezentacji i systemu

### Cel

Poprawić końcową jakość, szybkość i skalowalność dopiero po ustabilizowaniu
podziału odpowiedzialności.

Etap 4 odpowiada na pytanie: „która reprezentacja ma modelować dany sygnał?”.
Etap 5 odpowiada: „jak dobrze każda reprezentacja modeluje właściwy sobie
sygnał?”.

### Zakres jakościowy

- końcowe learning-rate schedules i harmonogramy adaptacji;
- ostrość, SH i budżet Gaussianów;
- subtelne struktury, density/color parameterization oraz budżet Voronoi;
- inicjalizacja obu reprezentacji bez oracle danych;
- kontrola footprintów, near/far i volume bounds;
- końcowe wagi mechanizmu disentanglementu;
- generalizacja na kilka scen i datasetów.

Nie akceptujemy poprawy samego hybrid PSNR, jeżeli layer diagnostics pokazują
większy entanglement.

### Renderer drugiej generacji

Obecny renderer ma pośredni OptiX hit buffer `[16,R]` i osobny kernel
compositingu. Kierunki pracy:

1. benchmark hit generation i hybrid traversal osobno;
2. usunięcie limitu 16 albo dowód braku truncation;
3. połączenie obu typów eventów w jednym traversal/launch;
4. szybszy starting-cell lookup;
5. zredukowane statystyki dla obu kontrolerów topologii;
6. zachowanie publicznego kontraktu `GaussianScene/VoronoiScene`.

Każda optymalizacja przechodzi forward, pure-mode, finite-difference,
gradient i pełno-rozdzielczy benchmark przeciw aktualnemu rendererowi.

### Duże sceny

Dla COLMAP/Mip-NeRF 360 potrzebne są:

- odporna normalizacja oraz jawne near/far;
- volume bounds odporne na outliery;
- budżety zależne od camera extent i coverage;
- chunked validation;
- kontrola pamięci AS, CSR i hit buffers;
- ewentualny spatial partitioning bez zmiany compositingu.

### Kryterium ukończenia

- poprawa jakości występuje w hybrid i właściwych component-only metrics;
- brak regresji disentanglementu;
- pełne 30k działa bez NaN/OOM na scenach syntetycznych;
- co najmniej jeden duży dataset COLMAP przechodzi end-to-end;
- raport zawiera training time, FPS, peak memory i liczności obu reprezentacji;
- renderer zachowuje testy forward/gradient po optymalizacjach.

## Wymagania wspólne dla wszystkich etapów

Każda nowa funkcja musi:

- mieć jawny config i zachowanie wyłączone odtwarzające poprzedni etap;
- mieć unit tests logiki oraz native integration smoke test;
- raportować własne sygnały i operacje do CSV oraz W&B;
- działać po checkpoint/resume;
- mieć co najmniej jeden pełny eksperyment, nie tylko pojedynczy render;
- zachowywać artefakty potrzebne do wizualnego porównania;
- aktualizować dokumentację renderera i walkthrough.

Minimalny raport eksperymentu zawiera:

- hybrid, Gaussian-only i Voronoi-only PSNR/SSIM/LPIPS;
- layer-specific foreground metrics;
- responsibility, contribution, overlap i leakage;
- liczności, zdarzenia topologii i rozkłady parametrów;
- training time, render FPS i peak GPU memory;
- config, seed, commit i źródło datasetu.

Nie przechodzimy do kolejnego etapu tylko dlatego, że pojedynczy run wygląda
obiecująco. Przejście wymaga spełnienia jawnych kryteriów oraz wyniku
powtarzalnego między seedami.
