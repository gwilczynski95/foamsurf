# Plan wdrażania po baseline B0

Ten dokument rozdziela to, co działa dzisiaj, od kolejnych hipotez badawczych.
Nie należy włączać kilku etapów naraz: B0 ma najpierw dostarczyć kontrolę i
diagnostykę entanglementu bez regularyzacji.

Dokument koncepcyjny projektu pozostaje w
[entanglement_proposition.md](../entanglement_proposition.md). Poniższy plan
przekłada te pomysły na granice modułów, eksperymenty i kryteria akceptacji.

## Stan wyjściowy: B0

B0 zawiera:

- surface 2D Gaussians i sparse volumetric Voronoi medium;
- wspólny depth-ordered renderer z gradientami obu reprezentacji;
- wyłącznie hybrid RGB MSE;
- stałą liczbę prymitywów;
- okresowy rebuild Delaunay po ruchu sites;
- trzy tryby walidacji, CSV, W&B, last/best checkpoints;
- unit, native renderer i end-to-end resume tests.

Pierwszy pełny run B0 powinien odpowiedzieć:

- czy Voronoi pozostaje medium, czy kopiuje powierzchnie;
- która gałąź najszybciej przejmuje rekonstrukcję;
- jak zmieniają się density, opacity, alpha i gradient norm;
- gdzie hybrid jest lepszy/gorszy od każdego izolowanego komponentu;
- czy czas i pamięć ogranicza OptiX hit buffer, Voronoi traversal czy batch.

Nie zmieniamy modelu przed obejrzeniem obrazów Gaussian-only, Voronoi-only,
transmittance oraz wykresów parametrów z kilku scen i seedów.

## Etap B0.1: wiarygodna diagnostyka odpowiedzialności

Przed regularyzacją renderer powinien opcjonalnie zwracać per-ray/per-primitive
statystyki bez zmiany obrazu:

- skumulowaną wagę compositingu Gaussianów i medium;
- długość drogi w aktywnych komórkach;
- liczbę Gaussian events i Voronoi crossings;
- `T` przed/po każdym typie reprezentacji w formie zredukowanej;
- widoczność, maksymalny błąd rayów i sumę gradient magnitude per primitive.

Nie wolno alokować pełnej macierzy `rays x primitives`. CUDA ma atomowo
redukować statystyki do buforów `[G]`, `[V]` i ewentualnie `[R]`.
Tryb statystyk musi dać identyczny RGB i gradient w tolerancji do zwykłego
renderu i być domyślnie wyłączony.

Kryterium akceptacji:

- test zgodności obrazu i gradientów statistics on/off;
- brak znaczącego narzutu, gdy statistics są wyłączone;
- W&B/CSV raportuje rozkłady contribution/visibility dla obu reprezentacji.

## Etap B1: kontrolowana densyfikacja/pruning Gaussianów

Najpierw warto odtworzyć standardowy cykl 3DGS/FlaRe w nowej strukturze:

- collect screen-space/ray error i visibility;
- split/clone prymitywów o wysokim błędzie;
- prune znikome opacity, contribution albo błędną geometrię;
- zachowaj/przenieś stan Adam;
- wykonaj `set_gaussian_geometry` po mutacji;
- loguj liczbę split/clone/prune i pamięć GPU.

Nie należy jeszcze aktywować adaptive Voronoi w tym samym eksperymencie.
Porównanie B0 vs B1 pokaże wpływ samej adaptacji powierzchni.

Kryterium akceptacji:

- unit test mapowania parametrów i momentów Adam;
- renderer equivalence przed/po no-op topology operation;
- pełny run osiąga zakładany budget Gaussianów bez NaN/OOM;
- ablation z tą samą inicjalizacją i seedem.

## Etap B2: pierwszy mechanizm disentanglement

Dopiero na podstawie B0/B1 należy wybrać najmniejszą regularyzację. Kandydaci z
propozycji entanglementu powinni być wdrażani pojedynczo, na przykład:

- powierzchnia: preferencja ostrej, zwartej odpowiedzi i spójności normalnych;
- medium: preferencja rozproszonej, niskoczęstotliwościowej absorpcji/emisji;
- penalizacja nakładania się dużej medium contribution z silną surface alpha
  w bardzo bliskim zakresie głębokości;
- stopniowany schedule, aby inicjalizacja jednej gałęzi nie zagłodziła drugiej.

Każdy składnik dostaje osobny klucz config, osobny log W&B/CSV i wagę równą
zero w configu B0. Nie sumujemy anonimowego `regularization_loss`.

Kryterium akceptacji:

- przy wadze zero loss i gradient są zgodne z B0;
- test gradientu pokazuje dokładnie, do której gałęzi działa składnik;
- raportujemy quality hybrid i component-only, a nie tylko PSNR hybrid;
- minimum trzy seedy i kilka scen, bo podział odpowiedzialności może być
  niestabilny.

## Etap B3: adaptacyjna densyfikacja i pruning Voronoi

To jest wymagany element finalnej metody. Nie powinien kopiować mechanizmu
Radiant Foam opartego na twardym surface error, ponieważ Voronoi ma modelować
medium.

### Statystyki dla komórki

Dla każdej komórki zbieramy wykładniczo wygładzone:

- `visibility`: liczba/masa rayów przechodzących przez komórkę;
- `path_weight`: suma długości odcinków ważona wejściową transmittance;
- `contribution`: suma `T * alpha_medium`;
- `residual_correlation`: korelacja jej wkładu z residualem RGB;
- `density_grad`, `color_grad`, `site_grad`;
- objętość komórki lub stabilny proxy z sąsiadów;
- overlap z Gaussian surface events wzdłuż tych samych rayów.

EMA i minimalny okres obserwacji są konieczne, aby pojedynczy batch nie
wywoływał zmiany topologii.

### Densyfikacja

Komórka jest kandydatem do splitu, gdy jest widoczna, ma istotną drogę/wkład i
utrzymujący się błąd lub wariancję gradientu. Nowy site powinien powstać:

- wzdłuż osi największej przestrzennej niepewności komórki;
- albo w ważonej residualem lokalizacji odcinków rayów;
- z minimalnym odstępem od istniejących sites;
- wewnątrz dozwolonego volume.

Nowe density/color dziedziczymy tak, aby pierwszy render po splicie możliwie
mało zmienił funkcję sceny. Momenty Adam są kopiowane lub kontrolowanie
zerowane i operacja zawsze kończy się pełnym rebuildem Delaunay.

### Pruning i merge

Nie wystarczy usuwać niskiej density. Przezroczysta komórka może być potrzebna
topologicznie jako przejście, a guard sites nie mogą zniknąć.

Kandydat do usunięcia/merge powinien przez odpowiednio długi czas mieć:

- małą visibility/path weight albo małą contribution;
- mały wpływ na residual;
- density/color podobne do najlepszego sąsiada;
- brak statusu guard;
- bezpieczną geometrię po symulowanym usunięciu.

Preferowany jest merge do sąsiada z podobnymi właściwościami zamiast ślepego
delete. Potrzebny jest minimalny i maksymalny budget punktów oraz cooldown po
każdej operacji.

### Własność kodu

`VoronoiTopologyController` powinien dostać trzy jawne fazy:

```text
collect_statistics(render_aux, residual)
propose_changes(iteration, budget) -> TopologyPlan
apply_plan(model, optimizer, plan)
```

`TopologyPlan` jest logowalny/testowalny przed mutacją. `apply_plan` musi
przebudować wszystkie `Parameter`, przenieść stan Adam, zaktualizować guard
mask, wykonać pełne Delaunay i zsynchronizować renderer. Trainer jedynie
wywołuje kontroler na schedule; nie zna kryteriów split/prune.

Kryterium akceptacji:

- deterministyczne unit tests planu na syntetycznych statystykach;
- test zachowania wszystkich pól i stanu Adam przy split/prune/merge;
- test topologii po setkach losowych mutacji bez błędnych indeksów;
- brak pruningu guard sites;
- obraz przed i po neutralnym splicie/merge zmienia się poniżej ustalonej
  tolerancji;
- liczność pozostaje w budżecie;
- pełna diagnostyka operacji w CSV/W&B.

## Etap B4: renderer drugiej generacji

Obecny renderer jest poprawnym baseline'em, ale ma pośredni OptiX hit buffer
`[16,R]` i osobny kernel compositingu. CUDA roadmapa:

1. dodać benchmark osobno dla hit generation i hybrid traversal;
2. usunąć twardy limit 16 albo udowodnić, że truncation nie występuje;
3. przenieść oba typy eventów do jednego traversal/launch;
4. przyspieszyć starting-cell lookup;
5. dodać opcjonalne zredukowane statystyki dla adaptacyjnej topologii;
6. zachować publiczny kontrakt `GaussianScene/VoronoiScene`.

Każdy etap porównujemy z `HybridRenderer` oracle i aktualnym native
rendererem: forward, pure modes, finite differences/gradcheck, przypadki
kolejności surface-before-medium i medium-before-surface oraz benchmark
pełnego wywołania dla rzeczywistej rozdzielczości.

## Etap B5: duże sceny

Dla Mip-NeRF 360/COLMAP potrzebne będą osobne eksperymenty:

- normalizacja i jawne near/far per scena;
- volume bounds odporne na outliery kamer/point cloudu;
- budżety primitive zależne od extent/coverage;
- chunked full-image validation;
- kontrola pamięci AS/CSR/hit buffers;
- ewentualny spatial partitioning bez zmiany wyniku compositingu.

Najpierw loader powinien przejść camera/ray regression tests przeciw 3DGS.
Następnie mały crop/scene smoke test, dopiero potem pełne runy.

## Macierz eksperymentów

Minimalna sekwencja porównań:

| wariant | surface adaptive | medium adaptive | disentanglement |
|---|---:|---:|---:|
| B0 | nie | nie | nie |
| B1 | tak | nie | nie |
| B2 | tak | nie | jedna regularyzacja |
| B3 | tak | tak | ta sama regularyzacja |

Dla każdego zachowujemy dataset/split, seed, budget rayów, liczbę iteracji i
renderer settings. Raport obejmuje hybrid/Gaussian-only/Voronoi-only
PSNR/SSIM, training time, FPS, peak memory, liczności prymitywów i wykresy
responsibility/contribution.

## Definicja ukończenia finalnej integracji

Etap można uznać za gotowy do merge dopiero, gdy:

- ma config z domyślnie jawnym zachowaniem;
- ma unit test logiki i native integration smoke test;
- zachowuje B0 przy wyłączonej opcji;
- loguje własne sygnały i operacje do CSV oraz W&B;
- działa po checkpoint/resume;
- ma wynik pełnego eksperymentu, nie tylko pojedynczy render;
- dokumentacja renderera i walkthrough odzwierciedlają faktyczny kod.
