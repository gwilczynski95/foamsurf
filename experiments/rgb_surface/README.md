# IUI3: czarne tło × overlap całego Voronoi × mocne TV

## Cel i hipotezy

Chcemy, żeby **Gaussian-only pokrywał dno bez dziur**, a Voronoi reprezentowało wodę.
Dobry PSNR hybrydy, ładne `direct` i mały loss overlap nie dowodzą separacji.

1. Czarne tło usuwa człon `T_G * T_a * C_bg` z direct. Wtedy źródłem direct
   są wyłącznie Gaussy, choć jego tłumienie nadal zależy od Voronoi.
2. Nowy overlap ogranicza medium lokalnie wokół wiarygodnych trafień Gaussianów.
   Gradient obejmuje absorpcję aRGB, backscatter bRGB, kolor B i położenia sites.
3. TV utrudnia medium tworzenie ostrych struktur dna. Badamy również wagi
   znacznie większe niż w poprzednich eksperymentach.

To **72 runy**, bez warmupu, po 30k iteracji, maksymalnie **7 jednocześnie**.
Nie ma tu bramki Gumbel-Softmax. To osobny mechanizm do następnego etapu.

## Środowisko i ścieżki

- Repo: `/shared/results/gmwilczn/foamsurf/foamsurf`.
- Conda: `/shared/results/gmwilczn/conda/envs/foamsurf` (`conda activate foamsurf`).
- Dataset: `/shared/sets/datasets/SeathruNeRF_dataset/IUI3-RedSea-Undistorted`.
- Kod/config/testy/sbatche: `experiments/rgb_surface/`.
- Baza: `experiments/rgb_smooth/sweep.yaml` → `configs/colmap_rgb_medium.yaml`
  → `configs/end_to_end_iui3.yaml` i wskazane przez niego konfiguracje.
- W&B online: organizacja `mlpolska`, projekt `foamsurf_iui3_rgb_surface`.
- Grupa W&B: `rgb-surface-ARRAY_JOB_ID`.
- Wyniki: `outputs/rgb_surface/ARRAY_JOB_ID/run_ID/`.
- Logi: `slurm_logs/rgb_surface_ARRAY_JOB_ID_TASK_ID.{out,err}`.
- Testy: `outputs/rgb_surface_tests/TEST_JOB_ID/passed.json` oraz raporty JUnit.

Nowy trainer wybierany jest wyłącznie przez nowy entrypoint. Pliki poprzednich
eksperymentów, głównego renderera i treningu nie są edytowane. Nowa ścieżka
regularyzacji używa istniejącego renderera CUDA, łącznie z jego backwardem granic.

## Macierz eksperymentów

Pełny iloczyn kartezjański, niezależne siły lossów:

| Czynnik | Wartości |
|---|---|
| Tło | white `[1,1,1]`, black `[0,0,0]` |
| Waga nowego overlap | 0, 0.03, 0.1 |
| Mnożnik TV | 0, 1, 3, 10 |
| Seed | 42, 43, 44 |

**TV ×1 to dawne `all_strong`, czyli wagi (a,b,B)=(0.1,0.1,1).**
TV ×3: (0.3,0.3,3). TV ×10: (1,1,10).
To nie jest ×1 z `rgb_disentangle`, gdzie bazą było słabsze (0.01,0.01,0.1).

Kolejność: tło → overlap → TV → seed. Numer:
`id = 36*background_index + 12*overlap_index + 3*tv_index + seed_index`.

| ID | Co porównujemy |
|---|---|
| 0–2 | białe tło, brak O i TV: odtworzenie kontroli |
| 3–5 | białe tło, brak O, TV ×1: odtworzenie dawnego mocnego TV |
| 6–11 | białe tło, brak O, TV ×3 i ×10 |
| 12–23 | białe tło, O=0.03, wszystkie TV |
| 24–35 | białe tło, O=0.1, wszystkie TV |
| 36–71 | dokładnie te same ustawienia na czarnym tle: odpowiednik ID = ID−36 |

Pełna lista: `python experiments/rgb_surface/train.py --list`.

## Co pozostaje wspólne i dlaczego

Zero kroków warmupu. Katalog `warmup/` przechowuje wyłącznie inicjalizację
iteracji 0. Gaussy startują z punktów COLMAP. Początkowe Voronoi: 25k sites,
25% punktów z COLMAP, spatial extent 2. Bez nowego ręcznego AABB.
Obie reprezentacje uczą się i mają adaptacyjne populacje. K pozostaje jak w bazie.

Binaryzacja opacity Gaussianów: 0.001, cosine ramp od 1k do 4k.
Próg opacity pruning: 0.005 → 0.10 w tym samym okresie; pruning do 30k,
maksymalnie 10% populacji na krok. Dodatkowy loss obrazu
`1-mean(A_G | A_G>0.1)` z poprzedniego sweepu nie jest włączony.

**Harmonogram populacji jest odziedziczony, a nie zmieniony w tym sweepie:**
Gaussian densification do 15k, Voronoi densification do 30k. Jest to znany
potencjalny czynnik sprzyjający Voronoi. Trzymamy go jednak wspólnie w całej
macierzy, aby oddzielić efekty trzech zamówionych czynników. Stop Voronoi na
15k wymaga osobnego porównania; tego sweepu nie wolno opisywać jako jego testu.
Po 15k Gaussy nadal uczą parametrów i podlegają pruningowi, ale już nie pączkują.
Dlatego mapy pokrycia na 10k/15k są równie istotne jak wynik na 30k.

IUI3 ma zdjęcia RGB (alpha=1). Zmiana tła renderera nie zmienia GT tych zdjęć.
Nie należy przenosić tego twierdzenia na syntetyczny dataset RGBA.

## Nowy overlap: dokładna definicja

To **nowy lokalny loss**, a nie identyczny `suspect_overlap` ze starych heatmap.
Poprzedni score różniczkował tylko b,B przez próbkowanie najbliższego site.
Nowy nie pozwala ukryć medium samym wyzerowaniem B ani przez tłumienie przed
wybranym odcinkiem: całkowanie zaczyna od nowa wewnątrz lokalnego okna.

Dla 128 pikseli z aktualnego obrazu, próbkowanych z powtórzeniami lokalnym RNG:

1. Native OptiX zwraca trafienia Gaussianów. Liczymy aktywowane alpha jak renderer,
   sortujemy zdarzenia po głębokości i liczymy `w_i = alpha_i * product_{j<i}(1-alpha_j)`.
2. `A=sum(w_i)`, `mu=sum(w_i*t_i)/A`, `var=sum(w_i*(t_i-mu)^2)/A`.
3. `L=min(far,8*scene_extent)`, `h=0.02*L`.
   `confidence=A*exp(-var/h^2)`. Brak trafień daje confidence=0.
4. Okno ma długość 2h i środek `clamp(mu,h,L-h)`, żeby mieściło się w [0,L].
   Jest lokalne wokół warstwy powierzchni; nie obejmuje całego słupa wody.
5. Pomocniczy, skalarny współczynnik każdej komórki:

   `k = mean_RGB(a + b + b*B) / (3*optical_scale)`; domyślnie optical_scale=1.

6. Native scalar CUDA tracer całkuje k w lokalnym oknie:

   `tau_local = integral_window k(x) ds`

   `occupancy = 1-exp(-tau_local)`

   `L_O = mean_sampled_pixels[stop_gradient(confidence) * occupancy]`.

Kara ma wagi: 0 do 1k, liniowy wzrost 1k–2k, potem pełna waga.
Średnia obejmuje także promienie z confidence=0; nie normalizujemy przez liczbę
trafień, żeby bardzo mała grupa Gaussianów nie dostawała nieproporcjonalnej wagi.

`k` jest **polem pomocniczym regularyzacji**, nie nowym fizycznym modelem wody.
Niezależne człony a i b wymuszają koszt również wtedy, gdy B=0. Człon bB daje
gradient do koloru. Przy stałych oknach obniżanie a,b,B nie zwiększa tej kary.

Gradienty:

- do aRGB, bRGB i B w odwiedzonych komórkach;
- do sites przez położenia przeciętych granic Voronoi i długości odcinków;
- **zero do Gaussianów**: trafienia, momenty, confidence i początki promieni są odłączone;
- topologia grafu i wybór komórki startowej są dyskretne. Są różniczkowane
  długości odcinków przy aktualnej topologii, nie same operacje przebudowy grafu.

"Całe Voronoi" oznacza wszystkie rodzaje jego parametrów. Nie oznacza niezerowego
gradientu każdej komórki na każdym kroku: nieodwiedzone komórki nie dostają O,
a geometryczny gradient w jednorodnym polu powinien być zerowy.

Pomocniczy renderer ma osobny stan, near=0, far=2h, threshold=0, limit 4096
przecięć. Przy ucięciu przejścia run kończy się błędem zamiast trenować na błędnej
całce. Topologia odświeża się po revision/pointer/count; ruch sites bez zmiany
topologii jest widoczny, bo używany jest aktualny tensor positions.

Ograniczenia: okno ma heurystyczną szerokość; może usunąć trochę prawidłowej wody
przy powierzchni. Kara może się nasycić przy dużej grubości optycznej — logujemy
`saturated_fraction`. Rozproszone trafienia G mają małą confidence. W miejscach
całkowicie pozbawionych Gaussianów loss nie inicjuje ich z powietrza.

## TV

Używamy istniejącego graph Huber/TV, wagi krawędzi równe 1. Osobno wygładzamy
log(a+1e-6), log(b+1e-6), aktywowane B. Delta dla współczynników=0.1,
dla koloru=0.05. Liniowy ramp 0–250. Średnia po nieskierowanych krawędziach i RGB.
Duże TV może wygładzić zarówno fałszywe dno, jak i prawdziwe zmiany wody;
nie zakładamy z góry, że ×10 będzie najlepsze.

## Testy i uruchomienie

Stan przygotowania: **19 testów CPU przeszło**; składnia obu sbatchy i skryptu
submit jest poprawna. Pierwszy test RTX4090 (job 518571) wykrył porównanie
`UInt32` na CUDA w diagnostyce lokalnego renderera. Porównanie zostało poprawione
przez konwersję liczby przecięć do `int64`. Wymagany jest nowy pełny test RTX4090;
certyfikat jobu 518571 ma status `failed` i nie dopuszcza treningów.

W przygotowaniu lokalnie używamy CPU. Obecny node zgłasza RTX2080 Ti;
pełne testy wymagają RTX4090 przez sbatch. Certyfikat kontroluje GPU,
komplet wymaganych grup bez skipów i fingerprint kodu/configów/sbatchy.
Zmiana tych plików po certyfikacji wymaga ponownych testów.

Testy obejmują analityczną całkę dwóch komórek, gradcheck optyki i sites,
odłączenie Gaussianów, czarne tło bez Gaussianów, niezależność alpha od tła,
72 manifesty, zero-warmup, adaptację, resume i sidecary. GPU porównuje natywne
gradienty z niezależnym rendererem referencyjnym i finite differences po sites,
sprawdza brak uszkodzenia backward głównego renderu, refresh po wymianie sites,
oraz pełną rozdzielczość IUI3 z produkcyjnym budżetem rayów. Bramka obejmuje
również wcześniejsze regresje RGB i scalar, w osobnych procesach dla ABI OptiX.

**Jedna komenda: testy, a po ich sukcesie cały sweep:**

```bash
cd /shared/results/gmwilczn/foamsurf/foamsurf
bash experiments/rgb_surface/submit.sh
```

Skrypt kolejkuje array z `afterok`, `--kill-on-invalid-dep=yes` oraz przekazuje
ścieżkę certyfikatu. Każdy element array ponownie sprawdza certyfikat przed
inicjalizacją. Limit równoległości: `%7`.

**Jeżeli najpierw chcesz tylko testy:**

```bash
sbatch experiments/rgb_surface/test.sbatch
```

**Po przejściu testu o ID TEST_JOB_ID:**

```bash
sbatch --export=ALL,RGB_SURFACE_TEST_CERTIFICATE=/shared/results/gmwilczn/foamsurf/foamsurf/outputs/rgb_surface_tests/TEST_JOB_ID/passed.json experiments/rgb_surface/train.sbatch
```

CPU: `python -m pytest -q experiments/rgb_surface/test_cpu.py`.
Dry run: `python experiments/rgb_surface/train.py --variant 69 --dry-run --output-dir /tmp/surface_preview`.

## Obrazy, metryki i sposób analizy

Co 500 iteracji są metryki/checkpointy oraz obrazy pierwszych 4 kamer testowych:
`run_ID/joint/evaluation/iteration_XXXXXX/`.

Nowy podpisany `*_coverage_panel.png` ma kolumny:

**GT | HYBRID | GAUSSIAN ONLY BLACK | ALPHA GAUSSIANS | RED: A_G≤0.1 | VORONOI ONLY**.

Towarzyszą mu osobne `*_gaussian_only_black.png`, `*_gaussian_alpha.png`
i bezstratne numeryczne `*_gaussian_alpha.npz` (float32), a także `coverage.csv`.
G-only black odzyskujemy z istniejącego nieclampowanego renderu jako
`I_G_train_bg - (1-A_G)*C_bg`; nie wymaga kolejnego pełnego renderowania.
W&B dostaje `coverage/view_XX` oraz `eval/gaussian_coverage/*`.

Czerwony oznacza brak pokrycia G, ale **obejmuje również otwartą wodę**.
Nie nazywamy całej czerwieni dziurami w dnie. Alpha>0.9 na całym obrazie może
oznaczać także błędne pokrycie wody; nie jest samodzielnym celem optymalizacji.

Raport CPU po zakończeniu albo dla wybranego wspólnego checkpointu:

```bash
python experiments/rgb_surface/analyze.py outputs/rgb_surface/ARRAY_JOB_ID --iteration 30000
python experiments/rgb_surface/analyze.py outputs/rgb_surface/ARRAY_JOB_ID --iteration 15000 --output outputs/rgb_surface/ARRAY_JOB_ID/analysis_15k
```

Powstają `RESULTS.md`, `factorial.png`, `runs.csv`, `paired_deltas.csv` oraz
galerie TV osobno dla każdego tła i siły overlap. Domyślna galeria: seed42,
kamera0; `--view-index 1/2/3` z osobnym `--output` służy do pozostałych kamer.
Raport pomija runy bez żądanego checkpointu, podaje rzeczywiste n seedów i nie
miesza ich ostatnich, różnych iteracji. Pole best_psnr obejmuje historię danego
runu i wymaga oddzielnego porównania od PSNR przy wspólnym kroku.

Opcjonalnie `--masks KATALOG` czyta stałe binarne maski dna `NAZWA_KAMERY.png`
w rozdzielczości zapisanej alpha. W masce liczy `floor_holes_01` (A_G≤0.1)
i `floor_coverage_09` (A_G>0.9). Bez masek te pola pozostają NaN, nie zastępujemy
ich D/(D+B), thresholdem jasności ani maską wygenerowaną przez badany model.
Maski służą tylko do ewaluacji, nigdy do treningu. Do porównania runów używamy
dokładnie tych samych kamer/masek i sprawdzamy `floor_views`.

Analiza czynników:

1. Tło: zestawić ID i ID+36 przy tej samej wadze O, TV i seedzie.
2. O: porównać 0 → .03 → .1 przy stałym tle, TV i seedzie.
3. TV: porównać 0 → 1 → 3 → 10 przy stałym tle, O i seedzie.
4. Kombinacje: sprawdzić, czy efekt O zmienia się po włączeniu czarnego tła/TV.
5. Dynamika: 500, 2k, 5k, 10k, 15k, 20k, 30k. Czy G pokrywa więcej dna przed
   zatrzymaniem densyfikacji? Czy później Voronoi znowu buduje strukturę dna?
6. Kontrola artefaktów: wzrost alpha bez zgodnej geometrii/tekstury, Gaussy
   w wodzie, zniknięcie wody, nasycenie lossu, pełny bufor trafień, koszty czasu.

Raport końcowy ma osobno wskazać najlepszy PSNR i najlepsze pokrycie dna.
`direct`, `backscatter`, G-only PSNR oraz liczba Gaussianów są informacjami
pomocniczymi. Dowodem postępu będą zgodne rendery G-only i mapy alpha dna
w wielu kamerach, przy zachowaniu jakości hybrydy i sensownego medium.
