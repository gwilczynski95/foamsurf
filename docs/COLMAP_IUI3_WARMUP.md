# Natywny warmup COLMAP / IUI3 w FoamSurf

## Zakres

Implementacja ścieżki COLMAP lokalnego RadFoam wewnątrz natywnego warmupu
FoamSurf, bez importowania checkpointu RadFoam. Nie zmienia domyślnego LEGO
ani jego parametrów treningu. Zmiana kalibracji dotyczy kamer COLMAP; kamery
Blendera nadal korzystają z dotychczasowej ścieżki FOV.

Repo: `/shared/results/gmwilczn/foamsurf/foamsurf`.
Env: `/shared/results/gmwilczn/conda/envs/foamsurf`.
Dane: `/shared/sets/datasets/SeathruNeRF_dataset/IUI3-RedSea-Undistorted`.
Źródłowe obrazy i model COLMAP nie są modyfikowane.

## Kontrakt inicjalizacji

`WarmupProfile.site_initialization: pcd` wybiera nową ścieżkę.
`training/warmup_pcd.py` implementuje:

- `floor(0.9 * N_COLMAP)` próbek ze zwracaniem, jitter N(0, 0.01²);
- 5000 dodatkowych sites N(0, 10²) wokół początku układu, jak w RadFoam;
- surowe density U(0,1) dla próbek COLMAP i -0.5 dla outer sites;
- fizyczne density `softplus(raw, beta=10)`, activation_scale=1;
- neutralny kolor RGB 0.5, bez inicjalizacji z kolorów chmury;
- parametryzację RadFoam SH0/DC: `max(0, 0.5 + C0 * raw_color)`,
  `C0=0.28209479177387814`, bez zależności od kierunku.

IUI3 ma 21907 punktów, więc startuje z 19716 + 5000 = **24716 komórek**.
Wartość `model.medium_points: 32` w wrapperze jest tylko placeholderem:
rzeczywista liczba jest odczytywana z COLMAP przed budową modelu i logowaniem.
Liczby nie ustalamy ręcznie dla każdej sceny. Model COLMAP zachowuje swoją skalę
i położenie; nie ma normalizacji ani przycinania do boxa LEGO.

Automatyczny AABB z pełnej chmury i kamer pozostaje metadanymi modelu;
organiczna adaptacja warmupu nie ogranicza nowych punktów do tego AABB.
Ręcznie zadany box dla PCD jest odrzucany. Outer sites są w konfiguracji
referencyjnej uczone i objęte organiczną adaptacją, nie stanowią sztywnej ściany.

## Kalibracja i dane

Loader przenosi `fx, fy, cx, cy`, przeskalowane do faktycznego rozmiaru obrazu.
Pełne promienie, sampler jednej kamery, sampler mieszany i mapa błędu adaptacji
używają tej samej kalibracji. Obsługiwane są undistorted PINHOLE/SIMPLE_PINHOLE;
zniekształcone modele nie są po cichu traktowane jako pinhole.
Uwzględniono też principal point w obliczaniu projekcji wsparcia Gaussianów.

RGB otrzymuje alpha=1, więc opacity loss jest nadzorem końcowej akumulacji,
nie maską oddzielającą wodę od solid. Nie ma GT osobnych warstw.
Split: posortowane nazwy zdjęć, co ósme do testu — 25 train, 4 test.
Nie deklarujemy zgodności ze splitem publikacyjnym WaterSplatting.
Jeśli nie ma `images_2/images_4`, loader zmniejsza obrazy w pamięci (PIL BILINEAR).

## Ustawienia porównania

Wrapper `configs/end_to_end_iui3.yaml`:

- seed 42, białe tło, RGB-only;
- natywny warmup 5000 iteracji, batch 1000000 mieszanych promieni;
- rozdzielczość 4 → 2 → 1 w krokach 0, 150, 500;
- SmoothL1 beta=1 + opacity MSE (waga 1) + depth-quantile (waga 1e-4,
  rampa 10000 kroków, zgodna z referencją RadFoam trenowaną przez 20k);
- harmonogramy warmupu: sites 2e-4 → 5e-6 / 18k, density 0.1 → 0.01 / 20k
  z warmupem 2k, color 0.005 → 0.0005 / 20k, eps Adam 1e-15;
- organiczny wzrost od 2k, horyzont 11k, factor 1.15, target 2097152;
- near=0, far=1000; bez SH, bez regularyzacji disentanglement w warmupie.

Wariant end-to-end następnie używa weighted_valley_w1_s2, inicjalizacji do
100k Gaussianów z wybranych sites i zwykłego etapu hybrydowego 30k.
Liczba Gaussianów zależy również od dostępnych sites SURFACE. Medium pozostaje
uczone, bez joint split/prune. Nie naprawiamy tu wcześniejszego błędu joint
densyfikacji i nie włączamy tej ścieżki.

## Checkpoint / resume / handoff

Warmup zapisuje oryginalne parametry density beta10 oraz DC i stany
optymalizatora do sekcji `warmup`. Wznowienie odtwarza również bazowy budżet
populacji, RNG i harmonogram, z uwzględnieniem permutacji topologii.

Sekcja `model` pozostaje kompatybilna ze standardowym FoamSurf: density beta1
i kolor sigmoid. Przy eksporcie kolory DC ograniczamy do [1e-6, 1-1e-6],
ponieważ reprezentacja hybrydowa nie przechowuje kolorów poza [0,1]. Liczba
wartości >1 jest zapisywana jako `warmup.color_clipped_values`. Jest to jawne
ograniczenie handoffu, nie ograniczenie uczenia kolorów w samym warmupie.

Nie obiecujemy bitowej identyczności treningu z RadFoam: pozostają inny backend
renderera, próbkowanie batchy i kolejność RNG/topologii. Inicjalizacja PCD ma
test tensor-po-tensor względem tych samych operacji referencyjnych; promienie
są porównywane z pycolmap. W ocenie jakości porównujemy te same iteracje,
split, rozdzielczość i tło. Trzeba obserwować truncation promieni i wpływ far.

## Uruchamianie

Testy można wykonać lokalnie na RTX4090 przez `scripts/colmap_warmup_gate.py`,
albo sbatchem. Launcher treningu wymaga certyfikatu pasującego do kodu.

```bash
cd /shared/results/gmwilczn/foamsurf/foamsurf
test_job=$(sbatch --parsable scripts/test_colmap_warmup.sbatch)
test_job=${test_job%%;*}
sbatch --dependency=afterok:${test_job} \
  --export=ALL,COLMAP_TEST_CERTIFICATE=$PWD/outputs/colmap_warmup_tests/${test_job}/passed.json \
  scripts/train_colmap.sbatch
```

Domyślnie uruchamia się **sam warmup 5k**:
`outputs/colmap_warmup/<JOB_ID>/`. Wariant 20k do porównania pełnej referencji:
do `--export` dodać `ITERATIONS=20000`.
Pełne end-to-end: dodać `TRAIN_MODE=end_to_end`; wyniki trafią do
`outputs/end_to_end_iui3/<JOB_ID>/`.
Konfiguracja używa W&B offline. Nic nie jest automatycznie wysyłane online.

Bez SLURM, w poprawnie ustawionym env GPU:

```bash
python scripts/train_colmap_warmup.py --output-dir outputs/colmap_warmup/manual
python scripts/train_end_to_end.py --config configs/end_to_end_iui3.yaml \
  --output-dir outputs/end_to_end_iui3/manual
```

Są to alternatywne uruchomienia, nie dwa kolejne kroki: drugie samo robi warmup.
Sam warmup można wznowić przez `--resume <checkpoint>` z identycznym profilem
i zaplanowaną długością. Nie traktować checkpointu warmupu jako bezpośredniego
resume zwykłego hybrydowego Trainer — używać handoffu.

## Testy i interpretacja

`tests/colmap_warmup/` obejmuje kalibrację na realnych kamerach w trzech
rozdzielczościach, samplery, inicjalizację, checkpoint/resume, organiczną
densyfikację, podział, handoff i trening hybrydowy. Test handoffu używa Otsu
na celowo niedotrenowanym warmupie 6 kroków; nie dowodzi jakości weighted valley
w 5k. Końcowa bramka obejmuje też istniejące regresje warmupu i end-to-end LEGO.

Osobny test produkcyjny wykonuje 4 kroki na pełnej populacji i zdjęciach
1383×917, z milionem promieni oraz przyspieszonym eventem densyfikacji.
To test poprawności startu i operacji, nie dowód zbieżności 5k/20k ani gwarancja
pamięci przy docelowych dwóch milionach sites. Ocena podziału wody i solid
wymaga obejrzenia faktycznie wytrenowanego warmupu.
