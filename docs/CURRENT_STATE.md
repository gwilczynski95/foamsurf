# FoamSurf — aktualny stan do audytu (2026-10-07)

Ten dokument dotyczy aktualnego kandydata IUI3: **sigmoid na RGB Gaussów oraz RGB komórek Voronoi**. To nie jest historyczny A4, mimo że bazuje na tym samym pliku YAML. Wynik jest obiecujący, ale nie dowodzi pełnego rozdzielenia powierzchni i medium.

## Lokalizacja i stan

- Repo: `/shared/results/gmwilczn/foamsurf/foamsurf`; branch `feature/voronoi-rgb-medium`; HEAD `0a39fe1` w chwili spisania.
- Środowisko: conda `foamsurf`, Python `/shared/results/gmwilczn/conda/envs/foamsurf/bin/python`; trening/testy GPU na RTX4090.
- Dataset: `/shared/sets/datasets/SeathruNeRF_dataset/IUI3-RedSea-Undistorted`, split COLMAP `llffhold=8`, rozdzielczość 1. Cztery widoki testowe: `MTN_5894`, `MTN_5903`, `MTN_5911`, `MTN_5928`.
- Bieżący run: `outputs/iui3_sigmoid/525722/both_sigmoid/`. **Źródło prawdy o konfiguracji:** `pipeline_config.json` oraz wygenerowane `joint.yaml` i `warmup.yaml` w tym katalogu.
- Najlepszy checkpoint: `both_sigmoid/joint/checkpoints/best.pt`, iteracja **19 000**, średni test hybrid PSNR **28,843 dB**. Trening ukończył 30k iteracji; końcowy test PSNR **28,493 dB**. Nie mylić best i final.
- Pełny lokalny raport: `outputs/iui3_sigmoid/525722/report.html`. Surowe metryki: `both_sigmoid/joint/metrics/train.csv` i `eval.csv`. Obrazy best: `both_sigmoid/joint/evaluation/iteration_019000/`.
- W&B: encja `mlpolska`, projekt `foamsurf_iui3_sigmoid`; lokalne pliki umożliwiają audyt bez W&B.

**Zmiany nie są jeszcze commitowane.** Aktualnie zmienione są `train.py`, `evaluation/evaluator.py`, `tests/test_train.py`, ten dokument, a `scripts/train_sigmoid.sbatch` jest nowym plikiem. Audyt należy prowadzić na roboczym drzewie, nie tylko na `HEAD`.

## Model i pipeline

IUI3 ma **0 iteracji warmupu**. Etap `warmup` tylko inicjalizuje i zapisuje stan Voronoi; dopiero `joint` optymalizuje obie reprezentacje przez 30k iteracji. To nie jest synthetic warmup ani przejęcie modelu z checkpointu RadFoam.

- Gaussy startują z **21 907 punktów COLMAP**; są densyfikowane do iteracji 15k i prunowane pod względem opacity do 30k. Końcowy run `both_sigmoid` ma **293 380** Gaussów, zero resetów opacity.
- Voronoi startuje z **5 000 losowanych przestrzennie sites**. `spatial_colmap_fraction=0`: punkty COLMAP służą Gaussom, nie jako sites Voronoi. Zakres samplowania wyznaczają robust bounds sceny i pozycje kamer. Liczba cells w joint jest stała (`voronoi_adaptation.enabled=false`, zero zdarzeń densyfikacji/pruningu), ale pozycje sites i parametry medium są optymalizowane.
- Medium ma tryb `rgb_backscatter`: po trzy dodatnie współczynniki attenuacji i backscatter (aktywacja `softplus` surowych parametrów) oraz RGB na każdą komórkę. W obecnym runie **kolor komórki = sigmoid(raw_color)**.
- Wygląd Gaussów to RGB bez SH; w obecnym runie **kolor Gaussa = sigmoid(parametru wyglądu)**. Opacity Gaussa ma osobny sigmoid, niezależny od eksperymentu z kolorem. Oba RGB należą do (0, 1), co nie wymusza poprawnej semantyki ani nie wyklucza saturacji blisko 1.
- Początkowy kolor Voronoi wynosi fizycznie 0,5 zarówno w wariancie DC, jak i sigmoidowym. Przy sigmoidzie Gaussy startują z tych samych odczytanych kolorów, tylko zapisanych jako logity.
- Renderer sortuje zdarzenia Gaussów i odcinki komórek wzdłuż rayu; bezpośredni składnik powierzchni/tła jest tłumiony przez medium, a backscatter akumuluje się po odcinkach. Hybryda sumuje te składniki; tło jest białe. **Direct nie jest Gaussian-only**: zawiera tłumienie przez Voronoi. Backscatter nie jest całym Voronoi-only.
- Rekonstrukcja: `l1_dssim` z wagą DSSIM 0,2. Aktywne także: binarizacja opacity (waga 0,001; start 1k; rampa cosine 3k), lokalny overlap (waga 0,03; start 1k; rampa 1k; 128 rays) i wygładzanie parametrów RGB medium (attenuation 0,1; backscatter 0,1; color 1,0; rampa 250). **Graph TV ma wagę 0**. Density cap, opaque-if-used, optical-thickness sparsity, repulsion oraz direct teacher są wyłączone.
- Uczenie korzysta z `full_image_batch=true` i wyboru kamer bez powtórzeń; nie losuje pikseli ze wszystkich obrazów jako jednej mieszanej puli.

Główne pliki: `train.py` (jedyny publiczny entrypoint), `training/colmap_joint.py` (bootstrap), `training/colmap_joint_trainer.py` (joint), `foamsurf_model/gaussians.py` i `medium.py` (parametry/aktywacje), `renderer/foamsurf/rgb_medium.py` oraz reszta `renderer/` (kompozycja), `evaluation/evaluator.py` (ewaluacja).

## Odtworzenie treningu

**Samo** `python train.py --config configs/iui3_5k.yaml` uruchamia stary wariant A4 z `Gaussian=identity` i `Voronoi=dc`. Aktualne warianty mają krótkie configi dziedziczące pełny A4: `configs/iui3_sigmoid_both.yaml` (oba sigmoidy) i `configs/iui3_sigmoid_gaussians.yaml` (tylko Gaussy; medium DC). Można uruchomić je jednym `--config`; poniższa komenda z flagami jest równoważna pierwszemu configowi:

~~~bash
cd /shared/results/gmwilczn/foamsurf/foamsurf
conda activate foamsurf
python train.py --config configs/iui3_5k.yaml \
  --output-dir outputs/iui3_sigmoid/manual_both \
  --gaussian-rgb-activation sigmoid \
  --medium-color-activation sigmoid \
  --wandb-project foamsurf_iui3_sigmoid
~~~

`scripts/train_sigmoid.sbatch` definiuje trzy warianty: indeks 0 = sigmoid tylko na Gaussach; 1 = tylko na medium; 2 = **oba sigmoidy**. Tylko aktualny kandydat: `mkdir -p slurm_logs && sbatch --array=2 scripts/train_sigmoid.sbatch`. Skrypt ustawia env `foamsurf`, partycję `rtx4090_batch` i zapisuje do `outputs/iui3_sigmoid/<job-id>/both_sigmoid/`.

Osobny `configs/synthetic.yaml` uruchamia Voronoi-only warmup, podział surface/non-surface weighted valley i joint. **Ta ścieżka nie dotyczy opisanego tu runu IUI3.**

## Wyniki ablacji RGB

Wszystkie trzy nowe warianty ukończyły 30k iteracji; seed 42. Historyczny A4 pochodzi z wcześniejszego sweepu. PSNR to średnia z czterech widoków testowych.

| Wariant | Gauss / medium RGB | Best PSNR test (iter.) | Final PSNR test | G-only black PSNR* | Alpha G > 0,5 | Proxy udział medium |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A4 historyczny | identity / DC | 28,716 (21k) | 28,510 | 9,76 | 52,36% | 87,43% |
| `gaussian_sigmoid` | sigmoid / DC | 28,742 (19k) | 28,064 | 11,96 | 47,05% | 71,46% |
| `medium_sigmoid` | identity / sigmoid | 28,733 (21k) | 28,446 | 9,58 | 49,66% | 89,66% |
| **`both_sigmoid`** | **sigmoid / sigmoid** | **28,843 (19k)** | **28,493** | **12,15** | **48,78%** | **70,03%** |

`*` PSNR Gaussian-only na czarnym tle względem **podwodnego GT** jest diagnostyką, nie miarą fizycznie poprawnej powierzchni bez wody. Diagnostyki w tabeli są z iteracji najlepszej danego wariantu. Proxy udział medium to udział wag renderera, a nie udział semantycznie poprawnego modelowania wody.

Best PSNR `both_sigmoid` jest o **0,127 dB** wyższy od best A4, lecz final PSNR o ok. **0,016 dB niższy**. Jeden seed i tak małe różnice nie wystarczają do mocnej tezy o przewadze jakości. Nie ma tu kontrolowanego benchmarku z Water-Splatting na identycznych ustawieniach.

## Obserwacje i ograniczenia

- W `both_sigmoid` surowy Gaussian-only nie ma pikseli z kanałem RGB > 1; dawne A4 taki problem miało. Udział prawie białych pikseli w obszarze alpha G > 0,5 spadł do ok. **1,2%**. Kolory dna w Gaussian-only black wyglądają naturalniej, lecz nadal są **czarne dziury**.
- Alpha G > 0,5 obejmuje 48,78% wszystkich pikseli, wobec 52,36% w A4. **To nie jest maska dna**; sama globalna liczba nie mówi, ile rzeczywistego dna pokrywają Gaussy.
- `voronoi_only` nadal pokazuje kształt i barwy dna; pojawiają się jasne plamy o strukturze komórek. Zatem oba sigmoidy **nie rozwiązują disentanglementu**. Medium wciąż pomaga odtworzyć część powierzchni.
- `eval/gaussian_raw_over_1_fraction` mierzy odsetek pikseli z dowolnym surowym kanałem G > 1, przed przycięciem do PNG. `eval/gaussian_near_white_on_covered` dotyczy pikseli z alpha G > 0,5. `eval/gaussian_coverage/coverage_gt_0.5` dotyczy całego obrazu, nie tylko dna. Do inspekcji rzeczywistych kolorów samych Gaussów służy `gaussian_only_black`, **nie direct**.
- Wysokie hybrid PSNR nie oznacza dobrego Gaussian-only. Spadek test PSNR między 19k a 30k wymaga porównania train/test i polityki wyboru checkpointu; sam w sobie nie dowodzi przyczyny przetrenowania.

## Lista kontrolna dla audytora

1. Sprawdzić niecommitowany diff i manifest: oba sigmoidy powinny działać **przed rendererem**, bez podwójnej aktywacji, dodatkowego tone mappingu lub clippingu używanego do lossu.
2. Potwierdzić zerowy warmup, przestrzenną inicjalizację 5k Voronoi bez sites z COLMAP i to, że stała populacja komórek nie oznacza zamrożenia ich parametrów.
3. Zweryfikować definicje `direct`, `backscatter`, `gaussian_only_black` i `voronoi_only` oraz znaczenie masek diagnostycznych. Nie utożsamiać wag proxy lub `D/(D+B)` z semantyką reprezentacji.
4. Obejrzeć te same cztery widoki w `both_sigmoid/joint/evaluation/iteration_019000/`: GT, hybrid, G-only black, alpha G, Voronoi-only. Osobno ocenić coverage dna, prawdziwe kolory Gaussów i kolor dna w medium.
5. Potwierdzić test split, best checkpoint na 19k, średni PSNR oraz przebiegi train/test. Własne hipotezy wyraźnie odróżniać od faktów. Nie wyciągać wniosków statystycznych z jednego seeda.

Testy implementacji wykonane po dodaniu configów na RTX4090: `python -m pytest -q tests/test_train.py`: **8 passed**, w tym krótkie uruchomienia GPU wariantów sigmoid i resume. `sbatch --test-only scripts/train_sigmoid.sbatch` również przeszedł. To weryfikuje ścieżkę kodu, nie pełną poprawność naukową rozdzielenia reprezentacji.
