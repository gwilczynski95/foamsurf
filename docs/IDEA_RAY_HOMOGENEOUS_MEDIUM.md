# Pomysł: medium jednorodne wzdłuż promienia (prior w stylu WaterSplatting)

Stan: pomysł, nic nie zaimplementowane (2026-10-21). Kontekst: [CURRENT_STATE.md](CURRENT_STATE.md), cel: wygrać z WaterSplatting na 4 scenach SeaThru.

## Skąd pomysł

WaterSplatting nie renderuje osobno Gaussów i medium. Ma jeden przebieg (`water-splatting/water_splatting/cuda/csrc/forward.cu:436-496`), w którym medium jest przeplatane z Gaussami wzdłuż promienia:

- Gauss `i` jest tłumiony przez `exp(-σ_attn · z_i)`.
- Backscatter odcinka między Gaussami to `T · (exp(-σ_bs·z_{i-1}) − exp(-σ_bs·z_i)) · c_med`, plus ogon za ostatnim Gaussem.
- Wyjścia `rgb_object`, `rgb_medium` i `rgb_clear` to tylko rozbicie tego samego przebiegu.

Nasz hybrydowy renderer robi w zasadzie to samo.

**Prawdziwa różnica jest w parametryzacji medium.** U WS `σ_attn`, `σ_bs` i `c_med` są **stałe wzdłuż całego promienia**: liczy je MLP z kierunku patrzenia (`water_splatting.py:767-787`). U nas każda z 5000 komórek Voronoi ma własne parametry, czyli medium jest niejednorodne w przestrzeni.

Ta swoboda pasuje do dwóch błędów, które widzimy:

- Na Curaçao medium zasłania daleki fragment dna ciemną smugą (MTN_1304).
- Na JG błąd niskich częstotliwości (kolor/mgła) wynosi 35e-4 vs 20.7e-4 u WS (sweep 543997, `outputs/seathru/report_543997_sh.html`).

U WS woda nie może „udawać” dna, bo wzdłuż promienia jest jednorodna.

**Odrzucony wariant:** trening z osobnymi renderami złożonymi wzorem SeaThru (`J·e^{-β_D z} + B∞(1−e^{-β_B z})`) i ewaluacja hybrydą. Ten wzór zgadza się z hybrydą tylko przy całkiem nieprzezroczystej powierzchni i jednorodnym medium. W wodzie i na półprzezroczystych krawędziach rozbieżność trening/ewaluacja poszłaby prosto w błąd.

## Definicje

Dla promienia `r` i współczynnika `x ∈ {attenuation[3], backscatter[3]}` średnia wzdłuż promienia ważona długością to

`x̄_r = Σ_s ℓ_s · x_{cell(s)} / Σ_s ℓ_s`,

gdzie `s` to odcinki promienia w komórkach na `[near, far]`. Kolor medium uśredniamy ważąc wkładem do backscatteru (`ℓ_s · backscatter_s`), żeby ciemne, rzadkie komórki nie przesuwały koloru.

**Otwarta decyzja:** czy średnia obejmuje komórki guard (poza bboxem), czy tylko odcinek wewnątrz bboxa. Wstępnie: tylko wewnątrz bboxa, a guard bierze `x̄_r`.

**Mieszanie:** `x_eff(s) = (1 − w) · x̄_r + w · x_{cell(s)}`, gdzie `w ∈ [0, 1]`.

- `w = 1` to obecny renderer.
- `w = 0` to medium jednorodne wzdłuż promienia, jak u WS.

## Punkt 1: diagnoza na istniejących checkpointach (bez treningu)

**Pytanie:** ile nasze wytrenowane modele korzystają z niejednorodności medium i gdzie.

**Procedura:**

1. Weź końcowe checkpointy z `outputs/seathru/543588` (kontrola) i `543997` (`_sh3`, `_dres`), seed 42, 4 sceny.
2. Wyrenderuj widoki testowe w trzech trybach:
   - **(a)** zwykła hybryda (`w = 1`);
   - **(b)** medium jednorodne per kamera: wszystkie komórki dostają parametry komórki zawierającej kamerę, czyli najbliższego site'u do `camera.origin`. Wystarczy podmienić tensory w `VoronoiMediumModel.render_scene()` (`foamsurf_model/medium.py:179`), bez zmian w kernelu;
   - **(c)** średnia po promieniu (`w = 0`). Wymaga trybu w kernelu z punktu 2. Jeśli kernela jeszcze nie ma, zrób (b), a (c) później.
3. Policz PSNR całości, dna i wody (maski SAM, `configs/water_sam`) oraz rozkład błędu na niskie i wysokie częstotliwości. Gotowe funkcje: `score_views` i `frequency_split` w `scripts/variants_vs_ws_report.py`.
4. Dodatkowo policz statystyki niejednorodności:
   - per promień: odchylenie std współczynników wzdłuż promienia, ważone długością;
   - per obraz: rozrzut `x̄_r` między pikselami;
   - mapa `|x̄_r − x_{cell przy dnie}|` nałożona na maskę dna.

**Interpretacja spadku PSNR (b)/(c) względem (a):**

| gdzie spada | znaczenie | wniosek |
|---|---|---|
| prawie nigdzie | niejednorodność jest nieużywana | curriculum jest bezpieczne, zysk tylko z lepszego rozdzielenia na starcie |
| głównie na **dnie** | medium robi robotę Gaussów (np. smuga na Curaçao) | potwierdza hipotezę, curriculum ma największy sens |
| głównie w **wodzie** | niejednorodność jest potrzebna (np. gradient głębokości na IUI3) | curriculum musi oddać swobodę w końcówce, czyli kończyć z `w = 1` |

Uwaga: modele wytrenowano z niejednorodnym medium, więc jakiś spadek będzie zawsze. Liczy się, **gdzie** występuje, a nie jego wartość bezwzględna.

**Wynik:** HTML z tabelą (scena × tryb × region) i parą obrazów (a)/(c) z mapą różnic dla 1–2 widoków na scenę.

## Punkt 2: curriculum `_raymed`

Trenuj z `w = 0` przez pierwsze `T_h` iteracji, potem rampą liniową do `w = 1` w ciągu `T_r` iteracji, a dalej normalnie. Ewaluacja i wszystkie raporty zawsze idą zwykłą hybrydą (`w = 1`), więc końcowy model jest spójny z ewaluacją.

Przy `w = 0` gradient przez `x̄_r` trafia do **wszystkich** komórek na promieniu, proporcjonalnie do `ℓ_s`. Komórki są więc trenowane od początku i przejście do `w = 1` nie startuje od wartości inicjalizacyjnych. Rampa wygładza skok między średnią a wartością komórki.

**Konfiguracja** (nowe pola w `ModelConfig` albo `OptimizationConfig`, `training/config.py`):

- `medium_ray_homogeneous_until: int = 0` (`T_h`; 0 = wyłączone, zachowanie jak dziś);
- `medium_ray_homogeneous_ramp: int = 2000` (`T_r`);
- `medium_ray_homogeneous_mode: str = "ray_mean"` (`"ray_mean"` | `"camera_cell"`).
  - `camera_cell` to tani prototyp bez kernela: per obraz stała z komórki kamery. Sampler losuje jedną kamerę na krok (`training/sampler.py:32`), więc to jest dobrze zdefiniowane.
  - Wada `camera_cell`: gradient dostają tylko komórki, w których stoją kamery.
- `w(iteration)` liczy trener (`training/trainer.py`) i przekazuje rendererowi; ewaluacja zawsze `w = 1`.

**Implementacja `ray_mean`** (fused kernel RGB, `renderer/native/hybrid_cuda.cu`):

- **forward** (`rgb_forward_kernel`, l. 1269): najpierw przebieg po komórkach na `[near, far]` sumujący `ℓ_s · x` i `ℓ_s` (tylko traversal Voronoi, bez Gaussów), potem obecne komponowanie z `x_eff(s)`. Przy `w = 1` pierwszy przebieg pomijamy, więc obecna ścieżka zostaje bez zmian.
- **backward** (`rgb_backward_kernel`, l. 1772; `rgb_backward_medium`, l. 1635): `∂L/∂x_cell(s) = w · g_s + (1 − w) · (ℓ_s / Σℓ) · Σ_t g_t`, gdzie `g_s` to obecny gradient dla odcinka `s`. Potrzebne `Σ_t g_t` i `Σℓ` per promień: drugi przebieg albo akumulacja w pierwszym. Gradient po długościach `ℓ_s`, czyli po pozycjach site'ów, dostaje człon od średniej. Jeśli `sites` się nie uczą w tej fazie, można go pominąć, ale trzeba to sprawdzić w `optimizer.py`.
- **Python:** `_RgbCudaFunction` (`renderer/foamsurf/cuda.py:186`) dostaje `w` jako skalar. `compose_fused` (`renderer/foamsurf/rgb_medium.py:10`) przekazuje go dalej. `six_pass` może zgłaszać błąd dla `w < 1`.
- **Referencja:** rozszerzyć `reference_segments` (`renderer/foamsurf/rgb_medium.py:45`) o `w`. To zwykły autograd, który posłuży za wyrocznię w testach.

**Testy** (obowiązkowe, `tests/`; testy GPU idą na 4090 przez `smoke_tests.sbatch`):

1. **CPU:** referencja z `w = 0` na scenie z różnymi komórkami = referencja z `w = 1` na scenie, w której każda komórka ma wartości `x̄_r`.
2. **CPU:** gradient referencji przy `w ∈ {0, 0.5}` zgadza się z różnicami skończonymi.
3. **GPU:** kernel z `w = 1` = obecny kernel (bitowo albo do 1e-6) na losowej scenie.
4. **GPU:** kernel vs referencja: forward i gradienty (attenuation, backscatter, kolor medium, kolory i opacity Gaussów) przy `w ∈ {0, 0.5}`.
5. **GPU:** scena z jednorodnym medium: wynik nie zależy od `w`.
6. Harmonogram `w(iteration)`: 0 do `T_h`, liniowo do 1, potem 1. Ewaluacja zawsze 1.
7. Krótki prawdziwy trening z `T_h > 0` (wzór: `tests/test_ws_recipe.py`): przechodzi, loguje `w` i zapisuje checkpoint.
8. `camera_cell`: wszystkie komórki dostają wartości komórki najbliższej kamerze. Gradient trafia tylko do tej komórki.

**Sweep** (`scripts/train_seathru.sbatch`, nowy sufiks w `variant_overrides`):

- `_rm<N>` → `medium_ray_homogeneous_until = N·1000`, mode `ray_mean`;
- `_rmc<N>` → to samo z `camera_cell`.

Proponowany zestaw na bazie najlepszego wariantu ze sweepu 544488 (albo `_sh3_dres`): `_rm5`, `_rm10`, `_rm15`, `_rmc10`, 4 sceny, seedy 42 i 43, maks. 7 jednocześnie. Kontrola to ta sama baza bez sufiksu.

**Kryteria sukcesu** (raport `scripts/variants_vs_ws_report.py`, ostatnia iteracja, średnia z 2 seedów):

- JG i IUI3: spadek błędu niskich częstotliwości i wzrost PSNR wody (IUI3: 34.5 vs 39.7 u WS);
- Curaçao: wzrost PSNR dna, brak smugi na MTN_1304;
- brak regresji ponad szum seeda (Curaçao ma ok. 1.7 dB rozrzutu, patrz niżej).

## Uwaga o szumie

Na Curaçao `_g1200k` nigdy nie przekroczył ~490k Gaussów, więc limit 1.2M nie zadziałał. To w praktyce konfiguracja kontroli, a wyszło −1.73 dB względem kontroli. Pojedynczy seed nie wystarcza do wniosków poniżej ~1 dB.
