# FoamSurf — aktualny stan (2026-10-20)

Hybryda: Gaussy 3D (powierzchnia/dno) + medium w komórkach Voronoi (woda). Scena referencyjna: IUI3 (SeaThru-NeRF).

## Lokalizacja

- Repo: `/shared/results/gmwilczn/foamsurf/foamsurf`, branch `dev/medium_fix`. Stan sprzed porządków: tag `archive/pre-cleanup` (tam są usunięte eksperymenty: prior RadFoam, prefiks/handoff medium, skalarne regularyzacje medium, `flare_movement`, proximity/coverage, globalny kolor medium).
- Środowisko: conda `foamsurf` (`/shared/results/gmwilczn/conda/envs/foamsurf/bin/python`); trening na `rtx4090_batch` z `--exclude=c22,c11`.
- Dataset: `/shared/sets/datasets/SeathruNeRF_dataset/IUI3-RedSea-Undistorted` (`sparse/0` albo płaskie `sparse/`), split `llffhold=8`, rozdzielczość 1.
- W&B: `mlpolska/foamsurf_iui3_medium_color`.

## Konfiguracja domyślna (od 2026-10-28): OceanSplat + PPISP

`configs/seathru_default.yaml` = `configs/iui3_default.yaml` (opis niżej) + wariant `_clamp05_minit_rs_rel_oceane_isp` ze sweepu 551590:

- skala Gaussa maks. 5% sceny (`_clamp05`), kolor medium inicjalizowany ze zdjęć treningowych (`_minit`);
- harmonogram rozdzielczości ¼ → ½ od 6k → pełna od 12k (`_rs`);
- strata WaterSplatting/OceanSplat: L1 i D-SSIM dzielone przez odłączoną predykcję (`_rel`);
- OceanSplat: spójność trójokularowa, głębia epipolarna, residuum i krycie zależne od głębi, 4k–24k (`_oceane`);
- PPISP (`_isp`, `training/camera_isp.py`): ekspozycja i kolor per zdjęcie, wspólna winieta i krzywa odpowiedzi; widok testowy dostaje ekspozycję i kolor 4 najbliższych (po środku kamery) zdjęć treningowych, wagi 1/odległość (`camera_isp_neighbours`, 0 = neutralna).

Wynik (PSNR, średnia z seedów 42 i 43): IUI3 31,10, Curaçao 33,77, JG 23,49, Panama 31,42, średnio 29,95 dB; WaterSplatting (nasze runy) 29,63. `sbatch scripts/train_seathru.sbatch` (zestaw `default`) trenuje ten config na 4 scenach × 2 seedy. Pozostałe zestawy budują sufiksy na `configs/iui3_default.yaml`, więc ich nazwy zachowują znaczenie; sufiksy na nowym defaulcie: `FS_CONFIG=configs/seathru_default.yaml`.

### Poprzedni default: `configs/iui3_default.yaml`

`configs/iui3_default.yaml` = `B0_chroma100_chthr02_scov3e3_cthr90_g700k_tden_water1e2_bk` ze sweepu 542092: `configs/iui3_5k.yaml` + sigmoid na RGB Gaussów i komórek, czarne tło, oraz:

- prior chromy backscatteru, waga 1, na wszystkich promieniach z α Gaussów > 0,02;
- jednostronne pokrycie dna 3e-3 na promieniach z α > 0,9;
- strata wodna 1e-2: α Gaussów < 0,3 pchana do 0, od 4k (pełna od 6k);
- densyfikacja kompensowana tłumieniem wody (gradient / transmitancja do kamery, maks. 20×);
- próg densyfikacji 2e-4, maks. 700k Gaussów.

Wynik na IUI3 (2 seedy): best PSNR 29,32 dB, otwarta woda zajęta przez Gaussy 0,0%, dno 0–30 px od wody pokryte (α > 0,5) w 25%, 30–60 px w 64%, całe dno α > 0,99 w 70%; raport `outputs/iui3_medium_fix/report_542092_seabed_focus.html`.

Inna scena: dodaj `--set warmup.dataset.source_path=<scena> --set joint.dataset.source_path=<scena>`. Maski woda/dno (SAM) są dla widoków testowych wszystkich czterech scen (`configs/water_sam/`); widoki bez otwartej wody mają metryki wody NaN.

~~~bash
python train.py --config configs/iui3_default.yaml --output-dir outputs/<nazwa>
# dowolny klucz manifestu:
python train.py --config configs/iui3_default.yaml --output-dir ... \
  --set medium_fix.surface_coverage_weight=3.0e-3 --set joint.gaussian_adaptation.max_points=700000
~~~

Sweepy: `scripts/sweep_medium_fix.sbatch` (składnia nazw wariantów w nagłówku), raport HTML: `scripts/medium_fix_report.py`, audyt checkpointu: `scripts/medium_audit.py`.

Ścieżka syntetyczna (`configs/synthetic.yaml`: warmup Voronoi-only → podział surface/non-surface → joint) zostaje do ablacji względem WaterSplatting.

## Priory medium (`medium_fix`, `training/losses.py`)

- `chroma_*`: na rayach z alpha Gaussów > `chroma_alpha_threshold` barwa backscatteru ma mieć odcień wody (EMA z rayów otwartej wody), margines 5°.
- `surface_coverage_*`: jednostronne pokrycie dna, kara (1 − A) na rayach z A > próg; `surface_coverage_end` wyłącza ją od danej iteracji.
- `water_clear_*`: strata wodna, kara A na rayach z A < próg (domyślnie 0,3, od 4k); z pokryciem dna progi muszą zostawiać martwą strefę.
- `joint.gaussian_adaptation.gradient_attenuation_compensation`: gradient do densyfikacji dzielony przez transmitancję wody do kamery (mediana tłumienia komórek), maks. `gradient_compensation_max`.
- `under_surface_*`: kara za backscatter medium renderowanego zza nieprzezroczystego dna.

## Wyniki (skrót)

- Domyślna konfiguracja + `scov 3e-3` + ok. 800k Gaussów: średnio **29,31 dB** PSNR test (WaterSplatting w artykule: ok. 29,3 dB; nasz pomiar WS: 29,93 dB, ok. 174 FPS na 4090).
- Pruning dużych Gaussów daleko od kamer (`far_prune_*`) obniża PSNR i nie czyści wody — odrzucony (sweep 534880). Wodę czyści strata wodna, a granicę dna poprawia densyfikacja kompensowana tłumieniem (sweepy 541528 i 542092).
- Problem otwarty: pas dna 0–30 px przy wodzie jest pokryty Gaussami tylko w ok. 25%.
- Metryki wody i dna: maski SAM widoków testowych wszystkich czterech scen SeaThru (`configs/water_sam/<scena>.npz`, `scripts/sam_water_mask.py`; Panama MTN_1539 i MTN_1547 bez otwartej wody); raport skupiony na dnie: `scripts/seabed_focus_report.py`.
- Raporty: `outputs/iui3_medium_fix/report_*.html`.

## Testy

`python -m pytest -q` (pełny zestaw, z testami GPU: `sbatch scripts/smoke_tests.sbatch` na 4090). Bez zbudowanych rozszerzeń CUDA dla lokalnej karty: `-k "not short_real and not finite_difference"`.

Stare katalogi runów wciąż się ładują: nieaktywne klucze usuniętych funkcji są pomijane (`drop_legacy_keys`, `drop_legacy_medium_fix`), a aktywne powodują błąd.
