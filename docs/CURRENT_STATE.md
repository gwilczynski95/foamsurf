# FoamSurf — aktualny stan (2026-10-20)

Hybryda: Gaussy 3D (powierzchnia/dno) + medium w komórkach Voronoi (woda). Scena referencyjna: IUI3 (SeaThru-NeRF).

## Lokalizacja

- Repo: `/shared/results/gmwilczn/foamsurf/foamsurf`, branch `dev/medium_fix`. Stan sprzed porządków: tag `archive/pre-cleanup` (tam są usunięte eksperymenty: prior RadFoam, prefiks/handoff medium, skalarne regularyzacje medium, `flare_movement`, proximity/coverage, globalny kolor medium).
- Środowisko: conda `foamsurf` (`/shared/results/gmwilczn/conda/envs/foamsurf/bin/python`); trening na `rtx4090_batch` z `--exclude=c22,c11`.
- Dataset: `/shared/sets/datasets/SeathruNeRF_dataset/IUI3-RedSea-Undistorted` (`sparse/0` albo płaskie `sparse/`), split `llffhold=8`, rozdzielczość 1.
- W&B: `mlpolska/foamsurf_iui3_medium_color`.

## Konfiguracja domyślna

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
