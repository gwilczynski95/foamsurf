# FoamSurf — aktualny stan (2026-10-15)

Hybryda: Gaussy 3D (powierzchnia/dno) + medium w komórkach Voronoi (woda). Scena referencyjna: IUI3 (SeaThru-NeRF).

## Lokalizacja

- Repo: `/shared/results/gmwilczn/foamsurf/foamsurf`, branch `dev/medium_fix`. Stan sprzed porządków: tag `archive/pre-cleanup` (tam są usunięte eksperymenty: prior RadFoam, prefiks/handoff medium, skalarne regularyzacje medium, `flare_movement`, proximity/coverage, globalny kolor medium).
- Środowisko: conda `foamsurf` (`/shared/results/gmwilczn/conda/envs/foamsurf/bin/python`); trening na `rtx4090_batch` z `--exclude=c22,c11`.
- Dataset: `/shared/sets/datasets/SeathruNeRF_dataset/IUI3-RedSea-Undistorted` (`sparse/0` albo płaskie `sparse/`), split `llffhold=8`, rozdzielczość 1.
- W&B: `mlpolska/foamsurf_iui3_medium_color`.

## Konfiguracja domyślna

`configs/iui3_default.yaml` = `B0_chroma100_bk`: `configs/iui3_5k.yaml` + sigmoid na RGB Gaussów i komórek, czarne tło, prior chromy backscatteru o wadze 1,0.

~~~bash
python train.py --config configs/iui3_default.yaml --output-dir outputs/<nazwa>
# dowolny klucz manifestu:
python train.py --config configs/iui3_default.yaml --output-dir ... \
  --set medium_fix.surface_coverage_weight=3.0e-3 --set joint.gaussian_adaptation.max_points=700000
~~~

Sweepy: `scripts/sweep_medium_fix.sbatch` (składnia nazw wariantów w nagłówku), raport HTML: `scripts/medium_fix_report.py`, audyt checkpointu: `scripts/medium_audit.py`.

Ścieżka syntetyczna (`configs/synthetic.yaml`: warmup Voronoi-only → podział surface/non-surface → joint) zostaje do ablacji względem WaterSplatting.

## Priory medium (`medium_fix`, `training/losses.py`)

- `chroma_*`: na rayach z alpha Gaussów > 0,5 barwa backscatteru ma mieć odcień wody (EMA z rayów otwartej wody), margines 5°.
- `surface_coverage_*`: jednostronne pokrycie dna, kara (1 − A) na rayach z A > 0,5.
- `under_surface_*`: kara za backscatter medium renderowanego zza nieprzezroczystego dna.

## Wyniki (skrót)

- Domyślna konfiguracja + `scov 3e-3` + ok. 800k Gaussów: średnio **29,31 dB** PSNR test (WaterSplatting w artykule: ok. 29,3 dB; nasz pomiar WS: 29,93 dB, ok. 174 FPS na 4090).
- Problem otwarty: przy dużej liczbie Gaussów część z nich (ok. 0,4%) wchodzi w otwartą wodę — duże i daleko od kamer. Następny krok: pruning Gaussów zbyt dużych względem odległości od najbliższej kamery treningowej.
- Raporty: `outputs/iui3_medium_fix/report_*.html`.

## Testy

`python -m pytest -q` (pełny zestaw, z testami GPU: `sbatch scripts/smoke_tests.sbatch` na 4090). Bez zbudowanych rozszerzeń CUDA dla lokalnej karty: `-k "not short_real and not finite_difference"`.

Stare katalogi runów wciąż się ładują: nieaktywne klucze usuniętych funkcji są pomijane (`drop_legacy_keys`, `drop_legacy_medium_fix`), a aktywne powodują błąd.
