# FoamSurf — stan projektu (2026-10-06)

## Aktualny baseline IUI3

Wariant A4 poprzedniego sweepu capacity: `init5_cap5_fixed`, 28,51 dB PSNR
testowego. To **5 000 komórek**, a nie 5 000 iteracji warmupu.
Pełna konfiguracja znajduje się w `configs/iui3_5k.yaml`.

- Dane: IUI3-RedSea-Undistorted, split COLMAP z `llffhold=8`.
- Gaussiany startują z punktów COLMAP. Voronoi startuje z 5 000 punktów
  przestrzennych, `spatial_colmap_fraction=0`.
- Warmup ma 0 iteracji; etap bootstrapu tylko inicjalizuje Voronoi i zapisuje
  checkpoint. Potem 30k iteracji treningu joint.
- Topologia Voronoi jest stała. Kod densyfikacji/pruningu **pozostaje**,
  lecz `joint.voronoi_adaptation.enabled=false` w tym baseline.
- RGB medium, białe tło, lokalna kara overlap `weight=0.03` i wygładzanie
  parametrów medium są aktywne. Dodatkowe repulsion/direct teacher mają wagę 0.
- Gaussian RGB ma aktywację `identity`; sigmoid jest zachowany jako opcja.

28,51 dB to istotna poprawa, lecz nie oznacza pełnego disentanglementu.
Oceniamy nadal oddzielnie hybrid, Gaussian-only, Voronoi-only i alpha Gaussów.
Kolory Gaussian-only mogą być przejaskrawione; nie należy utożsamiać renderu
`direct` z Gaussian-only.

## Synthetic

`configs/synthetic.yaml` uruchamia bez dodatkowego skryptu: Voronoi-only
warmup (5k iteracji), automatyczny podział surface/non-surface metodą weighted
valley, zastąpienie surface cells Gaussianami i 30k iteracji treningu
hybrydowego. Konfiguracja jest samodzielna: ma ustawienia obu etapów w jednym
YAML. Ta ścieżka nie używa zewnętrznego checkpointu RadFoam.

## Przepływ kodu

`train.py` ładuje YAML i wybiera IUI3 albo synthetic.
IUI3: `training/colmap_joint.py` tworzy bootstrap i uruchamia
`training/colmap_joint_trainer.py`, który zawiera
trener joint, lokalny overlap i wygładzanie RGB medium.
Synthetic: `training/end_to_end.py` prowadzi warmup, partition i joint.
W obu przypadkach renderer jest w `renderer/`, parametry w
`foamsurf_model/`, a checkpointy oraz ewaluacja trafiają do katalogu
`--output-dir`.

Lossy i ich konfiguracje są w `training/losses.py` oraz
`training/disentanglement.py`; lokalny renderer overlap jest w
`training/surface.py`. O aktywnych składnikach decydują wagi w YAML.

## Weryfikacja i użycie

```bash
conda activate foamsurf
python train.py --config configs/iui3_5k.yaml --dry-run
python train.py --config configs/synthetic.yaml --dry-run
python -m pytest -q tests
sbatch scripts/train.sbatch
```

Config IUI3 porównano pole po polu z resolverem A4: manifesty były identyczne.
Krótki replay na RTX4090 dał te same metryki i zgodne checkpointy
(z tolerancją około `1e-4` dla niedeterministycznych redukcji CUDA).
Jednorazowy test starego i nowego treningu został po wykonaniu usunięty.
Testy bieżącego kodu pozostają w `tests/`.

Historyczne sweepy, stare configi, raporty i prezentacje są odzyskiwalne pod
`/shared/results/gmwilczn/foamsurf/foamsurf_legacy_2026-10-06/`.
Nie usunięto żadnych katalogów `outputs/` ani danych treningowych.
