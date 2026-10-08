# FoamSurf

FoamSurf trenuje hybrydę 2D Gaussów (powierzchnia) i komórek Voronoi
(medium). Repo ma jeden publiczny punkt wejścia: `train.py`.

## Trening

W środowisku conda `foamsurf`:

```bash
python train.py --config configs/iui3_5k.yaml --output-dir outputs/moj_run
python train.py --config configs/synthetic.yaml --output-dir outputs/moje_lego
```

Pierwszy config odtwarza aktualny baseline IUI3: 5 000 komórek Voronoi
inicjalizowanych przestrzennie, bez punktów COLMAP w Voronoi, bez warmupu i bez
adaptacji topologii medium w treningu joint. Adaptacja Voronoi (densyfikacja
i pruning) nadal jest w kodzie i można ją ponownie włączyć w configu po
świadomej zmianie eksperymentu.

Drugi config uruchamia synthetic `lego_both`: Voronoi-only warmup, automatyczny
podział surface/non-surface, inicjalizację Gaussów i joint training. Oba etapy
są opisane w tym jednym pliku YAML.

Na Slurmie:

```bash
sbatch scripts/train.sbatch
sbatch --export=ALL,FS_CONFIG=configs/synthetic.yaml scripts/train.sbatch
```

Można ustawić `FS_OUTPUT` i `FS_RESUME=1`. Do podglądu konfiguracji bez
treningu: `python train.py --config configs/iui3_5k.yaml --dry-run`.

## Co jest w repo

- `train.py` — jedyny publiczny entrypoint.
- `configs/` — dwa aktualne scenariusze.
- `training/` — kod treningu, lossów, optymalizatorów i obu etapów inicjalizacji.
- `foamsurf_model/`, `foamsurf_data/`, `renderer/`, `evaluation/` —
  model, dane, renderowanie i ewaluacja.
- `tests/test_train.py` — jedyny skrypt testujący oba aktualne pipeline’y.

Aktualny stan, wyniki, ograniczenia i mapa modułów:
[docs/CURRENT_STATE.md](docs/CURRENT_STATE.md).

Budowanie rozszerzeń: `./build.sh` w środowisku `foamsurf`. Runtime nie
wymaga sąsiedniego repo RadFoam. Historyczne sweepy i raporty zostały
przeniesione poza repo do
`/shared/results/gmwilczn/foamsurf/foamsurf_legacy_2026-10-06/`;
nie są częścią aktualnego API treningowego.
