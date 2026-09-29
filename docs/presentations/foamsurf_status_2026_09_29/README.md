# Prezentacja FoamSurf — stan projektu (29.09.2026)

Gotowa prezentacja po polsku: [FoamSurf_stan_projektu_PL.pdf](FoamSurf_stan_projektu_PL.pdf) (10 slajdów). `slajd_01.png`–`slajd_10.png` to osobne obrazy slajdów.

Odtworzenie w środowisku projektu:

```bash
/shared/results/gmwilczn/conda/envs/foamsurf/bin/python docs/presentations/foamsurf_status_2026_09_29/build.py
```

Źródła i zastrzeżenia:

- IUI3: lokalne rendery `outputs/rgb_smooth/513211/run_1` (kontrola, seed 43) i `run_13` (silne TV, seed 43), kamera `MTN_5911`, iteracja 30 000. PSNR uśredniony po trzech seedach: [raport RGB](../../reports/rgb_disentanglement_2026_09/RESULTS.md). Runy z overlap + TV ×3 pochodzą z ukończonego sweepu `513646`.
- W pokazanych treningach joint IUI3 sampler wybiera jedną kamerę na iterację (tasowana talia bez powtórzeń) i renderuje wszystkie 1383 × 917 pikseli. Parametr `rays_per_batch=640000` nie ogranicza liczby promieni, gdy `full_image_batch=true`. Źródła: `training/sampler.py`, `training/trainer.py`, `outputs/rgb_smooth/513211/run_13/joint.yaml`.
- Szczegółowe definicje matematyczne i gradienty strat z dwóch slajdów: [notatki techniczne](WZORY_I_LOSSY.md).
- LEGO: `outputs/radfoam_prior/radfoam_prior_run30_502753/evaluation/iteration_030000`; metryki i zastrzeżenie dotyczące zewnętrznego prioru: [opis eksperymentu](../../RADFOAM_PRIOR_HANDOFF_TRAINING.md).
- WaterSplatting: 29,386 dB dla IUI3 Red Sea z [tabeli 1 publikacji](https://arxiv.org/html/2408.08206). To punkt odniesienia, **nie** bezpośredni benchmark na identycznym splicie, preprocessingu i protokole.
- `direct` zawiera promieniowanie Gaussów **oraz** tłumione tło. Ocenę przejęcia dna należy opierać na `gaussian_only`, nie na `direct` ani `D/(D+B)`.
- Zaplanowany sweep (czarne tło × poprawiony overlap × TV) nie był ukończony przy tworzeniu prezentacji; slajdy nie przypisują mu wyników.
