# Disentanglement: Gaussian-eviction sweep

## Purpose

This is the second disentanglement study, built from the Phase 1 observation
that the `min_opacity=0.10`, `opacity_prune_from=3000` run produced the first
visible transfer of some clouds from Gaussians to Voronoi while preserving a
strong hybrid reconstruction.

The study tests one concrete mechanism: give cloud Gaussians a route to low
opacity, delete them, and leave enough optimization time for Voronoi to fill
the released regions. It is not a general regularizer sweep. Graph TV,
medium-LR boosts, Gaussian freezing, opaque-if-used and alternative K values
are deliberately held out. They should be studied only after this experiment
identifies a reliable Gaussian-removal schedule.

Every run:

- trains from scratch for 30k iterations with seed 0;
- uses the qualified clean joint baseline and full-image L1 + 0.2 DSSIM;
- trains only on `lego_both`; aligned component images are evaluation-only;
- uses the K=64 renderer from `configs/baseline_clean_joint.yaml`;
- densifies Gaussians only before 15k;
- varies only the fields explicitly recorded in the manifest.

The source-of-truth manifest is
`training/disentanglement_eviction_sweeps.py`. It contains exactly 100 unique
configurations and stores both machine-readable factors and complete config
overrides for every array index.

## Mechanism implemented for this sweep

Gaussian densification and opacity pruning now have independent schedules.
After 15k, an experiment may continue opacity pruning without cloning,
splitting, screen-size pruning or world-size pruning. `opacity_prune_until` is
exclusive: `15000` gives the same last event at 14750 as the old baseline,
whereas `30000` continues through the last interval before training ends.

The pruning threshold can ramp from `min_opacity_initial` to `min_opacity`.
All threshold ramps in this matrix are linear and start at 0.01. A zero-length
ramp is a hard threshold. The pruning interval is independently configurable.

Opacity-binarization has an optional end and maintenance weight. At the end it
can switch off, fall to 10% of its configured strength, or remain fully active.
The code also supports a smooth end ramp, although this first 100-run matrix
uses exact step transitions so the late policies have simple causal contrasts.

Two safeguards prevent an opacity event from accidentally deleting an entire
population: `gaussian_adaptation.min_points` and
`gaussian_adaptation.max_opacity_prune_fraction`. Their backward-compatible
defaults (`1` and `1.0`) are held fixed in this matrix, so the Phase 1 winner is
reproduced without changing its behavior. Raw, accepted and rejected opacity
candidate counts are logged.

## The 100 experiments

| Indices | Count | Block | Factors |
|---:|---:|---|---|
| 0–35 | 36 | pruning main | target `{.05,.075,.10,.125,.15,.20}` × ramp `{0,2k,5k}` × end `{15k,30k}`; start 3k, interval 250 |
| 36–43 | 8 | pruning start | target `{.10,.15}` × start `{1k,5k}` × ramp `{2k,5k}`; end 30k, interval 250 |
| 44–45 | 2 | weak-pruning controls | target `.01` × end `{15k,30k}`; hard from 3k, interval 250 |
| 46–49 | 4 | pruning interval | target `{.10,.15}` × interval `{500,1000}`; start 3k, ramp 2k, end 30k |
| 50–97 | 48 | binarization + pruning | bin weight `{1e-4,3e-4,1e-3,3e-3}` × start `{1k,3k,5k}` × 4 late policies |
| 98–99 | 2 | early binarization | weight `{3e-4,1e-3}`, start 0, ramp 5k, stop at 15k; pruning through 30k |

Index 12 (`prune_a0.1_s3k_r0k_u15k`) reproduces the effective Phase 1 run 96
schedule and is the main reference. Indices 44–45 test the original 0.01
threshold rather than the Phase 1 winner.

All binarization+pruning runs use a hard opacity-pruning threshold of 0.10 from
3k, with interval 250. Binarization ramps in with a cosine schedule. The four
late policies are:

| Policy label | Binarization after 15k | Opacity pruning |
|---|---|---|
| `stop15_prune15` | off | stops at 15k |
| `stop15_prune30` | off | continues to 30k |
| `maint10_prune30` | 10% maintenance weight | continues to 30k |
| `full30_prune30` | full weight | continues to 30k |

## Hypotheses and matched comparisons

The central question is not whether fewer Gaussians improve a scalar loss. It
is whether Voronoi takes over cloud regions after Gaussian removal without
creating empty regions or damaging the solid object.

Use matched contrasts before ranking all runs:

1. For every main-pruning tuple, compare `u15k` with `u30k`. This isolates the
   effect of allowing removal after densification ends.
2. Within a fixed target/end, compare hard, 2k and 5k ramps. This tests whether
   a gradual threshold gives Voronoi time to replace disappearing Gaussians.
3. At fixed ramp and end, compare targets. This identifies the useful pruning
   pressure before reconstruction collapses.
4. Compare indices 36–43 against their start-3k counterparts to determine
   whether early or late selection decides which representation owns clouds at
   15k.
5. Compare interval 250 counterparts with indices 46–49. This tests whether
   frequent late removal is necessary or merely destabilizing.
6. For each bin weight/start pair, compare `stop15_prune15` with
   `stop15_prune30`; binarization is identical, so this isolates late pruning.
7. Then compare `stop15_prune30`, `maint10_prune30` and
   `full30_prune30`; pruning is identical, so this isolates whether continued
   binary pressure preserves separation or prevents recovery.

The expected useful sequence is visible in curves, not only at iteration 30k:

1. ambiguous cloud Gaussians move toward low opacity;
2. an opacity event removes them;
3. Gaussian cloud weight/alpha falls;
4. Voronoi cloud dominance and cloud-layer PSNR rise over the following evals;
5. hybrid PSNR recovers instead of leaving a persistent hole.

## How to analyse the results

First verify completeness and stability. A run must reach iteration 30000,
write both checkpoints and have finite losses/metrics. Record topology errors,
OOMs, peak memory, population collapse and abrupt PSNR drops separately; do not
treat failed runs as merely low-scoring observations.

Aggregate the two key snapshots with:

```bash
cd /shared/results/gmwilczn/foamsurf/foamsurf
python scripts/summarize_disentanglement_eviction.py
```

This writes
`outputs/disentanglement_eviction/eviction_summary.csv`. It includes the
manifest factors, 15k and 30k decomposition metrics, Gaussian counts and
opacity fractions, plus post-15k pruning and quality deltas.

Analyse outcomes in this order:

1. **Hybrid fidelity:** `hybrid_psnr_30k`, its change from 15k, SSIM in the raw
   eval CSV, and the rendered RGB images. Report a Pareto frontier rather than
   selecting the highest PSNR alone.
2. **Cloud evacuation:** lower
   `gaussian_weight_cloud_30k` and `gaussian_alpha_cloud_30k` are better.
   Inspect the cloud-only masks and Gaussian contribution heatmaps; averages
   can hide a remaining Gaussian cloud patch.
3. **Voronoi takeover:** higher `medium_layer_psnr_30k` and
   `medium_cloud_dominance_30k`, especially positive changes after 15k, show
   replacement. Fewer Gaussians without these gains indicates an empty hole,
   not disentanglement.
4. **Solid preservation:** `gaussian_layer_psnr_30k` should remain strong and
   `medium_weight_lego_30k` should not increase. Inspect edges and texture in
   the Gaussian-only render.
5. **Mechanism evidence:** relate `post15_pruned`, Gaussian-count change,
   opacity `<0.1`/`>0.9` fractions and the active binarization weight to the
   decomposition changes. Check train curves around every pruning event, not
   just endpoints.
6. **Spatial verification:** for all Pareto candidates and matched controls,
   compare hybrid, Gaussian-only, Voronoi-only, layer-error and responsibility
   images at 15k and 30k. A run is not accepted if clouds remain spatially in
   Gaussians despite an improved global average.

Because every run uses only seed 0, this is a screening study. After selecting
a small Pareto set, repeat those configurations with multiple seeds before
drawing a robustness conclusion. Do not add freezing, medium-LR boosts or
post-15k TV until this sweep tells us whether removal alone creates a genuine
handoff window.

## Submission

Submit the full array from a login/CPU node:

```bash
cd /shared/results/gmwilczn/foamsurf/foamsurf
sbatch scripts/sweep_disentanglement_eviction.sbatch
```

The job requests one RTX 4090 and runs at most eight tasks concurrently. To
submit a block or retry selected indices:

```bash
sbatch --array=0-49%8 scripts/sweep_disentanglement_eviction.sbatch
sbatch --array=50-99%8 scripts/sweep_disentanglement_eviction.sbatch
sbatch --array=12,44,50 scripts/sweep_disentanglement_eviction.sbatch
```

Set `WANDB_MODE=offline` or `disabled` at submission time if needed. Outputs
are placed under `outputs/disentanglement_eviction`; Slurm logs go to
`slurm_logs/dis_evict_<job>_<task>.{out,err}`. `OUTPUT_ROOT` can redirect the
whole array while still preserving one unique directory per task.
