"""Targeted initialization-only experiment; two controls plus seven seed pairs."""
from dataclasses import replace
from .warmup_sweep import configuration as old_configuration
from .warmup_profile import WarmupProfile


def variants():
    base = replace(WarmupProfile(), site_initialization="random", ray_sampling="mixed",
                   population_policy="radfoam", topology_schedule="radfoam",
                   outer_positions_trainable=True)
    rows = [
        dict(id=0, name="control_previous_1", seed=0, source=1, profile=None, inner=350000),
        dict(id=1, name="control_previous_7", seed=0, source=7, profile=None, inner=350000),
    ]
    cases = [
        ("organic", base, 131072 - 5000),
        ("single_camera", replace(base, ray_sampling="single_camera"), 131072 - 5000),
        ("foamsurf_population", replace(base, population_policy="foamsurf"), 131072 - 5000),
        ("no_density_reset", replace(base, reset_low_contribution=False), 131072 - 5000),
        ("stratified", replace(base, site_initialization="stratified"), 131072 - 5000),
        ("no_quantile", replace(base, quantile_weight=0.), 131072 - 5000),
        ("start_350k", base, 350000),
    ]
    for name, profile, inner in cases:
        for seed in (0, 42):
            rows.append(dict(id=len(rows), name=f"{name}_seed{seed}", seed=seed,
                             source=7, profile=profile.as_dict(), inner=inner))
    return rows


def configuration(index, output_dir, iterations=6000):
    rows = variants()
    if index not in range(len(rows)):
        raise ValueError("organic sweep index must be 0..15")
    row = dict(rows[index])
    config, profile, _ = old_configuration(row["source"], output_dir, max(iterations, 6000))
    config.optimization.iterations = iterations
    if row["profile"] is not None:
        profile = WarmupProfile.from_dict(row["profile"])
        config.model.medium_points = row["inner"]
        config.optimization.rays_per_batch = 1000000
        config.voronoi_adaptation.densify_from = 2000
        config.voronoi_adaptation.max_points = profile.population_target
        # This is only a capacity safety setting for the original-policy ablation.
        # Organic pruning uses its own activity rule, without the shared min/cap.
        config.voronoi_adaptation.min_points = 4
    config.seed = row["seed"]
    config.experiment_name = f"organic_{index:02d}_{row['name']}"
    config.wandb.name = config.experiment_name
    config.wandb.project = "foamsurf_warmup_organic"
    config.voronoi_warmup.milestone_iterations = sorted(set(
        [i for i in (0,500,1000,1999,2000,2500,3000,4000,5000,6000) if i <= iterations]
        + [iterations]))
    row.update(profile=profile.as_dict(), legacy=False, suite="organic")
    config.validate()
    return config, profile, row
