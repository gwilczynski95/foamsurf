"""Explicit 50-run initialization-only manifest, with one legacy control."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import math
from .config import ExperimentConfig
from .warmup_profile import WarmupProfile

ROOT = Path(__file__).resolve().parents[1]


def variants():
    r = WarmupProfile()
    old = replace(r, density_beta=1., initial_sigma=.005, outer_points=0,
        loss="l1_dssim", schedule="constant", sites_lr=5e-5, density_lr=.01,
        color_lr=.0025, adam_eps=1e-8, resolutions=[1], resolution_steps=[0],
        opacity_weight=0., quantile_weight=0.)
    rows = []
    def add(name, profile=r, **overrides):
        profile.validate()
        rows.append({"id": len(rows), "name": name, "profile": profile.as_dict(),
                     "config_overrides": overrides, "legacy": len(rows) == 0})
    add("legacy_control", old)
    a = replace(old, loss="smooth_l1")
    add("ladder_smooth_l1", a)
    a = replace(a, density_beta=10.)
    add("ladder_beta10", a)
    a = replace(r, initial_sigma=.005, outer_points=0, resolutions=[1], resolution_steps=[0], quantile_weight=0.)
    add("ladder_cosine", a)
    a = replace(a, outer_points=5000)
    add("ladder_outer", a)
    a = replace(a, resolutions=[4,2,1], resolution_steps=[0,150,500])
    add("ladder_progressive", a)
    add("ladder_quantile", replace(a, quantile_weight=1e-4))
    add("reference_R")
    add("R_without_quantile", replace(r, quantile_weight=0.))
    add("R_without_outer", replace(r, outer_points=0))
    for beta in (1.,5.,20.):
        add(f"beta_{beta:g}_matched_sigma", replace(r,density_beta=beta))
    # .005 is already ID6: .01 replaces the duplicate from the discussion.
    for sigma in (.0006715,.01,.02,.04,.10,.20):
        add(f"sigma_{sigma:g}",replace(r,initial_sigma=sigma))
    add("density_without_lr_warmup",replace(r,density_warmup=0))
    for weight in (1e-6,3e-6,1e-5,3e-5,3e-4,1e-3):
        add(f"quantile_weight_{weight:g}",replace(r,quantile_weight=weight))
    for ramp in (0,500,1000,2000,5000):
        add(f"quantile_ramp_{ramp}",replace(r,quantile_ramp=ramp))
    for start in (500,1000,2000):
        add(f"quantile_start_{start}",replace(r,quantile_start=start,quantile_ramp=2000))
    add("fixed_quantiles_09_05",replace(r,fixed_quantiles=[.9,.5]))
    add("corner_guards",replace(r,outer_points=8,corner_guards=True))
    for n in (500,2000,10000):
        add(f"outer_count_{n}",replace(r,outer_points=n))
    for scale in (5.,20.):
        add(f"outer_scale_{scale:g}",replace(r,outer_scale=scale))
    add("outer_movable",replace(r,outer_positions_trainable=True))
    add("full_resolution",replace(r,resolutions=[1],resolution_steps=[0]))
    add("slow_progressive",replace(r,resolution_steps=[0,500,1500]))
    add("four_resolutions",replace(r,resolutions=[8,4,2,1],resolution_steps=[0,150,500,1500]))
    add("capacity_131072",inner_points=131072)
    add("capacity_500000",inner_points=500000,max_inner_points=700000)
    add("fixed_population",adaptation=False)
    add("aggressive_densification",densify_fraction=.15,densify_from=2000,max_inner_points=1000000)
    add("reference_seed1",seed=1)
    assert len(rows)==50
    return rows


def configuration(index, output_dir, iterations=5000):
    row = deepcopy(variants()[index])
    config = ExperimentConfig.load(ROOT / "configs/voronoi_prior_warmup.yaml")
    config.output_dir = str(output_dir)
    config.experiment_name = f"warmup_{index:02d}_{row['name']}"
    config.wandb.name = config.experiment_name
    config.wandb.project = "foamsurf_warmup_radfoam_like"
    config.optimization.iterations = iterations
    milestones = [0,150,500,1000,2000,3000,4000,5000,8000,11000,15000,20000]
    config.voronoi_warmup.milestone_iterations = sorted(set([i for i in milestones if i <= iterations]+[iterations]))
    profile = WarmupProfile.from_dict(row["profile"])
    overrides = row["config_overrides"]
    config.seed = overrides.get("seed",0)
    config.model.medium_points = overrides.get("inner_points",350000)
    config.model.medium_guard_points = 0
    # max/min are total populations in the shared controller; preserve the
    # specified INNER budgets by adding the scaffold count.
    config.voronoi_adaptation.max_points = overrides.get("max_inner_points",500000)+profile.outer_points
    config.voronoi_adaptation.min_points = 100000+profile.outer_points
    config.voronoi_adaptation.enabled = True
    if not overrides.get("adaptation", True):
        config.voronoi_adaptation.freeze_at_capacity = True
        config.voronoi_adaptation.max_points = config.model.medium_points + profile.outer_points
    config.voronoi_adaptation.densify_fraction = overrides.get("densify_fraction",.02)
    config.voronoi_adaptation.densify_from = overrides.get("densify_from",500)
    config.voronoi_adaptation.densify_until = min(iterations,11000 if iterations>5000 else 5000)+1
    config.validate()
    return config,profile,row
