from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from .config import ExperimentConfig


@dataclass(frozen=True)
class VoronoiCapacitySpec:
    label: str
    seed: int
    max_points: int
    child_jitter_fraction: float
    iterations: int
    densify_until: int
    lr_decay_from: int


# With 2048 initial sites, 5% growth every 500 iterations reaches these
# capacities around 15k, 22k, 29k and 36.5k respectively.  Each run retains
# roughly 15k post-capacity optimization steps under the LR scheduler.
_CAPACITY_TIMING = {
    8_192: (30_000, 20_000, 15_000),
    16_384: (38_000, 23_000, 22_000),
    32_768: (45_000, 30_000, 29_000),
    65_536: (53_000, 37_000, 36_500),
}


VORONOI_CAPACITY_SWEEP_SPECS = tuple(
    VoronoiCapacitySpec(
        label=(
            f"cap{capacity // 1024}k_j{jitter:g}_s{seed}"
        ),
        seed=seed,
        max_points=capacity,
        child_jitter_fraction=jitter,
        iterations=_CAPACITY_TIMING[capacity][0],
        densify_until=_CAPACITY_TIMING[capacity][1],
        lr_decay_from=_CAPACITY_TIMING[capacity][2],
    )
    for capacity in (8_192, 16_384, 32_768, 65_536)
    for jitter in (0.025, 0.10)
    for seed in (42, 1337)
)


def build_voronoi_capacity_config(
    base: ExperimentConfig,
    index: int,
    output_dir: str,
    wandb_name: str,
    wandb_mode: str = "online",
) -> ExperimentConfig:
    config = deepcopy(base)
    spec = VORONOI_CAPACITY_SWEEP_SPECS[index]
    config.seed = spec.seed
    config.output_dir = output_dir
    config.experiment_name = wandb_name
    config.wandb.name = wandb_name
    config.wandb.mode = wandb_mode
    config.wandb.project = "foamsurf_stage2_voronoi_capacity_sweep"
    config.wandb.tags = list(config.wandb.tags) + [spec.label]

    # Exact vref_462246_28 combo_slow_sched recipe.
    config.model.medium_points = 2_048
    config.model.medium_initial_density = 0.005
    config.optimization.iterations = spec.iterations
    config.optimization.medium_sites_lr = 5.0e-5
    config.optimization.medium_density_lr = 1.0e-2
    config.optimization.medium_color_lr = 2.5e-3
    config.optimization.medium_sites_lr_final = 5.0e-6
    config.optimization.medium_density_lr_final = 1.0e-3
    config.optimization.medium_color_lr_final = 5.0e-4
    config.optimization.medium_lr_decay_from = spec.lr_decay_from
    config.optimization.medium_lr_decay_until = spec.iterations
    config.voronoi_adaptation.enabled = True
    config.voronoi_adaptation.densify_from = 1_000
    config.voronoi_adaptation.densify_until = spec.densify_until
    config.voronoi_adaptation.interval = 500
    config.voronoi_adaptation.densify_fraction = 0.05
    config.voronoi_adaptation.max_points = spec.max_points
    config.voronoi_adaptation.contribution_threshold = 1.0e-6
    config.voronoi_adaptation.gradient_threshold = 0.0
    config.voronoi_adaptation.child_jitter_fraction = (
        spec.child_jitter_fraction
    )

    if config.optimization.reconstruction_loss != "l1_dssim":
        raise ValueError("Voronoi capacity sweep requires L1+DSSIM")
    if config.optimization.dssim_weight != 0.2:
        raise ValueError("Voronoi capacity sweep pins DSSIM weight to 0.2")
    if list(config.background) != [0.0, 0.0, 0.0]:
        raise ValueError("Voronoi capacity sweep requires a black background")
    config.validate()
    return config
