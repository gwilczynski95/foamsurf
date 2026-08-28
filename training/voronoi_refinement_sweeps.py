from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from .config import ExperimentConfig


@dataclass(frozen=True)
class VoronoiRefinementSpec:
    label: str
    question: str
    model_overrides: tuple[tuple[str, object], ...] = ()
    optimization_overrides: tuple[tuple[str, object], ...] = ()
    adaptation_overrides: tuple[tuple[str, object], ...] = ()


def _values(**kwargs: object) -> tuple[tuple[str, object], ...]:
    return tuple(kwargs.items())


VORONOI_REFINEMENT_SWEEP_SPECS = (
    VoronoiRefinementSpec("baseline", "exact robust black L1+DSSIM winner"),
    VoronoiRefinementSpec(
        "growth_f0.05_i500", "slower population growth",
        adaptation_overrides=_values(densify_fraction=0.05, interval=500),
    ),
    VoronoiRefinementSpec(
        "growth_f0.025_i500", "very gradual population growth",
        adaptation_overrides=_values(densify_fraction=0.025, interval=500),
    ),
    VoronoiRefinementSpec(
        "growth_f0.10_i1000", "less frequent baseline-sized adaptation",
        adaptation_overrides=_values(densify_fraction=0.10, interval=1000),
    ),
    VoronoiRefinementSpec(
        "growth_f0.05_i1000", "slow and infrequent population growth",
        adaptation_overrides=_values(densify_fraction=0.05, interval=1000),
    ),
    VoronoiRefinementSpec(
        "lr_d0.0075_c0.0025", "joint density/color learning-rate balance",
        optimization_overrides=_values(
            medium_density_lr=0.0075, medium_color_lr=0.0025
        ),
    ),
    VoronoiRefinementSpec(
        "lr_d0.01_c0.0025", "winner density rate with slower color",
        optimization_overrides=_values(
            medium_density_lr=0.01, medium_color_lr=0.0025
        ),
    ),
    VoronoiRefinementSpec(
        "lr_d0.015_c0.0025", "faster density with slower color",
        optimization_overrides=_values(
            medium_density_lr=0.015, medium_color_lr=0.0025
        ),
    ),
    VoronoiRefinementSpec(
        "lr_d0.0075_c0.005", "intermediate density rate",
        optimization_overrides=_values(
            medium_density_lr=0.0075, medium_color_lr=0.005
        ),
    ),
    VoronoiRefinementSpec(
        "lr_d0.015_c0.005", "higher density rate control",
        optimization_overrides=_values(
            medium_density_lr=0.015, medium_color_lr=0.005
        ),
    ),
    *(
        VoronoiRefinementSpec(
            f"jitter_{jitter:g}", "child placement jitter",
            adaptation_overrides=_values(child_jitter_fraction=jitter),
        )
        for jitter in (0.0, 0.025, 0.10)
    ),
    VoronoiRefinementSpec(
        "recycle_c1e-5_g5e-5_r0.02_p2",
        "activate conservative low-activity recycling",
        adaptation_overrides=_values(
            contribution_threshold=1.0e-5, gradient_threshold=5.0e-5,
            prune_cell_radius_fraction=0.02, prune_patience=2,
        ),
    ),
    VoronoiRefinementSpec(
        "recycle_c1e-5_g1e-4_r0.05_p2",
        "recycle small inactive regions",
        adaptation_overrides=_values(
            contribution_threshold=1.0e-5, gradient_threshold=1.0e-4,
            prune_cell_radius_fraction=0.05, prune_patience=2,
        ),
    ),
    VoronoiRefinementSpec(
        "recycle_c1e-4_g1e-4_r0.05_p2",
        "stronger low-activity recycling",
        adaptation_overrides=_values(
            contribution_threshold=1.0e-4, gradient_threshold=1.0e-4,
            prune_cell_radius_fraction=0.05, prune_patience=2,
        ),
    ),
    VoronoiRefinementSpec(
        "recycle_c1e-4_g1e-4_r0.05_p1",
        "fast low-activity recycling",
        adaptation_overrides=_values(
            contribution_threshold=1.0e-4, gradient_threshold=1.0e-4,
            prune_cell_radius_fraction=0.05, prune_patience=1,
        ),
    ),
    VoronoiRefinementSpec(
        "init_density_0.001", "reduce untrained-cell haze",
        model_overrides=_values(medium_initial_density=0.001),
    ),
    VoronoiRefinementSpec(
        "init_density_0.005", "lower initial medium density",
        model_overrides=_values(medium_initial_density=0.005),
    ),
    VoronoiRefinementSpec(
        "topology_i10", "more frequent topology refresh",
        optimization_overrides=_values(topology_update_interval=10),
    ),
    VoronoiRefinementSpec(
        "topology_i100", "less frequent topology refresh",
        optimization_overrides=_values(topology_update_interval=100),
    ),
    VoronoiRefinementSpec(
        "topology_full", "full instead of incremental topology rebuild",
        optimization_overrides=_values(topology_incremental=False),
    ),
    VoronoiRefinementSpec(
        "sched_gentle_from15k", "gentle joint medium LR stabilization",
        optimization_overrides=_values(
            medium_sites_lr_final=1.0e-5,
            medium_density_lr_final=2.5e-3,
            medium_color_lr_final=1.0e-3,
            medium_lr_decay_from=15000,
        ),
    ),
    *(
        VoronoiRefinementSpec(
            f"sched_x10_from{start // 1000}k",
            "tenfold joint LR decay with varied onset",
            optimization_overrides=_values(
                medium_sites_lr_final=5.0e-6,
                medium_density_lr_final=1.0e-3,
                medium_color_lr_final=5.0e-4,
                medium_lr_decay_from=start,
            ),
        )
        for start in (10000, 15000, 20000)
    ),
    VoronoiRefinementSpec(
        "sched_freeze_sites_from15k", "strong site stabilization",
        optimization_overrides=_values(
            medium_sites_lr_final=1.0e-6,
            medium_density_lr_final=2.5e-3,
            medium_color_lr_final=1.0e-3,
            medium_lr_decay_from=15000,
        ),
    ),
    VoronoiRefinementSpec(
        "sched_density_only_from15k", "isolate late density polarization",
        optimization_overrides=_values(
            medium_density_lr_final=1.0e-3,
            medium_lr_decay_from=15000,
        ),
    ),
    VoronoiRefinementSpec(
        "combo_slow_sched",
        "combine slower growth, lower initialization, jitter, and LR decay",
        model_overrides=_values(medium_initial_density=0.005),
        optimization_overrides=_values(
            medium_color_lr=0.0025,
            medium_sites_lr_final=5.0e-6,
            medium_density_lr_final=1.0e-3,
            medium_color_lr_final=5.0e-4,
            medium_lr_decay_from=15000,
        ),
        adaptation_overrides=_values(
            densify_fraction=0.05, interval=500,
            child_jitter_fraction=0.025,
        ),
    ),
    VoronoiRefinementSpec(
        "combo_recycle_sched",
        "combine slow growth, recycling, and LR stabilization",
        model_overrides=_values(medium_initial_density=0.005),
        optimization_overrides=_values(
            medium_color_lr=0.0025,
            medium_sites_lr_final=5.0e-6,
            medium_density_lr_final=1.0e-3,
            medium_color_lr_final=5.0e-4,
            medium_lr_decay_from=15000,
        ),
        adaptation_overrides=_values(
            densify_fraction=0.05, interval=500,
            contribution_threshold=1.0e-5, gradient_threshold=5.0e-5,
            prune_cell_radius_fraction=0.05, prune_patience=2,
            child_jitter_fraction=0.025,
        ),
    ),
)


def build_voronoi_refinement_config(
    base: ExperimentConfig,
    index: int,
    output_dir: str,
    wandb_name: str,
    wandb_mode: str = "online",
) -> ExperimentConfig:
    config = deepcopy(base)
    spec = VORONOI_REFINEMENT_SWEEP_SPECS[index]
    config.seed = 42
    config.output_dir = output_dir
    config.experiment_name = wandb_name
    config.wandb.name = wandb_name
    config.wandb.mode = wandb_mode
    config.wandb.project = "foamsurf_stage2_voronoi_refinement_sweep"
    config.wandb.tags = list(config.wandb.tags) + [spec.label]

    for name, value in spec.model_overrides:
        setattr(config.model, name, value)
    for name, value in spec.optimization_overrides:
        setattr(config.optimization, name, value)
    for name, value in spec.adaptation_overrides:
        setattr(config.voronoi_adaptation, name, value)

    if config.optimization.reconstruction_loss != "l1_dssim":
        raise ValueError("Voronoi refinement sweep requires L1+DSSIM")
    if config.optimization.dssim_weight != 0.2:
        raise ValueError("Voronoi refinement sweep pins DSSIM weight to 0.2")
    config.validate()
    return config
