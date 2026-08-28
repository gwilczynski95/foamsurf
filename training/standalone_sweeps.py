from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from .config import ExperimentConfig


@dataclass(frozen=True)
class StandaloneSweepSpec:
    label: str
    family: str
    question: str
    seed: int
    initial_points: int | None = None
    sites_lr: float | None = None
    density_lr: float | None = None
    adaptive: bool | None = None
    model_overrides: tuple[tuple[str, object], ...] = ()
    gaussian_overrides: tuple[tuple[str, object], ...] = ()
    optimization_overrides: tuple[tuple[str, object], ...] = ()


def _overrides(**values: object) -> tuple[tuple[str, object], ...]:
    return tuple(values.items())


_CONTROL_SPECS = tuple(
    [
        StandaloneSweepSpec(
            label=f"ctrl_baseline_s{seed}",
            family="control",
            question="exact 25.309 dB fixed-topology control",
            seed=seed,
        )
        for seed in (0, 42, 1337, 2025)
    ]
    + [
        StandaloneSweepSpec(
            label=f"ctrl_points{points}_s{seed}",
            family="control",
            question="fixed-topology initial population",
            seed=seed,
            model_overrides=_overrides(gaussian_initial_points=points),
        )
        for points in (25_000, 50_000, 75_000, 150_000, 200_000, 300_000)
        for seed in (0, 42)
    ]
    + [
        StandaloneSweepSpec(
            label=f"ctrl_opacity{opacity:g}_s{seed}",
            family="control",
            question="fixed-topology initial opacity",
            seed=seed,
            model_overrides=_overrides(gaussian_initial_opacity=opacity),
        )
        for opacity in (0.0025, 0.005, 0.02, 0.05, 0.1)
        for seed in (0, 42)
    ]
    + [
        StandaloneSweepSpec(
            label=f"ctrl_knn{neighbors}_s{seed}",
            family="control",
            question="fixed-topology scale initialization neighbors",
            seed=seed,
            model_overrides=_overrides(
                gaussian_initial_scale_neighbors=neighbors
            ),
        )
        for neighbors in (1, 2, 4, 8, 16)
        for seed in (0, 42)
    ]
    + [
        StandaloneSweepSpec(
            label=f"ctrl_reset_i{interval}_v{value:g}",
            family="control",
            question="opacity reset without topology changes",
            seed=0,
            gaussian_overrides=_overrides(
                enabled=True,
                policy="fixed",
                max_points=None,
                runaway_max_points=None,
                densify_until=30_001,
                opacity_reset_interval=interval,
                opacity_reset_value=value,
                opacity_reset_at_densify_from=False,
            ),
        )
        for interval in (1_500, 3_000, 6_000)
        for value in (0.005, 0.01)
    ]
    + [
        StandaloneSweepSpec(
            label=f"ctrl_scalereg{weight:g}",
            family="control",
            question="fixed-topology Gaussian scale regularization",
            seed=0,
            optimization_overrides=_overrides(
                gaussian_scale_regularization=weight
            ),
        )
        for weight in (0.0, 1.0e-4, 5.0e-4, 5.0e-3, 1.0e-2, 2.0e-2)
    ]
)


_RESET_PROFILES = (
    ("r0", 0, 0.01, False, True),
    ("r3k", 3_000, 0.01, False, True),
    ("r6ksoft", 6_000, 0.05, False, False),
)
_GRAPHDECO_CORE_SPECS = tuple(
    StandaloneSweepSpec(
        label=(
            f"g3d_grad{gradient:g}_i{interval}_{reset_name}"
            f"_op{min_opacity:g}_pd{percent_dense:g}"
        ),
        family="graphdeco",
        question="3DGS/current-FlaRe densification and opacity-reset grid",
        seed=0,
        gaussian_overrides=_overrides(
            enabled=True,
            policy="graphdeco",
            max_points=None,
            runaway_max_points=None,
            densify_from=500,
            densify_until=15_000,
            densification_interval=interval,
            gradient_threshold=gradient,
            min_opacity=min_opacity,
            opacity_prune_from=500,
            percent_dense=percent_dense,
            size_prune_from=3_000,
            screen_pruning_enabled=False,
            max_world_scale_fraction=0.10,
            opacity_reset_interval=reset_interval,
            opacity_reset_value=reset_value,
            opacity_reset_at_densify_from=reset_at_start,
            opacity_reset_optimizer_state=reset_optimizer,
        ),
    )
    for gradient in (5.0e-5, 1.0e-4, 2.0e-4, 4.0e-4)
    for interval in (50, 100, 250)
    for (
        reset_name,
        reset_interval,
        reset_value,
        reset_at_start,
        reset_optimizer,
    ) in _RESET_PROFILES
    for min_opacity in (0.0025, 0.01)
    for percent_dense in (0.005, 0.02)
)


_SCALE_SHAPE_SPECS = tuple(
    StandaloneSweepSpec(
        label=(
            f"shape_reg{regularization:g}_min{minimum:g}"
            f"_max{maximum:g}_lr{scale_lr:g}"
        ),
        family="scale_shape",
        question="joint Gaussian size regularization, clamp, and scale LR",
        seed=0,
        gaussian_overrides=_overrides(
            enabled=True,
            policy="graphdeco",
            max_points=None,
            runaway_max_points=None,
            densify_from=500,
            densify_until=15_000,
            densification_interval=100,
            gradient_threshold=2.0e-4,
            min_opacity=0.005,
            percent_dense=0.01,
            opacity_reset_interval=0,
            opacity_reset_at_densify_from=False,
        ),
        optimization_overrides=_overrides(
            gaussian_scale_regularization=regularization,
            gaussian_min_scale=minimum,
            gaussian_max_scale_fraction=maximum,
            gaussian_scales_lr=scale_lr,
            gaussian_scales_lr_final=scale_lr,
        ),
    )
    for regularization in (0.0, 1.0e-4, 1.0e-3, 1.0e-2)
    for minimum in (1.0e-5, 1.0e-4, 5.0e-4)
    for maximum in (0.025, 0.05, 0.10)
    for scale_lr in (0.02, 0.08)
)


_MOVEMENT_RESETS = (
    ("r0", 0, 0.01, True),
    ("r3k", 3_000, 0.01, True),
)
_FLARE_MOVEMENT_SPECS = tuple(
    StandaloneSweepSpec(
        label=(
            f"fmove_m{movement:g}_i{interval}_op{min_opacity:g}"
            f"_es{effective_scale:g}_{reset_name}"
        ),
        family="flare_movement",
        question="working flare_new post-step movement densification",
        seed=0,
        gaussian_overrides=_overrides(
            enabled=True,
            policy="flare_movement",
            max_points=None,
            runaway_max_points=None,
            densify_from=800,
            densify_until=24_000,
            densification_interval=interval,
            movement_threshold=movement,
            min_opacity=min_opacity,
            min_effective_scale=effective_scale,
            opacity_reset_interval=reset_interval,
            opacity_reset_value=reset_value,
            opacity_reset_at_densify_from=False,
            opacity_reset_optimizer_state=reset_optimizer,
        ),
    )
    for movement in (2.5e-4, 5.0e-4, 1.0e-3)
    for interval in (50, 100, 250)
    for min_opacity in (0.005, 0.02, 1.0 / 28.0)
    for effective_scale in (0.0, 1.0e-5)
    for (
        reset_name,
        reset_interval,
        reset_value,
        reset_optimizer,
    ) in _MOVEMENT_RESETS
)


GAUSSIAN_SWEEP_SPECS = (
    _CONTROL_SPECS
    + _GRAPHDECO_CORE_SPECS
    + _SCALE_SHAPE_SPECS
    + _FLARE_MOVEMENT_SPECS
)
GAUSSIAN_SWEEP_FAMILY_COUNTS = {
    family: sum(spec.family == family for spec in GAUSSIAN_SWEEP_SPECS)
    for family in ("control", "graphdeco", "scale_shape", "flare_movement")
}


VORONOI_INITIAL_POINTS = (2_048, 4_096, 8_192, 16_384, 32_768)
_VORONOI_MAX_POINTS = {2_048: 8_192, 4_096: 16_384, 8_192: 32_768, 16_384: 65_536, 32_768: 65_536}
VORONOI_SWEEP_SPECS = tuple(
    [
        StandaloneSweepSpec(
            family="voronoi",
            label=f"fixed_n{points}_s{seed}",
            question="fixed-population control",
            seed=seed, initial_points=points, adaptive=False,
        )
        for points in VORONOI_INITIAL_POINTS for seed in (42, 1337)
    ]
    + [
        StandaloneSweepSpec(
            family="voronoi",
            label=f"adapt_n{points}_slr{sites_lr:g}_dlr{density_lr:g}_s{seed}",
            question="joint effect of initial sparsity and medium optimizer rates",
            seed=seed, initial_points=points, sites_lr=sites_lr,
            density_lr=density_lr, adaptive=True,
        )
        for points in VORONOI_INITIAL_POINTS
        for sites_lr in (5e-5, 1e-4, 2e-4)
        for density_lr in (5e-3, 1e-2)
        for seed in (42, 1337)
    ]
)


def build_standalone_sweep_config(
    base: ExperimentConfig,
    phase: str,
    index: int,
    output_dir: str,
    wandb_name: str,
    wandb_mode: str = "online",
) -> ExperimentConfig:
    config = deepcopy(base)
    specs = GAUSSIAN_SWEEP_SPECS if phase == "gaussian" else VORONOI_SWEEP_SPECS
    spec = specs[index]
    config.seed = spec.seed
    config.output_dir = output_dir
    config.experiment_name = wandb_name
    config.wandb.name = wandb_name
    config.wandb.mode = wandb_mode
    config.wandb.project = f"foamsurf_stage{1 if phase == 'gaussian' else 2}_{phase}_sweep"
    config.wandb.tags = list(config.wandb.tags) + [spec.family, spec.label]
    if phase == "gaussian":
        # The massive sweep intentionally has no population/runaway cap.
        config.gaussian_adaptation.max_points = None
        config.gaussian_adaptation.runaway_max_points = None
        config.validation.eval_interval = 2_000
        config.validation.checkpoint_interval = 2_000
        config.validation.image_views = 1
        config.diagnostics.save_per_primitive = False
        for name, value in spec.model_overrides:
            setattr(config.model, name, value)
        for name, value in spec.gaussian_overrides:
            setattr(config.gaussian_adaptation, name, value)
        for name, value in spec.optimization_overrides:
            setattr(config.optimization, name, value)
    else:
        config.model.medium_points = int(spec.initial_points)
        config.voronoi_adaptation.enabled = bool(spec.adaptive)
        config.voronoi_adaptation.max_points = _VORONOI_MAX_POINTS[int(spec.initial_points)]
        if spec.sites_lr is not None:
            config.optimization.medium_sites_lr = spec.sites_lr
        if spec.density_lr is not None:
            config.optimization.medium_density_lr = spec.density_lr
    config.validate()
    return config
