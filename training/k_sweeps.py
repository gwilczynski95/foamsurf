from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from .config import ExperimentConfig


K_VALUES = (32, 64, 128, 16)


@dataclass(frozen=True)
class GaussianKSweepRecipe:
    label: str
    family: str
    model: tuple[tuple[str, object], ...] = ()
    adaptation: tuple[tuple[str, object], ...] = ()
    optimization: tuple[tuple[str, object], ...] = ()


def _values(**kwargs: object) -> tuple[tuple[str, object], ...]:
    return tuple(kwargs.items())


_GRAPHDECO = _values(
    enabled=True,
    policy="graphdeco",
    max_points=None,
    runaway_max_points=None,
    densify_from=500,
    densify_until=15_000,
    densification_interval=250,
    gradient_threshold=4.0e-4,
    min_opacity=0.01,
    opacity_prune_from=500,
    percent_dense=0.02,
    size_prune_from=3_000,
    screen_pruning_enabled=False,
    max_world_scale_fraction=0.10,
    opacity_reset_value=0.01,
    opacity_reset_at_densify_from=False,
    opacity_reset_optimizer_state=True,
)

_FLARE_MOVEMENT = _values(
    enabled=True,
    policy="flare_movement",
    max_points=None,
    runaway_max_points=None,
    densify_from=800,
    densify_until=24_000,
    densification_interval=250,
    movement_threshold=1.0e-3,
    min_opacity=0.005,
    min_effective_scale=1.0e-5,
    opacity_reset_value=0.01,
    opacity_reset_at_densify_from=False,
    opacity_reset_optimizer_state=True,
)


def _with(
    base: tuple[tuple[str, object], ...], **updates: object
) -> tuple[tuple[str, object], ...]:
    values = dict(base)
    values.update(updates)
    return tuple(values.items())


K_SWEEP_RECIPES = (
    GaussianKSweepRecipe(
        "fixed_25k", "fixed",
        model=_values(gaussian_initial_points=25_000),
        adaptation=_values(enabled=False),
    ),
    GaussianKSweepRecipe(
        "fixed_100k_scalereg", "fixed",
        adaptation=_values(enabled=False),
        optimization=_values(gaussian_scale_regularization=0.02),
    ),
    GaussianKSweepRecipe(
        "fixed_100k_reset", "fixed",
        adaptation=_values(
            enabled=True, policy="fixed", max_points=None,
            runaway_max_points=None, densify_until=30_001,
            opacity_reset_interval=3_000, opacity_reset_value=0.005,
            opacity_reset_at_densify_from=False,
            opacity_reset_optimizer_state=True,
        ),
    ),
    GaussianKSweepRecipe(
        "fixed_100k_scalereg_reset", "fixed",
        adaptation=_values(
            enabled=True, policy="fixed", max_points=None,
            runaway_max_points=None, densify_until=30_001,
            opacity_reset_interval=3_000, opacity_reset_value=0.005,
            opacity_reset_at_densify_from=False,
            opacity_reset_optimizer_state=True,
        ),
        optimization=_values(gaussian_scale_regularization=0.02),
    ),
    GaussianKSweepRecipe(
        "g3d_best_reset", "graphdeco",
        adaptation=_with(_GRAPHDECO, opacity_reset_interval=3_000),
    ),
    GaussianKSweepRecipe(
        "g3d_best_noreset", "graphdeco",
        adaptation=_with(_GRAPHDECO, opacity_reset_interval=0),
    ),
    GaussianKSweepRecipe(
        "g3d_best_scalereg_reset", "graphdeco",
        adaptation=_with(_GRAPHDECO, opacity_reset_interval=3_000),
        optimization=_values(gaussian_scale_regularization=0.02),
    ),
    GaussianKSweepRecipe(
        "g3d_best_scalereg_noreset", "graphdeco",
        adaptation=_with(_GRAPHDECO, opacity_reset_interval=0),
        optimization=_values(gaussian_scale_regularization=0.02),
    ),
    GaussianKSweepRecipe(
        "fmove_best_reset", "flare_movement",
        adaptation=_with(_FLARE_MOVEMENT, opacity_reset_interval=3_000),
    ),
    GaussianKSweepRecipe(
        "fmove_best_scalereg_noreset", "flare_movement",
        adaptation=_with(_FLARE_MOVEMENT, opacity_reset_interval=0),
        optimization=_values(gaussian_scale_regularization=0.02),
    ),
    GaussianKSweepRecipe(
        "g3d_scalereg_l1", "loss",
        adaptation=_with(_GRAPHDECO, opacity_reset_interval=0),
        optimization=_values(
            gaussian_scale_regularization=0.02,
            reconstruction_loss="l1",
        ),
    ),
    GaussianKSweepRecipe(
        "g3d_scalereg_mse", "loss",
        adaptation=_with(_GRAPHDECO, opacity_reset_interval=0),
        optimization=_values(
            gaussian_scale_regularization=0.02,
            reconstruction_loss="mse",
        ),
    ),
)

K_SWEEP_SIZE = len(K_VALUES) * len(K_SWEEP_RECIPES)


def k_sweep_case(index: int) -> tuple[int, GaussianKSweepRecipe]:
    if not 0 <= index < K_SWEEP_SIZE:
        raise IndexError(f"K sweep index must be in [0, {K_SWEEP_SIZE - 1}]")
    return K_VALUES[index // len(K_SWEEP_RECIPES)], K_SWEEP_RECIPES[
        index % len(K_SWEEP_RECIPES)
    ]


def build_k_sweep_config(
    base: ExperimentConfig,
    index: int,
    output_dir: str,
    wandb_name: str,
    wandb_mode: str = "online",
) -> ExperimentConfig:
    k, recipe = k_sweep_case(index)
    config = deepcopy(base)
    config.seed = 0
    config.output_dir = output_dir
    config.experiment_name = wandb_name
    config.renderer.gaussian_hit_buffer_size = k
    config.gaussian_adaptation.max_points = None
    config.gaussian_adaptation.runaway_max_points = None
    config.validation.eval_interval = 2_000
    config.validation.checkpoint_interval = 2_000
    config.validation.image_views = 1
    config.diagnostics.save_per_primitive = False
    for name, value in recipe.model:
        setattr(config.model, name, value)
    for name, value in recipe.adaptation:
        setattr(config.gaussian_adaptation, name, value)
    for name, value in recipe.optimization:
        setattr(config.optimization, name, value)
    config.wandb.mode = wandb_mode
    config.wandb.project = "foamsurf_stage1_gaussian_k_sweep"
    config.wandb.name = wandb_name
    config.wandb.tags = list(config.wandb.tags) + [
        "k-sweep", f"k{k}", recipe.family, recipe.label,
    ]
    config.validate()
    return config
