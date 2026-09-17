from __future__ import annotations

from .config import MediumHandoffConfig, OptimizationConfig


def handoff_freeze_active(
    config: MediumHandoffConfig, iteration: int
) -> bool:
    """Whether the extra handoff freeze contains this main-loop iteration."""

    start = config.gaussian_freeze_start
    return (
        start is not None
        and start <= iteration < start + config.gaussian_freeze_duration
    )


def gaussian_is_frozen(
    optimization: OptimizationConfig,
    handoff: MediumHandoffConfig,
    iteration: int,
) -> bool:
    """Combine the legacy initial freeze and the finite handoff window."""

    return (
        iteration <= optimization.gaussian_freeze_until
        or handoff_freeze_active(handoff, iteration)
    )


def medium_lr_multipliers(
    config: MediumHandoffConfig, iteration: int
) -> dict[str, float]:
    """Return component-specific boosts, active only during handoff freeze."""

    if not handoff_freeze_active(config, iteration):
        return {
            "medium_sites": 1.0,
            "medium_density": 1.0,
            "medium_color": 1.0,
        }
    return {
        "medium_sites": float(config.medium_sites_lr_multiplier),
        "medium_density": float(config.medium_density_lr_multiplier),
        "medium_color": float(config.medium_color_lr_multiplier),
    }


def apply_medium_lr_multipliers(
    optimizer, rates: dict[str, float], multipliers: dict[str, float]
) -> dict[str, float]:
    """Apply boosts after the normal main-iteration LR schedule."""

    adjusted = {
        name: float(rate) * float(multipliers[name])
        for name, rate in rates.items()
    }
    for group in optimizer.param_groups:
        name = group.get("name")
        if name in adjusted:
            group["lr"] = adjusted[name]
    return adjusted


def handoff_boundary_iterations(config: MediumHandoffConfig) -> set[int]:
    """Main iterations evaluated around the freeze transition."""

    start = config.gaussian_freeze_start
    if start is None:
        return set()
    end = start + config.gaussian_freeze_duration
    return {start - 1, start, end - 1, end}
