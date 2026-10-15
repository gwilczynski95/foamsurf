from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from .config import DisentanglementConfig


def scheduled_weight(
    final_weight: float,
    iteration: int,
    start: int,
    ramp: int,
    shape: str,
    end: int | None = None,
    end_ramp: int = 0,
    end_ramp_shape: str = "linear",
    maintenance_scale: float = 1.0,
) -> float:
    """Return a deterministic ramp-up and optional maintenance schedule."""
    if final_weight <= 0.0 or iteration < start:
        return 0.0
    if ramp == 0:
        weight = float(final_weight)
    else:
        progress = min(max((iteration - start) / float(ramp), 0.0), 1.0)
        if shape == "linear":
            factor = progress
        elif shape == "cosine":
            factor = 0.5 - 0.5 * math.cos(math.pi * progress)
        elif shape == "quadratic":
            factor = progress * progress
        else:
            raise ValueError(f"unknown ramp shape: {shape}")
        weight = float(final_weight * factor)
    if end is None or iteration < end:
        return weight
    maintenance = float(final_weight * maintenance_scale)
    if end_ramp == 0:
        return maintenance
    progress = min(max((iteration - end) / float(end_ramp), 0.0), 1.0)
    if end_ramp_shape == "linear":
        factor = progress
    elif end_ramp_shape == "cosine":
        factor = 0.5 - 0.5 * math.cos(math.pi * progress)
    elif end_ramp_shape == "quadratic":
        factor = progress * progress
    else:
        raise ValueError(f"unknown end ramp shape: {end_ramp_shape}")
    return float(final_weight + (maintenance - final_weight) * factor)


def opacity_binarization_loss(opacity: torch.Tensor) -> torch.Tensor:
    if opacity.numel() == 0:
        return opacity.new_zeros(())
    return (opacity * (1.0 - opacity)).mean()


def undirected_edges(
    adjacency: torch.Tensor,
    offsets: torch.Tensor,
    point_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    adjacency = adjacency.to(torch.long)
    offsets = offsets.to(torch.long)
    if offsets.shape != (point_count + 1,):
        raise ValueError("offsets must have point_count + 1 entries")
    degrees = offsets[1:] - offsets[:-1]
    if bool((degrees < 0).any()) or int(degrees.sum().item()) != adjacency.numel():
        raise ValueError("adjacency and offsets are inconsistent")
    owners = torch.repeat_interleave(
        torch.arange(point_count, device=adjacency.device), degrees
    )
    valid = (
        (adjacency >= 0)
        & (adjacency < point_count)
        & (owners < adjacency)
    )
    return owners[valid], adjacency[valid]


@dataclass(frozen=True)
class DisentanglementLoss:
    total: torch.Tensor
    opacity_binarization: torch.Tensor
    opacity_binarization_weight: float

    def metrics(self) -> dict[str, float | int]:
        raw = float(self.opacity_binarization.detach().item())
        return {
            "disentanglement/total": float(self.total.detach().item()),
            "disentanglement/opacity_binarization_raw": raw,
            "disentanglement/opacity_binarization_weight": self.opacity_binarization_weight,
            "disentanglement/opacity_binarization_weighted": raw * self.opacity_binarization_weight,
        }


class DisentanglementRegularizer:
    """Scheduled per-Gaussian opacity binarization (pushes opacities to 0 or 1)."""

    def __init__(self, config: DisentanglementConfig) -> None:
        self.config = config

    def weight(self, iteration: int) -> float:
        config = self.config
        return scheduled_weight(
            config.opacity_binarization_weight, iteration,
            config.opacity_binarization_start, config.opacity_binarization_ramp,
            config.opacity_binarization_ramp_shape,
            end=config.opacity_binarization_end,
            end_ramp=config.opacity_binarization_end_ramp,
            end_ramp_shape=config.opacity_binarization_end_ramp_shape,
            maintenance_scale=config.opacity_binarization_maintenance_scale,
        )

    def compute(self, iteration: int, model: Any, topology: Any) -> DisentanglementLoss:
        weight = self.weight(iteration)
        if self.config.opacity_binarization_weight > 0.0:
            value = opacity_binarization_loss(model.gaussians.opacities)
        else:
            value = next(model.parameters()).new_zeros(())
        return DisentanglementLoss(total=weight * value, opacity_binarization=value,
                                   opacity_binarization_weight=weight)

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Stateless; older checkpoints stored density-cap calibration, now unused."""
