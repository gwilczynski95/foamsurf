from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F

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


def opaque_if_used_loss(
    opacity: torch.Tensor,
    contribution: torch.Tensor,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    opacity = opacity.reshape(-1)
    contribution = contribution.detach().reshape(-1)
    if opacity.shape != contribution.shape:
        raise ValueError("opacity and contribution must have equal length")
    if opacity.numel() == 0:
        return opacity.new_zeros(())
    nonnegative = contribution.clamp_min(0.0)
    denominator = nonnegative.sum()
    numerator = (
        nonnegative * -torch.log(opacity.clamp_min(epsilon))
    ).sum()
    return torch.where(
        denominator > 0.0,
        numerator / denominator.clamp_min(epsilon),
        numerator * 0.0,
    )


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


def graph_total_variation_loss(
    density: torch.Tensor,
    adjacency: torch.Tensor,
    offsets: torch.Tensor,
    mode: str,
    huber_delta: float,
) -> tuple[torch.Tensor, int]:
    density = density.reshape(-1)
    first, second = undirected_edges(
        adjacency, offsets, density.numel()
    )
    if first.numel() == 0:
        return density.new_zeros(()), 0
    difference = density[first] - density[second]
    if mode == "huber":
        value = F.huber_loss(
            difference,
            torch.zeros_like(difference),
            reduction="mean",
            delta=huber_delta,
        )
    elif mode == "l1":
        value = difference.abs().mean()
    elif mode == "l2":
        value = difference.square().mean()
    else:
        raise ValueError(f"unknown graph TV mode: {mode}")
    return value, int(first.numel())


def relative_density_cap_loss(
    density: torch.Tensor,
    cap: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    density = density.reshape(-1)
    if density.numel() == 0:
        zero = density.new_zeros(())
        return zero, zero
    cap_tensor = torch.as_tensor(
        cap, dtype=density.dtype, device=density.device
    ).detach()
    if not bool(torch.isfinite(cap_tensor)) or float(cap_tensor) <= 0.0:
        raise ValueError("density cap must be finite and positive")
    excess = (density / cap_tensor - 1.0).clamp_min(0.0)
    return excess.square().mean(), (excess > 0.0).float().mean()


def optical_thickness_loss(
    medium_transmittance: torch.Tensor,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    if medium_transmittance.numel() == 0:
        return medium_transmittance.new_zeros(())
    return -torch.log(
        medium_transmittance.clamp(min=epsilon, max=1.0)
    ).mean()


@dataclass(frozen=True)
class DisentanglementLoss:
    total: torch.Tensor
    opacity_binarization: torch.Tensor
    opaque_if_used: torch.Tensor
    graph_tv: torch.Tensor
    density_cap: torch.Tensor
    optical_thickness: torch.Tensor
    opacity_binarization_weight: float
    opaque_if_used_weight: float
    graph_tv_weight: float
    density_cap_weight: float
    optical_thickness_weight: float
    density_cap_value: float
    density_cap_quantile: float
    density_cap_violation_fraction: float
    graph_edge_count: int

    def metrics(self) -> dict[str, float | int]:
        raw = {
            "opacity_binarization": self.opacity_binarization,
            "opaque_if_used": self.opaque_if_used,
            "graph_tv": self.graph_tv,
            "density_cap": self.density_cap,
            "optical_thickness": self.optical_thickness,
        }
        weights = {
            "opacity_binarization": self.opacity_binarization_weight,
            "opaque_if_used": self.opaque_if_used_weight,
            "graph_tv": self.graph_tv_weight,
            "density_cap": self.density_cap_weight,
            "optical_thickness": self.optical_thickness_weight,
        }
        result: dict[str, float | int] = {
            "disentanglement/total": float(self.total.detach().item()),
            "disentanglement/density_cap_value": self.density_cap_value,
            "disentanglement/density_cap_quantile": self.density_cap_quantile,
            "disentanglement/density_cap_violation_fraction": (
                self.density_cap_violation_fraction
            ),
            "disentanglement/graph_edge_count": self.graph_edge_count,
        }
        for name, value in raw.items():
            result[f"disentanglement/{name}_raw"] = float(
                value.detach().item()
            )
            result[f"disentanglement/{name}_weight"] = weights[name]
            result[f"disentanglement/{name}_weighted"] = float(
                (value.detach() * weights[name]).item()
            )
        return result


class DisentanglementRegularizer:
    """Normalized surface/medium priors with resumable cap calibration."""

    def __init__(self, config: DisentanglementConfig) -> None:
        self.config = config
        self._density_stat_ema: float | None = None
        self._density_cap_value: float | None = None
        self._density_cap_frozen = False
        self._last_cap_update_iteration = -1

    def _weight(self, name: str, iteration: int) -> float:
        schedule_tail: dict[str, Any] = {}
        if name == "opacity_binarization":
            schedule_tail = {
                "end": self.config.opacity_binarization_end,
                "end_ramp": self.config.opacity_binarization_end_ramp,
                "end_ramp_shape": (
                    self.config.opacity_binarization_end_ramp_shape
                ),
                "maintenance_scale": (
                    self.config.opacity_binarization_maintenance_scale
                ),
            }
        return scheduled_weight(
            getattr(self.config, f"{name}_weight"),
            iteration,
            getattr(self.config, f"{name}_start"),
            getattr(self.config, f"{name}_ramp"),
            getattr(self.config, f"{name}_ramp_shape"),
            **schedule_tail,
        )

    def requires_contribution_diagnostics(self, iteration: int) -> bool:
        return self._weight("opaque_if_used", iteration) > 0.0

    def requires_medium_transmittance(self, iteration: int) -> bool:
        return self._weight("optical_thickness", iteration) > 0.0

    def _density_values(self, model: Any) -> torch.Tensor:
        density = model.medium.densities.reshape(-1)
        guard_mask = model.medium.guard_mask.reshape(-1)
        if guard_mask.shape != density.shape:
            raise ValueError("medium guard mask does not match density")
        values = density[~guard_mask]
        return values if values.numel() else density

    @torch.no_grad()
    def _update_density_cap(
        self, values: torch.Tensor, iteration: int
    ) -> tuple[float, float]:
        config = self.config
        statistic = float(
            torch.quantile(values.detach(), config.density_cap_quantile).item()
        )
        policy = config.density_cap_policy
        if policy == "fixed":
            self._density_cap_value = config.density_cap_fixed_value
            return self._density_cap_value, statistic

        should_update = (
            self._density_cap_value is None
            or iteration - self._last_cap_update_iteration
            >= config.density_cap_update_interval
        )
        if policy == "warmup_freeze" and self._density_cap_frozen:
            should_update = False
        if should_update:
            if policy == "rolling":
                reference = statistic
            else:
                if self._density_stat_ema is None:
                    self._density_stat_ema = statistic
                else:
                    decay = config.density_cap_ema_decay
                    self._density_stat_ema = (
                        decay * self._density_stat_ema
                        + (1.0 - decay) * statistic
                    )
                reference = self._density_stat_ema
            self._density_cap_value = max(
                config.density_cap_minimum,
                config.density_cap_multiplier * reference,
            )
            self._last_cap_update_iteration = iteration
        if (
            policy == "warmup_freeze"
            and iteration >= config.density_cap_warmup_until
        ):
            self._density_cap_frozen = True
        if self._density_cap_value is None:
            raise RuntimeError("density cap was not initialized")
        return self._density_cap_value, statistic

    def compute(
        self,
        iteration: int,
        model: Any,
        topology: Any,
        diagnostics: Any | None = None,
        medium_transmittance: torch.Tensor | None = None,
    ) -> DisentanglementLoss:
        reference = next(model.parameters())
        zero = reference.new_zeros(())
        weights = {
            name: self._weight(name, iteration)
            for name in (
                "opacity_binarization",
                "opaque_if_used",
                "graph_tv",
                "density_cap",
                "optical_thickness",
            )
        }

        opacity_bin = zero
        opaque = zero
        graph_tv = zero
        cap_loss = zero
        tau = zero
        cap_value = float("nan")
        cap_quantile = float("nan")
        cap_violation = 0.0
        edge_count = 0

        if self.config.opacity_binarization_weight > 0.0:
            opacity_bin = opacity_binarization_loss(
                model.gaussians.opacities
            )
        if weights["opaque_if_used"] > 0.0:
            if diagnostics is None:
                raise ValueError("opaque-if-used requires render diagnostics")
            opaque = opaque_if_used_loss(
                model.gaussians.opacities,
                diagnostics.gaussian[:, 0],
                self.config.epsilon,
            )
        if self.config.graph_tv_weight > 0.0:
            graph_tv, edge_count = graph_total_variation_loss(
                model.medium.densities,
                topology.adjacency,
                topology.offsets,
                self.config.graph_tv_mode,
                self.config.graph_tv_huber_delta,
            )

        if self.config.density_cap_weight > 0.0:
            values = self._density_values(model)
            cap_value, cap_quantile = self._update_density_cap(
                values, iteration
            )
            cap_loss, violation = relative_density_cap_loss(
                values, cap_value
            )
            cap_violation = float(violation.detach().item())

        if weights["optical_thickness"] > 0.0:
            if medium_transmittance is None:
                raise ValueError(
                    "optical-thickness loss requires medium transmittance"
                )
            tau = optical_thickness_loss(
                medium_transmittance, self.config.epsilon
            )

        total = (
            weights["opacity_binarization"] * opacity_bin
            + weights["opaque_if_used"] * opaque
            + weights["graph_tv"] * graph_tv
            + weights["density_cap"] * cap_loss
            + weights["optical_thickness"] * tau
        )
        return DisentanglementLoss(
            total=total,
            opacity_binarization=opacity_bin,
            opaque_if_used=opaque,
            graph_tv=graph_tv,
            density_cap=cap_loss,
            optical_thickness=tau,
            opacity_binarization_weight=weights["opacity_binarization"],
            opaque_if_used_weight=weights["opaque_if_used"],
            graph_tv_weight=weights["graph_tv"],
            density_cap_weight=weights["density_cap"],
            optical_thickness_weight=weights["optical_thickness"],
            density_cap_value=cap_value,
            density_cap_quantile=cap_quantile,
            density_cap_violation_fraction=cap_violation,
            graph_edge_count=edge_count,
        )

    def state_dict(self) -> dict[str, float | int | bool | None]:
        return {
            "density_stat_ema": self._density_stat_ema,
            "density_cap_value": self._density_cap_value,
            "density_cap_frozen": self._density_cap_frozen,
            "last_cap_update_iteration": self._last_cap_update_iteration,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._density_stat_ema = (
            None
            if state.get("density_stat_ema") is None
            else float(state["density_stat_ema"])
        )
        self._density_cap_value = (
            None
            if state.get("density_cap_value") is None
            else float(state["density_cap_value"])
        )
        self._density_cap_frozen = bool(
            state.get("density_cap_frozen", False)
        )
        self._last_cap_update_iteration = int(
            state.get("last_cap_update_iteration", -1)
        )
