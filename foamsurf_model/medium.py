from __future__ import annotations

from collections.abc import Mapping
import itertools
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from foamsurf import VoronoiScene
from training.config import ModelConfig


def _inverse_softplus(value: float) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64)
    return float((tensor + torch.log(-torch.expm1(-tensor))).item())


def _logit(value: torch.Tensor) -> torch.Tensor:
    epsilon = torch.finfo(value.dtype).eps
    value = value.clamp(epsilon, 1.0 - epsilon)
    return torch.log(value) - torch.log1p(-value)


def scene_bounds(
    surface_points: np.ndarray,
    camera_origins: torch.Tensor,
    config: ModelConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = camera_origins.device
    if config.medium_bbox_min is not None:
        lower = torch.tensor(
            config.medium_bbox_min, dtype=torch.float32, device=device
        )
        upper = torch.tensor(
            config.medium_bbox_max, dtype=torch.float32, device=device
        )
        return lower, upper

    surface_points = np.asarray(surface_points, dtype=np.float32)
    robust_lower = np.percentile(surface_points, 1.0, axis=0)
    robust_upper = np.percentile(surface_points, 99.0, axis=0)
    surface_lower = torch.as_tensor(robust_lower, dtype=torch.float32, device=device)
    surface_upper = torch.as_tensor(robust_upper, dtype=torch.float32, device=device)
    lower = torch.minimum(surface_lower, camera_origins.amin(dim=0))
    upper = torch.maximum(surface_upper, camera_origins.amax(dim=0))
    extent = (upper - lower).clamp_min(1.0e-3)
    lower = lower - config.medium_bbox_margin * extent
    upper = upper + config.medium_bbox_margin * extent
    return lower, upper


def _stratified_sites(
    count: int,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    side = max(1, math.ceil(count ** (1.0 / 3.0)))
    coordinates = torch.stack(
        torch.meshgrid(
            torch.arange(side, device=lower.device),
            torch.arange(side, device=lower.device),
            torch.arange(side, device=lower.device),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)
    selection = torch.randperm(
        coordinates.shape[0], device=lower.device
    )[:count]
    coordinates = coordinates[selection].to(torch.float32)
    jitter = torch.rand_like(coordinates)
    unit = (coordinates + jitter) / float(side)
    return lower + unit * (upper - lower)


def _guard_sites(
    lower: torch.Tensor,
    upper: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    center = 0.5 * (lower + upper)
    half_extent = 0.5 * (upper - lower) * scale
    signs = torch.tensor(
        list(itertools.product((-1.0, 1.0), repeat=3)),
        dtype=torch.float32,
        device=lower.device,
    )
    guards = center + signs * half_extent
    jitter = torch.randn_like(guards) * (
        1.0e-5 * half_extent.norm().clamp_min(1.0)
    )
    return guards + jitter


class VoronoiMediumModel(nn.Module):
    """Sparse piecewise-constant emission/absorption medium."""

    PARAMETER_NAMES = ("sites", "raw_density", "raw_color")

    def __init__(
        self,
        lower: torch.Tensor,
        upper: torch.Tensor,
        config: ModelConfig,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.medium_type = config.medium_type
        if self.medium_type not in ("scalar", "rgb_backscatter"):
            raise ValueError("Unknown medium_type")
        self.register_buffer("bbox_min", lower.to(self.device))
        self.register_buffer("bbox_max", upper.to(self.device))
        guard_mask = torch.zeros(config.medium_points, dtype=torch.bool, device=self.device)
        if config.medium_guard_points:
            guard_mask[-config.medium_guard_points:] = True
        self.register_buffer("guard_mask", guard_mask)

        inner_count = config.medium_points - config.medium_guard_points
        sites = _stratified_sites(
            inner_count, self.bbox_min, self.bbox_max
        )
        if config.medium_guard_points:
            sites = torch.cat(
                (
                    sites,
                    _guard_sites(
                        self.bbox_min,
                        self.bbox_max,
                        config.medium_guard_scale,
                    ),
                ),
                dim=0,
            )

        raw_density = torch.full(
            (config.medium_points, 6 if self.medium_type == "rgb_backscatter" else 1),
            _inverse_softplus(config.medium_initial_density),
            dtype=torch.float32,
            device=self.device,
        )
        initial_color = torch.tensor(
            config.medium_initial_color,
            dtype=torch.float32,
            device=self.device,
        ).expand(config.medium_points, 3)
        raw_color = _logit(initial_color)

        self.sites = nn.Parameter(sites)
        self.raw_density = nn.Parameter(raw_density)
        self.raw_color = nn.Parameter(raw_color.clone())

    @property
    def point_count(self) -> int:
        return int(self.sites.shape[0])

    @property
    def densities(self) -> torch.Tensor:
        values = self.optical_coefficients
        return values[:, :3].mean(-1, keepdim=True) if self.is_rgb else values

    @property
    def is_rgb(self) -> bool:
        return getattr(self, "medium_type", "scalar") == "rgb_backscatter"

    @property
    def optical_coefficients(self) -> torch.Tensor:
        return F.softplus(self.raw_density)

    @property
    def color_mode(self) -> str:
        return "global" if hasattr(self, "raw_global_color") else "per_cell"

    def _activate_color(self, raw: torch.Tensor) -> torch.Tensor:
        if getattr(self, "color_activation", "sigmoid") == "dc":
            return (0.5 + 0.28209479177387814 * raw).clamp_min(0)
        return torch.sigmoid(raw)

    def _inverse_color(self, color: torch.Tensor) -> torch.Tensor:
        if getattr(self, "color_activation", "sigmoid") == "dc":
            return (color - 0.5) / 0.28209479177387814
        return _logit(color)

    @property
    def global_color(self) -> torch.Tensor:
        """One scene-wide backscatter color; only densities stay per cell."""
        return self._activate_color(self.raw_global_color)

    def enable_global_color(self, color: torch.Tensor) -> None:
        """Replace per-cell backscatter colors by one shared RGB parameter.

        ``raw_color`` stays registered so topology permutations and Voronoi
        adaptation are unchanged, but it no longer reaches the renderer.
        """
        color = torch.as_tensor(color, dtype=torch.float32, device=self.device)
        if color.shape != (3,):
            raise ValueError("global medium color must contain RGB")
        self.raw_global_color = nn.Parameter(self._inverse_color(color).clone())

    @property
    def colors(self) -> torch.Tensor:
        if self.color_mode == "global":
            # Broadcast, so the renderer sees ordinary per-cell colors and its
            # gradients sum into the single shared parameter.
            return self.global_color.expand(self.point_count, 3).contiguous()
        return self._activate_color(self.raw_color)

    def render_scene(self) -> VoronoiScene:
        if self.is_rgb:
            from foamsurf.types import RGBVoronoiScene
            values = self.optical_coefficients
            return RGBVoronoiScene(self.sites, self.densities, self.colors,
                                   values[:, :3].contiguous(), values[:, 3:].contiguous())
        return VoronoiScene(
            sites=self.sites,
            densities=self.densities,
            colors=self.colors,
        )

    def named_primitive_parameters(self) -> dict[str, nn.Parameter]:
        return {
            name: getattr(self, name) for name in self.PARAMETER_NAMES
        }

    @torch.no_grad()
    def reorder_without_optimizer(
        self, permutation: torch.Tensor
    ) -> None:
        self.guard_mask = self.guard_mask[permutation].contiguous()
        for name in self.PARAMETER_NAMES:
            value = getattr(self, name)
            setattr(
                self,
                name,
                nn.Parameter(value.detach()[permutation].contiguous()),
            )

    def checkpoint_metadata(self) -> dict[str, object]:
        return {"point_count": self.point_count, "guard_count": int(self.guard_mask.sum().item()),
                "color_activation": getattr(self, "color_activation", "sigmoid"),
                "color_mode": self.color_mode,
                "medium_type": getattr(self, "medium_type", "scalar")}

    @classmethod
    def from_checkpoint(
        cls,
        state: Mapping[str, torch.Tensor],
        metadata: Mapping[str, object],
        bbox_min: torch.Tensor,
        bbox_max: torch.Tensor,
        device: torch.device | str,
    ) -> "VoronoiMediumModel":
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.device = torch.device(device)
        instance.medium_type = metadata.get("medium_type", "scalar")
        if instance.medium_type not in ("scalar", "rgb_backscatter"):
            raise ValueError("Unknown medium_type in checkpoint")
        expected = 6 if instance.medium_type == "rgb_backscatter" else 1
        if state["raw_density"].shape != (int(metadata["point_count"]), expected):
            raise ValueError("medium coefficient shape disagrees with checkpoint type")
        instance.color_activation = metadata.get("color_activation", "sigmoid")
        if instance.color_activation not in ("sigmoid", "dc"):
            raise ValueError("Unknown medium color activation")
        instance.register_buffer(
            "bbox_min", bbox_min.to(instance.device)
        )
        instance.register_buffer(
            "bbox_max", bbox_max.to(instance.device)
        )
        guard_count = int(metadata.get("guard_count", 0))
        guard_mask = torch.zeros(int(metadata["point_count"]), dtype=torch.bool, device=instance.device)
        if guard_count:
            guard_mask[-guard_count:] = True
        instance.register_buffer("guard_mask", guard_mask)
        for name in cls.PARAMETER_NAMES:
            setattr(
                instance,
                name,
                nn.Parameter(state[name].to(instance.device)),
            )
        if instance.point_count != int(metadata["point_count"]):
            raise ValueError("medium checkpoint point count mismatch")
        color_mode = metadata.get("color_mode", "per_cell")
        if color_mode == "global":
            # The value is restored by load_state_dict when it is not supplied.
            raw = state.get("raw_global_color", torch.zeros(3))
            instance.raw_global_color = nn.Parameter(
                torch.as_tensor(raw, dtype=torch.float32).to(instance.device).clone()
            )
        elif color_mode != "per_cell":
            raise ValueError("Unknown medium color mode in checkpoint")
        return instance
