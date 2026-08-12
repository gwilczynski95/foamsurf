from __future__ import annotations

from dataclasses import dataclass

import torch


def _require_matrix(name: str, value: torch.Tensor, width: int) -> None:
    if value.ndim != 2 or value.shape[1] != width:
        raise ValueError(f"{name} must have shape [N, {width}]")


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True)
class GaussianScene:
    """Pre-baked directional appearance and planar Gaussian geometry.

    Quaternion ordering and parameter transforms intentionally match the
    FALCON base renderer: quaternions are (w, x, y, z), scales are stored in
    log space, opacity is a logit, and kappa is 1 + softplus(raw_kappa).
    """

    means: torch.Tensor
    log_scales: torch.Tensor
    quaternions: torch.Tensor
    colors: torch.Tensor
    opacity_logits: torch.Tensor
    raw_kappa: torch.Tensor

    def validate(self) -> None:
        _require_matrix("means", self.means, 3)
        _require_matrix("log_scales", self.log_scales, 2)
        _require_matrix("quaternions", self.quaternions, 4)
        _require_matrix("colors", self.colors, 3)
        _require_matrix("opacity_logits", self.opacity_logits, 1)
        _require_matrix("raw_kappa", self.raw_kappa, 1)
        count = self.means.shape[0]
        tensors = (
            self.log_scales,
            self.quaternions,
            self.colors,
            self.opacity_logits,
            self.raw_kappa,
        )
        if any(tensor.shape[0] != count for tensor in tensors):
            raise ValueError("all Gaussian tensors must have the same length")
        if any(tensor.device != self.means.device for tensor in tensors):
            raise ValueError("all Gaussian tensors must use the same device")
        if any(tensor.dtype != self.means.dtype for tensor in tensors):
            raise ValueError("all Gaussian tensors must use the same dtype")
        for name, tensor in (
            ("means", self.means),
            ("log_scales", self.log_scales),
            ("quaternions", self.quaternions),
            ("colors", self.colors),
            ("opacity_logits", self.opacity_logits),
            ("raw_kappa", self.raw_kappa),
        ):
            _require_finite(name, tensor)
        if count and (self.quaternions.square().sum(-1) <= 1.0e-12).any():
            raise ValueError("Gaussian quaternions must have non-zero norm")


@dataclass(frozen=True)
class VoronoiScene:
    """Piecewise-constant optical attributes attached to Voronoi sites."""

    sites: torch.Tensor
    densities: torch.Tensor
    colors: torch.Tensor

    def validate(self) -> None:
        _require_matrix("sites", self.sites, 3)
        _require_matrix("densities", self.densities, 1)
        _require_matrix("colors", self.colors, 3)
        count = self.sites.shape[0]
        if count == 0:
            raise ValueError("a Voronoi scene must contain at least one site")
        if self.densities.shape[0] != count or self.colors.shape[0] != count:
            raise ValueError("all Voronoi tensors must have the same length")
        if self.densities.device != self.sites.device:
            raise ValueError("Voronoi densities must be on the sites device")
        if self.colors.device != self.sites.device:
            raise ValueError("Voronoi colors must be on the sites device")
        if self.densities.dtype != self.sites.dtype:
            raise ValueError("Voronoi densities must use the sites dtype")
        if self.colors.dtype != self.sites.dtype:
            raise ValueError("Voronoi colors must use the sites dtype")
        _require_finite("sites", self.sites)
        _require_finite("densities", self.densities)
        _require_finite("colors", self.colors)


@dataclass(frozen=True)
class HybridRender:
    image: torch.Tensor
    alpha: torch.Tensor
    transmittance: torch.Tensor
    surface_depth: torch.Tensor
    surface_index: torch.Tensor

