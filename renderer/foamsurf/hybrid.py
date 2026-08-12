from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .types import GaussianScene, HybridRender, VoronoiScene


class HybridRenderer(nn.Module):
    """Differentiable correctness renderer for FoamSurf event fusion.

    Each ray gets one depth-sorted event stream containing planar Gaussian
    hits and Voronoi bisector crossings. Medium segments are integrated with
    Beer--Lambert compositing before the surface event at the same depth.

    This implementation deliberately favors a transparent mathematical
    contract over speed. Its O(R * (G + V^2)) construction is the oracle for
    tests and for the future fused OptiX/CUDA implementation.
    """

    def __init__(
        self,
        near: float = 0.0,
        far: float = 100.0,
        gaussian_radius_squared: float = 11.3449,
        epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if far <= near:
            raise ValueError("far must be greater than near")
        if gaussian_radius_squared <= 0.0:
            raise ValueError("gaussian_radius_squared must be positive")
        self.near = float(near)
        self.far = float(far)
        self.gaussian_radius_squared = float(gaussian_radius_squared)
        self.epsilon = float(epsilon)

    @staticmethod
    def _rotation_rows(quaternion: torch.Tensor) -> torch.Tensor:
        a, b, c, d = quaternion.unbind(-1)
        norm_factor = 2.0 / quaternion.square().sum(-1).clamp_min(1.0e-12)
        bs = b * norm_factor
        cs = c * norm_factor
        ds = d * norm_factor
        ab = a * bs
        ac = a * cs
        ad = a * ds
        bb = b * bs
        bc = b * cs
        bd = b * ds
        cc = c * cs
        cd = c * ds
        dd = d * ds
        return torch.stack(
            (
                1.0 - cc - dd,
                bc + ad,
                bd - ac,
                bc - ad,
                1.0 - bb - dd,
                cd + ab,
                bd + ac,
                cd - ab,
                1.0 - bb - cc,
            ),
            dim=-1,
        ).reshape(-1, 3, 3)

    def _gaussian_hits(
        self,
        origin: torch.Tensor,
        direction: torch.Tensor,
        scene: GaussianScene,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = self._rotation_rows(scene.quaternions)
        normals = rows[:, 2]
        denominator = (normals * direction).sum(-1)
        safe_denominator = torch.where(
            denominator.abs() > self.epsilon,
            denominator,
            torch.ones_like(denominator),
        )
        depth = (normals * (scene.means - origin)).sum(-1) / safe_denominator
        hit = origin + depth[:, None] * direction
        local = hit - scene.means
        inverse_scales = torch.exp(-scene.log_scales)
        u = (rows[:, 0] * local).sum(-1) * inverse_scales[:, 0]
        v = (rows[:, 1] * local).sum(-1) * inverse_scales[:, 1]
        radius_squared = u.square() + v.square()
        kappa = 1.0 + F.softplus(scene.raw_kappa[:, 0])
        radial_power = radius_squared.clamp_min(self.epsilon).pow(kappa)
        response = torch.exp(-0.5 * radial_power / kappa)
        alpha = torch.sigmoid(scene.opacity_logits[:, 0]) * response
        valid = (
            (denominator.abs() > self.epsilon)
            & (depth >= self.near)
            & (depth <= self.far)
            & (radius_squared <= self.gaussian_radius_squared)
        )
        return depth, alpha, valid

    def _voronoi_crossings(
        self,
        origin: torch.Tensor,
        direction: torch.Tensor,
        sites: torch.Tensor,
    ) -> torch.Tensor:
        count = sites.shape[0]
        if count < 2:
            return sites.new_empty((0,))
        pairs = torch.triu_indices(count, count, offset=1, device=sites.device)
        first = sites[pairs[0]]
        second = sites[pairs[1]]
        difference = second - first
        denominator = 2.0 * (difference * direction).sum(-1)
        first_distance = (origin - first).square().sum(-1)
        second_distance = (origin - second).square().sum(-1)
        numerator = second_distance - first_distance
        safe_denominator = torch.where(
            denominator.abs() > self.epsilon,
            denominator,
            torch.ones_like(denominator),
        )
        depths = numerator / safe_denominator
        valid = (
            (denominator.abs() > self.epsilon)
            & (depths > self.near)
            & (depths < self.far)
        )
        return depths[valid]

    def _integrate_medium(
        self,
        rgb: torch.Tensor,
        transmittance: torch.Tensor,
        origin: torch.Tensor,
        direction: torch.Tensor,
        start: torch.Tensor,
        end: torch.Tensor,
        scene: VoronoiScene | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if scene is None:
            return rgb, transmittance
        length = (end - start).clamp_min(0.0)
        midpoint = origin + (0.5 * (start + end)) * direction
        cell = (scene.sites - midpoint).square().sum(-1).argmin()
        density = scene.densities[cell, 0].clamp_min(0.0)
        alpha = 1.0 - torch.exp(-density * length)
        rgb = rgb + transmittance * alpha * scene.colors[cell]
        transmittance = transmittance * (1.0 - alpha)
        return rgb, transmittance

    def _render_ray(
        self,
        origin: torch.Tensor,
        direction: torch.Tensor,
        gaussians: GaussianScene | None,
        voronoi: VoronoiScene | None,
        background: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = origin.dtype
        device = origin.device
        near = torch.as_tensor(self.near, dtype=dtype, device=device)
        far = torch.as_tensor(self.far, dtype=dtype, device=device)

        event_depths = [near, far]
        event_tags = [-1, -1]
        gaussian_alpha = origin.new_empty((0,))
        gaussian_valid_indices = torch.empty(0, dtype=torch.long, device=device)
        if voronoi is not None:
            crossings = self._voronoi_crossings(
                origin, direction, voronoi.sites
            )
            event_depths.extend(crossings.unbind(0))
            event_tags.extend([-1] * crossings.numel())
        if gaussians is not None:
            depths, gaussian_alpha, valid = self._gaussian_hits(
                origin, direction, gaussians
            )
            gaussian_valid_indices = torch.nonzero(valid).flatten()
            event_depths.extend(depths[valid].unbind(0))
            event_tags.extend(gaussian_valid_indices.tolist())

        depths = torch.stack(event_depths)
        tags = torch.tensor(event_tags, dtype=torch.long, device=device)
        order = torch.argsort(depths, stable=True)
        depths = depths[order]
        tags = tags[order]

        rgb = torch.zeros(3, dtype=dtype, device=device)
        transmittance = torch.ones((), dtype=dtype, device=device)
        first_surface_depth = torch.full((), -1.0, dtype=dtype, device=device)
        first_surface_index = torch.full((), -1, dtype=torch.long, device=device)
        previous = depths[0]
        for event_index in range(1, depths.numel()):
            depth = depths[event_index]
            rgb, transmittance = self._integrate_medium(
                rgb,
                transmittance,
                origin,
                direction,
                previous,
                depth,
                voronoi,
            )
            tag = int(tags[event_index].item())
            if tag >= 0 and gaussians is not None:
                alpha = gaussian_alpha[tag]
                rgb = rgb + transmittance * alpha * gaussians.colors[tag]
                transmittance = transmittance * (1.0 - alpha)
                if int(first_surface_index.item()) < 0:
                    first_surface_depth = depth
                    first_surface_index = tags[event_index]
            previous = depth

        image = rgb + transmittance * background
        alpha = 1.0 - transmittance
        return (
            image,
            alpha,
            transmittance,
            first_surface_depth,
            first_surface_index,
        )

    def forward(
        self,
        rays: torch.Tensor,
        gaussians: GaussianScene | None = None,
        voronoi: VoronoiScene | None = None,
        background: torch.Tensor | tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> HybridRender:
        if gaussians is None and voronoi is None:
            raise ValueError("at least one representation must be provided")
        if rays.shape[-1] != 6:
            raise ValueError("rays must have shape [..., 6]")
        if gaussians is not None:
            gaussians.validate()
        if voronoi is not None:
            voronoi.validate()

        original_shape = rays.shape[:-1]
        flat_rays = rays.reshape(-1, 6)
        origins = flat_rays[:, :3]
        raw_directions = flat_rays[:, 3:]
        if not torch.isfinite(flat_rays).all():
            raise ValueError("rays must contain only finite values")
        if (raw_directions.square().sum(-1) <= self.epsilon**2).any():
            raise ValueError("ray directions must have non-zero norm")
        directions = F.normalize(raw_directions, dim=-1)
        reference = gaussians.means if gaussians is not None else voronoi.sites
        if rays.device != reference.device or rays.dtype != reference.dtype:
            raise ValueError("rays and scene tensors must share device and dtype")
        background_tensor = torch.as_tensor(
            background, dtype=rays.dtype, device=rays.device
        )
        if background_tensor.shape != (3,):
            raise ValueError("background must contain exactly three values")

        outputs = [
            self._render_ray(
                origin,
                direction,
                gaussians,
                voronoi,
                background_tensor,
            )
            for origin, direction in zip(origins, directions)
        ]
        images, alphas, transmittances, depths, indices = zip(*outputs)
        return HybridRender(
            image=torch.stack(images).reshape(*original_shape, 3),
            alpha=torch.stack(alphas).reshape(*original_shape, 1),
            transmittance=torch.stack(transmittances).reshape(*original_shape, 1),
            surface_depth=torch.stack(depths).reshape(*original_shape, 1),
            surface_index=torch.stack(indices).reshape(*original_shape, 1),
        )

    def render_gaussians(
        self,
        rays: torch.Tensor,
        scene: GaussianScene,
        background: torch.Tensor | tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> HybridRender:
        return self(rays, gaussians=scene, background=background)

    def render_voronoi(
        self,
        rays: torch.Tensor,
        scene: VoronoiScene,
        background: torch.Tensor | tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> HybridRender:
        return self(rays, voronoi=scene, background=background)

