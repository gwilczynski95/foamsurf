from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn

from training.config import GaussianAdaptationConfig
from .gaussians import SurfaceGaussianModel


@dataclass(frozen=True)
class GaussianTopologyPlan:
    """Explicit GraphDECO clone/split/prune operation."""

    keep_indices: torch.Tensor
    clone_indices: torch.Tensor
    split_indices: torch.Tensor
    original_count: int
    densification_skipped: bool = False

    @property
    def prune_count(self) -> int:
        return self.original_count - int(self.keep_indices.numel()) - int(
            self.split_indices.numel()
        )

    @property
    def clone_count(self) -> int:
        return int(self.clone_indices.numel())

    @property
    def split_count(self) -> int:
        return int(self.split_indices.numel())

    @property
    def final_count(self) -> int:
        return (
            int(self.keep_indices.numel())
            + self.clone_count
            + 2 * self.split_count
        )

    @property
    def is_noop(self) -> bool:
        return (
            self.prune_count == 0
            and self.clone_count == 0
            and self.split_count == 0
        )


def scheduled_opacity_threshold(
    config: GaussianAdaptationConfig,
    iteration: int,
) -> float:
    """Return the opacity-pruning threshold at a given iteration."""
    initial = (
        config.min_opacity
        if config.min_opacity_initial is None
        else config.min_opacity_initial
    )
    if config.opacity_prune_ramp == 0:
        return float(config.min_opacity)
    progress = min(
        max(
            (iteration - config.opacity_prune_from)
            / float(config.opacity_prune_ramp),
            0.0,
        ),
        1.0,
    )
    if config.opacity_prune_ramp_shape == "linear":
        factor = progress
    elif config.opacity_prune_ramp_shape == "cosine":
        factor = 0.5 - 0.5 * math.cos(math.pi * progress)
    elif config.opacity_prune_ramp_shape == "quadratic":
        factor = progress * progress
    else:
        raise ValueError(
            f"unknown opacity-pruning ramp: "
            f"{config.opacity_prune_ramp_shape}"
        )
    return float(initial + factor * (config.min_opacity - initial))


def _quaternion_rotation(quaternion: torch.Tensor) -> torch.Tensor:
    q = torch.nn.functional.normalize(quaternion, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)


class GaussianAdaptationController:
    """GraphDECO clone/split/prune adapted to planar ray-traced splats."""

    def __init__(
        self,
        model: SurfaceGaussianModel,
        optimizer: torch.optim.Adam,
        config: GaussianAdaptationConfig,
        scene_extent: float,
        support_radius: float = 3.0,
        camera_centers: torch.Tensor | None = None,
    ) -> None:
        if config.far_prune_distance is not None and camera_centers is None:
            raise ValueError("far_prune_distance needs the training camera centers")
        self.camera_centers = camera_centers
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.scene_extent = float(scene_extent)
        self.support_radius = float(support_radius)
        self.total_cloned = 0
        self.total_split = 0
        self.total_pruned = 0
        self.total_budget_skips = 0
        self.total_opacity_resets = 0
        self.last_opacity_reset_changed = 0
        self.revision = 0
        self.last_plan_statistics = {
            "gradient_p50": 0.0,
            "gradient_p95": 0.0,
            "gradient_max": 0.0,
            "gradient_candidates": 0,
            "clone_candidates": 0,
            "split_candidates": 0,
            "prune_candidates": 0,
            "opacity_prune": 0,
            "opacity_prune_raw": 0,
            "opacity_prune_rejected": 0,
            "opacity_threshold": 0.0,
            "densification_active": 0,
            "opacity_pruning_active": 0,
            "screen_prune": 0,
            "world_prune": 0,
            "far_prune": 0,
            "gradient_opacity_overlap": 0,
            "gradient_screen_overlap": 0,
            "gradient_world_overlap": 0,
            "budget_rejected": 0,
            "screen_replacements": 0,
            "radius_p50": 0.0,
            "radius_p95": 0.0,
            "radius_p99": 0.0,
            "radius_max": 0.0,
        }
        self._reset_statistics()

    def _reset_statistics(self) -> None:
        count = self.model.point_count
        device = self.model.means.device
        self.gradient_sum = torch.zeros(count, device=device)
        self.gradient_denom = torch.zeros(count, device=device)
        self.max_radii2d = torch.zeros(count, device=device)

    def should_collect(self, iteration: int) -> bool:
        return (
            self.config.enabled
            and self.config.policy == "graphdeco"
            and iteration < self.config.densify_until
        )

    def should_densify(self, iteration: int) -> bool:
        return (
            self.config.enabled
            and self.config.policy == "graphdeco"
            and self.config.densify_from < iteration
            and iteration < self.config.densify_until
            and iteration % self.config.densification_interval == 0
        )

    def should_prune(self, iteration: int) -> bool:
        if (
            not self.config.enabled
            or self.config.policy != "graphdeco"
            or iteration <= self.config.opacity_prune_from
        ):
            return False
        if self.config.opacity_prune_until is None:
            in_range = iteration < self.config.densify_until
        else:
            in_range = iteration < self.config.opacity_prune_until
        interval = (
            self.config.densification_interval
            if self.config.opacity_prune_interval is None
            else self.config.opacity_prune_interval
        )
        return in_range and iteration % interval == 0

    def should_adapt(self, iteration: int) -> bool:
        if self.config.policy == "graphdeco":
            return self.should_densify(iteration) or self.should_prune(
                iteration
            )
        return False

    def should_reset_opacity(self, iteration: int) -> bool:
        if (
            not self.config.enabled
            or self.config.opacity_reset_interval <= 0
            or iteration >= self.config.densify_until
        ):
            return False
        periodic = iteration % self.config.opacity_reset_interval == 0
        white_background_start = (
            self.config.opacity_reset_at_densify_from
            and iteration == self.config.densify_from
        )
        return periodic or white_background_start

    @torch.no_grad()
    def reset_opacity(self) -> int:
        """Clamp Gaussian opacity and optionally clear its Adam moments."""
        parameter = self.model.opacity_logits
        opacity = parameter.sigmoid()
        reset_value = opacity.new_tensor(self.config.opacity_reset_value)
        changed = int((opacity > reset_value).sum().item())
        clamped = torch.minimum(opacity, reset_value)
        parameter.copy_(torch.logit(clamped.clamp(1.0e-6, 1.0 - 1.0e-6)))
        parameter.grad = None
        if self.config.opacity_reset_optimizer_state:
            state = self.optimizer.state.get(parameter, {})
            for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                value = state.get(name)
                if torch.is_tensor(value):
                    value.zero_()
        self.total_opacity_resets += 1
        self.last_opacity_reset_changed = changed
        return changed

    @torch.no_grad()
    def far_from_cameras(self, points: torch.Tensor) -> torch.Tensor:
        """Points farther than far_prune_distance from every training camera."""
        distance = torch.full(points.shape[:1], float("inf"), device=points.device)
        centers = self.camera_centers.to(points.device, points.dtype)
        for chunk in torch.split(torch.arange(points.shape[0], device=points.device), 262144):
            distance[chunk] = torch.cdist(points[chunk], centers).amin(dim=1)
        return distance > self.config.far_prune_distance

    @torch.no_grad()
    def projected_radii(self, camera) -> torch.Tensor:
        """Return the exact 2DGS homographic support radius in pixels.

        This is the PyTorch equivalent of ``compute_transmat`` followed by
        ``compute_aabb`` in hbb1/diff-surfel-rasterization. A Jacobian of the
        projection is insufficient for tilted surfels because perspective is
        not affine across their tangent plane.
        """

        relative = self.model.means - camera.origin
        x = (relative * camera.right).sum(dim=-1)
        y = (relative * camera.down).sum(dim=-1)
        z = (relative * camera.forward).sum(dim=-1)
        focal_x = camera.width / (
            2.0 * torch.tan(z.new_tensor(0.5 * camera.fov_x))
        )
        focal_y = camera.height / (
            2.0 * torch.tan(z.new_tensor(0.5 * camera.fov_y))
        )
        center_x = z.new_tensor(0.5 * (camera.width - 1))
        center_y = z.new_tensor(0.5 * (camera.height - 1))
        if getattr(camera, 'intrinsics', None) is not None:
            fx, fy, cx, cy = camera.intrinsics
            focal_x, focal_y = z.new_tensor(fx), z.new_tensor(fy)
            center_x, center_y = z.new_tensor(cx - 0.5), z.new_tensor(cy - 0.5)

        rotation = _quaternion_rotation(self.model.quaternions)
        tangents = rotation[:, :, :2] * self.model.scales[:, None, :]
        tangent_x = torch.einsum("i,nij->nj", camera.right, tangents)
        tangent_y = torch.einsum("i,nij->nj", camera.down, tangents)
        tangent_z = torch.einsum("i,nij->nj", camera.forward, tangents)

        x_coefficients = torch.cat(
            (
                focal_x * tangent_x + center_x * tangent_z,
                (focal_x * x + center_x * z)[:, None],
            ),
            dim=-1,
        )
        y_coefficients = torch.cat(
            (
                focal_y * tangent_y + center_y * tangent_z,
                (focal_y * y + center_y * z)[:, None],
            ),
            dim=-1,
        )
        depth_coefficients = torch.cat((tangent_z, z[:, None]), dim=-1)

        cutoff = z.new_tensor(self.support_radius)
        conic_metric = torch.stack(
            (cutoff.square(), cutoff.square(), z.new_tensor(-1.0))
        )
        denominator = (
            conic_metric * depth_coefficients.square()
        ).sum(dim=-1)
        valid = denominator != 0
        safe_denominator = torch.where(
            valid, denominator, torch.ones_like(denominator)
        )
        dual_conic = conic_metric / safe_denominator[:, None]

        center_px = torch.stack(
            (
                (dual_conic * x_coefficients * depth_coefficients).sum(-1),
                (dual_conic * y_coefficients * depth_coefficients).sum(-1),
            ),
            dim=-1,
        )
        extent_squared = center_px.square() - torch.stack(
            (
                (dual_conic * x_coefficients.square()).sum(dim=-1),
                (dual_conic * y_coefficients.square()).sum(dim=-1),
            ),
            dim=-1,
        )
        extent = torch.sqrt(extent_squared.clamp_min(1.0e-4))
        filter_radius = cutoff * z.new_tensor(2.0).sqrt() / 2.0
        radius = torch.ceil(
            torch.maximum(extent.amax(dim=-1), filter_radius)
        )

        tile_size = 16.0
        grid_x = math.ceil(camera.width / tile_size)
        grid_y = math.ceil(camera.height / tile_size)
        rect_min_x = torch.trunc(
            (center_px[:, 0] - radius) / tile_size
        ).clamp(0, grid_x)
        rect_min_y = torch.trunc(
            (center_px[:, 1] - radius) / tile_size
        ).clamp(0, grid_y)
        rect_max_x = torch.trunc(
            (center_px[:, 0] + radius + tile_size - 1) / tile_size
        ).clamp(0, grid_x)
        rect_max_y = torch.trunc(
            (center_px[:, 1] + radius + tile_size - 1) / tile_size
        ).clamp(0, grid_y)
        overlaps_image = (rect_max_x > rect_min_x) & (
            rect_max_y > rect_min_y
        )
        valid &= (
            (z > 0.2)
            & overlaps_image
            & torch.isfinite(center_px).all(dim=-1)
            & torch.isfinite(radius)
        )
        return torch.where(valid, radius, torch.zeros_like(radius))

    @torch.no_grad()
    def collect_statistics(
        self,
        screen_gradient: torch.Tensor | None,
        gaussian_statistics: torch.Tensor,
        camera,
    ) -> None:
        expected = (self.model.point_count, 4)
        if gaussian_statistics.shape != expected:
            raise ValueError(
                f"gaussian statistics must have shape {expected}"
            )
        if screen_gradient is None:
            return
        # Official 2DGS accumulates densification statistics for every
        # primitive whose rasterizer radius is non-zero. This is deliberately
        # broader than "contributed to a processed ray": an in-frustum but
        # occluded surfel still contributes to the denominator.
        radii = self.projected_radii(camera)
        visible = radii > 0
        self.gradient_sum[visible] += screen_gradient.detach()[visible].norm(
            dim=-1
        )
        self.gradient_denom[visible] += 1
        self.max_radii2d[visible] = torch.maximum(
            self.max_radii2d[visible], radii[visible]
        )

    @torch.no_grad()
    def propose_plan(
        self, iteration: int, *, allow_densification: bool = True
    ) -> GaussianTopologyPlan:
        count = self.model.point_count
        device = self.model.means.device
        indices = torch.arange(count, device=device)
        if not self.config.enabled:
            return GaussianTopologyPlan(
                indices, indices[:0], indices[:0], count
            )

        opacities = self.model.opacities.squeeze(-1)
        scales = self.model.scales.amax(dim=-1)
        densification_active = (
            allow_densification and self.should_densify(iteration)
        )
        opacity_pruning_active = self.should_prune(iteration)
        opacity_threshold = scheduled_opacity_threshold(
            self.config, iteration
        )
        raw_opacity_prune = torch.zeros_like(opacities, dtype=torch.bool)
        if opacity_pruning_active:
            raw_opacity_prune = opacities < opacity_threshold
        size_pruning_active = (
            densification_active
            and iteration > self.config.size_prune_from
        )
        screen_prune = torch.zeros_like(raw_opacity_prune)
        world_prune = torch.zeros_like(raw_opacity_prune)
        if size_pruning_active:
            if self.config.screen_pruning_enabled:
                screen_prune = (
                    self.max_radii2d > self.config.max_screen_size
                )
            world_prune = scales > (
                self.config.max_world_scale_fraction * self.scene_extent
            )
        far_prune = torch.zeros_like(raw_opacity_prune)
        if self.config.far_prune_distance is not None and (
            opacity_pruning_active or size_pruning_active
        ):
            far_prune = self.far_from_cameras(self.model.means) & (
                scales > self.config.far_prune_scale
            )
        mandatory_prune = screen_prune | world_prune | far_prune
        opacity_prune = raw_opacity_prune & mandatory_prune
        optional_opacity = indices[
            raw_opacity_prune & ~mandatory_prune
        ]
        maximum_opacity_prunes = int(
            math.floor(
                count * self.config.max_opacity_prune_fraction
            )
        )
        remaining_opacity_budget = max(
            maximum_opacity_prunes - int(opacity_prune.sum().item()), 0
        )
        remaining_population_budget = max(
            count
            - self.config.min_points
            - int(mandatory_prune.sum().item()),
            0,
        )
        accepted_count = min(
            int(optional_opacity.numel()),
            remaining_opacity_budget,
            remaining_population_budget,
        )
        if accepted_count:
            order = torch.argsort(
                opacities[optional_opacity], stable=True
            )
            opacity_prune[optional_opacity[order[:accepted_count]]] = True
        prune = opacity_prune | mandatory_prune

        average_gradient = self.gradient_sum / self.gradient_denom
        average_gradient.nan_to_num_(0.0)
        gradient_candidates = (
            average_gradient >= self.config.gradient_threshold
        ) & densification_active
        densify = gradient_candidates.clone()

        # GraphDECO creates replacements before pruning the old parents.
        # Filtering this mask with ~prune caused the destructive collapse in
        # run 449540: useful screen-oversized parents disappeared without
        # clone or split children.
        split_threshold = self.config.percent_dense * self.scene_extent
        clone_mask = densify & (scales <= split_threshold)
        split_mask = densify & (scales > split_threshold)

        # Children which the following pruning phase would immediately
        # remove do not affect the final model. Avoid allocating them. New
        # children have no accumulated screen-radius history, so screen
        # pruning removes the oversized parent, not its replacement.
        clone_mask &= ~opacity_prune & ~world_prune & ~far_prune
        split_mask &= ~opacity_prune
        if self.config.far_prune_distance is not None:
            # Split children are scaled by 1/1.6 and stay near the parent.
            split_mask &= ~(far_prune & (scales / 1.6 > self.config.far_prune_scale))
        if size_pruning_active:
            child_world_prune = scales / 1.6 > (
                self.config.max_world_scale_fraction * self.scene_extent
            )
            split_mask &= ~child_world_prune

        # Keep the final topology within max_points without disabling the
        # entire densification event. Pruning first creates room for useful
        # replacements. If the provisional plan is still too large, reject
        # the lowest-gradient clone/split candidates until it fits.
        clone_indices = indices[clone_mask]
        split_indices = indices[split_mask]
        base_count = int((~prune).sum().item())
        clone_costs = torch.ones_like(clone_indices)
        split_costs = 1 + prune[split_indices].long()
        candidate_indices = torch.cat((clone_indices, split_indices))
        candidate_costs = torch.cat((clone_costs, split_costs))
        provisional_count = base_count + int(candidate_costs.sum().item())
        budget_rejected = 0
        densification_skipped = (
            self.config.max_points is not None
            and provisional_count > self.config.max_points
        )
        if densification_skipped and candidate_indices.numel():
            excess = provisional_count - self.config.max_points
            priorities = average_gradient[candidate_indices]
            rejection_order = torch.argsort(
                priorities, descending=False, stable=True
            )
            cumulative_reduction = torch.cumsum(
                candidate_costs[rejection_order], dim=0
            )
            reject_count = int(
                torch.searchsorted(
                    cumulative_reduction,
                    cumulative_reduction.new_tensor(excess),
                    right=False,
                ).item()
            ) + 1
            rejected = rejection_order[:reject_count]
            clone_candidate_count = int(clone_indices.numel())
            rejected_clones = rejected[rejected < clone_candidate_count]
            rejected_splits = rejected[rejected >= clone_candidate_count]
            if rejected_clones.numel():
                clone_mask[clone_indices[rejected_clones]] = False
            if rejected_splits.numel():
                split_mask[
                    split_indices[
                        rejected_splits - clone_candidate_count
                    ]
                ] = False
            budget_rejected = int(rejected.numel())

        clone = indices[clone_mask]
        split = indices[split_mask]
        keep = indices[~prune & ~split_mask]

        observed = self.gradient_denom > 0
        observed_gradient = average_gradient[observed]
        if observed_gradient.numel():
            gradient_p50 = float(
                torch.quantile(observed_gradient, 0.50).item()
            )
            gradient_p95 = float(
                torch.quantile(observed_gradient, 0.95).item()
            )
            gradient_max = float(observed_gradient.max().item())
            observed_radii = self.max_radii2d[observed]
            radius_p50 = float(
                torch.quantile(observed_radii, 0.50).item()
            )
            radius_p95 = float(
                torch.quantile(observed_radii, 0.95).item()
            )
            radius_p99 = float(
                torch.quantile(observed_radii, 0.99).item()
            )
            radius_max = float(observed_radii.max().item())
        else:
            gradient_p50 = gradient_p95 = gradient_max = 0.0
            radius_p50 = radius_p95 = radius_p99 = radius_max = 0.0

        self.last_plan_statistics = {
            "gradient_p50": gradient_p50,
            "gradient_p95": gradient_p95,
            "gradient_max": gradient_max,
            "gradient_candidates": int(
                gradient_candidates.sum().item()
            ),
            "clone_candidates": int(clone_mask.sum().item()),
            "split_candidates": int(split_mask.sum().item()),
            "prune_candidates": int(prune.sum().item()),
            "opacity_prune": int(opacity_prune.sum().item()),
            "opacity_prune_raw": int(raw_opacity_prune.sum().item()),
            "opacity_prune_rejected": int(
                raw_opacity_prune.sum().item()
                - opacity_prune.sum().item()
            ),
            "opacity_threshold": opacity_threshold,
            "densification_active": int(densification_active),
            "opacity_pruning_active": int(opacity_pruning_active),
            "screen_prune": int(screen_prune.sum().item()),
            "world_prune": int(world_prune.sum().item()),
            "far_prune": int(far_prune.sum().item()),
            "gradient_opacity_overlap": int(
                (gradient_candidates & opacity_prune).sum().item()
            ),
            "gradient_screen_overlap": int(
                (gradient_candidates & screen_prune).sum().item()
            ),
            "gradient_world_overlap": int(
                (gradient_candidates & world_prune).sum().item()
            ),
            "budget_rejected": budget_rejected,
            "screen_replacements": int(
                (screen_prune & (clone_mask | split_mask)).sum().item()
            ),
            "radius_p50": radius_p50,
            "radius_p95": radius_p95,
            "radius_p99": radius_p99,
            "radius_max": radius_max,
        }
        return GaussianTopologyPlan(
            keep,
            clone,
            split,
            count,
            densification_skipped=densification_skipped,
        )

    @staticmethod
    def _mapped_state(
        state: dict[str, Any],
        old: nn.Parameter,
        keep: torch.Tensor,
        new_count: int,
    ) -> dict[str, Any]:
        mapped: dict[str, Any] = {}
        extension = new_count - int(keep.numel())
        for key, value in state.items():
            if torch.is_tensor(value) and value.shape == old.shape:
                zeros = value.new_zeros((extension, *value.shape[1:]))
                mapped[key] = torch.cat((value[keep], zeros), dim=0)
            elif torch.is_tensor(value):
                mapped[key] = value.clone()
            else:
                mapped[key] = value
        return mapped

    @torch.no_grad()
    def apply_plan(self, plan: GaussianTopologyPlan) -> dict[str, int]:
        if plan.original_count != self.model.point_count:
            raise ValueError("Gaussian plan does not match current model")
        if plan.final_count < 1:
            raise ValueError("Gaussian pruning would remove every primitive")
        if plan.is_noop:
            self.total_budget_skips += int(plan.densification_skipped)
            self._reset_statistics()
            return {
                "clone": 0,
                "split": 0,
                "prune": 0,
                "budget_skip": int(plan.densification_skipped),
            }

        old_parameters = self.model.named_primitive_parameters()
        split_count = plan.split_count
        split_offsets = None
        if split_count:
            source_scales = self.model.scales[plan.split_indices]
            local = torch.randn(
                (2 * split_count, 3),
                device=self.model.means.device,
                dtype=self.model.means.dtype,
            )
            local[:, :2] *= source_scales.repeat(2, 1)
            local[:, 2] = 0.0
            rotations = _quaternion_rotation(
                self.model.quaternions[plan.split_indices]
            ).repeat(2, 1, 1)
            split_offsets = torch.bmm(
                rotations, local.unsqueeze(-1)
            ).squeeze(-1)

        opacity_source = None
        split_opacity_children = None
        if self.config.preserve_clone_opacity:
            opacity = self.model.opacities.detach().squeeze(-1)
            opacity_source = self.model.opacity_logits.detach().clone()
            kept_mask = torch.zeros(
                plan.original_count,
                dtype=torch.bool,
                device=self.model.means.device,
            )
            kept_mask[plan.keep_indices] = True
            distributed_indices = plan.clone_indices[
                kept_mask[plan.clone_indices]
            ]
            if distributed_indices.numel():
                distributed = 1.0 - torch.sqrt(
                    (1.0 - opacity[distributed_indices]).clamp_min(0.0)
                )
                opacity_source[distributed_indices] = torch.logit(
                    distributed.clamp(1.0e-6, 1.0 - 1.0e-6)
                ).unsqueeze(-1)
        if self.config.preserve_split_opacity and split_count:
            opacity = self.model.opacities.detach().squeeze(-1)
            distributed = 1.0 - torch.sqrt(
                (1.0 - opacity[plan.split_indices]).clamp_min(0.0)
            )
            split_opacity_children = torch.logit(
                distributed.clamp(1.0e-6, 1.0 - 1.0e-6)
            ).unsqueeze(-1).repeat(2, 1)

        for name, old in old_parameters.items():
            source = old.detach()
            if name == "opacity_logits" and opacity_source is not None:
                source = opacity_source
            parts = [source[plan.keep_indices]]
            if plan.clone_count:
                parts.append(source[plan.clone_indices])
            if split_count:
                children = source[plan.split_indices].repeat(
                    (2,) + (1,) * (old.ndim - 1)
                )
                if (
                    name == "opacity_logits"
                    and split_opacity_children is not None
                ):
                    children = split_opacity_children
                elif name == "means":
                    children = children + split_offsets
                elif name == "log_scales":
                    children = children - torch.log(
                        children.new_tensor(1.6)
                    )
                parts.append(children)
            value = torch.cat(parts, dim=0).contiguous()
            new_parameter = nn.Parameter(value)

            group_name = f"gaussian_{name}"
            group = next(
                group
                for group in self.optimizer.param_groups
                if group.get("name") == group_name
            )
            if len(group["params"]) != 1 or group["params"][0] is not old:
                raise RuntimeError(f"optimizer group mismatch for {name}")
            old_state = self.optimizer.state.pop(old, None)
            group["params"] = [new_parameter]
            if old_state is not None:
                self.optimizer.state[new_parameter] = self._mapped_state(
                    old_state, old, plan.keep_indices, plan.final_count
                )
            setattr(self.model, name, new_parameter)

        self.total_cloned += plan.clone_count
        self.total_split += plan.split_count
        self.total_pruned += plan.prune_count
        self.total_budget_skips += int(plan.densification_skipped)
        self.revision += 1
        self._reset_statistics()
        return {
            "clone": plan.clone_count,
            "split": plan.split_count,
            "prune": plan.prune_count,
            "budget_skip": int(plan.densification_skipped),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "gradient_sum": self.gradient_sum,
            "gradient_denom": self.gradient_denom,
            "max_radii2d": self.max_radii2d,
            "total_cloned": self.total_cloned,
            "total_split": self.total_split,
            "total_pruned": self.total_pruned,
            "total_budget_skips": self.total_budget_skips,
            "total_opacity_resets": self.total_opacity_resets,
            "last_opacity_reset_changed": self.last_opacity_reset_changed,
            "revision": self.revision,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for name in (
            "gradient_sum",
            "gradient_denom",
            "max_radii2d",
        ):
            value = state[name].to(self.model.means.device)
            if value.shape != (self.model.point_count,):
                raise ValueError("Gaussian adaptation checkpoint size mismatch")
            setattr(self, name, value)
        self.total_cloned = int(state["total_cloned"])
        self.total_split = int(state["total_split"])
        self.total_pruned = int(state["total_pruned"])
        self.total_budget_skips = int(state.get("total_budget_skips", 0))
        self.total_opacity_resets = int(state.get("total_opacity_resets", 0))
        self.last_opacity_reset_changed = int(
            state.get("last_opacity_reset_changed", 0)
        )
        self.revision = int(state["revision"])
