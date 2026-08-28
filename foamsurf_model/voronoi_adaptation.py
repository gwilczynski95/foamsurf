from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from training.config import VoronoiAdaptationConfig
from .medium import VoronoiMediumModel
from .topology import VoronoiTopologyController


@dataclass(frozen=True)
class VoronoiTopologyPlan:
    keep_indices: torch.Tensor
    parent_indices: torch.Tensor
    child_sites: torch.Tensor
    original_count: int

    @property
    def prune_count(self) -> int:
        return self.original_count - int(self.keep_indices.numel())

    @property
    def densify_count(self) -> int:
        return int(self.parent_indices.numel())

    @property
    def is_noop(self) -> bool:
        return self.prune_count == 0 and self.densify_count == 0


class VoronoiAdaptationController:
    """Error-guided Voronoi split/prune for a participating medium.

    This deliberately ports only RadFoam's useful topology ideas: split cells
    with high reconstruction demand and remove redundant, inactive small
    cells. It does not use RadFoam's depth-quantile/surface regularizer.
    """

    def __init__(
        self,
        model: VoronoiMediumModel,
        optimizer: torch.optim.Adam,
        topology: VoronoiTopologyController,
        config: VoronoiAdaptationConfig,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.topology = topology
        self.config = config
        self.revision = 0
        self.total_densified = 0
        self.total_pruned = 0
        self._reset_statistics()
        self.low_activity_streak = torch.zeros(
            model.point_count, dtype=torch.int32, device=model.sites.device
        )
        self.last_statistics = self._empty_summary()

    @staticmethod
    def _empty_summary() -> dict[str, float | int]:
        return {
            "densified": 0,
            "pruned": 0,
            "duplicate_repairs": 0,
            "triangulation_retries": 0,
            "score_p95": 0.0,
            "cell_radius_p95": 0.0,
            "active_fraction": 0.0,
            "observed_fraction": 0.0,
        }

    def _reset_statistics(self) -> None:
        count = self.model.point_count
        device = self.model.sites.device
        self.contribution = torch.zeros(count, device=device)
        self.visibility = torch.zeros(count, device=device)
        self.path_length = torch.zeros(count, device=device)
        self.weighted_path_length = torch.zeros(count, device=device)
        self.max_error = torch.zeros(count, device=device)
        self.gradient = torch.zeros(count, device=device)

    def should_collect(self, iteration: int) -> bool:
        return (
            self.config.enabled
            and iteration < self.config.densify_until
            and not self._frozen_at_capacity()
        )

    def _frozen_at_capacity(self) -> bool:
        return (
            self.config.freeze_at_capacity
            and self.config.max_points is not None
            and self.model.point_count >= self.config.max_points
        )

    def should_adapt(self, iteration: int) -> bool:
        return (
            self.config.enabled
            and not self._frozen_at_capacity()
            and self.config.densify_from <= iteration < self.config.densify_until
            and (iteration - self.config.adaptation_offset)
            % self.config.interval
            == 0
        )

    @torch.no_grad()
    def collect_statistics(self, statistics: torch.Tensor) -> None:
        expected = (self.model.point_count, 5)
        if statistics.shape != expected:
            raise ValueError(f"medium statistics must have shape {expected}")
        self.contribution += statistics[:, 0]
        self.visibility += statistics[:, 1]
        self.path_length += statistics[:, 2]
        self.weighted_path_length += statistics[:, 3]
        self.max_error = torch.maximum(self.max_error, statistics[:, 4])
        squared = torch.zeros_like(self.gradient)
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                squared += parameter.grad.detach().reshape(
                    parameter.shape[0], -1
                ).square().sum(dim=1)
        self.gradient += squared.sqrt()

    @torch.no_grad()
    def reorder_statistics(self, permutation: torch.Tensor | None) -> None:
        if permutation is None:
            return
        for name in (
            "contribution", "visibility", "path_length",
            "weighted_path_length", "max_error", "gradient",
            "low_activity_streak",
        ):
            setattr(self, name, getattr(self, name)[permutation].contiguous())

    @torch.no_grad()
    def _cell_geometry(self) -> tuple[torch.Tensor, torch.Tensor]:
        count = self.model.point_count
        offsets = self.topology.offsets.long()
        adjacency = self.topology.adjacency.long()
        degree = offsets[1:] - offsets[:-1]
        owners = torch.repeat_interleave(
            torch.arange(count, device=self.model.sites.device), degree
        )
        radii = torch.zeros(count, device=self.model.sites.device)
        farthest = torch.arange(count, device=self.model.sites.device)
        if adjacency.numel() == 0:
            return radii, farthest
        distances = (self.model.sites[adjacency] - self.model.sites[owners]).norm(dim=-1)
        radii.scatter_reduce_(0, owners, distances, reduce="amax", include_self=True)
        edge_ids = torch.arange(adjacency.numel(), device=adjacency.device)
        candidates = torch.where(
            distances >= radii[owners] - 1.0e-7,
            edge_ids,
            torch.full_like(edge_ids, adjacency.numel()),
        )
        selected = torch.full(
            (count,), adjacency.numel(), dtype=torch.long, device=adjacency.device
        )
        selected.scatter_reduce_(0, owners, candidates, reduce="amin", include_self=True)
        valid = selected < adjacency.numel()
        farthest[valid] = adjacency[selected[valid]]
        return radii, farthest

    @torch.no_grad()
    def _repair_duplicate_children(
        self,
        keep_indices: torch.Tensor,
        child_sites: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Perturb only exact child collisions before Delaunay construction.

        A zero configured jitter may place a midpoint exactly on another site.
        The triangulation rejects such populations.  This safety perturbation
        is several orders of magnitude below the explicit child jitter and is
        applied only to colliding children.
        """

        if child_sites.numel() == 0:
            return child_sites, 0
        occupied = self.model.sites[keep_indices]
        repaired = child_sites.clone()
        repaired_total = 0
        diagonal = float(
            (self.model.bbox_max - self.model.bbox_min).norm().item()
        )
        base_step = max(diagonal * 1.0e-6, 1.0e-7)
        child_ids = torch.arange(
            repaired.shape[0], device=repaired.device, dtype=repaired.dtype
        )

        for attempt in range(8):
            combined = torch.cat((occupied, repaired), dim=0)
            _, inverse, counts = torch.unique(
                combined, dim=0, return_inverse=True, return_counts=True
            )
            duplicate = counts[inverse[occupied.shape[0]:]] > 1
            duplicate_count = int(duplicate.sum().item())
            if duplicate_count == 0:
                return repaired, repaired_total
            repaired_total += duplicate_count

            phase = child_ids + float(1 + attempt * repaired.shape[0])
            offsets = torch.stack(
                (
                    torch.sin(phase * 12.9898),
                    torch.sin(phase * 78.233 + 0.5),
                    torch.sin(phase * 37.719 + 1.0),
                ),
                dim=-1,
            )
            offsets = offsets / offsets.norm(
                dim=-1, keepdim=True
            ).clamp_min(1.0e-8)
            repaired[duplicate] += (
                offsets[duplicate] * base_step * float(2**attempt)
            )
            repaired = torch.maximum(
                torch.minimum(repaired, self.model.bbox_max),
                self.model.bbox_min,
            )

        raise RuntimeError(
            "failed to separate duplicate Voronoi child sites after 8 attempts"
        )

    @torch.no_grad()
    def propose_plan(self) -> VoronoiTopologyPlan:
        count = self.model.point_count
        device = self.model.sites.device
        indices = torch.arange(count, device=device)
        radii, farthest = self._cell_geometry()
        observed = self.visibility >= self.config.min_visibility
        normalized_contribution = self.contribution / self.visibility.clamp_min(1.0)
        normalized_gradient = self.gradient / self.visibility.clamp_min(1.0)
        normalized_weighted_path = (
            self.weighted_path_length / self.visibility.clamp_min(1.0)
        )
        active = observed & (
            (normalized_contribution >= self.config.contribution_threshold)
            | (normalized_gradient >= self.config.gradient_threshold)
        )
        if self.config.gradient_threshold <= 0:
            active = observed & (normalized_contribution >= self.config.contribution_threshold)

        adjacency = self.topology.adjacency.long()
        offsets = self.topology.offsets.long()
        degrees = offsets[1:] - offsets[:-1]
        owners = torch.repeat_interleave(indices, degrees)
        active_neighbors = torch.zeros(count, dtype=torch.int32, device=device)
        if adjacency.numel():
            active_neighbors.scatter_add_(0, owners, active[adjacency].to(torch.int32))
        diagonal = (self.model.bbox_max - self.model.bbox_min).norm()
        small = radii < self.config.prune_cell_radius_fraction * diagonal
        low = (~active) & (active_neighbors == 0) & small & (~self.model.guard_mask)
        self.low_activity_streak = torch.where(
            low, self.low_activity_streak + 1, torch.zeros_like(self.low_activity_streak)
        )
        prune = self.low_activity_streak >= self.config.prune_patience
        maximum_prune = min(
            int(self.config.max_prune_fraction * count),
            max(0, count - self.config.min_points),
        )
        prune_indices = indices[prune]
        if prune_indices.numel() > maximum_prune:
            order = torch.argsort(normalized_contribution[prune_indices], stable=True)
            prune_indices = prune_indices[order[:maximum_prune]]
        prune_mask = torch.zeros(count, dtype=torch.bool, device=device)
        prune_mask[prune_indices] = True
        keep = indices[~prune_mask]

        score = (
            self.max_error
            * normalized_contribution.clamp_min(0).sqrt()
            * normalized_weighted_path.clamp_min(0).sqrt()
            + normalized_gradient
        ) * radii * observed.float()
        score[self.model.guard_mask | prune_mask] = 0
        candidate_count = int((score > 0).sum().item())
        requested = min(
            candidate_count,
            max(0, int(self.config.densify_fraction * int((~self.model.guard_mask).sum().item()))),
        )
        if self.config.max_points is not None:
            requested = min(requested, max(0, self.config.max_points - int(keep.numel())))
        parents = torch.topk(score, requested, sorted=False).indices if requested else indices[:0]
        direction = self.model.sites[farthest[parents]] - self.model.sites[parents]
        child_sites = self.model.sites[parents] + 0.5 * direction
        if requested and self.config.child_jitter_fraction > 0:
            noise = torch.randn_like(child_sites)
            noise = noise / noise.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
            child_sites += noise * direction.norm(dim=-1, keepdim=True) * self.config.child_jitter_fraction
        child_sites = torch.maximum(
            torch.minimum(child_sites, self.model.bbox_max), self.model.bbox_min
        )
        child_sites, duplicate_repairs = self._repair_duplicate_children(
            keep, child_sites
        )

        observed_values = score[observed]
        self.last_statistics = {
            "densified": int(parents.numel()),
            "pruned": int(prune_indices.numel()),
            "duplicate_repairs": duplicate_repairs,
            "score_p95": float(torch.quantile(observed_values, 0.95).item()) if observed_values.numel() else 0.0,
            "cell_radius_p95": float(torch.quantile(radii, 0.95).item()),
            "active_fraction": float(active.float().mean().item()),
            "observed_fraction": float(observed.float().mean().item()),
        }
        return VoronoiTopologyPlan(keep, parents, child_sites, count)

    @torch.no_grad()
    def apply_plan(self, plan: VoronoiTopologyPlan) -> dict[str, int]:
        if plan.original_count != self.model.point_count:
            raise RuntimeError("stale Voronoi topology plan")
        if plan.is_noop:
            self._reset_statistics()
            return {"densify": 0, "prune": 0}
        old_streak = self.low_activity_streak
        combined_streak = torch.cat(
            (old_streak[plan.keep_indices], torch.zeros(plan.densify_count, dtype=old_streak.dtype, device=old_streak.device))
        )
        self.topology.replace_population(
            self.model, self.optimizer, plan.keep_indices,
            plan.parent_indices, plan.child_sites,
        )
        self.last_statistics["triangulation_retries"] = (
            self.topology.last_replacement_retries
        )
        self.low_activity_streak = combined_streak[self.topology.last_permutation]
        self.total_densified += plan.densify_count
        self.total_pruned += plan.prune_count
        self.revision += 1
        self._reset_statistics()
        return {"densify": plan.densify_count, "prune": plan.prune_count}

    def state_dict(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "total_densified": self.total_densified,
            "total_pruned": self.total_pruned,
            "low_activity_streak": self.low_activity_streak.detach().cpu(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        streak = state["low_activity_streak"].to(self.model.sites.device)
        if streak.numel() != self.model.point_count:
            raise ValueError("Voronoi adaptation checkpoint count mismatch")
        self.low_activity_streak = streak
        self.revision = int(state["revision"])
        self.total_densified = int(state["total_densified"])
        self.total_pruned = int(state["total_pruned"])
