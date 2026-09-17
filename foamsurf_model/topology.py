from __future__ import annotations

import torch
from torch import nn

import foamsurf_topology

from .medium import VoronoiMediumModel


class VoronoiTopologyController:
    """Own Delaunay topology and safe optimizer-state permutations.

    B0 updates connectivity but never changes the number of sites. Future
    densification and pruning belong here, not in the training loop.
    """

    GROUP_NAMES = {
        "sites": "medium_sites",
        "raw_density": "medium_density",
        "raw_color": "medium_color",
    }

    def __init__(
        self,
        medium: VoronoiMediumModel,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        with torch.no_grad():
            self.triangulation = foamsurf_topology.Triangulation(
                medium.sites.detach()
            )
            permutation = self.triangulation.permutation().to(torch.long)
            self.last_permutation = permutation
            if optimizer is None:
                medium.reorder_without_optimizer(permutation)
            else:
                self._permute_with_optimizer(
                    medium, optimizer, permutation
                )
        self.revision = 0
        self.last_replacement_retries = 0
        self.last_replacement_failed = False
        self._read_topology()

    def _read_topology(self) -> None:
        self.adjacency = (
            self.triangulation.point_adjacency()
            .to(torch.uint32)
            .contiguous()
        )
        self.offsets = (
            self.triangulation.point_adjacency_offsets()
            .to(torch.uint32)
            .contiguous()
        )

    @torch.no_grad()
    def _permute_with_optimizer(
        self,
        medium: VoronoiMediumModel,
        optimizer: torch.optim.Optimizer,
        permutation: torch.Tensor,
    ) -> None:
        medium.guard_mask = medium.guard_mask[permutation].contiguous()
        groups = {
            group.get("name"): group for group in optimizer.param_groups
        }
        for attribute, group_name in self.GROUP_NAMES.items():
            old_parameter = getattr(medium, attribute)
            group = groups.get(group_name)
            if group is None or group["params"][0] is not old_parameter:
                raise RuntimeError(
                    f"optimizer group {group_name!r} does not own "
                    f"medium.{attribute}"
                )
            old_state = optimizer.state.pop(old_parameter, {})
            new_parameter = nn.Parameter(
                old_parameter.detach()[permutation].contiguous()
            )
            setattr(medium, attribute, new_parameter)
            group["params"][0] = new_parameter

            new_state = {}
            for name, value in old_state.items():
                if (
                    torch.is_tensor(value)
                    and value.ndim > 0
                    and value.shape[0] == permutation.numel()
                ):
                    new_state[name] = value[permutation].contiguous()
                else:
                    new_state[name] = (
                        value.clone() if torch.is_tensor(value) else value
                    )
            if new_state:
                optimizer.state[new_parameter] = new_state

    @torch.no_grad()
    def replace_population(
        self,
        medium: VoronoiMediumModel,
        optimizer: torch.optim.Optimizer,
        keep_indices: torch.Tensor,
        parent_indices: torch.Tensor,
        child_sites: torch.Tensor,
        max_attempts: int = 12,
    ) -> bool:
        """Atomically replace sites and retry ambiguous triangulations.

        Candidate geometry is triangulated before model or optimizer state is
        mutated. If every attempt fails, the pre-event state remains intact
        and ``False`` is returned. A single degenerate adaptation proposal is
        recoverable and must not terminate a multi-hour training run.
        """
        old_count = medium.point_count
        child_count = int(parent_indices.numel())
        candidate_guard_mask = torch.cat(
            (
                medium.guard_mask[keep_indices],
                torch.zeros(
                    child_count, dtype=torch.bool, device=medium.sites.device
                ),
            )
        )
        survivor_sites = medium.sites.detach()[keep_indices]
        original_sites = torch.cat(
            (survivor_sites, child_sites.detach()), dim=0
        ).contiguous()
        candidate_sites = original_sites.clone()
        diagonal = float((medium.bbox_max - medium.bbox_min).norm().item())
        base_step = max(diagonal * 1.0e-7, 1.0e-8)
        triangulation = None
        failure_count = 0
        while triangulation is None:
            try:
                triangulation = foamsurf_topology.Triangulation(candidate_sites)
            except foamsurf_topology.TriangulationFailedError:
                failure_count += 1
                if failure_count >= max_attempts:
                    self.last_replacement_retries = failure_count
                    self.last_replacement_failed = True
                    return False

                # First move inserted sites. If ambiguity is inherited from
                # survivors, broaden the microscopic deterministic jitter to
                # all non-guard sites. Domain-defining guards never move.
                movable = torch.zeros(
                    original_sites.shape[0], dtype=torch.bool,
                    device=original_sites.device,
                )
                if child_count:
                    movable[-child_count:] = True
                if failure_count >= 4 or child_count == 0:
                    movable |= ~candidate_guard_mask
                ids = torch.arange(
                    original_sites.shape[0], device=original_sites.device,
                    dtype=original_sites.dtype,
                )
                phase = ids + float(
                    failure_count * original_sites.shape[0] + 1
                )
                directions = torch.stack(
                    (
                        torch.sin(phase * 12.9898),
                        torch.sin(phase * 78.233 + 0.5),
                        torch.sin(phase * 37.719 + 1.0),
                    ),
                    dim=-1,
                )
                directions = directions / directions.norm(
                    dim=-1, keepdim=True
                ).clamp_min(1.0e-8)
                offsets = (
                    directions * base_step * float(2 ** (failure_count - 1))
                )
                candidate_sites = original_sites.clone()
                candidate_sites[movable] += offsets[movable]
                candidate_sites = torch.maximum(
                    torch.minimum(candidate_sites, medium.bbox_max),
                    medium.bbox_min,
                ).contiguous()

        groups = {
            group.get("name"): group for group in optimizer.param_groups
        }
        for attribute, group_name in self.GROUP_NAMES.items():
            parameter = getattr(medium, attribute)
            group = groups.get(group_name)
            if group is None or group["params"][0] is not parameter:
                raise RuntimeError(
                    f"optimizer group {group_name!r} does not own "
                    f"medium.{attribute}"
                )

        self.last_replacement_retries = failure_count
        self.last_replacement_failed = False
        permutation = triangulation.permutation().to(torch.long)
        for attribute, group_name in self.GROUP_NAMES.items():
            old_parameter = getattr(medium, attribute)
            group = groups[group_name]
            old_state = optimizer.state.pop(old_parameter, {})
            survivor = old_parameter.detach()[keep_indices]
            child = old_parameter.detach()[parent_indices]
            values = (
                candidate_sites
                if attribute == "sites"
                else torch.cat((survivor, child), dim=0).contiguous()
            )
            new_parameter = nn.Parameter(values)
            setattr(medium, attribute, new_parameter)
            group["params"][0] = new_parameter
            new_state = {}
            for name, value in old_state.items():
                if (
                    torch.is_tensor(value)
                    and value.ndim > 0
                    and value.shape[0] == old_count
                ):
                    zeros = torch.zeros(
                        (child_count, *value.shape[1:]),
                        dtype=value.dtype,
                        device=value.device,
                    )
                    new_state[name] = torch.cat(
                        (value[keep_indices], zeros), dim=0
                    ).contiguous()
                else:
                    new_state[name] = (
                        value.clone() if torch.is_tensor(value) else value
                    )
            if new_state:
                optimizer.state[new_parameter] = new_state

        medium.guard_mask = candidate_guard_mask
        self.triangulation = triangulation
        self._permute_with_optimizer(medium, optimizer, permutation)
        self.last_permutation = permutation
        self._read_topology()
        self.revision += 1
        return True

    @torch.no_grad()
    def update(
        self,
        medium: VoronoiMediumModel,
        optimizer: torch.optim.Optimizer,
        incremental: bool = True,
        max_attempts: int = 25,
    ) -> bool:
        """Rebuild connectivity and preserve Adam state if ordering changes."""
        points = medium.sites.detach()
        if not torch.isfinite(points).all():
            raise RuntimeError("non-finite Voronoi site position")

        candidate = points
        perturbation = 1.0e-6
        needs_permutation = False
        failure_count = 0
        while True:
            try:
                needs_permutation = bool(
                    self.triangulation.rebuild(
                        candidate, incremental=incremental
                    )
                )
                break
            except foamsurf_topology.TriangulationFailedError:
                failure_count += 1
                if failure_count > max_attempts:
                    raise RuntimeError(
                        "Voronoi triangulation failed after "
                        f"{max_attempts} retries"
                    )
                perturbation *= 2.0
                incremental = False
                candidate = (
                    points + perturbation * torch.randn_like(points)
                )

        if failure_count > 5:
            medium.sites.copy_(candidate)
        if needs_permutation:
            permutation = self.triangulation.permutation().to(torch.long)
            self._permute_with_optimizer(
                medium, optimizer, permutation
            )

        self.last_permutation = (
            permutation if needs_permutation else None
        )
        self._read_topology()
        self.revision += 1
        return needs_permutation

    @torch.no_grad()
    def synchronize_renderer(self, renderer, medium) -> None:
        renderer.set_voronoi_geometry(
            medium.render_scene(),
            self.adjacency,
            self.offsets,
        )
