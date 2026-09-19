"""RadFoam-inspired population dynamics for initialization ONLY.

The hybrid controller is intentionally not edited or registered globally.
"""
import math
import torch
from foamsurf_model.voronoi_adaptation import VoronoiAdaptationController, VoronoiTopologyPlan
from .warmup_population_math import population_masks, sample_parents, offspring_sites, growth_interval
from .warmup_profile import inverse_density


class WarmupPopulation(VoronoiAdaptationController):
    @staticmethod
    def _empty_summary():
        return {**VoronoiAdaptationController._empty_summary(), "density_reset": 0}

    def __init__(self, model, optimizer, topology, config, profile, initial_count):
        super().__init__(model, optimizer, topology, config)
        self.profile = profile
        self.initial_count = initial_count
        self.next_iteration = config.densify_from
        self.event_iteration = config.densify_from
        if profile.population_target <= initial_count or profile.population_horizon <= config.densify_from:
            raise ValueError("organic growth requires a larger target and later horizon")
        self.reset_mask = None
        self.total_density_reset = 0

    def should_collect(self, iteration):
        return False  # Fresh multi-camera map collected immediately before each event.

    def should_adapt(self, iteration):
        self.event_iteration = iteration
        return (self.config.enabled and iteration >= self.next_iteration
                and self.config.densify_from <= iteration < self.config.densify_until
                and self.model.point_count < .9 * self.profile.population_target)

    @torch.no_grad()
    def set_error_map(self, error, contribution):
        if error.shape != (self.model.point_count,) or contribution.shape != error.shape:
            raise ValueError("error map population mismatch")
        if not bool(torch.isfinite(error).all() and torch.isfinite(contribution).all()):
            raise FloatingPointError("non-finite error map")
        self.gradient.copy_(error)
        self.contribution.copy_(contribution)

    @torch.no_grad()
    def propose_plan(self):
        p = self.profile
        radii, farthest = self._cell_geometry()
        prune = population_masks(self.contribution, radii, self.topology.adjacency,
                                 self.topology.offsets, p.prune_contribution, p.prune_radius)
        ids = torch.arange(self.model.point_count, device=radii.device)
        # Keep at least four non-coplanar candidates in practice: defer pathological
        # all-empty proposals, rather than deleting the tessellation.
        if int((~prune).sum()) < 4:
            prune.zero_()
        keep = ids[~prune]
        requested = min(int((p.population_factor - 1) * len(ids)),
                        max(0, p.population_target - len(keep)))
        parents = sample_parents(self.gradient, radii, requested)
        children = offspring_sites(self.model.sites, farthest, parents)
        # Raw=-1 in beta10 is a physical sigma; honor it even in beta ablations.
        self.reset_mask = (self.contribution < p.reset_contribution) & p.reset_low_contribution
        self.last_statistics = self._empty_summary()
        self.last_statistics.update(
            densified=len(parents), pruned=int(prune.sum()),
            score_p95=float(torch.quantile(self.gradient * radii, .95)),
            cell_radius_p95=float(torch.quantile(radii, .95)),
            active_fraction=float((self.contribution > p.prune_contribution).float().mean()),
            observed_fraction=float((self.contribution > 0).float().mean()),
            density_reset=0)
        return VoronoiTopologyPlan(keep, parents, children, len(ids))

    @torch.no_grad()
    def apply_plan(self, plan):
        if plan.original_count != self.model.point_count or self.reset_mask is None:
            raise RuntimeError("missing/stale organic population plan")
        original_density = self.model.raw_density.detach().clone()
        reset_count = int(self.reset_mask.sum())
        sigma = math.log1p(math.exp(-10)) / 10
        reset_raw = inverse_density(self.model.raw_density.new_tensor(sigma),
                                    self.profile.density_beta, self.profile.activation_scale)
        self.model.raw_density[self.reset_mask] = reset_raw
        try:
            result = super().apply_plan(plan)
        except Exception:
            # Shared replacement is transactional until population commit.
            if self.model.raw_density.shape == original_density.shape:
                self.model.raw_density.copy_(original_density)
            raise
        if result["skipped"]:
            self.model.raw_density.copy_(original_density)
            reset_count = 0
        self.total_density_reset += reset_count
        self.last_statistics["density_reset"] = reset_count
        self.next_iteration = self.event_iteration + growth_interval(
            self.model.point_count, self.initial_count, self.profile.population_target,
            self.profile.population_factor, self.config.densify_from,
            self.profile.population_horizon, self.profile.population_min_interval)
        self.reset_mask = None
        return result

    def state_dict(self):
        return {**super().state_dict(), "organic_next_iteration": self.next_iteration,
                "organic_total_density_reset": self.total_density_reset}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.next_iteration = int(state["organic_next_iteration"])
        self.total_density_reset = int(state["organic_total_density_reset"])
