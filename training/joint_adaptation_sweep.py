"""Controlled post-handoff experiments; the ordinary Trainer is unchanged."""
from copy import deepcopy
from math import ceil, floor

VARIANTS = (
    ("fixed_binary", "fixed", True),
    ("prune", "prune", False),
    ("prune_binary", "prune", True),
    ("adaptive", "adaptive", False),
    ("adaptive_binary", "adaptive", True),
    ("delayed", "delayed", False),
    ("delayed_binary", "delayed", True),
)


def validate_initialization(payload):
    if payload.get("iteration") != 0:
        raise ValueError("Sweep requires the iteration-zero initialization, not a trained checkpoint")
    if payload.get("config", {}).get("representation_mode") != "hybrid":
        raise ValueError("Expected hybrid initialization")
    for key in ("gaussian_optimizer", "medium_optimizer"):
        if payload[key]["state"]:
            raise ValueError("Initialization must have fresh optimizer states")
    return int(payload["model_metadata"]["medium"]["point_count"])


def build_variant(base, index, count, settings, output_dir):
    if not 0 <= index < len(VARIANTS):
        raise ValueError("variant must be 0..6")
    name, policy, binary = VARIANTS[index]
    config = deepcopy(base)
    config.output_dir = str(output_dir)
    config.experiment_name = f"joint-{index:02d}-{name}"
    config.wandb.mode = "online"
    config.wandb.name = config.experiment_name
    config.wandb.tags = list(base.wandb.tags) + ["joint-adaptation", policy, "binary" if binary else "original-opacity"]
    config.model.medium_points = count
    a = config.voronoi_adaptation
    a.enabled = policy != "fixed"
    a.interval = int(settings["interval"])
    a.adaptation_offset = 0
    a.densify_from = a.interval + (int(settings["delay"]) if policy == "delayed" else 0)
    a.densify_until = config.optimization.iterations
    a.densify_fraction = float(settings["split_fraction"]) if policy in ("adaptive", "delayed") else 0.0
    a.max_prune_fraction = float(settings["prune_fraction"]) if a.enabled else 0.0
    a.min_points = max(32, ceil(count * float(settings["minimum_population_ratio"])))
    a.max_points = floor(count * float(settings["maximum_population_ratio"])) if a.densify_fraction else count
    a.min_visibility = 1.0
    a.contribution_threshold = 1e-6
    a.gradient_threshold = 0.0
    a.prune_cell_radius_fraction = 0.02
    a.prune_patience = 2
    a.child_jitter_fraction = 0.025
    a.freeze_at_capacity = False
    if binary:
        d = config.disentanglement
        d.opacity_binarization_weight = float(settings["binarization_weight"])
        d.opacity_binarization_start = int(settings["binarization_start"])
        d.opacity_binarization_ramp = int(settings["binarization_ramp"])
        d.opacity_binarization_ramp_shape = "cosine"
        d.opacity_binarization_end = None
        config.gaussian_adaptation.opacity_prune_until = config.optimization.iterations
    config.validate()
    return config


class PopulationBoundsTopology:
    """Preserve outer sites during the existing triangulation retry fallback.

    Only replace_population sees the expanded box. Scoring, render bounds and
    child proposals retain the original scene box. Restore even on failure.
    """
    def __init__(self, topology):
        self.inner = topology

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def replace_population(self, medium, optimizer, keep_indices, parent_indices, child_sites, **kwargs):
        import torch
        lo, hi = medium.bbox_min, medium.bbox_max
        points = torch.cat((medium.sites.detach(), child_sites.detach()))
        medium.bbox_min = torch.minimum(lo, points.amin(dim=0))
        medium.bbox_max = torch.maximum(hi, points.amax(dim=0))
        try:
            return self.inner.replace_population(medium, optimizer, keep_indices, parent_indices, child_sites, **kwargs)
        finally:
            medium.bbox_min, medium.bbox_max = lo, hi


def install_sweep_controller(trainer):
    from foamsurf_model.voronoi_adaptation import VoronoiAdaptationController

    class WindowedController(VoronoiAdaptationController):
        def should_collect(self, iteration):
            return iteration > self.config.densify_from - self.config.interval and super().should_collect(iteration)

    previous = trainer.voronoi_adaptation
    controller = WindowedController(trainer.model.medium, trainer.medium_optimizer,
                                    PopulationBoundsTopology(trainer.topology), trainer.config.voronoi_adaptation)
    controller.load_state_dict(previous.state_dict())
    trainer.voronoi_adaptation = controller
