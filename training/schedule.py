"""A shorter (or longer) joint training with every iteration-based schedule scaled by the same factor:
learning-rate decays, the resolution schedule, the OceanSplat window, densification and pruning,
SH degree steps and the start / ramp of every regulariser. Logging, evaluation and checkpoint cadence and
the renderer's topology updates are not schedules and stay. Applied to a resolved IUI3 manifest after
all overrides (so it scales the values the suffixes set), from the override ``schedule.iterations=N``.
"""
from training.config import ExperimentConfig
from training.losses import CapacityLossConfig, MediumFixConfig, SmoothConfig, SurfaceConfig

JOINT = {
    "optimization": ("iterations", "training_downscale_interval", "gaussian_means_lr_delay_steps",
                     "gaussian_means_lr_max_steps", "gaussian_freeze_until", "gaussian_sh_degree_interval",
                     "medium_lr_decay_from", "medium_lr_decay_until", "trinocular_from", "trinocular_until",
                     "ocean_from", "ocean_until", "ocean_alpha_until"),
    "disentanglement": ("opacity_binarization_start", "opacity_binarization_ramp", "opacity_binarization_end",
                        "opacity_binarization_end_ramp"),
    "gaussian_adaptation": ("densify_from", "densify_until", "densification_interval", "min_opacity_late_from",
                            "opacity_prune_from", "opacity_prune_ramp", "opacity_prune_until",
                            "opacity_prune_interval", "size_prune_from", "opacity_reset_interval",
                            "medium_freeze_after_reset", "medium_freeze_interval", "medium_freeze_from",
                            "medium_freeze_steps"),
    "voronoi_adaptation": ("densify_from", "densify_until", "interval"),
}
SECTIONS = {
    "rgb_smoothing": (SmoothConfig, ("ramp_steps",)),
    "rgb_surface": (SurfaceConfig, ("start", "ramp_steps")),
    "capacity_loss": (CapacityLossConfig, ("repulsion_start", "repulsion_ramp", "direct_start", "direct_ramp")),
    "medium_fix": (MediumFixConfig, ("chroma_start", "chroma_ramp", "surface_coverage_start", "surface_coverage_ramp",
                                     "surface_coverage_end", "water_clear_start", "water_clear_ramp",
                                     "under_surface_start", "under_surface_ramp", "camera_water_start",
                                     "camera_water_ramp", "water_anchor_start", "water_anchor_ramp")),
}
# Intervals stay >= 1 (0 means 'off' and stays 0).
INTERVALS = {"training_downscale_interval", "gaussian_sh_degree_interval", "densification_interval",
             "opacity_prune_interval", "opacity_reset_interval", "medium_freeze_interval", "interval"}


def pop_schedule(overrides):
    """(iterations or None, the other overrides) from a list of ``key=value`` overrides."""
    iterations, rest = None, []
    for item in overrides or ():
        key, _, value = item.partition("=")
        if key == "schedule.iterations":
            iterations = int(value)
            if iterations < 1:
                raise ValueError("schedule.iterations must be >= 1")
        else:
            rest.append(item)
    return iterations, rest


def _scale(value, factor, interval):
    if value is None or value == 0:
        return value
    scaled = int(round(value * factor))
    return max(scaled, 1) if interval else scaled


def scale_schedule(manifest, iterations):
    """Scales the joint stage's schedules in place so training ends at ``iterations``."""
    defaults = ExperimentConfig().as_dict()
    joint = manifest["joint"]
    current = joint["optimization"].get("iterations", defaults["optimization"]["iterations"])
    factor = iterations / current
    for section, keys in JOINT.items():
        target = joint.setdefault(section, {})
        for key in keys:
            value = target.get(key, defaults[section][key])
            target[key] = _scale(value, factor, key in INTERVALS)
    for section, (config_class, keys) in SECTIONS.items():
        target = manifest.setdefault(section, {})
        base = config_class().__dict__
        for key in keys:
            target[key] = _scale(target.get(key, base[key]), factor, False)
    joint["optimization"]["iterations"] = iterations
    return manifest
