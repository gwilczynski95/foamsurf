"""Configuration/math for the initialization stage only (no hybrid defaults)."""
from dataclasses import asdict, dataclass, field, fields
import math


@dataclass
class WarmupProfile:
    # Opt-in initialization dynamics; defaults preserve the first warmup sweep.
    site_initialization: str = "stratified"
    spatial_count: int = 25000
    spatial_colmap_fraction: float = 0.1
    spatial_extent: float = 1.0
    # Share of the spatial sites sampled uniformly in the volume the training cameras
    # see, up to spatial_view_depth x the median COLMAP point distance to a camera.
    spatial_view_fraction: float = 0.0
    spatial_view_depth: float = 2.0
    ray_sampling: str = "single_camera"
    population_policy: str = "foamsurf"
    population_factor: float = 1.15
    population_target: int = 2097152
    population_horizon: int = 11000
    population_min_interval: int = 100
    error_map_downsample: int = 2
    error_map_chunk: int = 65536
    reset_low_contribution: bool = True
    prune_contribution: float = 1e-2
    reset_contribution: float = 1e-3
    prune_radius: float = 0.1
    topology_schedule: str = "fixed"
    density_beta: float = 10.0
    activation_scale: float = 1.0
    initial_sigma: float = math.log(2) / 10
    outer_points: int = 5000
    outer_scale: float = 10.0
    outer_sigma: float = math.log1p(math.exp(-5)) / 10
    outer_positions_trainable: bool = False
    outer_density_trainable: bool = True
    outer_color_trainable: bool = True
    corner_guards: bool = False
    loss: str = "smooth_l1"
    smooth_l1_beta: float = 1.0
    opacity_weight: float = 1.0
    schedule: str = "cosine"
    sites_lr: float = 2e-4
    sites_final: float = 5e-6
    sites_steps: int = 18000
    density_lr: float = 0.1
    density_final: float = 0.01
    density_warmup: int = 2000
    density_steps: int = 20000
    color_lr: float = 0.005
    color_final: float = 0.0005
    color_steps: int = 20000
    adam_eps: float = 1e-15
    resolutions: list[int] = field(default_factory=lambda: [4, 2, 1])
    resolution_steps: list[int] = field(default_factory=lambda: [0, 150, 500])
    quantile_weight: float = 1e-4
    quantile_start: int = 0
    quantile_ramp: int = 10000
    quantile_end: int | None = None
    fixed_quantiles: list[float] | None = None

    @classmethod
    def from_dict(cls, values):
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown warmup profile fields: {sorted(unknown)}")
        result = cls(**values)
        result.validate()
        return result

    def validate(self):
        if self.site_initialization not in ("stratified", "random", "pcd", "spatial"):
            raise ValueError("invalid site initialization")
        if self.site_initialization == "spatial":
            if not isinstance(self.spatial_count, int) or self.spatial_count < 32:
                raise ValueError("spatial_count must be >=32")
            if not math.isfinite(self.spatial_colmap_fraction) or not 0 <= self.spatial_colmap_fraction <= 1:
                raise ValueError("invalid spatial COLMAP fraction")
            if not math.isfinite(self.spatial_extent) or self.spatial_extent <= 0:
                raise ValueError("invalid spatial extent")
            if not math.isfinite(self.spatial_view_fraction) or not 0 <= self.spatial_view_fraction <= 1:
                raise ValueError("invalid spatial view fraction")
            if not math.isfinite(self.spatial_view_depth) or self.spatial_view_depth <= 0:
                raise ValueError("invalid spatial view depth")
            if self.outer_points != 0 or self.corner_guards:
                raise ValueError("spatial sites are ordinary medium sites, not guards")
        if self.site_initialization in ("pcd", "spatial") and (self.corner_guards or self.density_beta != 10 or self.activation_scale != 1):
            raise ValueError("RadFoam PCD initialization requires beta=10, scale=1, no corner guards")
        if self.ray_sampling not in ("single_camera", "mixed"):
            raise ValueError("invalid ray sampling")
        if self.ray_sampling == "mixed" and self.loss != "smooth_l1":
            raise ValueError("mixed rays require smooth_l1")
        if self.population_policy not in ("foamsurf", "radfoam"):
            raise ValueError("invalid population policy")
        if self.topology_schedule not in ("fixed", "radfoam"):
            raise ValueError("invalid topology schedule")
        if not math.isfinite(self.population_factor) or not 1 < self.population_factor <= 2:
            raise ValueError("population_factor must be in (1,2]")
        for name in ("population_target", "population_horizon", "population_min_interval",
                     "error_map_downsample", "error_map_chunk"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("prune_contribution", "reset_contribution", "prune_radius"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.population_policy == "radfoam" and not (
                self.outer_positions_trainable and self.outer_density_trainable
                and self.outer_color_trainable):
            raise ValueError("radfoam population requires trainable outer sites")

        for name in ("density_beta", "activation_scale", "initial_sigma", "outer_sigma",
                     "outer_scale", "smooth_l1_beta", "adam_eps"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.outer_points < 0 or (self.corner_guards and self.outer_points != 8):
            raise ValueError("invalid outer point count")
        if self.loss not in ("smooth_l1", "l1_dssim") or self.schedule not in ("cosine", "constant"):
            raise ValueError("invalid warmup loss/schedule")
        if (not self.resolutions or len(self.resolutions) != len(self.resolution_steps)
                or self.resolution_steps[0] != 0 or min(self.resolutions) < 1
                or any(a >= b for a, b in zip(self.resolution_steps, self.resolution_steps[1:]))):
            raise ValueError("invalid progressive resolution schedule")
        if self.resolutions[-1] != 1:
            raise ValueError("progressive resolution must finish at base resolution")
        if not math.isfinite(self.opacity_weight) or self.opacity_weight < 0:
            raise ValueError("opacity_weight must be finite and nonnegative")
        if self.quantile_weight < 0 or self.quantile_start < 0 or self.quantile_ramp < 0:
            raise ValueError("invalid quantile schedule")
        if self.quantile_end is not None and self.quantile_end <= self.quantile_start:
            raise ValueError("quantile_end must exceed start")
        if self.fixed_quantiles is not None and (
            len(self.fixed_quantiles) != 2 or not 0 < self.fixed_quantiles[1] <= self.fixed_quantiles[0] < 1
        ):
            raise ValueError("fixed quantiles must be descending and in (0,1)")
        for prefix in ("sites", "density", "color"):
            if getattr(self, prefix + "_steps") <= 0:
                raise ValueError("LR horizon must be positive")
            for suffix in ("_lr", "_final"):
                value = getattr(self, prefix + suffix)
                if not math.isfinite(value) or value < 0:
                    raise ValueError("LR must be finite and nonnegative")
        if not 0 <= self.density_warmup < self.density_steps:
            raise ValueError("density warmup must precede horizon")

    def as_dict(self):
        return asdict(self)

    def resolution(self, step):
        return next(r for r, s in reversed(list(zip(self.resolutions, self.resolution_steps))) if step >= s)

    def quantile_strength(self, step):
        if step < self.quantile_start or (self.quantile_end is not None and step >= self.quantile_end):
            return 0.0
        return self.quantile_weight * (min((step - self.quantile_start) / self.quantile_ramp, 1.0)
                                      if self.quantile_ramp else 1.0)

    def rates(self, step):
        return {"medium_" + target: cosine_lr(
            step, getattr(self, source + "_lr"), getattr(self, source + "_final"),
            self.density_warmup if source == "density" else 0,
            getattr(self, source + "_steps"), self.schedule,
        ) for target, source in (("sites", "sites"), ("density", "density"), ("color", "color"))}


def cosine_lr(step, initial, final, warmup, horizon, mode="cosine"):
    if mode == "constant":
        return initial
    if step > horizon:
        return 0.0
    if warmup and step < warmup:
        return initial * step / warmup
    t = min(max((step - warmup) / (horizon - warmup), 0.0), 1.0)
    return final + 0.5 * (initial - final) * (1 + math.cos(math.pi * t))


def inverse_density(sigma, beta=1.0, scale=1.0):
    import torch
    y = sigma / scale
    return y + torch.log(-torch.expm1(-beta * y)) / beta


def opacity_loss(rendered_alpha, target_alpha):
    """RadFoam opacity objective; real-image loaders provide target alpha=1."""
    if rendered_alpha.shape != target_alpha.shape:
        raise ValueError("rendered and target alpha must have equal shape")
    return ((rendered_alpha - target_alpha.detach()) ** 2).mean()


def quantile_loss(depths):
    valid = (depths > 0).all(dim=-1)
    width = (depths[..., 0] - depths[..., 1]).abs()
    return (width * valid).mean(), valid, width
