from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


# Removed features (RadFoam prior, medium handoff prefix, scalar medium
# regularizers, flare_movement). Configs and manifests written before the
# removal still carry them; they are dropped when inactive.
_LEGACY_SECTIONS = {
    "radfoam_prior": lambda section: not section.get("enabled", False),
    "medium_handoff": lambda section: (
        section.get("prefix_mode", "none") == "none"
        and not section.get("prefix_iterations", 0)
        and section.get("gaussian_freeze_start") is None
    ),
}
_LEGACY_ADAPTATION = {"min_opacity_densify": None, "opacity_reset_prune_grace": 0}
_LEGACY_DISENTANGLEMENT_PREFIXES = (
    "opaque_if_used_", "graph_tv_", "density_cap_", "optical_thickness_",
)


def drop_legacy_keys(raw: dict[str, Any]) -> dict[str, Any]:
    """Return a stage config dict without inactive keys of removed features."""

    raw = dict(raw)
    for name, inactive in _LEGACY_SECTIONS.items():
        if name in raw:
            if not inactive(raw[name] or {}):
                raise ValueError(f"{name} is no longer supported")
            del raw[name]
    disentanglement = raw.get("disentanglement")
    if disentanglement is not None:
        disentanglement = dict(disentanglement)
        for key in list(disentanglement):
            if key.startswith(_LEGACY_DISENTANGLEMENT_PREFIXES) or key == "epsilon":
                if key.endswith("_weight") and disentanglement[key] != 0:
                    raise ValueError(f"disentanglement.{key} is no longer supported")
                del disentanglement[key]
        raw["disentanglement"] = disentanglement
    adaptation = raw.get("gaussian_adaptation")
    if adaptation is not None:
        if adaptation.get("policy") == "flare_movement":
            raise ValueError("gaussian_adaptation.policy flare_movement is no longer supported")
        adaptation = dict(adaptation)
        for key, inactive in _LEGACY_ADAPTATION.items():
            if key in adaptation:
                if adaptation[key] != inactive:
                    raise ValueError(f"gaussian_adaptation.{key} is no longer supported")
                del adaptation[key]
        adaptation.pop("movement_threshold", None)
        raw["gaussian_adaptation"] = adaptation
    return raw


def _construct(cls, values: dict[str, Any] | None):
    values = {} if values is None else dict(values)
    valid = {item.name for item in fields(cls)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(
            f"unknown {cls.__name__} fields: {sorted(unknown)}"
        )
    return cls(**values)


@dataclass
class DatasetConfig:
    source_path: str = ""
    resolution: int = 1
    llffhold: int = 8
    gaussian_ground_truth_path: str | None = None
    medium_ground_truth_path: str | None = None
    train_target: str = "both"


@dataclass
class ModelConfig:
    medium_type: str = "scalar"
    appearance: str = "rgb"
    gaussian_rgb_activation: str = "identity"
    sh_degree: int = 3
    gaussian_initial_points: int = 100_000
    gaussian_initial_opacity: float = 0.1
    gaussian_initial_kappa: float = 1.01
    gaussian_initial_scale_neighbors: int = 1
    medium_points: int = 2_048
    medium_guard_points: int = 8
    medium_guard_scale: float = 1.25
    medium_bbox_margin: float = 0.10
    medium_bbox_min: list[float] | None = None
    medium_bbox_max: list[float] | None = None
    medium_initial_density: float = 0.01
    medium_initial_color: list[float] = field(
        default_factory=lambda: [0.5, 0.5, 0.5]
    )
    # Joint stage: every cell starts with the mean colour of the training photos instead of the
    # warmup's colour, so open water starts close to the water (not a grey wall).
    medium_initial_color_from_data: bool = False


@dataclass
class RendererConfig:
    near: float = 0.1
    far: float = 6.0
    training_transmittance_threshold: float = 1.0e-4
    validation_transmittance_threshold: float = 1.0e-3
    max_intersections: int = 1_024
    number_of_sides: int = 8
    gaussian_radius_squared: float = 11.3449
    gaussian_hit_buffer_size: int = 64


@dataclass
class OptimizationConfig:
    iterations: int = 35_000
    rays_per_batch: int = 65_536
    full_image_batch: bool = False
    # Coarse-to-fine (WaterSplatting: 2 levels every 3000 steps = 1/4, 1/2, then full
    # resolution from 6000): downscale 2^max(levels - iteration // interval, 0).
    training_downscale_levels: int = 0
    training_downscale_interval: int = 3000
    cameras_without_replacement: bool = False
    reconstruction_loss: str = "mse"
    dssim_weight: float = 0.2
    gaussian_appearance_lr: float = 0.0025
    gaussian_appearance_rest_lr_scale: float = 0.05
    gaussian_appearance_lr_decay: float = 0.0
    gaussian_appearance_lr_final: float | None = None
    gaussian_opacity_lr: float = 0.025
    gaussian_opacity_lr_decay: float = 0.0
    gaussian_opacity_lr_final: float | None = None
    gaussian_kappa_lr: float = 0.10
    gaussian_kappa_lr_decay: float = 0.0
    gaussian_kappa_lr_final: float | None = None
    gaussian_means_lr: float = 0.00016
    gaussian_means_lr_decay: float | None = None
    gaussian_means_lr_final: float | None = 0.0000016
    gaussian_means_lr_delay_steps: int = 0
    gaussian_means_lr_delay_mult: float = 0.01
    gaussian_means_lr_max_steps: int = 30_000
    gaussian_means_spatial_scale: bool = True
    gaussian_scales_lr: float = 0.005
    gaussian_scales_lr_decay: float = 0.0
    gaussian_scales_lr_final: float | None = None
    gaussian_quaternions_lr: float = 0.001
    gaussian_quaternions_lr_decay: float = 0.0
    gaussian_quaternions_lr_final: float | None = None
    gaussian_adam_eps: float = 1.0e-15
    gaussian_freeze_until: int = 0
    gaussian_sh_degree_interval: int = 1_000
    gaussian_scale_regularization: float = 0.0
    gaussian_min_scale: float | None = None
    gaussian_max_scale_fraction: float | None = None
    # Cap every Gaussian scale at this fraction of the scene extent (1.1 x camera radius, as 3DGS)
    # from iteration 0, initial scales included. Replaces gaussian_max_scale_fraction, whose
    # reference (the point cloud, outliers included) can be half the scene. None: off.
    gaussian_max_scale_scene_fraction: float | None = None
    medium_sites_lr: float = 1.0e-4
    medium_density_lr: float = 1.0e-2
    medium_color_lr: float = 5.0e-3
    medium_sites_lr_final: float | None = None
    medium_density_lr_final: float | None = None
    medium_color_lr_final: float | None = None
    medium_lr_decay_from: int = 0
    medium_lr_decay_until: int = 30_000
    topology_update_interval: int = 25
    topology_incremental: bool = True


@dataclass
class ValidationConfig:
    eval_interval: int = 1_000
    log_interval: int = 10
    checkpoint_interval: int = 1_000
    max_eval_views: int = -1
    image_views: int = 4


@dataclass
class DiagnosticsConfig:
    enabled: bool = False
    save_per_primitive: bool = True
    contribution_epsilon: float = 1.0e-6
    layer_alpha_threshold: float = 0.01


@dataclass
class DisentanglementConfig:
    """Scheduled per-Gaussian opacity binarization."""

    opacity_binarization_weight: float = 0.0
    opacity_binarization_start: int = 0
    opacity_binarization_ramp: int = 0
    opacity_binarization_ramp_shape: str = "linear"
    opacity_binarization_end: int | None = None
    opacity_binarization_end_ramp: int = 0
    opacity_binarization_end_ramp_shape: str = "linear"
    opacity_binarization_maintenance_scale: float = 1.0


@dataclass
class VoronoiWarmupConfig:
    """Native full-scene Voronoi pretraining before Gaussian spawning."""

    enabled: bool = False
    milestone_iterations: list[int] = field(default_factory=list)


@dataclass
class GaussianAdaptationConfig:
    enabled: bool = False
    # graphdeco: screen-gradient clone/split/prune (3DGS/current FlaRe)
    # fixed: fixed topology with optional opacity reset
    policy: str = "graphdeco"
    max_points: int | None = 200_000
    densify_from: int = 500
    densify_until: int = 15_000
    densification_interval: int = 100
    gradient_threshold: float = 2.0e-4
    min_opacity: float = 0.005
    min_opacity_initial: float | None = None
    # A second, usually lower threshold from min_opacity_late_from on (WaterSplatting:
    # 0.5 while densifying, 0.1 from 10000). None: min_opacity throughout.
    min_opacity_late: float | None = None
    min_opacity_late_from: int = 10000
    opacity_prune_from: int = 500
    opacity_prune_ramp: int = 0
    opacity_prune_ramp_shape: str = "linear"
    opacity_prune_until: int | None = None
    opacity_prune_interval: int | None = None
    min_points: int = 1
    max_opacity_prune_fraction: float = 1.0
    percent_dense: float = 0.01
    size_prune_from: int = 3_000
    # The pinned official 2DGS implementation resets max_radii2D inside
    # densification_postfix before its final prune, so its advertised
    # screen-space test is effectively inactive. Keep this explicit rather
    # than reproducing that ordering side effect.
    screen_pruning_enabled: bool = False
    max_screen_size: float = 20.0
    max_world_scale_fraction: float = 0.10
    # Gaussians whose largest scale exceeds far_prune_scale and whose center is
    # farther than far_prune_distance from every training camera (both in world
    # units) are pruned at every opacity-prune or size-prune event; they fill
    # open water. None disables.
    far_prune_distance: float | None = None
    far_prune_scale: float = 0.05
    # Divide each Gaussian's densification gradient by the medium transmittance
    # to the camera, T = mean_c exp(-a_c d) (a: median attenuation of the cells,
    # d: distance), capped at gradient_compensation_max: Gaussians far behind
    # water otherwise receive gradients attenuated by it and are never densified.
    gradient_attenuation_compensation: bool = False
    # gradient_threshold holds for images this wide; other widths use
    # threshold * reference / width (larger images densify more). None: as given.
    gradient_threshold_reference_width: int | None = None
    gradient_compensation_max: float = 20.0
    # AbsGS / WaterSplatting: accumulate sum over pixels of |dL/dscreen| per axis
    # instead of |sum over pixels|, so opposite pixel gradients do not cancel.
    # Larger than the plain statistic: raise gradient_threshold (WS: 4x).
    absolute_gradient: bool = False
    # Budgeted densification: the cap grows linearly from the initial count at densify_from
    # to max_points at this iteration, so the budget is spread over the window instead of
    # being used up in the first few thousand iterations. None: max_points from the start.
    max_points_ramp_until: int | None = None
    opacity_reset_interval: int = 0
    opacity_reset_value: float = 0.01
    opacity_reset_at_densify_from: bool = False
    opacity_reset_optimizer_state: bool = True
    medium_freeze_after_reset: int = 0
    medium_freeze_interval: int = 0
    medium_freeze_from: int = 0
    medium_freeze_steps: int = 0
    preserve_clone_opacity: bool = False
    preserve_split_opacity: bool = False
    runaway_max_points: int | None = None
    min_effective_scale: float = 1.0e-5


@dataclass
class VoronoiAdaptationConfig:
    """Adaptive medium population without RadFoam solid-surface regularization."""

    enabled: bool = False
    densify_from: int = 1_000
    densify_until: int = 20_000
    interval: int = 500
    adaptation_offset: int = 0
    densify_fraction: float = 0.10
    max_points: int | None = 65_536
    min_points: int = 1_024
    min_visibility: float = 1.0
    contribution_threshold: float = 1.0e-6
    gradient_threshold: float = 0.0
    prune_cell_radius_fraction: float = 0.02
    max_prune_fraction: float = 0.10
    prune_patience: int = 2
    child_jitter_fraction: float = 0.05
    freeze_at_capacity: bool = False


@dataclass
class WandbConfig:
    mode: str = "offline"
    project: str = "foamsurf"
    entity: str | None = None
    name: str | None = None
    tags: list[str] = field(default_factory=lambda: ["B0"])


@dataclass
class ExperimentConfig:
    experiment_name: str = "b0"
    output_dir: str = "outputs/b0"
    seed: int = 42
    device: str = "cuda"
    representation_mode: str = "hybrid"
    background: list[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0]
    )
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    renderer: RendererConfig = field(default_factory=RendererConfig)
    optimization: OptimizationConfig = field(
        default_factory=OptimizationConfig
    )
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    disentanglement: DisentanglementConfig = field(
        default_factory=DisentanglementConfig
    )
    voronoi_warmup: VoronoiWarmupConfig = field(
        default_factory=VoronoiWarmupConfig
    )
    gaussian_adaptation: GaussianAdaptationConfig = field(
        default_factory=GaussianAdaptationConfig
    )
    voronoi_adaptation: VoronoiAdaptationConfig = field(
        default_factory=VoronoiAdaptationConfig
    )
    wandb: WandbConfig = field(default_factory=WandbConfig)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        raw = drop_legacy_keys(yaml.safe_load(path.read_text()) or {})
        valid = {
            "experiment_name",
            "output_dir",
            "seed",
            "device",
            "representation_mode",
            "background",
            "dataset",
            "model",
            "renderer",
            "optimization",
            "validation",
            "diagnostics",
            "disentanglement",
            "voronoi_warmup",
            "gaussian_adaptation",
            "voronoi_adaptation",
            "wandb",
        }
        unknown = set(raw) - valid
        if unknown:
            raise ValueError(
                f"unknown ExperimentConfig fields: {sorted(unknown)}"
            )
        config = cls(
            experiment_name=raw.get("experiment_name", "b0"),
            output_dir=raw.get("output_dir", "outputs/b0"),
            seed=int(raw.get("seed", 42)),
            device=raw.get("device", "cuda"),
            representation_mode=raw.get("representation_mode", "hybrid"),
            background=list(raw.get("background", [1.0, 1.0, 1.0])),
            dataset=_construct(DatasetConfig, raw.get("dataset")),
            model=_construct(ModelConfig, raw.get("model")),
            renderer=_construct(RendererConfig, raw.get("renderer")),
            optimization=_construct(
                OptimizationConfig, raw.get("optimization")
            ),
            validation=_construct(
                ValidationConfig, raw.get("validation")
            ),
            diagnostics=_construct(
                DiagnosticsConfig, raw.get("diagnostics")
            ),
            disentanglement=_construct(
                DisentanglementConfig, raw.get("disentanglement")
            ),
            voronoi_warmup=_construct(
                VoronoiWarmupConfig, raw.get("voronoi_warmup")
            ),
            gaussian_adaptation=_construct(
                GaussianAdaptationConfig, raw.get("gaussian_adaptation")
            ),
            voronoi_adaptation=_construct(
                VoronoiAdaptationConfig, raw.get("voronoi_adaptation")
            ),
            wandb=_construct(WandbConfig, raw.get("wandb")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.model.medium_type not in ("scalar", "rgb_backscatter"):
            raise ValueError("Unknown medium_type")
        if not self.dataset.source_path:
            raise ValueError("dataset.source_path is required")
        if self.representation_mode not in {"hybrid", "gaussian_only", "voronoi_only"}:
            raise ValueError("invalid representation_mode")
        if self.dataset.train_target not in {"both", "lego", "clouds"}:
            raise ValueError("dataset.train_target must be both, lego, or clouds")
        warmup = self.voronoi_warmup
        if warmup.enabled:
            if self.representation_mode != "voronoi_only":
                raise ValueError(
                    "Voronoi warmup requires representation_mode=voronoi_only"
                )
            if self.dataset.train_target != "both":
                raise ValueError(
                    "Voronoi warmup requires dataset.train_target=both"
                )
        else:
            expected_target = {
                "hybrid": "both",
                "gaussian_only": "lego",
                "voronoi_only": "clouds",
            }[self.representation_mode]
            # Gaussians alone may also fit the whole photo (real scenes have no layer ground truth).
            whole_photo = self.representation_mode == "gaussian_only" and self.dataset.train_target == "both"
            if self.dataset.train_target != expected_target and not whole_photo:
                raise ValueError(
                    f"{self.representation_mode} requires "
                    f"dataset.train_target={expected_target}"
                )
        layer_paths = (
            self.dataset.gaussian_ground_truth_path,
            self.dataset.medium_ground_truth_path,
        )
        if (layer_paths[0] is None) != (layer_paths[1] is None):
            raise ValueError(
                "dataset Gaussian and medium ground-truth paths must be set together"
            )
        if self.dataset.train_target != "both" and layer_paths[0] is None:
            raise ValueError("standalone training requires both layer ground-truth paths")
        if warmup.enabled:
            # Separated layers are optional diagnostics, never initialization inputs.
            milestones = warmup.milestone_iterations
            if not milestones or milestones[0] != 0:
                raise ValueError(
                    "voronoi_warmup.milestone_iterations must start at 0"
                )
            if milestones != sorted(set(milestones)):
                raise ValueError(
                    "voronoi_warmup.milestone_iterations must be sorted and unique"
                )
            if milestones[-1] > self.optimization.iterations:
                raise ValueError(
                    "Voronoi warmup milestone exceeds training iterations"
                )
            if self.model.gaussian_initial_points != 1:
                raise ValueError(
                    "Voronoi warmup requires exactly one inactive Gaussian placeholder"
                )
            if self.model.medium_guard_points != 0:
                raise ValueError("Voronoi warmup does not use guard sites")
            if not self.voronoi_adaptation.enabled:
                raise ValueError(
                    "Voronoi warmup requires adaptive Voronoi population"
                )
            if self.gaussian_adaptation.enabled:
                raise ValueError(
                    "Voronoi warmup cannot enable Gaussian adaptation"
                )
        if self.dataset.resolution < 1:
            raise ValueError("dataset.resolution must be positive")
        if self.model.appearance not in {"rgb", "sh"}:
            raise ValueError("model.appearance must be rgb or sh")
        if self.model.gaussian_rgb_activation not in {"identity", "sigmoid"}:
            raise ValueError("model.gaussian_rgb_activation must be identity or sigmoid")
        if not 0 <= self.model.sh_degree <= 3:
            raise ValueError("model.sh_degree must be between 0 and 3")
        if self.model.gaussian_initial_points < 1:
            raise ValueError("gaussian_initial_points must be positive")
        if self.model.gaussian_initial_scale_neighbors < 1:
            raise ValueError("gaussian_initial_scale_neighbors must be positive")
        if not 0.0 < self.model.gaussian_initial_opacity < 1.0:
            raise ValueError("gaussian_initial_opacity must be in (0, 1)")
        if self.model.medium_points < 32:
            raise ValueError("medium_points must be at least 32")
        if not 0 <= self.model.medium_guard_points < self.model.medium_points:
            raise ValueError(
                "medium_guard_points must be in [0, medium_points)"
            )
        if self.model.medium_guard_points not in {0, 8}:
            raise ValueError("B0 supports either 0 or 8 guard points")
        if self.model.medium_initial_density <= 0:
            raise ValueError("medium_initial_density must be positive")
        fraction = self.optimization.gaussian_max_scale_scene_fraction
        if fraction is not None and not 0.0 < fraction <= 1.0:
            raise ValueError("gaussian_max_scale_scene_fraction must be in (0, 1]")
        if len(self.model.medium_initial_color) != 3:
            raise ValueError("medium_initial_color must contain RGB")
        manual_bounds = (
            self.model.medium_bbox_min,
            self.model.medium_bbox_max,
        )
        if (manual_bounds[0] is None) != (manual_bounds[1] is None):
            raise ValueError(
                "medium_bbox_min and medium_bbox_max must be set together"
            )
        if manual_bounds[0] is not None:
            if len(manual_bounds[0]) != 3 or len(manual_bounds[1]) != 3:
                raise ValueError("manual medium bounds must contain XYZ")
            if any(
                upper <= lower
                for lower, upper in zip(
                    manual_bounds[0], manual_bounds[1]
                )
            ):
                raise ValueError("medium_bbox_max must exceed medium_bbox_min")
        if self.renderer.far <= self.renderer.near:
            raise ValueError("renderer.far must exceed renderer.near")
        if self.renderer.gaussian_hit_buffer_size not in {16, 32, 64, 128, 256}:
            raise ValueError(
                "renderer.gaussian_hit_buffer_size must be one of "
                "16, 32, 64, 128, 256"
            )
        if self.optimization.iterations < 1:
            raise ValueError("optimization.iterations must be positive")
        if self.optimization.rays_per_batch < 1:
            raise ValueError("rays_per_batch must be positive")
        if self.optimization.reconstruction_loss not in {"mse", "l1", "l1_dssim",
                "reg_l1_dssim", "l1_reg_dssim", "reg_l2_dssim",
                "reg_l1_reg_dssim", "reg_l2_reg_dssim"}:
            raise ValueError(
                "unknown optimization.reconstruction_loss"
            )
        if not 0.0 <= self.optimization.dssim_weight <= 1.0:
            raise ValueError("optimization.dssim_weight must be in [0, 1]")
        if self.optimization.gaussian_means_lr_final is not None and (
            self.optimization.gaussian_means_lr_final <= 0
        ):
            raise ValueError("gaussian_means_lr_final must be positive")
        for name in (
            "gaussian_appearance_lr_final",
            "gaussian_opacity_lr_final",
            "gaussian_kappa_lr_final",
            "gaussian_scales_lr_final",
            "gaussian_quaternions_lr_final",
        ):
            value = getattr(self.optimization, name)
            if value is not None and value <= 0:
                raise ValueError(f"optimization.{name} must be positive")
        if self.optimization.gaussian_means_lr_delay_steps < 0:
            raise ValueError(
                "gaussian_means_lr_delay_steps must be non-negative"
            )
        if self.optimization.gaussian_means_lr_max_steps < 1:
            raise ValueError("gaussian_means_lr_max_steps must be positive")
        if self.optimization.gaussian_sh_degree_interval < 0:
            raise ValueError("gaussian_sh_degree_interval must be non-negative")
        if self.optimization.gaussian_freeze_until < 0:
            raise ValueError("gaussian_freeze_until must be non-negative")
        levels, interval = self.optimization.training_downscale_levels, self.optimization.training_downscale_interval
        if type(levels) is not int or levels < 0 or type(interval) is not int or interval < 1:
            raise ValueError("training_downscale_levels must be >= 0 and training_downscale_interval >= 1")
        if levels and not self.optimization.full_image_batch:
            raise ValueError("training_downscale_levels requires full_image_batch")
        if self.optimization.gaussian_adam_eps <= 0:
            raise ValueError("gaussian_adam_eps must be positive")
        if self.optimization.gaussian_scale_regularization < 0:
            raise ValueError("gaussian_scale_regularization must be non-negative")
        for name in (
            "medium_sites_lr",
            "medium_density_lr",
            "medium_color_lr",
        ):
            if getattr(self.optimization, name) <= 0:
                raise ValueError(f"optimization.{name} must be positive")
        medium_finals = (
            "medium_sites_lr_final",
            "medium_density_lr_final",
            "medium_color_lr_final",
        )
        for name in medium_finals:
            value = getattr(self.optimization, name)
            if value is not None and value <= 0:
                raise ValueError(f"optimization.{name} must be positive")
        if self.optimization.medium_lr_decay_from < 0:
            raise ValueError("medium_lr_decay_from must be non-negative")
        if (
            any(getattr(self.optimization, name) is not None for name in medium_finals)
            and self.optimization.medium_lr_decay_until
            <= self.optimization.medium_lr_decay_from
        ):
            raise ValueError(
                "medium_lr_decay_until must exceed medium_lr_decay_from "
                "when a medium LR schedule is enabled"
            )
        if (
            self.optimization.gaussian_min_scale is not None
            and self.optimization.gaussian_min_scale <= 0
        ):
            raise ValueError("gaussian_min_scale must be positive")
        if (
            self.optimization.gaussian_max_scale_fraction is not None
            and self.optimization.gaussian_max_scale_fraction <= 0
        ):
            raise ValueError("gaussian_max_scale_fraction must be positive")
        if not 0 < self.optimization.gaussian_appearance_rest_lr_scale <= 1:
            raise ValueError("invalid Gaussian SH-rest LR scale")
        if self.optimization.topology_update_interval < 1:
            raise ValueError("topology_update_interval must be positive")
        for name in (
            "eval_interval",
            "log_interval",
            "checkpoint_interval",
        ):
            if getattr(self.validation, name) < 1:
                raise ValueError(f"validation.{name} must be positive")
        if not 0.0 <= self.diagnostics.layer_alpha_threshold <= 1.0:
            raise ValueError(
                "diagnostics.layer_alpha_threshold must be in [0, 1]"
            )
        if self.diagnostics.contribution_epsilon < 0:
            raise ValueError(
                "diagnostics.contribution_epsilon must be non-negative"
            )
        disentanglement = self.disentanglement
        for name in ("opacity_binarization",):
            if getattr(disentanglement, f"{name}_weight") < 0.0:
                raise ValueError(
                    f"disentanglement.{name}_weight must be non-negative"
                )
            for suffix in ("start", "ramp"):
                if getattr(disentanglement, f"{name}_{suffix}") < 0:
                    raise ValueError(
                        f"disentanglement.{name}_{suffix} must be non-negative"
                    )
            if getattr(disentanglement, f"{name}_ramp_shape") not in {
                "linear", "cosine", "quadratic"
            }:
                raise ValueError(
                    f"invalid disentanglement.{name}_ramp_shape"
                )
        if disentanglement.opacity_binarization_end_ramp < 0:
            raise ValueError(
                "opacity_binarization_end_ramp must be non-negative"
            )
        if disentanglement.opacity_binarization_end_ramp_shape not in {
            "linear", "cosine", "quadratic"
        }:
            raise ValueError(
                "invalid opacity_binarization_end_ramp_shape"
            )
        if not 0.0 <= (
            disentanglement.opacity_binarization_maintenance_scale
        ) <= 1.0:
            raise ValueError(
                "opacity_binarization_maintenance_scale must be in [0, 1]"
            )
        opacity_end = disentanglement.opacity_binarization_end
        if opacity_end is not None:
            if opacity_end < (
                disentanglement.opacity_binarization_start
                + disentanglement.opacity_binarization_ramp
            ):
                raise ValueError(
                    "opacity_binarization_end must not overlap its ramp-up"
                )
            if (
                opacity_end
                + disentanglement.opacity_binarization_end_ramp
                > self.optimization.iterations
            ):
                raise ValueError(
                    "opacity_binarization end schedule exceeds training "
                    "iterations"
                )
        if (
            disentanglement.opacity_binarization_weight > 0.0
            and self.representation_mode == "voronoi_only"
        ):
            raise ValueError(
                "Gaussian disentanglement losses require Gaussians"
            )
        adaptation = self.gaussian_adaptation
        if adaptation.policy not in {"graphdeco", "fixed"}:
            raise ValueError("gaussian_adaptation.policy must be graphdeco or fixed")
        if adaptation.enabled and self.representation_mode == "voronoi_only":
            raise ValueError("Gaussian adaptation cannot run in voronoi_only mode")
        if adaptation.max_points is not None and adaptation.max_points < 1:
            raise ValueError("gaussian_adaptation.max_points must be positive")
        if (
            adaptation.enabled
            and adaptation.max_points is not None
            and self.model.gaussian_initial_points > adaptation.max_points
        ):
            raise ValueError("initial Gaussian count exceeds adaptation budget")
        if (
            adaptation.densify_from < 1
            or adaptation.densify_until < adaptation.densify_from
        ):
            raise ValueError("invalid Gaussian adaptation iteration range")
        if adaptation.densification_interval < 1:
            raise ValueError(
                "gaussian_adaptation.densification_interval must be positive"
            )
        if adaptation.size_prune_from < 0:
            raise ValueError(
                "gaussian_adaptation.size_prune_from must be non-negative"
            )
        late = adaptation.min_opacity_late
        if late is not None and not 0.0 <= late < 1.0:
            raise ValueError("gaussian_adaptation.min_opacity_late must be in [0, 1)")
        if adaptation.opacity_prune_from < 0:
            raise ValueError(
                "gaussian_adaptation.opacity_prune_from must be non-negative"
            )
        if adaptation.opacity_prune_ramp < 0:
            raise ValueError(
                "gaussian_adaptation.opacity_prune_ramp must be non-negative"
            )
        if adaptation.opacity_prune_ramp_shape not in {
            "linear", "cosine", "quadratic"
        }:
            raise ValueError(
                "invalid gaussian_adaptation.opacity_prune_ramp_shape"
            )
        prune_until = adaptation.opacity_prune_until
        if prune_until is None:
            prune_until = adaptation.densify_until
        if prune_until < adaptation.opacity_prune_from:
            raise ValueError("invalid Gaussian opacity-pruning range")
        if (
            adaptation.opacity_prune_until is not None
            and prune_until > self.optimization.iterations
        ):
            raise ValueError(
                "gaussian_adaptation.opacity_prune_until exceeds training"
            )
        prune_interval = (
            adaptation.densification_interval
            if adaptation.opacity_prune_interval is None
            else adaptation.opacity_prune_interval
        )
        if prune_interval < 1:
            raise ValueError(
                "gaussian_adaptation.opacity_prune_interval must be positive"
            )
        if adaptation.min_points < 1:
            raise ValueError("gaussian_adaptation.min_points must be positive")
        if adaptation.min_points > self.model.gaussian_initial_points:
            raise ValueError(
                "gaussian_adaptation.min_points exceeds initial population"
            )
        if not 0.0 < adaptation.max_opacity_prune_fraction <= 1.0:
            raise ValueError(
                "max_opacity_prune_fraction must be in (0, 1]"
            )
        if adaptation.min_opacity_initial is not None and not (
            0.0 <= adaptation.min_opacity_initial < 1.0
        ):
            raise ValueError(
                "gaussian_adaptation.min_opacity_initial must be in [0, 1)"
            )
        for name in (
            "opacity_reset_interval",
            "medium_freeze_after_reset",
            "medium_freeze_interval",
            "medium_freeze_from",
            "medium_freeze_steps",
        ):
            if getattr(adaptation, name) < 0:
                raise ValueError(
                    f"gaussian_adaptation.{name} must be non-negative"
                )
        if adaptation.min_opacity >= 1.0:
            raise ValueError(
                "gaussian_adaptation.min_opacity must be smaller than 1"
            )
        if not 0.0 < adaptation.opacity_reset_value < 1.0:
            raise ValueError(
                "gaussian_adaptation.opacity_reset_value must be in (0, 1)"
            )
        if (
            adaptation.runaway_max_points is not None
            and adaptation.runaway_max_points < 1
        ):
            raise ValueError(
                "gaussian_adaptation.runaway_max_points must be positive"
            )
        for name in (
            "gradient_threshold",
            "min_opacity",
            "percent_dense",
            "max_screen_size",
            "max_world_scale_fraction",
            "min_effective_scale",
        ):
            if getattr(adaptation, name) < 0:
                raise ValueError(
                    f"gaussian_adaptation.{name} must be non-negative"
                )
        if adaptation.far_prune_distance is not None and not (
            adaptation.far_prune_distance > 0 and adaptation.far_prune_scale > 0
        ):
            raise ValueError(
                "gaussian_adaptation.far_prune_distance and far_prune_scale must be positive"
            )
        reference = adaptation.gradient_threshold_reference_width
        if reference is not None and (type(reference) is not int or reference < 1):
            raise ValueError("gaussian_adaptation.gradient_threshold_reference_width must be a positive integer")
        if not adaptation.gradient_compensation_max >= 1.0:
            raise ValueError("gaussian_adaptation.gradient_compensation_max must be at least 1")
        ramp = adaptation.max_points_ramp_until
        if ramp is not None and (adaptation.max_points is None or ramp <= adaptation.densify_from):
            raise ValueError("gaussian_adaptation.max_points_ramp_until needs max_points and must exceed densify_from")
        if adaptation.absolute_gradient and self.model.medium_type != "rgb_backscatter":
            raise ValueError("gaussian_adaptation.absolute_gradient needs model.medium_type=rgb_backscatter")
        medium_adaptation = self.voronoi_adaptation
        if medium_adaptation.enabled and self.representation_mode == "gaussian_only":
            raise ValueError("Voronoi adaptation cannot run in gaussian_only mode")
        if medium_adaptation.interval < 1:
            raise ValueError("voronoi_adaptation.interval must be positive")
        if not 0 <= medium_adaptation.adaptation_offset < medium_adaptation.interval:
            raise ValueError(
                "voronoi_adaptation.adaptation_offset must be in [0, interval)"
            )
        if medium_adaptation.densify_from < 1 or medium_adaptation.densify_until < medium_adaptation.densify_from:
            raise ValueError("invalid Voronoi adaptation iteration range")
        if medium_adaptation.min_points < self.model.medium_guard_points:
            raise ValueError("Voronoi min_points must preserve guard sites")
        if medium_adaptation.max_points is not None and medium_adaptation.max_points < self.model.medium_points:
            raise ValueError("initial medium count exceeds Voronoi adaptation budget")
        for name in ("densify_fraction", "max_prune_fraction"):
            if not 0.0 <= getattr(medium_adaptation, name) <= 1.0:
                raise ValueError(f"voronoi_adaptation.{name} must be in [0, 1]")
        if medium_adaptation.prune_patience < 1:
            raise ValueError("voronoi_adaptation.prune_patience must be positive")

        if len(self.background) != 3:
            raise ValueError("background must contain RGB")
        if self.wandb.mode not in {"online", "offline", "disabled"}:
            raise ValueError(
                "wandb.mode must be online, offline, or disabled"
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.as_dict(), sort_keys=False)
        )
