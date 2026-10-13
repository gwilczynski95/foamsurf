from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


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


@dataclass
class RendererConfig:
    near: float = 0.1
    far: float = 6.0
    training_transmittance_threshold: float = 1.0e-4
    validation_transmittance_threshold: float = 1.0e-3
    max_intersections: int = 1_024
    number_of_sides: int = 8
    gaussian_radius_squared: float = 11.3449
    gaussian_hit_buffer_size: int = 16


@dataclass
class OptimizationConfig:
    iterations: int = 35_000
    rays_per_batch: int = 65_536
    full_image_batch: bool = False
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
    """Normalized surface/medium priors used after the B0 baseline."""

    opacity_binarization_weight: float = 0.0
    opacity_binarization_start: int = 0
    opacity_binarization_ramp: int = 0
    opacity_binarization_ramp_shape: str = "linear"
    opacity_binarization_end: int | None = None
    opacity_binarization_end_ramp: int = 0
    opacity_binarization_end_ramp_shape: str = "linear"
    opacity_binarization_maintenance_scale: float = 1.0
    opaque_if_used_weight: float = 0.0
    opaque_if_used_start: int = 0
    opaque_if_used_ramp: int = 0
    opaque_if_used_ramp_shape: str = "linear"
    graph_tv_weight: float = 0.0
    graph_tv_start: int = 0
    graph_tv_ramp: int = 0
    graph_tv_ramp_shape: str = "linear"
    graph_tv_mode: str = "huber"
    graph_tv_huber_delta: float = 0.02
    density_cap_weight: float = 0.0
    density_cap_start: int = 0
    density_cap_ramp: int = 0
    density_cap_ramp_shape: str = "linear"
    density_cap_policy: str = "fixed"
    density_cap_fixed_value: float = 1.0
    density_cap_quantile: float = 0.95
    density_cap_multiplier: float = 2.0
    density_cap_minimum: float = 0.005
    density_cap_update_interval: int = 25
    density_cap_ema_decay: float = 0.99
    density_cap_warmup_until: int = 5_000
    optical_thickness_weight: float = 0.0
    optical_thickness_start: int = 0
    optical_thickness_ramp: int = 0
    optical_thickness_ramp_shape: str = "linear"
    epsilon: float = 1.0e-8


@dataclass
class MediumHandoffConfig:
    """Optional phases that give the medium a controlled optimization window."""

    prefix_mode: str = "none"
    prefix_iterations: int = 0
    prefix_reset_medium_optimizer: bool = True
    gaussian_freeze_start: int | None = None
    gaussian_freeze_duration: int = 0
    medium_sites_lr_multiplier: float = 1.0
    medium_density_lr_multiplier: float = 1.0
    medium_color_lr_multiplier: float = 1.0


@dataclass
class RadFoamPriorConfig:
    """Optional spatial initialization from a separated RadFoam checkpoint."""

    enabled: bool = False
    checkpoint_path: str = ""
    labels_path: str = ""
    gaussian_count: int = 100_000
    pca_neighbors: int = 8
    reset_density: float = 1.0e-8


@dataclass
class VoronoiWarmupConfig:
    """Native full-scene Voronoi pretraining before Gaussian spawning."""

    enabled: bool = False
    milestone_iterations: list[int] = field(default_factory=list)


@dataclass
class GaussianAdaptationConfig:
    enabled: bool = False
    # graphdeco: screen-gradient clone/split/prune (3DGS/current FlaRe)
    # flare_movement: duplicate points moving during one optimizer step
    # fixed: fixed topology with optional opacity reset
    policy: str = "graphdeco"
    max_points: int | None = 200_000
    densify_from: int = 500
    densify_until: int = 15_000
    densification_interval: int = 100
    gradient_threshold: float = 2.0e-4
    min_opacity: float = 0.005
    min_opacity_initial: float | None = None
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
    # Water-Splatting style culling: while densifying, prune below this
    # opacity (WS: 0.5) instead of the scheduled threshold, if higher.
    min_opacity_densify: float | None = None
    opacity_reset_interval: int = 0
    opacity_reset_value: float = 0.01
    opacity_reset_at_densify_from: bool = False
    opacity_reset_optimizer_state: bool = True
    # Iterations after an opacity reset during which min_opacity_densify is
    # not applied, so clamped Gaussians can recover before being culled.
    opacity_reset_prune_grace: int = 0
    medium_freeze_after_reset: int = 0
    medium_freeze_interval: int = 0
    medium_freeze_from: int = 0
    medium_freeze_steps: int = 0
    preserve_clone_opacity: bool = False
    preserve_split_opacity: bool = False
    runaway_max_points: int | None = None
    movement_threshold: float = 1.0e-3
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
    medium_handoff: MediumHandoffConfig = field(
        default_factory=MediumHandoffConfig
    )
    radfoam_prior: RadFoamPriorConfig = field(
        default_factory=RadFoamPriorConfig
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
        raw = yaml.safe_load(path.read_text()) or {}
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
            "medium_handoff",
            "radfoam_prior",
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
            medium_handoff=_construct(
                MediumHandoffConfig, raw.get("medium_handoff")
            ),
            radfoam_prior=_construct(
                RadFoamPriorConfig, raw.get("radfoam_prior")
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
        if self.model.medium_type == "rgb_backscatter":
            if self.radfoam_prior.enabled:
                raise ValueError("RGB medium does not support scalar RadFoam priors")
            if any(getattr(self.disentanglement, n) > 0 for n in
                   ("graph_tv_weight", "density_cap_weight", "optical_thickness_weight", "opaque_if_used_weight")):
                raise ValueError("Scalar medium regularizers are not calibrated for RGB medium")
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
            if self.dataset.train_target != expected_target:
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
            if self.radfoam_prior.enabled:
                raise ValueError("Voronoi warmup cannot use a RadFoam prior")
            if self.medium_handoff.prefix_mode != "none":
                raise ValueError("Voronoi warmup cannot use a handoff prefix")
        if self.dataset.resolution < 1:
            raise ValueError("dataset.resolution must be positive")
        prior = self.radfoam_prior
        if prior.enabled:
            if self.representation_mode != "hybrid":
                raise ValueError(
                    "RadFoam prior initialization requires hybrid mode"
                )
            if not prior.checkpoint_path or not prior.labels_path:
                raise ValueError(
                    "enabled RadFoam prior requires checkpoint_path and "
                    "labels_path"
                )
            if prior.gaussian_count < 1:
                raise ValueError(
                    "radfoam_prior.gaussian_count must be positive"
                )
            if prior.gaussian_count != self.model.gaussian_initial_points:
                raise ValueError(
                    "radfoam_prior.gaussian_count must equal "
                    "model.gaussian_initial_points"
                )
            if prior.pca_neighbors < 3:
                raise ValueError(
                    "radfoam_prior.pca_neighbors must be at least 3"
                )
            if prior.reset_density <= 0.0:
                raise ValueError(
                    "radfoam_prior.reset_density must be positive"
                )
            if self.voronoi_adaptation.enabled:
                raise ValueError(
                    "RadFoam prior requires "
                    "voronoi_adaptation.enabled=false to preserve the "
                    "imported population"
                )
        if self.model.appearance not in {"rgb", "sh"}:
            raise ValueError("model.appearance must be rgb or sh")
        if self.model.gaussian_rgb_activation not in {"identity", "sigmoid"}:
            raise ValueError("model.gaussian_rgb_activation must be identity or sigmoid")
        if self.model.gaussian_rgb_activation == "sigmoid" and self.model.appearance != "rgb":
            raise ValueError("sigmoid Gaussian RGB requires model.appearance=rgb")
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
        loss_names = (
            "opacity_binarization",
            "opaque_if_used",
            "graph_tv",
            "density_cap",
            "optical_thickness",
        )
        for name in loss_names:
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
        if disentanglement.graph_tv_mode not in {"huber", "l1", "l2"}:
            raise ValueError(
                "disentanglement.graph_tv_mode must be huber, l1, or l2"
            )
        if disentanglement.graph_tv_huber_delta <= 0.0:
            raise ValueError(
                "disentanglement.graph_tv_huber_delta must be positive"
            )
        if disentanglement.density_cap_policy not in {
            "fixed", "rolling", "ema", "warmup_freeze"
        }:
            raise ValueError("invalid disentanglement.density_cap_policy")
        if disentanglement.density_cap_fixed_value <= 0.0:
            raise ValueError("density_cap_fixed_value must be positive")
        if not 0.0 < disentanglement.density_cap_quantile < 1.0:
            raise ValueError("density_cap_quantile must be in (0, 1)")
        if disentanglement.density_cap_multiplier <= 0.0:
            raise ValueError("density_cap_multiplier must be positive")
        if disentanglement.density_cap_minimum <= 0.0:
            raise ValueError("density_cap_minimum must be positive")
        if disentanglement.density_cap_update_interval < 1:
            raise ValueError("density_cap_update_interval must be positive")
        if not 0.0 <= disentanglement.density_cap_ema_decay < 1.0:
            raise ValueError("density_cap_ema_decay must be in [0, 1)")
        if disentanglement.density_cap_warmup_until < 0:
            raise ValueError("density_cap_warmup_until must be non-negative")
        if disentanglement.epsilon <= 0.0:
            raise ValueError("disentanglement.epsilon must be positive")
        gaussian_losses_enabled = (
            disentanglement.opacity_binarization_weight > 0.0
            or disentanglement.opaque_if_used_weight > 0.0
        )
        medium_losses_enabled = (
            disentanglement.graph_tv_weight > 0.0
            or disentanglement.density_cap_weight > 0.0
            or disentanglement.optical_thickness_weight > 0.0
        )
        if (
            gaussian_losses_enabled
            and self.representation_mode == "voronoi_only"
        ):
            raise ValueError(
                "Gaussian disentanglement losses require Gaussians"
            )
        if (
            medium_losses_enabled
            and self.representation_mode == "gaussian_only"
        ):
            raise ValueError(
                "medium disentanglement losses require Voronoi"
            )
        handoff = self.medium_handoff
        if handoff.prefix_mode not in {"none", "medium_only", "joint"}:
            raise ValueError(
                "medium_handoff.prefix_mode must be none, medium_only, or joint"
            )
        if handoff.prefix_iterations < 0:
            raise ValueError(
                "medium_handoff.prefix_iterations must be non-negative"
            )
        if (handoff.prefix_mode == "none") != (handoff.prefix_iterations == 0):
            raise ValueError(
                "prefix_mode none requires zero prefix_iterations and "
                "a nonzero prefix requires medium_only or joint"
            )
        if handoff.prefix_iterations and self.representation_mode != "hybrid":
            raise ValueError("medium handoff prefix requires hybrid mode")
        if handoff.gaussian_freeze_duration < 0:
            raise ValueError("gaussian_freeze_duration must be non-negative")
        if (handoff.gaussian_freeze_start is None) != (
            handoff.gaussian_freeze_duration == 0
        ):
            raise ValueError(
                "gaussian freeze start and positive duration must be set together"
            )
        if handoff.gaussian_freeze_start is not None:
            if handoff.gaussian_freeze_start < 1:
                raise ValueError("gaussian_freeze_start must be positive")
            if (
                handoff.gaussian_freeze_start
                + handoff.gaussian_freeze_duration
                > self.optimization.iterations + 1
            ):
                raise ValueError("Gaussian freeze window exceeds training")
            if self.representation_mode != "hybrid":
                raise ValueError("Gaussian handoff freeze requires hybrid mode")
        multipliers = (
            handoff.medium_sites_lr_multiplier,
            handoff.medium_density_lr_multiplier,
            handoff.medium_color_lr_multiplier,
        )
        if any(value <= 0.0 for value in multipliers):
            raise ValueError("medium handoff LR multipliers must be positive")
        if handoff.gaussian_freeze_start is None and any(
            value != 1.0 for value in multipliers
        ):
            raise ValueError("medium LR boosts require a Gaussian freeze window")

        adaptation = self.gaussian_adaptation
        if adaptation.policy not in {"graphdeco", "flare_movement", "fixed"}:
            raise ValueError(
                "gaussian_adaptation.policy must be graphdeco, "
                "flare_movement, or fixed"
            )
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
        if adaptation.min_opacity_densify is not None and not (
            0.0 <= adaptation.min_opacity_densify < 1.0
        ):
            raise ValueError(
                "gaussian_adaptation.min_opacity_densify must be in [0, 1)"
            )
        for name in (
            "opacity_reset_interval",
            "opacity_reset_prune_grace",
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
            "movement_threshold",
            "min_effective_scale",
        ):
            if getattr(adaptation, name) < 0:
                raise ValueError(
                    f"gaussian_adaptation.{name} must be non-negative"
                )
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
