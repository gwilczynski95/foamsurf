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


@dataclass
class ModelConfig:
    appearance: str = "rgb"
    sh_degree: int = 3
    gaussian_initial_points: int = 100_000
    gaussian_initial_opacity: float = 0.1
    gaussian_initial_kappa: float = 1.01
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
    gaussian_opacity_lr: float = 0.025
    gaussian_kappa_lr: float = 0.10
    gaussian_means_lr: float = 0.00016
    gaussian_means_lr_final: float | None = 0.0000016
    gaussian_means_lr_delay_steps: int = 0
    gaussian_means_lr_delay_mult: float = 0.01
    gaussian_means_lr_max_steps: int = 30_000
    gaussian_means_spatial_scale: bool = True
    gaussian_scales_lr: float = 0.005
    gaussian_quaternions_lr: float = 0.001
    gaussian_adam_eps: float = 1.0e-15
    gaussian_sh_degree_interval: int = 1_000
    medium_sites_lr: float = 1.0e-4
    medium_density_lr: float = 1.0e-2
    medium_color_lr: float = 5.0e-3
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
class GaussianAdaptationConfig:
    enabled: bool = False
    max_points: int | None = 200_000
    densify_from: int = 500
    densify_until: int = 15_000
    densification_interval: int = 100
    gradient_threshold: float = 2.0e-4
    min_opacity: float = 0.005
    opacity_prune_from: int = 500
    percent_dense: float = 0.01
    size_prune_from: int = 3_000
    # The pinned official 2DGS implementation resets max_radii2D inside
    # densification_postfix before its final prune, so its advertised
    # screen-space test is effectively inactive. Keep this explicit rather
    # than reproducing that ordering side effect.
    screen_pruning_enabled: bool = False
    max_screen_size: float = 20.0
    max_world_scale_fraction: float = 0.10
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
    gaussian_adaptation: GaussianAdaptationConfig = field(
        default_factory=GaussianAdaptationConfig
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
            "background",
            "dataset",
            "model",
            "renderer",
            "optimization",
            "validation",
            "diagnostics",
            "gaussian_adaptation",
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
            gaussian_adaptation=_construct(
                GaussianAdaptationConfig, raw.get("gaussian_adaptation")
            ),
            wandb=_construct(WandbConfig, raw.get("wandb")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.dataset.source_path:
            raise ValueError("dataset.source_path is required")
        layer_paths = (
            self.dataset.gaussian_ground_truth_path,
            self.dataset.medium_ground_truth_path,
        )
        if (layer_paths[0] is None) != (layer_paths[1] is None):
            raise ValueError(
                "dataset Gaussian and medium ground-truth paths must be set together"
            )
        if self.dataset.resolution < 1:
            raise ValueError("dataset.resolution must be positive")
        if self.model.appearance not in {"rgb", "sh"}:
            raise ValueError("model.appearance must be rgb or sh")
        if not 0 <= self.model.sh_degree <= 3:
            raise ValueError("model.sh_degree must be between 0 and 3")
        if self.model.gaussian_initial_points < 1:
            raise ValueError("gaussian_initial_points must be positive")
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
        if self.optimization.iterations < 1:
            raise ValueError("optimization.iterations must be positive")
        if self.optimization.rays_per_batch < 1:
            raise ValueError("rays_per_batch must be positive")
        if self.optimization.reconstruction_loss not in {"mse", "l1", "l1_dssim"}:
            raise ValueError(
                "optimization.reconstruction_loss must be mse, l1, or l1_dssim"
            )
        if not 0.0 <= self.optimization.dssim_weight <= 1.0:
            raise ValueError("optimization.dssim_weight must be in [0, 1]")
        if self.optimization.gaussian_means_lr_final is not None and (
            self.optimization.gaussian_means_lr_final <= 0
        ):
            raise ValueError("gaussian_means_lr_final must be positive")
        if self.optimization.gaussian_means_lr_delay_steps < 0:
            raise ValueError(
                "gaussian_means_lr_delay_steps must be non-negative"
            )
        if self.optimization.gaussian_means_lr_max_steps < 1:
            raise ValueError("gaussian_means_lr_max_steps must be positive")
        if self.optimization.gaussian_sh_degree_interval < 0:
            raise ValueError("gaussian_sh_degree_interval must be non-negative")
        if self.optimization.gaussian_adam_eps <= 0:
            raise ValueError("gaussian_adam_eps must be positive")
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
        adaptation = self.gaussian_adaptation
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
        ):
            if getattr(adaptation, name) < 0:
                raise ValueError(
                    f"gaussian_adaptation.{name} must be non-negative"
                )
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
