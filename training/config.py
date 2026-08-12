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


@dataclass
class ModelConfig:
    appearance: str = "rgb"
    sh_degree: int = 3
    gaussian_initial_points: int = 100_000
    gaussian_initial_opacity: float = 0.01
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
    gaussian_appearance_lr: float = 0.005
    gaussian_opacity_lr: float = 0.05
    gaussian_kappa_lr: float = 0.10
    gaussian_means_lr: float = 0.0025
    gaussian_scales_lr: float = 0.0396538025
    gaussian_quaternions_lr: float = 0.01
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
            wandb=_construct(WandbConfig, raw.get("wandb")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.dataset.source_path:
            raise ValueError("dataset.source_path is required")
        if self.dataset.resolution < 1:
            raise ValueError("dataset.resolution must be positive")
        if self.model.appearance not in {"rgb", "sh"}:
            raise ValueError("model.appearance must be rgb or sh")
        if not 0 <= self.model.sh_degree <= 3:
            raise ValueError("model.sh_degree must be between 0 and 3")
        if self.model.gaussian_initial_points < 1:
            raise ValueError("gaussian_initial_points must be positive")
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
        if self.optimization.topology_update_interval < 1:
            raise ValueError("topology_update_interval must be positive")
        for name in (
            "eval_interval",
            "log_interval",
            "checkpoint_interval",
        ):
            if getattr(self.validation, name) < 1:
                raise ValueError(f"validation.{name} must be positive")
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
