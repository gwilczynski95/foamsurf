from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from .config import ExperimentConfig


@dataclass(frozen=True)
class VoronoiBlackL1DSSIMSpec:
    label: str
    question: str
    seed: int
    initial_points: int
    sites_lr: float
    density_lr: float
    color_lr: float = 5.0e-3
    adaptive: bool = True
    max_points: int | None = None


def _paired(
    prefix: str,
    question: str,
    initial_points: int,
    sites_lr: float,
    density_lr: float,
    *,
    adaptive: bool = True,
) -> tuple[VoronoiBlackL1DSSIMSpec, ...]:
    return tuple(
        VoronoiBlackL1DSSIMSpec(
            label=f"{prefix}_s{seed}",
            question=question,
            seed=seed,
            initial_points=initial_points,
            sites_lr=sites_lr,
            density_lr=density_lr,
            adaptive=adaptive,
        )
        for seed in (42, 1337)
    )


VORONOI_BLACK_L1DSSIM_SWEEP_SPECS = (
    *_paired(
        "adapt_n2048_slr5e-05_dlr0.005_clr0.005",
        "winning sparse adaptive configuration under Gaussian-matched supervision",
        2_048, 5.0e-5, 5.0e-3,
    ),
    *_paired(
        "adapt_n2048_slr0.0001_dlr0.005_clr0.005",
        "second-best site learning rate under Gaussian-matched supervision",
        2_048, 1.0e-4, 5.0e-3,
    ),
    *_paired(
        "adapt_n2048_slr0.0002_dlr0.005_clr0.005",
        "faster site optimization under Gaussian-matched supervision",
        2_048, 2.0e-4, 5.0e-3,
    ),
    *_paired(
        "adapt_n4096_slr5e-05_dlr0.005_clr0.005",
        "strong 4096-site configuration corresponding to previous run 23",
        4_096, 5.0e-5, 5.0e-3,
    ),
    *_paired(
        "adapt_n2048_slr5e-05_dlr0.01_clr0.005",
        "higher density learning rate control under the corrected loss",
        2_048, 5.0e-5, 1.0e-2,
    ),
    *_paired(
        "fixed_n8192_slr5e-05_dlr0.005_clr0.005",
        "fixed 8192-site control at the adaptive winner's final capacity",
        8_192, 5.0e-5, 5.0e-3, adaptive=False,
    ),
    VoronoiBlackL1DSSIMSpec(
        label="adapt_n2048_cap16384_slr5e-05_dlr0.005_clr0.005_s42",
        question="larger adaptive capacity from the winning sparse initialization",
        seed=42, initial_points=2_048, sites_lr=5.0e-5,
        density_lr=5.0e-3, max_points=16_384,
    ),
    VoronoiBlackL1DSSIMSpec(
        label="adapt_n2048_slr5e-05_dlr0.005_clr0.001_s42",
        question="lower color learning rate against late appearance drift",
        seed=42, initial_points=2_048, sites_lr=5.0e-5,
        density_lr=5.0e-3, color_lr=1.0e-3,
    ),
    VoronoiBlackL1DSSIMSpec(
        label="adapt_n2048_slr5e-05_dlr0.0025_clr0.005_s42",
        question="lower density learning rate against late density polarization",
        seed=42, initial_points=2_048, sites_lr=5.0e-5,
        density_lr=2.5e-3,
    ),
    VoronoiBlackL1DSSIMSpec(
        label="adapt_n8192_slr5e-05_dlr0.005_clr0.005_s42",
        question="larger initialization and 32768-site adaptive capacity control",
        seed=42, initial_points=8_192, sites_lr=5.0e-5,
        density_lr=5.0e-3, max_points=32_768,
    ),
)


_DEFAULT_MAX_POINTS = {2_048: 8_192, 4_096: 16_384, 8_192: 32_768}


def build_voronoi_black_l1dssim_config(
    base: ExperimentConfig,
    index: int,
    output_dir: str,
    wandb_name: str,
    wandb_mode: str = "online",
) -> ExperimentConfig:
    config = deepcopy(base)
    spec = VORONOI_BLACK_L1DSSIM_SWEEP_SPECS[index]
    config.seed = spec.seed
    config.output_dir = output_dir
    config.experiment_name = wandb_name
    config.wandb.name = wandb_name
    config.wandb.mode = wandb_mode
    config.wandb.project = "foamsurf_stage2_voronoi_black_l1dssim_sweep"
    config.wandb.tags = list(config.wandb.tags) + [spec.label]

    config.model.medium_points = spec.initial_points
    config.optimization.medium_sites_lr = spec.sites_lr
    config.optimization.medium_density_lr = spec.density_lr
    config.optimization.medium_color_lr = spec.color_lr
    config.voronoi_adaptation.enabled = spec.adaptive
    if not spec.adaptive:
        config.voronoi_adaptation.max_points = spec.initial_points
    else:
        config.voronoi_adaptation.max_points = (
            spec.max_points
            if spec.max_points is not None
            else _DEFAULT_MAX_POINTS[spec.initial_points]
        )
    config.validate()
    return config
