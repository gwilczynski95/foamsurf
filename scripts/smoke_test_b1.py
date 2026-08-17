#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from scripts.smoke_test_b0 import _make_dataset
from training.config import (
    DatasetConfig,
    DiagnosticsConfig,
    ExperimentConfig,
    GaussianAdaptationConfig,
    ModelConfig,
    OptimizationConfig,
    RendererConfig,
    ValidationConfig,
    WandbConfig,
)
from training.trainer import Trainer


def main() -> int:
    base = Path(tempfile.gettempdir()) / "foamsurf_b1_smoke"
    if base.exists():
        shutil.rmtree(base)
    dataset = base / "dataset"
    output = base / "output"
    _make_dataset(dataset)

    config = ExperimentConfig(
        experiment_name="b1-smoke",
        output_dir=str(output),
        seed=11,
        device="cuda",
        dataset=DatasetConfig(source_path=str(dataset)),
        model=ModelConfig(
            appearance="rgb",
            gaussian_initial_points=32,
            gaussian_initial_opacity=0.1,
            medium_points=40,
            medium_guard_points=8,
        ),
        renderer=RendererConfig(near=0.1, far=8.0),
        optimization=OptimizationConfig(
            iterations=4,
            rays_per_batch=64,
            gaussian_appearance_lr=0.0025,
            gaussian_appearance_rest_lr_scale=0.05,
            gaussian_opacity_lr=0.025,
            gaussian_means_lr=0.00016,
            gaussian_means_lr_final=0.0000016,
            gaussian_means_lr_delay_mult=0.01,
            gaussian_means_lr_max_steps=30_000,
            gaussian_means_spatial_scale=True,
            gaussian_scales_lr=0.005,
            gaussian_quaternions_lr=0.001,
            gaussian_adam_eps=1.0e-15,
            topology_update_interval=1,
        ),
        validation=ValidationConfig(
            eval_interval=2,
            log_interval=1,
            checkpoint_interval=1,
            max_eval_views=1,
            image_views=1,
        ),
        diagnostics=DiagnosticsConfig(enabled=True),
        gaussian_adaptation=GaussianAdaptationConfig(
            enabled=True,
            max_points=64,
            densify_from=1,
            densify_until=5,
            densification_interval=1,
            gradient_threshold=2.0e-4,
            min_opacity=0.0,
            percent_dense=1.0,
            size_prune_from=100,
            max_screen_size=20.0,
            max_world_scale_fraction=0.1,
        ),
        wandb=WandbConfig(mode="disabled"),
    )
    config.validate()
    first = Trainer(config).fit()
    if not 32 < int(first["gaussian_count"]) <= 64:
        raise RuntimeError(f"B1 did not densify within budget: {first}")

    config.optimization.iterations = 6
    resumed = Trainer(
        config, resume=output / "checkpoints" / "last.pt"
    ).fit()
    if int(resumed["gaussian_count"]) != int(first["gaussian_count"]):
        raise RuntimeError("B1 resume changed the post-schedule point count")

    with (output / "metrics" / "train.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    required = {
        "adaptation/cloned",
        "adaptation/split",
        "adaptation/pruned",
        "adaptation/total_cloned",
        "adaptation/revision",
        "adaptation/gradient_p95",
        "adaptation/gradient_max",
        "adaptation/gradient_candidates",
        "adaptation/clone_candidates",
        "adaptation/split_candidates",
        "adaptation/gradient_screen_overlap",
        "adaptation/screen_replacements",
        "adaptation/budget_rejected",
        "adaptation/radius_p95",
        "optimization/gaussian_means_lr",
        "memory/peak_allocated_mib",
    }
    if not required.issubset(rows[0]):
        raise RuntimeError("B1 train.csv is missing adaptation diagnostics")
    if max(int(float(row["model/gaussian_count"])) for row in rows) > 64:
        raise RuntimeError("B1 exceeded its Gaussian budget")

    print("FoamSurf B1 native train/resume smoke test: OK")
    print(json.dumps(resumed, indent=2))
    print(f"artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
