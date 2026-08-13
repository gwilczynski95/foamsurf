#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys
REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
import shutil
import tempfile

import numpy as np
from PIL import Image

from training.config import (
    DatasetConfig,
    DiagnosticsConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizationConfig,
    RendererConfig,
    ValidationConfig,
    WandbConfig,
)
from training.trainer import Trainer


def _camera(z: float, x: float = 0.0) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _make_dataset(root: Path) -> None:
    root.mkdir(parents=True)
    height = width = 16
    y, x = np.mgrid[:height, :width]
    distance = ((x - width / 2) ** 2 + (y - height / 2) ** 2)
    alpha = (distance < 30).astype(np.uint8) * 255
    rgb = np.stack(
        (
            np.full_like(alpha, 180),
            (x / (width - 1) * 255).astype(np.uint8),
            (y / (height - 1) * 255).astype(np.uint8),
            alpha,
        ),
        axis=-1,
    )
    frames = []
    for index, (z, offset) in enumerate(((4.0, 0.0), (4.2, 0.1))):
        name = f"train_{index}"
        Image.fromarray(rgb, mode="RGBA").save(root / f"{name}.png")
        frames.append(
            {
                "file_path": name,
                "transform_matrix": _camera(z, offset),
            }
        )
    metadata = {"camera_angle_x": 0.7, "frames": frames}
    (root / "transforms_train.json").write_text(
        json.dumps(metadata)
    )

    test_name = "test_0"
    Image.fromarray(rgb, mode="RGBA").save(root / f"{test_name}.png")
    (root / "transforms_test.json").write_text(
        json.dumps(
            {
                "camera_angle_x": 0.7,
                "frames": [
                    {
                        "file_path": test_name,
                        "transform_matrix": _camera(4.1, 0.05),
                    }
                ],
            }
        )
    )


def main() -> int:
    base = Path(tempfile.gettempdir()) / "foamsurf_b0_smoke"
    if base.exists():
        shutil.rmtree(base)
    dataset = base / "dataset"
    output = base / "output"
    _make_dataset(dataset)

    config = ExperimentConfig(
        experiment_name="b0-smoke",
        output_dir=str(output),
        seed=7,
        device="cuda",
        background=[1.0, 1.0, 1.0],
        dataset=DatasetConfig(source_path=str(dataset)),
        model=ModelConfig(
            appearance="rgb",
            gaussian_initial_points=32,
            gaussian_initial_opacity=0.1,
            medium_points=40,
            medium_guard_points=8,
            medium_initial_density=0.01,
        ),
        renderer=RendererConfig(near=0.1, far=8.0),
        optimization=OptimizationConfig(
            iterations=2,
            rays_per_batch=64,
            topology_update_interval=1,
        ),
        validation=ValidationConfig(
            eval_interval=1,
            log_interval=1,
            checkpoint_interval=1,
            max_eval_views=1,
            image_views=1,
        ),
        diagnostics=DiagnosticsConfig(enabled=True),
        wandb=WandbConfig(mode="disabled"),
    )
    config.validate()
    summary = Trainer(config).fit()
    config.optimization.iterations = 3
    summary = Trainer(
        config, resume=output / "checkpoints" / "last.pt"
    ).fit()

    required = (
        output / "metrics" / "train.csv",
        output / "metrics" / "eval.csv",
        output / "checkpoints" / "last.pt",
        output / "checkpoints" / "best.pt",
        output / "summary.json",
        output
        / "evaluation"
        / "iteration_000003"
        / "0000_test_0_panel.png",
        output
        / "evaluation"
        / "iteration_000003"
        / "0000_test_0_responsibility.png",
        output
        / "evaluation"
        / "iteration_000003"
        / "gaussian_primitives.csv",
        output
        / "evaluation"
        / "iteration_000003"
        / "medium_primitives.csv",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing B0 artifacts: {missing}")
    print("FoamSurf B0 smoke test: OK")
    print(json.dumps(summary, indent=2))
    print(f"artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
