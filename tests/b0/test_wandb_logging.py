from __future__ import annotations

import csv

import numpy as np
import torch

from training.config import ExperimentConfig, WandbConfig
from training.experiment_logging import ExperimentLogger


def test_wandb_offline_logs_b01_scalars_images_and_histograms(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("WANDB_SILENT", "true")
    output = tmp_path / "run"
    config = ExperimentConfig(
        experiment_name="wandb-offline-test",
        output_dir=str(output),
        wandb=WandbConfig(
            mode="offline",
            project="foamsurf-tests",
            name="offline-logger",
        ),
    )

    logger = ExperimentLogger(config, output, resume=False)
    logger.log_train(
        {
            "iteration": 1,
            "train/loss": 0.25,
            "train/psnr": 6.0206,
            "gradient/gaussian_active_fraction": 0.4,
            "gradient/medium_active_fraction": 0.8,
        }
    )
    eval_row = {
        "iteration": 1,
        "eval/hybrid_psnr": 7.0,
        "responsibility/gaussian_active_fraction": 0.3,
        "responsibility/medium_active_fraction": 0.7,
        "responsibility/gaussian_contribution_total": 10.0,
        "responsibility/medium_contribution_total": 5.0,
    }
    logger.log_eval(
        eval_row,
        {
            "comparison/view_00": np.zeros(
                (2, 4, 3), dtype=np.uint8
            ),
            "responsibility/view_00": np.zeros(
                (2, 3, 3), dtype=np.uint8
            ),
        },
        {
            "gaussian_contribution": torch.tensor([0.0, 0.2]),
            "medium_contribution": torch.tensor([0.1, 0.4]),
            "gaussian_gradient_magnitude": torch.tensor([0.0, 0.3]),
            "medium_gradient_magnitude": torch.tensor([0.2, 0.5]),
        },
    )
    logger.close()

    train_path = output / "metrics" / "train.csv"
    eval_path = output / "metrics" / "eval.csv"
    assert train_path.is_file()
    assert eval_path.is_file()
    with train_path.open(newline="") as file:
        train_row = next(csv.DictReader(file))
    with eval_path.open(newline="") as file:
        written_eval = next(csv.DictReader(file))
    assert "gradient/gaussian_active_fraction" in train_row
    assert set(eval_row).issubset(written_eval)
    assert list((output / "wandb").glob("offline-run-*"))
