from __future__ import annotations

import numpy as np
import torch

from training.config import ExperimentConfig, WandbConfig
from training.experiment_logging import ExperimentLogger


def test_wandb_offline_logs_train_eval_images_and_histograms(
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
        }
    )
    logger.log_eval(
        {
            "iteration": 1,
            "eval/hybrid_psnr": 7.0,
        },
        {"comparison": np.zeros((2, 4, 3), dtype=np.uint8)},
        {"medium_density": torch.tensor([0.01, 0.02])},
    )
    logger.close()

    assert (output / "metrics" / "train.csv").is_file()
    assert (output / "metrics" / "eval.csv").is_file()
    assert list((output / "wandb").glob("offline-run-*"))
