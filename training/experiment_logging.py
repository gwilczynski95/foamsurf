from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ExperimentConfig


class CsvTable:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames: list[str] | None = None
        if path.is_file() and path.stat().st_size:
            with path.open(newline="") as file:
                reader = csv.reader(file)
                self.fieldnames = next(reader)

    def append(self, row: dict[str, Any]) -> None:
        if self.fieldnames is None:
            self.fieldnames = list(row)
            with self.path.open("w", newline="") as file:
                writer = csv.DictWriter(
                    file, fieldnames=self.fieldnames
                )
                writer.writeheader()
                writer.writerow(row)
            return
        if set(row) != set(self.fieldnames):
            raise ValueError(
                f"CSV schema mismatch for {self.path}: "
                f"expected {self.fieldnames}, received {list(row)}"
            )
        with self.path.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writerow(row)


class ExperimentLogger:
    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: Path,
        resume: bool,
    ) -> None:
        self.train_csv = CsvTable(output_dir / "metrics" / "train.csv")
        self.eval_csv = CsvTable(output_dir / "metrics" / "eval.csv")
        self.wandb_run = None
        if config.wandb.mode != "disabled":
            import wandb

            self.wandb_run = wandb.init(
                project=config.wandb.project,
                entity=config.wandb.entity,
                name=config.wandb.name or config.experiment_name,
                tags=config.wandb.tags,
                mode=config.wandb.mode,
                config=config.as_dict(),
                dir=str(output_dir),
                resume="allow" if resume else None,
            )
            self.wandb_run.define_metric("iteration")
            self.wandb_run.define_metric("*", step_metric="iteration")

    def log_train(self, row: dict[str, Any]) -> None:
        self.train_csv.append(row)
        if self.wandb_run is not None:
            self.wandb_run.log(row)

    def log_eval(
        self,
        row: dict[str, Any],
        images: dict[str, np.ndarray],
        histograms: dict[str, torch.Tensor],
    ) -> None:
        self.eval_csv.append(row)
        if self.wandb_run is None:
            return
        import wandb

        payload: dict[str, Any] = dict(row)
        for name, image in images.items():
            payload[f"images/{name}"] = wandb.Image(image)
        for name, values in histograms.items():
            flat = values.detach().float().reshape(-1).cpu()
            if flat.numel() > 100_000:
                indices = torch.linspace(
                    0, flat.numel() - 1, 100_000
                ).long()
                flat = flat[indices]
            payload[f"histograms/{name}"] = wandb.Histogram(
                flat.numpy()
            )
        self.wandb_run.log(payload)

    def close(self) -> None:
        if self.wandb_run is not None:
            self.wandb_run.finish()
