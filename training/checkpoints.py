from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from foamsurf_model.scene import HybridSceneModel
from .config import ExperimentConfig


def checkpoint_payload(
    model: HybridSceneModel,
    gaussian_optimizer: torch.optim.Optimizer,
    medium_optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    iteration: int,
    training_seconds: float,
    best_psnr: float,
    topology_revision: int,
    gaussian_adaptation_state: dict[str, Any] | None = None,
    voronoi_adaptation_state: dict[str, Any] | None = None,
    medium_frozen_until: int = 0,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "baseline": "B0",
        "iteration": int(iteration),
        "training_seconds": float(training_seconds),
        "best_psnr": float(best_psnr),
        "topology_revision": int(topology_revision),
        "config": config.as_dict(),
        "model_metadata": model.checkpoint_metadata(),
        "model": model.state_dict(),
        "gaussian_optimizer": gaussian_optimizer.state_dict(),
        "medium_optimizer": medium_optimizer.state_dict(),
        "gaussian_adaptation": gaussian_adaptation_state,
        "voronoi_adaptation": voronoi_adaptation_state,
        "medium_frozen_until": int(medium_frozen_until),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: str | Path,
) -> dict[str, Any]:
    payload = torch.load(
        Path(path), map_location="cpu", weights_only=False
    )
    if payload.get("format_version") != 1:
        raise ValueError("unsupported FoamSurf checkpoint format")
    if payload.get("baseline") != "B0":
        raise ValueError("checkpoint is not a FoamSurf B0 run")
    return payload
