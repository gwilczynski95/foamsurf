from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch

from foamsurf_model import (
    HybridSceneModel,
    SurfaceGaussianModel,
    VoronoiMediumModel,
)
from .checkpoints import load_checkpoint
from .config import ExperimentConfig


def _checkpoint_config(payload: Mapping[str, Any], label: str) -> dict[str, Any]:
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{label} checkpoint does not contain a config")
    return config


def _require_equal(
    gaussian_config: Mapping[str, Any],
    medium_config: Mapping[str, Any],
    path: tuple[str, ...],
) -> None:
    left: Any = gaussian_config
    right: Any = medium_config
    for name in path:
        left = left[name]
        right = right[name]
    if left != right:
        dotted = ".".join(path)
        raise ValueError(
            f"checkpoint compatibility failure for {dotted}: "
            f"Gaussian={left!r}, Voronoi={right!r}"
        )


def validate_oracle_sources(
    gaussian_payload: Mapping[str, Any],
    medium_payload: Mapping[str, Any],
    oracle_config: ExperimentConfig,
) -> dict[str, Any]:
    """Validate the two standalone sources before combining their branches."""

    gaussian_config = _checkpoint_config(gaussian_payload, "Gaussian")
    medium_config = _checkpoint_config(medium_payload, "Voronoi")
    if gaussian_config.get("representation_mode") != "gaussian_only":
        raise ValueError("Gaussian checkpoint must use gaussian_only mode")
    if gaussian_config.get("dataset", {}).get("train_target") != "lego":
        raise ValueError("Gaussian checkpoint must have train_target=lego")
    if medium_config.get("representation_mode") != "voronoi_only":
        raise ValueError("Voronoi checkpoint must use voronoi_only mode")
    if medium_config.get("dataset", {}).get("train_target") != "clouds":
        raise ValueError("Voronoi checkpoint must have train_target=clouds")

    for path in (
        ("background",),
        ("dataset", "source_path"),
        ("dataset", "gaussian_ground_truth_path"),
        ("dataset", "medium_ground_truth_path"),
        ("dataset", "resolution"),
        ("dataset", "llffhold"),
        ("model", "appearance"),
    ):
        _require_equal(gaussian_config, medium_config, path)

    if gaussian_config["renderer"] != asdict(oracle_config.renderer):
        raise ValueError(
            "oracle renderer settings must exactly match the Gaussian "
            "checkpoint renderer settings"
        )
    if gaussian_config["background"] != oracle_config.background:
        raise ValueError("oracle background must match the source checkpoints")
    if gaussian_config["dataset"]["source_path"] != oracle_config.dataset.source_path:
        raise ValueError("oracle source_path must match the source checkpoints")

    gaussian_metadata = gaussian_payload["model_metadata"]["gaussians"]
    if gaussian_metadata["appearance_mode"] != oracle_config.model.appearance:
        raise ValueError("Gaussian checkpoint appearance does not match oracle config")

    return {
        "appearance": gaussian_metadata["appearance_mode"],
        "dataset_source_path": gaussian_config["dataset"]["source_path"],
        "resolution": gaussian_config["dataset"]["resolution"],
        "renderer_matches_gaussian_source": True,
        "medium_source_renderer": medium_config["renderer"],
        "oracle_renderer": asdict(oracle_config.renderer),
    }


def _branch_state(
    payload: Mapping[str, Any], prefix: str
) -> dict[str, torch.Tensor]:
    state = payload.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint does not contain model state")
    dotted = prefix + "."
    return {
        name[len(dotted) :]: value
        for name, value in state.items()
        if name.startswith(dotted)
    }


def branch_tensor_digest(payload: Mapping[str, Any], prefix: str) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(_branch_state(payload, prefix).items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_oracle_model(
    gaussian_payload: Mapping[str, Any],
    medium_payload: Mapping[str, Any],
    device: torch.device | str,
) -> HybridSceneModel:
    """Import only the qualified branch from each standalone checkpoint."""

    gaussian_state = _branch_state(gaussian_payload, "gaussians")
    missing_gaussian = set(SurfaceGaussianModel.PARAMETER_NAMES) - set(
        gaussian_state
    )
    if missing_gaussian:
        raise ValueError(
            f"Gaussian checkpoint is missing parameters: {sorted(missing_gaussian)}"
        )
    gaussians = SurfaceGaussianModel.from_checkpoint(
        gaussian_state,
        gaussian_payload["model_metadata"]["gaussians"],
        device,
    )

    medium_state = _branch_state(medium_payload, "medium")
    required_medium = {
        *VoronoiMediumModel.PARAMETER_NAMES,
        "bbox_min",
        "bbox_max",
    }
    missing_medium = required_medium - set(medium_state)
    if missing_medium:
        raise ValueError(
            f"Voronoi checkpoint is missing state: {sorted(missing_medium)}"
        )
    medium_metadata = medium_payload["model_metadata"]["medium"]
    medium = VoronoiMediumModel.from_checkpoint(
        medium_state,
        medium_metadata,
        medium_state["bbox_min"],
        medium_state["bbox_max"],
        device,
    )
    if "guard_mask" in medium_state:
        medium.guard_mask.copy_(medium_state["guard_mask"].to(device))

    model = HybridSceneModel(gaussians, medium)
    model.requires_grad_(False)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("oracle model parameters were not frozen")
    return model


def load_oracle_model(
    gaussian_checkpoint: str | Path,
    medium_checkpoint: str | Path,
    oracle_config: ExperimentConfig,
    device: torch.device | str,
) -> tuple[HybridSceneModel, dict[str, Any], dict[str, Any], dict[str, Any]]:
    gaussian_payload = load_checkpoint(gaussian_checkpoint)
    medium_payload = load_checkpoint(medium_checkpoint)
    compatibility = validate_oracle_sources(
        gaussian_payload, medium_payload, oracle_config
    )
    model = import_oracle_model(gaussian_payload, medium_payload, device)
    return model, gaussian_payload, medium_payload, compatibility
