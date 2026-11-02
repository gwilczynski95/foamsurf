from __future__ import annotations

from pathlib import Path
import torch

from .blender import BlenderDataset
from .colmap import ColmapDataset, colmap_model_dir


def detect_dataset_type(root: str | Path) -> str:
    root = Path(root)
    if (root / "transforms_train.json").is_file():
        return "blender"
    if colmap_model_dir(root) is not None:
        return "colmap"
    raise ValueError(
        f"could not detect dataset type at {root}; expected Blender transforms "
        "or a COLMAP model in sparse/0 or sparse"
    )


def load_dataset(
    root: str | Path,
    resolution: int,
    device: str | torch.device,
    llffhold: int = 8,
    validation_hold: int = 0,
):
    if detect_dataset_type(root) == "blender":
        if validation_hold:
            raise ValueError("validation_hold is supported for COLMAP datasets only")
        return BlenderDataset(root, resolution=resolution, device=device)
    return ColmapDataset(
        root,
        resolution=resolution,
        device=device,
        llffhold=llffhold,
        validation_hold=validation_hold,
    )
