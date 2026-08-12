from __future__ import annotations

from pathlib import Path
import torch

from .blender import BlenderDataset
from .colmap import ColmapDataset


def detect_dataset_type(root: str | Path) -> str:
    root = Path(root)
    if (root / "transforms_train.json").is_file():
        return "blender"
    sparse = root / "sparse" / "0"
    binary = (sparse / "cameras.bin").is_file() and (
        sparse / "images.bin"
    ).is_file()
    text = (sparse / "cameras.txt").is_file() and (
        sparse / "images.txt"
    ).is_file()
    if binary or text:
        return "colmap"
    raise ValueError(
        f"could not detect dataset type at {root}; expected Blender transforms "
        "or a COLMAP sparse/0 model"
    )


def load_dataset(
    root: str | Path,
    resolution: int,
    device: str | torch.device,
    llffhold: int = 8,
):
    if detect_dataset_type(root) == "blender":
        return BlenderDataset(root, resolution=resolution, device=device)
    return ColmapDataset(
        root,
        resolution=resolution,
        device=device,
        llffhold=llffhold,
    )
