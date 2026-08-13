from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .blender import BlenderDataset, Camera


def _camera_values(camera: Camera) -> tuple[torch.Tensor, ...]:
    return (
        camera.origin,
        camera.right,
        camera.down,
        camera.forward,
    )


def _validate_alignment(
    reference: list[Camera],
    candidate: list[Camera],
    label: str,
) -> None:
    if len(candidate) != len(reference):
        raise ValueError(
            f"{label} has {len(candidate)} test views; expected "
            f"{len(reference)}"
        )
    for index, (expected, actual) in enumerate(zip(reference, candidate)):
        prefix = f"{label} view {index}"
        if actual.name != expected.name:
            raise ValueError(
                f"{prefix} is named {actual.name!r}; expected "
                f"{expected.name!r}"
            )
        if (actual.height, actual.width) != (
            expected.height,
            expected.width,
        ):
            raise ValueError(f"{prefix} has a different image resolution")
        if abs(actual.fov_x - expected.fov_x) > 1.0e-6 or abs(
            actual.fov_y - expected.fov_y
        ) > 1.0e-6:
            raise ValueError(f"{prefix} has different camera intrinsics")
        for candidate_value, reference_value in zip(
            _camera_values(actual), _camera_values(expected)
        ):
            if not torch.allclose(
                candidate_value.detach().cpu(),
                reference_value.detach().cpu(),
                atol=1.0e-6,
                rtol=1.0e-6,
            ):
                raise ValueError(f"{prefix} has a different camera pose")


@dataclass(frozen=True)
class LayerGroundTruth:
    """Camera-aligned, evaluation-only decomposition targets on CPU."""

    lego_lego: tuple[Camera, ...]
    lego_clouds: tuple[Camera, ...]

    @classmethod
    def load(
        cls,
        reference_test: list[Camera],
        gaussian_path: str | Path,
        medium_path: str | Path,
        resolution: int,
    ) -> "LayerGroundTruth":
        gaussian_dataset = BlenderDataset(
            gaussian_path,
            resolution=resolution,
            device="cpu",
            splits=("test",),
        )
        medium_dataset = BlenderDataset(
            medium_path,
            resolution=resolution,
            device="cpu",
            splits=("test",),
        )
        _validate_alignment(
            reference_test, gaussian_dataset.test, "lego_lego"
        )
        _validate_alignment(
            reference_test, medium_dataset.test, "lego_clouds"
        )
        return cls(
            lego_lego=tuple(gaussian_dataset.test),
            lego_clouds=tuple(medium_dataset.test),
        )
