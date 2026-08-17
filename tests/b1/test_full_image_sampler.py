from __future__ import annotations

from dataclasses import dataclass
import random

import torch

from training.sampler import RandomCameraRaySampler


@dataclass
class _Camera:
    name: str
    origin: torch.Tensor
    right: torch.Tensor
    down: torch.Tensor
    forward: torch.Tensor
    fov_x: float
    fov_y: float
    foreground: torch.Tensor
    alpha: torch.Tensor

    @property
    def width(self) -> int:
        return int(self.foreground.shape[1])

    @property
    def height(self) -> int:
        return int(self.foreground.shape[0])


class _Dataset:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.train = []
        for index in range(3):
            self.train.append(
                _Camera(
                    name=f"camera_{index}",
                    origin=torch.tensor([float(index), 0.0, 0.0]),
                    right=torch.tensor([1.0, 0.0, 0.0]),
                    down=torch.tensor([0.0, 1.0, 0.0]),
                    forward=torch.tensor([0.0, 0.0, 1.0]),
                    fov_x=1.0,
                    fov_y=1.0,
                    foreground=torch.full((2, 2, 3), float(index)),
                    alpha=torch.ones((2, 2, 1)),
                )
            )

    def stacked_training_data(self) -> dict[str, torch.Tensor]:
        return {
            "origins": torch.stack([camera.origin for camera in self.train]),
            "rights": torch.stack([camera.right for camera in self.train]),
            "downs": torch.stack([camera.down for camera in self.train]),
            "forwards": torch.stack([camera.forward for camera in self.train]),
            "fov_x": torch.tensor([camera.fov_x for camera in self.train]),
            "fov_y": torch.tensor([camera.fov_y for camera in self.train]),
        }


def test_full_image_sampler_exhausts_shuffled_camera_stack() -> None:
    random.seed(42)
    sampler = RandomCameraRaySampler(
        _Dataset(),
        rays_per_batch=1,
        background=(1.0, 1.0, 1.0),
        full_image=True,
        cameras_without_replacement=True,
    )

    batches = [sampler.next() for _ in range(3)]

    assert len({batch.camera_name for batch in batches}) == 3
    for batch in batches:
        assert batch.rays.origins.shape == (4, 3)
        torch.testing.assert_close(
            batch.rays.pixel_indices, torch.arange(4)
        )
        expected = float(batch.camera_name.rsplit("_", 1)[1])
        torch.testing.assert_close(
            batch.target, torch.full((4, 3), expected)
        )
