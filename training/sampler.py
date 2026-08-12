from __future__ import annotations

from dataclasses import dataclass
import random

import torch

from foamsurf_data.rays import RayBundle, generate_indexed_rays


@dataclass(frozen=True)
class TrainingBatch:
    rays: RayBundle
    target: torch.Tensor
    camera_origin: torch.Tensor
    camera_name: str


class RandomCameraRaySampler:
    """Sample one camera per step so SH colors are well-defined."""

    def __init__(
        self,
        dataset,
        rays_per_batch: int,
        background: tuple[float, float, float],
    ) -> None:
        self.dataset = dataset
        self.rays_per_batch = int(rays_per_batch)
        self.background = torch.tensor(
            background,
            dtype=torch.float32,
            device=dataset.device,
        )
        self.data = dataset.stacked_training_data()

    def next(self) -> TrainingBatch:
        camera_index = random.randrange(len(self.dataset.train))
        camera = self.dataset.train[camera_index]
        pixel_count = camera.width * camera.height
        sample_count = min(self.rays_per_batch, pixel_count)
        pixel_indices = torch.randint(
            0,
            pixel_count,
            (sample_count,),
            device=self.dataset.device,
        )
        rays = generate_indexed_rays(
            pixel_indices,
            self.data["origins"][camera_index : camera_index + 1],
            self.data["rights"][camera_index : camera_index + 1],
            self.data["downs"][camera_index : camera_index + 1],
            self.data["forwards"][camera_index : camera_index + 1],
            self.data["fov_x"][camera_index : camera_index + 1],
            self.data["fov_y"][camera_index : camera_index + 1],
            camera.width,
            camera.height,
        )
        foreground = camera.foreground.reshape(-1, 3)[pixel_indices]
        alpha = camera.alpha.reshape(-1, 1)[pixel_indices]
        target = foreground + self.background * (1.0 - alpha)
        return TrainingBatch(
            rays=rays,
            target=target,
            camera_origin=camera.origin,
            camera_name=camera.name,
        )
