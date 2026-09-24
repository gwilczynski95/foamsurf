from __future__ import annotations

from dataclasses import dataclass
import random

import torch

from foamsurf_data.rays import RayBundle, generate_indexed_rays


@dataclass(frozen=True)
class TrainingBatch:
    rays: RayBundle
    target: torch.Tensor
    alpha: torch.Tensor
    camera: object

    @property
    def camera_origin(self) -> torch.Tensor:
        return self.camera.origin

    @property
    def camera_name(self) -> str:
        return self.camera.name


class RandomCameraRaySampler:
    """Sample one camera per step so SH colors are well-defined."""

    def __init__(
        self,
        dataset,
        rays_per_batch: int,
        background: tuple[float, float, float],
        *,
        full_image: bool = False,
        cameras_without_replacement: bool = False,
    ) -> None:
        self.dataset = dataset
        self.rays_per_batch = int(rays_per_batch)
        self.full_image = bool(full_image)
        self.cameras_without_replacement = bool(
            cameras_without_replacement
        )
        self._camera_stack: list[int] = []
        self.background = torch.tensor(
            background,
            dtype=torch.float32,
            device=dataset.device,
        )
        self.data = dataset.stacked_training_data()

    def next(self) -> TrainingBatch:
        if self.cameras_without_replacement:
            if not self._camera_stack:
                self._camera_stack = list(range(len(self.dataset.train)))
                random.shuffle(self._camera_stack)
            camera_index = self._camera_stack.pop()
        else:
            camera_index = random.randrange(len(self.dataset.train))
        camera = self.dataset.train[camera_index]
        pixel_count = camera.width * camera.height
        if self.full_image:
            pixel_indices = torch.arange(
                pixel_count, device=self.dataset.device
            )
        else:
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
            intrinsics=(self.data['intrinsics'][camera_index:camera_index+1]
                        if 'intrinsics' in self.data else None),
        )
        foreground = camera.foreground.reshape(-1, 3)[pixel_indices]
        alpha = camera.alpha.reshape(-1, 1)[pixel_indices]
        target = foreground + self.background * (1.0 - alpha)
        return TrainingBatch(
            rays=rays,
            target=target,
            alpha=alpha,
            camera=camera,
        )
