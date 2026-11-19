from __future__ import annotations

from dataclasses import dataclass
import random

import torch

import torch.nn.functional as F

from foamsurf_data.rays import RayBundle, generate_camera_rays, generate_indexed_rays


@dataclass(frozen=True)
class TrainingBatch:
    rays: RayBundle
    target: torch.Tensor
    alpha: torch.Tensor
    camera: object
    # (height, width) of a full-image batch when it is not the camera's own size.
    image_size: tuple[int, int] | None = None

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
        jitter: bool = False,
    ) -> None:
        self.dataset = dataset
        self.rays_per_batch = int(rays_per_batch)
        # Each ray through a uniformly random point of its pixel (the target is the pixel value).
        self.jitter = bool(jitter)
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
        # Targets come from camera.foreground: no second copy of every image on the GPU.
        self.data = dataset.stacked_training_data(pixels=False)
        # Full images are rendered and compared at 1/downscale of their resolution.
        self.downscale = 1

    def next(self) -> TrainingBatch:
        if self.cameras_without_replacement:
            if not self._camera_stack:
                self._camera_stack = list(range(len(self.dataset.train)))
                random.shuffle(self._camera_stack)
            camera_index = self._camera_stack.pop()
        else:
            camera_index = random.randrange(len(self.dataset.train))
        camera = self.dataset.train[camera_index]
        if self.downscale > 1:
            return self._downscaled(camera)
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
            pixel_offsets=(torch.rand(len(pixel_indices), 2, device=pixel_indices.device) - 0.5
                           if self.jitter else None),
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

    def _downscaled(self, camera) -> TrainingBatch:
        """The whole image at 1/downscale: intrinsics scaled as in nerfstudio, ground truth
        resized bilinearly with antialiasing (as WaterSplatting)."""
        if not self.full_image:
            raise ValueError("downscaled training needs full-image batches")
        factor = self.downscale
        height, width = camera.height // factor, camera.width // factor
        intrinsics = (None if camera.intrinsics is None
                      else tuple(value / factor for value in camera.intrinsics))
        offsets = (torch.rand(height, width, 2, device=camera.origin.device) - 0.5) if self.jitter else None
        bundle = generate_camera_rays(camera.origin, camera.right, camera.down, camera.forward, width, height,
                                      camera.fov_x, camera.fov_y, intrinsics=intrinsics, pixel_offsets=offsets)
        rays = RayBundle(bundle.origins.reshape(-1, 3), bundle.directions.reshape(-1, 3))

        def resize(image):
            image = image.permute(2, 0, 1).unsqueeze(0)
            return F.interpolate(image, size=(height, width), mode="bilinear", antialias=True,
                                 align_corners=False)[0].permute(1, 2, 0).reshape(height * width, -1)

        alpha = resize(camera.alpha.reshape(camera.height, camera.width, 1))
        target = resize(camera.foreground) + self.background * (1.0 - alpha)
        return TrainingBatch(rays=rays, target=target, alpha=alpha, camera=camera, image_size=(height, width))
