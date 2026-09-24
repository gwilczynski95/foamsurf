"""Multi-view pixel sampling used exclusively by the Voronoi initialization."""
from dataclasses import dataclass
import torch
from foamsurf_data.rays import generate_indexed_rays


@dataclass(frozen=True)
class MixedBatch:
    rays: object
    target: torch.Tensor
    alpha: torch.Tensor
    camera: object = None


class MixedRaySampler:
    """Uniform pixels with replacement across all TRAIN views.

    A single index tensor selects ray, RGB and alpha. Uses the checkpointed
    Torch RNG; no prefetch RNG state or independently shuffled target buffers.
    """
    def __init__(self, dataset, rays_per_batch, background):
        if rays_per_batch < 1 or not dataset.train:
            raise ValueError("mixed sampler requires rays and train cameras")
        self.dataset = dataset
        self.rays_per_batch = int(rays_per_batch)
        camera = dataset.train[0]
        self.width, self.height = camera.width, camera.height
        if any((c.width, c.height) != (self.width, self.height) for c in dataset.train):
            raise ValueError("mixed sampler requires equal image sizes")
        self.data = dataset.stacked_training_data()
        self.background = torch.as_tensor(background, device=dataset.device, dtype=torch.float32)
        self._camera_stack = []  # checkpoint compatibility; no stack used

    def batch_from_indices(self, indices):
        data = self.data
        rays = generate_indexed_rays(
            indices, data["origins"], data["rights"], data["downs"],
            data["forwards"], data["fov_x"], data["fov_y"], self.width, self.height, intrinsics=data.get("intrinsics"))
        alpha = data["alpha"][indices]
        target = data["foreground"][indices] + self.background * (1 - alpha)
        return MixedBatch(rays, target, alpha)

    def next(self):
        count = len(self.dataset.train) * self.width * self.height
        indices = torch.randint(count, (self.rays_per_batch,), device=self.dataset.device)
        return self.batch_from_indices(indices)
