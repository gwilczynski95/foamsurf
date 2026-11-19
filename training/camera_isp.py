"""The camera after the water: PPISP (Deutsch et al.; third_party/ppisp, Apache-2.0) applied to the
rendered image. Per training photo an exposure offset and a colour correction, per camera (one here) a
vignetting falloff and a response curve, all learned with the scene and regularised as in the paper (mean
exposure and colour over the photos at zero, vignetting at the image centre and non-positive).
Novel (test) views get the exposure and colour of their nearest training photos: inverse-distance weighted
over the `neighbours` nearest camera centres (0 = neutral, the mean over the photos). The exposure of a
handheld sequence drifts with the camera's path (Panama: -0.14..+0.23 EV), and one neighbour alone can be
an outlier (Curaçao MTN_1289: +0.45 EV). Only training parameters and the test pose are used, no controller
and nothing fitted on test photos. Vignetting and the response curve apply to every image of the camera.
"""
import sys
from pathlib import Path

import torch

PPISP_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "ppisp"


def load_ppisp():
    """The vendored ppisp package, its CUDA extension compiled at first use (TORCH_CUDA_ARCH_LIST)."""
    if "ppisp_cuda" not in sys.modules:
        from torch.utils.cpp_extension import load
        source = PPISP_ROOT / "ppisp"
        sys.modules["ppisp_cuda"] = load(
            name="ppisp_cuda", sources=[str(source / "ext.cpp"), str(source / "src" / "ppisp_impl.cu")],
            extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
    if str(PPISP_ROOT) not in sys.path:
        sys.path.insert(0, str(PPISP_ROOT))
    import ppisp
    return ppisp


def pixel_grid(height: int, width: int, device) -> torch.Tensor:
    """Pixel-centre coordinates (x + 0.5, y + 0.5) of a full image, [H * W, 2] in row-major order."""
    y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    return torch.stack((x, y), -1).reshape(-1, 2).float() + 0.5


class CameraISP:
    """PPISP for one camera and the given training photos (by name)."""

    def __init__(self, frames: list[str], iterations: int, device="cuda", centres: torch.Tensor | None = None,
                 neighbours: int = 0):
        ppisp = load_ppisp()
        self.apply_parameters = ppisp.ppisp_apply
        self.frame_index = {name: index for index, name in enumerate(frames)}
        if neighbours > 0 and (centres is None or len(centres) != len(frames)):
            raise ValueError("novel-view neighbours need one camera centre per training photo")
        self.centres = None if centres is None else centres.reshape(-1, 3).float().to(device)
        self.neighbours = min(neighbours, len(frames))
        config = ppisp.PPISPConfig(use_controller=False, scheduler_decay_max_steps=iterations)
        self.module = ppisp.PPISP(num_cameras=1, num_frames=len(frames), config=config).to(device)
        self.optimizers = self.module.create_optimizers()
        self.schedulers = self.module.create_schedulers(self.optimizers, iterations)

    def apply(self, rgb: torch.Tensor, coords: torch.Tensor, size: tuple[int, int], frame: str | None,
              centre: torch.Tensor | None = None) -> torch.Tensor:
        """rgb [..., 3] at pixel coords [..., 2] of an image of size (height, width); frame = training photo
        name, or None for a novel view with camera centre `centre` (neutral without neighbours)."""
        height, width = size
        if frame is not None:
            return self.module(rgb, coords, (width, height), camera_idx=0, frame_idx=self.frame_index[frame])
        if self.neighbours == 0:
            return self.module(rgb, coords, (width, height), camera_idx=0, frame_idx=-1)
        weights = self.novel_view_weights(centre)
        exposure = (weights * self.module.exposure_params).sum().reshape(1)
        colour = (weights[:, None] * self.module.color_params).sum(0, keepdim=True)
        return self.apply_parameters(exposure, self.module.vignetting_params, colour, self.module.crf_params,
                                     rgb, coords, width, height, 0, 0)

    def novel_view_weights(self, centre: torch.Tensor) -> torch.Tensor:
        """Weights over the training photos: 1 / distance over the nearest `neighbours` camera centres."""
        distance = (self.centres - centre.reshape(1, 3).to(self.centres)).norm(dim=1)
        nearest = distance.argsort()[:self.neighbours]
        weights = torch.zeros_like(distance)
        weights[nearest] = 1 / distance[nearest].clamp(min=1e-6)
        return weights / weights.sum()

    def regularization(self) -> torch.Tensor:
        return self.module.get_regularization_loss()

    def zero_grad(self) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()
        for scheduler in self.schedulers:
            scheduler.step()

    def state_dict(self) -> dict:
        return {"module": self.module.state_dict(), "optimizers": [o.state_dict() for o in self.optimizers],
                "schedulers": [s.state_dict() for s in self.schedulers], "frames": list(self.frame_index)}

    def load_state_dict(self, state: dict) -> None:
        if state["frames"] != list(self.frame_index):
            raise ValueError("camera ISP checkpoint was trained on other photos")
        self.module.load_state_dict(state["module"])
        for optimizer, saved in zip(self.optimizers, state["optimizers"]):
            optimizer.load_state_dict(saved)
        for scheduler, saved in zip(self.schedulers, state["schedulers"]):
            scheduler.load_state_dict(saved)
