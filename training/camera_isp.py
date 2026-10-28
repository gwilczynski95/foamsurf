"""The camera after the water: PPISP (Deutsch et al., CVPR 2026; third_party/ppisp, Apache-2.0) applied to the
rendered image. Per training photo an exposure offset and a colour correction, per camera (one here) a
vignetting falloff and a response curve, all learned with the scene and regularised as in the paper (mean
exposure and colour over the photos at zero, vignetting at the image centre and non-positive).
Test views are rendered with neutral exposure and colour (no controller, nothing fitted on test photos);
vignetting and the response curve apply to every image of the camera.
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

    def __init__(self, frames: list[str], iterations: int, device="cuda"):
        ppisp = load_ppisp()
        self.frame_index = {name: index for index, name in enumerate(frames)}
        config = ppisp.PPISPConfig(use_controller=False, scheduler_decay_max_steps=iterations)
        self.module = ppisp.PPISP(num_cameras=1, num_frames=len(frames), config=config).to(device)
        self.optimizers = self.module.create_optimizers()
        self.schedulers = self.module.create_schedulers(self.optimizers, iterations)

    def apply(self, rgb: torch.Tensor, coords: torch.Tensor, size: tuple[int, int], frame: str | None) -> torch.Tensor:
        """rgb [..., 3] at pixel coords [..., 2] of an image of size (height, width); frame = training photo
        name, or None for a novel view (neutral exposure and colour)."""
        height, width = size
        index = -1 if frame is None else self.frame_index[frame]
        return self.module(rgb, coords, (width, height), camera_idx=0, frame_idx=index)

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
