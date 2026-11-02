"""Direction-dependent water light (optimization.water_direction, the _wdir suffix).

The medium cells have one colour each, seen alike from every direction, but the light in water comes from above:
open water gets brighter the higher a ray looks (Curacao MTN_1288 is ~10/255 too dark above the elevation of every
training photo). One gain per colour channel scales the backscatter of each ray,

    image = direct + exp(W d) * backscatter,   W [3 channels x 3], d = unit world ray direction,

initialised at W = 0 (the plain model). Nine parameters for the whole scene: a smooth trend in direction that
extrapolates, unlike the cells. Applied before the camera model, in training and evaluation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class WaterDirectionGain:
    def __init__(self, device="cuda", lr: float = 0.005) -> None:
        self.weights = torch.zeros(3, 3, device=device, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.weights], lr=lr)

    def gain(self, directions: torch.Tensor) -> torch.Tensor:
        """Per-channel gain [..., 3] for ray directions [..., 3] (any length)."""
        return torch.exp(F.normalize(directions, dim=-1) @ self.weights.T)

    def apply(self, image: torch.Tensor, backscatter: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        """The image with its backscatter part scaled by the gain: image + (gain - 1) backscatter."""
        return image + (self.gain(directions) - 1.0) * backscatter

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def step(self) -> None:
        self.optimizer.step()

    def state_dict(self) -> dict:
        return {"weights": self.weights.detach().clone(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        with torch.no_grad():
            self.weights.copy_(state["weights"].to(self.weights))
        self.optimizer.load_state_dict(state["optimizer"])
