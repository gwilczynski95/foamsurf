"""Directional light in the water: a Henyey-Greenstein phase function on the backscatter (optimization.water_phase,
the _wphase suffix).

Under water the light comes from above and the particles scatter it mostly forward, so open water is brighter the
more a ray looks toward the light (Curacao MTN_1288 is ~4-10/255 too dark above the elevation of every training
photo; the medium cells have one colour each, seen alike from every direction). With single scattering of a
directional light the backscatter of each ray is scaled by the phase function relative to isotropic scattering,

    image = direct + gain(d) * backscatter,   gain = (1 - g^2) / (1 + g^2 - 2 g cos theta)^(3/2),

cos theta = d . l, d = unit ray direction, l = unit direction toward the light (learned, started at the cameras'
mean "up"), g in (-1, 1) per colour channel (the anisotropy, started at 0 = isotropic = the plain model). Four
parameters per scene plus one per channel, shared by every cell: a physical trend in direction that extrapolates
beyond the training views, unlike the cells.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

MAX_ANISOTROPY = 0.95


class WaterPhase:
    def __init__(self, up: torch.Tensor, device="cuda", lr: float = 0.01, fixed_light: bool = False) -> None:
        # fixed_light (_wphaseup): the light stays at `up` (sun overhead), only g is learned
        self.toward_light = F.normalize(up.reshape(3).float(), dim=0).to(device).clone().requires_grad_(not fixed_light)
        self.raw_anisotropy = torch.zeros(3, device=device, requires_grad=True)
        self.optimizer = torch.optim.Adam(([] if fixed_light else [self.toward_light]) + [self.raw_anisotropy], lr=lr)

    @property
    def anisotropy(self) -> torch.Tensor:
        """g per colour channel, in (-MAX_ANISOTROPY, MAX_ANISOTROPY)."""
        return MAX_ANISOTROPY * torch.tanh(self.raw_anisotropy)

    def gain(self, directions: torch.Tensor) -> torch.Tensor:
        """Henyey-Greenstein phase function over the isotropic one, [..., 3], for ray directions [..., 3]."""
        cosine = (F.normalize(directions, dim=-1) @ F.normalize(self.toward_light, dim=0))[..., None]
        g = self.anisotropy
        return (1.0 - g * g) / (1.0 + g * g - 2.0 * g * cosine).clamp_min(1e-6) ** 1.5

    def apply(self, image: torch.Tensor, backscatter: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        """The image with its backscatter part scaled by the phase gain: image + (gain - 1) backscatter."""
        return image + (self.gain(directions) - 1.0) * backscatter

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def step(self) -> None:
        self.optimizer.step()

    def state_dict(self) -> dict:
        return {"toward_light": self.toward_light.detach().clone(), "raw_anisotropy": self.raw_anisotropy.detach().clone(),
                "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        with torch.no_grad():
            self.toward_light.copy_(state["toward_light"].to(self.toward_light))
            self.raw_anisotropy.copy_(state["raw_anisotropy"].to(self.raw_anisotropy))
        self.optimizer.load_state_dict(state["optimizer"])
