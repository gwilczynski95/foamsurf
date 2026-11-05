"""Water light beyond the training directions (optimization.water_extrapolation, the _wext suffix), evaluation only.

Open water brightens toward the surface: light comes from above and the particles scatter it mostly forward. The
medium cells reproduce that trend only over the directions the training photos saw; a novel view looking higher
(Curacao MTN_1288: up to ~4 deg above every training photo) gets the colour of the highest seen direction, too dark
by 7-9/255 there. The trend itself is measured on the training photos: their open-water pixels (rays where the
trained Gaussians cover < 2%) are fitted with a Henyey-Greenstein phase function of the angle to the vertical,

    I_c = s_{photo,c} * HG_c(cos theta),   HG = (1 - g^2) / (1 + g^2 - 2 g cos theta)^(3/2),

cos theta = d . up (up = the cameras' mean "up": light from above), g per colour channel, a free scale per photo and
channel absorbing exposure and the level. At evaluation the backscatter of a ray is scaled by

    HG(cos theta) / HG(min(cos theta, cos_max)),   cos_max = the highest training open-water direction,

which is exactly 1 inside the training range (the cells stay as trained) and continues the measured trend beyond it.
Nothing is trained: the fit uses only the training photos and the trained model.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

MAX_ANISOTROPY = 0.95


def henyey_greenstein(cosine: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """HG phase over the isotropic one, [..., 3] for cosines [...] and g [3]."""
    return (1.0 - g * g) / (1.0 + g * g - 2.0 * g * cosine[..., None]).clamp_min(1e-6) ** 1.5


class WaterExtrapolation:
    def __init__(self, up: torch.Tensor, anisotropy: torch.Tensor, cos_max: float) -> None:
        self.up = F.normalize(up.reshape(3).float(), dim=0)
        self.anisotropy = anisotropy.reshape(3).float().to(self.up)
        self.cos_max = float(cos_max)

    @classmethod
    def fit(cls, up: torch.Tensor, directions: torch.Tensor, colours: torch.Tensor, photo: torch.Tensor,
            iterations: int = 4000, lr: float = 0.02) -> "WaterExtrapolation":
        """Least squares in log space over open-water samples: unit-or-not ray directions [n, 3], photo colours
        [n, 3] (> 0), the index of their photo [n]."""
        up = F.normalize(up.reshape(3).float(), dim=0).to(directions)
        cosine = F.normalize(directions, dim=-1) @ up
        target = torch.log(colours.clamp_min(1e-4))
        count = int(photo.max()) + 1
        with torch.enable_grad():
            raw = torch.zeros(3, device=directions.device, requires_grad=True)
            log_scale = target.mean(0).repeat(count, 1).clone().requires_grad_(True)
            optimizer = torch.optim.Adam([raw, log_scale], lr=lr)
            for _ in range(iterations):
                optimizer.zero_grad()
                g = MAX_ANISOTROPY * torch.tanh(raw)
                ((log_scale[photo] + torch.log(henyey_greenstein(cosine, g)) - target) ** 2).mean().backward()
                optimizer.step()
        return cls(up, (MAX_ANISOTROPY * torch.tanh(raw)).detach(), float(cosine.max()))

    def ratio(self, directions: torch.Tensor) -> torch.Tensor:
        """[..., 3]: 1 inside the training range, the HG trend beyond it."""
        cosine = F.normalize(directions, dim=-1) @ self.up.to(directions)
        g = self.anisotropy.to(directions)
        return henyey_greenstein(cosine, g) / henyey_greenstein(cosine.clamp(max=self.cos_max), g)

    def apply(self, image: torch.Tensor, backscatter: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        return image + (self.ratio(directions) - 1.0) * backscatter

    def state_dict(self) -> dict:
        return {"up": self.up.tolist(), "anisotropy": self.anisotropy.tolist(), "cos_max": self.cos_max}
