"""Local optical occupancy, differentiated through native Voronoi boundaries."""
from dataclasses import asdict, dataclass
import math

import torch

from experiments.rgb_disentangle.losses import ramp


@dataclass(frozen=True)
class SurfaceConfig:
    weight: float = 0.0
    start: int = 1000
    ramp_steps: int = 1000
    rays: int = 128
    range_multiplier: float = 8.0
    band_fraction: float = 0.02
    optical_scale: float = 1.0

    def __post_init__(self):
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError('Invalid overlap weight')
        for name in ('start', 'ramp_steps', 'rays'):
            if type(getattr(self, name)) is not int or getattr(self, name) < (1 if name == 'rays' else 0):
                raise ValueError(f'Invalid {name}')
        for name in ('range_multiplier', 'band_fraction', 'optical_scale'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'Invalid {name}')
        if self.band_fraction > 0.5:
            raise ValueError('Band must fit inside the ray interval')

    def as_dict(self):
        return asdict(self)

    def weight_at(self, iteration):
        return self.weight * ramp(iteration, self.start, self.ramp_steps)


def optical_density(coefficients, colors, scale=1.0):
    """k = mean_RGB(a + b + b*B) / (3*scale), in inverse scene units.

    Independent a and b terms prevent reducing B alone from removing the loss.
    This is an auxiliary occupancy field, NOT a new physical renderer model.
    """
    if coefficients.ndim != 2 or coefficients.shape[1] != 6 or colors.shape != (len(coefficients), 3):
        raise ValueError('Expected [N,6] optical coefficients and [N,3] colors')
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid optical scale')
    a, b = coefficients.split(3, dim=-1)
    return ((a + b + b * colors).mean(-1, keepdim=True) / (3 * scale)).contiguous()


@torch.no_grad()
def event_moments(depth, alpha, valid, band):
    """Front-to-back Gaussian weights; exact event depths, no depth bins."""
    depth = torch.where(valid, depth, torch.full_like(depth, float('inf')))
    depth, order = depth.sort(dim=-1)
    alpha = torch.where(valid, alpha, torch.zeros_like(alpha)).gather(-1, order).clamp(0, 1 - 1e-7)
    depth = torch.where(torch.isfinite(depth), depth, torch.zeros_like(depth))
    survival = torch.cumprod(1 - alpha, dim=-1)
    incoming = torch.cat((torch.ones_like(survival[:, :1]), survival[:, :-1]), dim=-1)
    weights = incoming * alpha
    opacity = weights.sum(-1)
    mean = (weights * depth).sum(-1) / opacity.clamp_min(1e-20)
    variance = (weights * (depth - mean[:, None]).square()).sum(-1) / opacity.clamp_min(1e-20)
    confidence = opacity * torch.exp(-variance / (band * band))
    return mean, confidence, opacity


@torch.no_grad()
def local_rays(rays, depth, band, length):
    """Constant-length window translated at near/far limits to stay in [0,L]."""
    direction = torch.nn.functional.normalize(rays[:, 3:], dim=-1)
    center = depth.clamp(min=band, max=length - band)
    origin = rays[:, :3] + (center - band)[:, None] * direction
    return torch.cat((origin, direction), dim=-1).contiguous()


def occupancy_loss(alpha, confidence):
    # Mean over ALL sampled pixels, including empty Gaussian rays.
    return (alpha.reshape(-1) * confidence.detach()).mean()
