from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import torch.nn.functional as F

import torch
from torchmetrics.functional.image import (
    structural_similarity_index_measure,
)


@dataclass(frozen=True)
class B0Loss:
    total: torch.Tensor
    rgb_mse: torch.Tensor
    psnr: torch.Tensor
    rgb_l1: torch.Tensor
    ssim: torch.Tensor
    gaussian_scale_regularization: torch.Tensor


def b0_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mode: str = "mse",
    dssim_weight: float = 0.2,
    image_size: tuple[int, int] | None = None,
) -> B0Loss:
    """Image objective; reg variants use detached prediction weights."""
    rgb_mse = torch.mean((prediction - target) ** 2)
    rgb_l1 = torch.mean(torch.abs(prediction - target))
    psnr = -10.0 * torch.log10(rgb_mse.clamp_min(1.0e-12))
    if mode == "mse":
        total = rgb_mse
        ssim = rgb_mse.new_tensor(float("nan"))
    elif mode == "l1":
        total = rgb_l1
        ssim = rgb_mse.new_tensor(float("nan"))
    elif mode in {"l1_dssim", "reg_l1_dssim", "l1_reg_dssim",
                  "reg_l2_dssim", "reg_l1_reg_dssim", "reg_l2_reg_dssim"}:
        if prediction.ndim == 2 and image_size is not None:
            height, width = image_size
            prediction_image = prediction.reshape(height, width, 3)
            target_image = target.reshape(height, width, 3)
        elif prediction.ndim == 3 and prediction.shape[-1] == 3:
            prediction_image = prediction
            target_image = target
        else:
            raise ValueError(
                "l1_dssim requires a full-image batch and image_size"
            )
        prediction_nchw = prediction_image.permute(2, 0, 1).unsqueeze(0)
        target_nchw = target_image.permute(2, 0, 1).unsqueeze(0)
        if mode.endswith("reg_dssim"):
            weight = (prediction_image.detach() + 1.0e-3).reciprocal()
            prediction_nchw = (prediction_image * weight).permute(2, 0, 1).unsqueeze(0)
            target_nchw = (target_image * weight).permute(2, 0, 1).unsqueeze(0)
        ssim = structural_similarity_index_measure(prediction_nchw, target_nchw, data_range=1.0)
        if mode.startswith("reg_l1"):
            pixel = (torch.abs(prediction - target) / (prediction.detach() + 1.0e-3)).mean()
        elif mode.startswith("reg_l2"):
            pixel = (((prediction - target) / (prediction.detach() + 1.0e-3)) ** 2).mean()
        else:
            pixel = rgb_l1
        total = (1.0 - dssim_weight) * pixel + dssim_weight * (1.0 - ssim)
    else:
        raise ValueError(f"unknown reconstruction loss: {mode}")
    return B0Loss(
        total=total,
        rgb_mse=rgb_mse,
        psnr=psnr,
        rgb_l1=rgb_l1,
        ssim=ssim,
        gaussian_scale_regularization=rgb_mse.new_zeros(()),
    )



@dataclass(frozen=True)
class SmoothConfig:
    attenuation: float = 0.
    backscatter: float = 0.
    color: float = 0.
    coefficient_delta: float = .1
    color_delta: float = .05
    epsilon: float = 1e-6
    ramp_steps: int = 250

    def __post_init__(self):
        for name in ('attenuation','backscatter','color'):
            value=getattr(self,name)
            if not math.isfinite(value) or value<0:raise ValueError('Invalid smoothing weight')
        for name in ('coefficient_delta','color_delta','epsilon'):
            value=getattr(self,name)
            if not math.isfinite(value) or value<=0:raise ValueError('Invalid smoothing scale')
        if type(self.ramp_steps) is not int or self.ramp_steps<0:
            raise ValueError('Invalid smoothing ramp')

    def as_dict(self):return asdict(self)

def smooth_loss(coefficients, colors, first, second, settings, iteration):
    """Mean over undirected edges AND RGB, all edge weights exactly one.

    No gradient through discrete topology. Actual activated physical coefficients
    and color are differentiable. Do not smooth raw softplus/DC parameters.
    """
    if coefficients.ndim!=2 or coefficients.shape[1]!=6 or colors.shape!=(len(coefficients),3):
        raise ValueError('Expected activated coefficients [N,6] and colors [N,3]')
    if first.shape!=second.shape or first.ndim!=1:raise ValueError('Invalid graph edges')
    zero=coefficients.sum()*0+colors.sum()*0
    a=b=c=zero
    if first.numel():
        def huber(values,delta):
            difference=values[first]-values[second]
            return F.huber_loss(difference,torch.zeros_like(difference),delta=delta,reduction='mean')
        av=coefficients[:,:3] if settings.attenuation else coefficients[:,:3].detach()
        bv=coefficients[:,3:] if settings.backscatter else coefficients[:,3:].detach()
        cv=colors if settings.color else colors.detach()
        a=huber(torch.log(av+settings.epsilon),settings.coefficient_delta)
        b=huber(torch.log(bv+settings.epsilon),settings.coefficient_delta)
        c=huber(cv,settings.color_delta)
    ramp=1. if settings.ramp_steps==0 else min(max(iteration/settings.ramp_steps,0.),1.)
    total=zero+ramp*(settings.attenuation*a+settings.backscatter*b+settings.color*c)
    return total,dict(attenuation=a,backscatter=b,color=c,ramp=ramp,edge_count=first.numel())


def ramp(iteration, start, duration):
    if iteration < start: return 0.0
    if duration == 0: return 1.0
    return min(max((iteration - start) / duration, 0.0), 1.0)

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


@dataclass(frozen=True)
class CapacityLossConfig:
    repulsion_weight: float = 0.0
    repulsion_fraction: float = 0.5
    repulsion_start: int = 1000
    repulsion_ramp: int = 3000
    direct_weight: float = 0.0
    direct_start: int = 5000
    direct_ramp: int = 5000
    alpha_threshold: float = 0.9

    def __post_init__(self):
        for name in ("repulsion_weight", "direct_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {name}")
        if not 0 < self.repulsion_fraction <= 1:
            raise ValueError("repulsion_fraction must be in (0, 1]")
        if not 0 < self.alpha_threshold < 1:
            raise ValueError("alpha_threshold must be in (0, 1)")
        for name in ("repulsion_start", "repulsion_ramp", "direct_start", "direct_ramp"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"Invalid {name}")

    def as_dict(self):
        return asdict(self)

def ramp_weight(weight, iteration, start, duration):
    if iteration < start:
        return 0.0
    return weight * (1.0 if duration == 0 else min((iteration - start) / duration, 1.0))

def repulsion_loss(sites, guard_mask, first, second, bbox_min, bbox_max, fraction):
    """Penalize only abnormally close non-guard Delaunay neighbors.

    Reference spacing follows volume^(1/3) / N^(1/3), not the current
    nearest-neighbor median, so a collapsed population cannot shrink its
    own penalty scale.
    """
    if first.numel() == 0:
        return sites.sum() * 0, sites.new_zeros(())
    active = ~(guard_mask[first] | guard_mask[second])
    if not bool(active.any()):
        return sites.sum() * 0, sites.new_zeros(())
    count = int((~guard_mask).sum().item())
    volume = (bbox_max - bbox_min).clamp_min(1.0e-8).prod()
    spacing = (volume / max(count, 1)).pow(1.0 / 3.0).detach()
    distances = (sites[first[active]] - sites[second[active]]).norm(dim=-1)
    normalized = distances / spacing.clamp_min(1.0e-8)
    short = (normalized < fraction).float().mean().detach()
    return F.relu(fraction - normalized).square().mean(), short

def masked_direct_consistency(gaussian_image, direct_image, gaussian_alpha, threshold):
    """Direct is a detached teacher; alpha is a detached selector.

    The caller must detach Gaussian geometry/opacity in the auxiliary render,
    otherwise the model can evade this appearance-only loss by deleting
    Gaussians. No selected rays gives exact zero with a valid backward path.
    """
    if gaussian_image.shape != direct_image.shape or gaussian_alpha.shape != gaussian_image.shape[:-1] + (1,):
        raise ValueError("Incompatible direct-consistency shapes")
    selected = (gaussian_alpha.detach() >= threshold).to(gaussian_image.dtype)
    difference = F.smooth_l1_loss(gaussian_image, direct_image.detach(), reduction="none")
    loss = (difference * selected).sum() / (3 * selected.sum().clamp_min(1))
    return loss, selected.mean().detach()


# Removed medium_fix options and their inactive values, still present in
# manifests and resume sidecars written before the cleanup.
_LEGACY_MEDIUM_FIX = {
    "color_mode": "per_cell", "coverage_weight": 0.0, "proximity_multiplier": 1.0,
}
_LEGACY_MEDIUM_FIX_PREFIXES = ("coverage_", "proximity_")


def drop_legacy_medium_fix(values):
    """Return medium_fix settings without inactive removed options."""

    values = dict(values)
    for key in list(values):
        if key in _LEGACY_MEDIUM_FIX or key.startswith(_LEGACY_MEDIUM_FIX_PREFIXES):
            if key in _LEGACY_MEDIUM_FIX and values[key] != _LEGACY_MEDIUM_FIX[key]:
                raise ValueError(f"medium_fix.{key}={values[key]!r} is no longer supported")
            del values[key]
    return values


@dataclass(frozen=True)
class MediumFixConfig:
    """Seabed/backscatter priors of the IUI3 joint stage, all off by default.

    Every prior uses the same random subset of ``prior_rays`` training rays and
    one Gaussian-only render of it.
    """
    prior_rays: int = 131072
    # Backscatter chroma prior: on rays where Gaussians are opaque, backscatter
    # should have the water's color (any brightness); see backscatter_chroma_loss.
    chroma_weight: float = 0.0
    chroma_start: int = 1000
    chroma_ramp: int = 1000
    chroma_margin_degrees: float = 5.0
    chroma_alpha_threshold: float = 0.5
    chroma_water_alpha: float = 0.01
    chroma_ema: float = 0.99
    # One-sided coverage: on rays already covered more than surface_coverage_threshold
    # by Gaussians, penalize the uncovered rest (1 - alpha). Open-water rays are untouched.
    surface_coverage_weight: float = 0.0
    surface_coverage_start: int = 1000
    surface_coverage_ramp: int = 1000
    surface_coverage_threshold: float = 0.5
    # Iteration from which the coverage loss is off again (None: never).
    surface_coverage_end: int | None = None
    # Water clearing, the mirror of the coverage loss: on rays whose Gaussian-only
    # alpha is still below water_clear_threshold, penalize that alpha (push it to 0).
    # Starts after the opacity-pruning ramp has removed the initial Gaussians from the water.
    water_clear_weight: float = 0.0
    water_clear_start: int = 4000
    water_clear_ramp: int = 2000
    water_clear_threshold: float = 0.3
    # Medium under the seabed: render the medium alone from just behind the depth
    # where Gaussians become opaque and penalize any backscatter produced there.
    under_surface_weight: float = 0.0
    under_surface_start: int = 1000
    under_surface_ramp: int = 1000
    under_surface_threshold: float = 0.5
    under_surface_offset: float = 0.01

    def __post_init__(self):
        if type(self.prior_rays) is not int or self.prior_rays < 1:
            raise ValueError('Invalid medium_fix.prior_rays')
        for prefix in ('chroma', 'surface_coverage', 'under_surface', 'water_clear'):
            weight = getattr(self, f'{prefix}_weight')
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f'Invalid medium_fix.{prefix}_weight')
            for name in (f'{prefix}_start', f'{prefix}_ramp'):
                if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                    raise ValueError(f'Invalid medium_fix.{name}')
        if not 0 <= self.chroma_margin_degrees < 90:
            raise ValueError('medium_fix.chroma_margin_degrees must be in [0, 90)')
        if not 0 < self.chroma_water_alpha < self.chroma_alpha_threshold < 1:
            raise ValueError('medium_fix needs 0 < chroma_water_alpha < chroma_alpha_threshold < 1')
        if not 0 <= self.chroma_ema < 1:
            raise ValueError('medium_fix.chroma_ema must be in [0, 1)')
        for prefix in ('surface_coverage', 'under_surface', 'water_clear'):
            if not 0 < getattr(self, f'{prefix}_threshold') < 1:
                raise ValueError(f'medium_fix.{prefix}_threshold must be in (0, 1)')
        end = self.surface_coverage_end
        if end is not None and (type(end) is not int or end <= self.surface_coverage_start):
            raise ValueError('medium_fix.surface_coverage_end must be an iteration after surface_coverage_start')
        if (self.water_clear_weight > 0 and self.surface_coverage_weight > 0
                and self.water_clear_threshold >= self.surface_coverage_threshold):
            # Both losses would pull the same rays in opposite directions.
            raise ValueError('medium_fix.water_clear_threshold must be below surface_coverage_threshold')
        if not math.isfinite(self.under_surface_offset) or self.under_surface_offset < 0:
            raise ValueError('medium_fix.under_surface_offset must be non-negative')

    def as_dict(self):
        return asdict(self)

    def chroma_weight_at(self, iteration):
        return ramp_weight(self.chroma_weight, iteration, self.chroma_start, self.chroma_ramp)

    def surface_coverage_weight_at(self, iteration):
        if self.surface_coverage_end is not None and iteration >= self.surface_coverage_end:
            return 0.0
        return ramp_weight(self.surface_coverage_weight, iteration, self.surface_coverage_start,
                           self.surface_coverage_ramp)

    def water_clear_weight_at(self, iteration):
        return ramp_weight(self.water_clear_weight, iteration, self.water_clear_start, self.water_clear_ramp)

    def under_surface_weight_at(self, iteration):
        return ramp_weight(self.under_surface_weight, iteration, self.under_surface_start,
                           self.under_surface_ramp)


def water_direction(backscatter, gaussian_alpha, water_alpha):
    """Unit RGB direction of backscatter summed over open-water rays, or None."""
    water = gaussian_alpha.reshape(-1) < water_alpha
    if not bool(water.any()):
        return None
    total = backscatter.reshape(-1, 3)[water].detach().sum(0)
    norm = total.norm()
    return None if float(norm) <= 1e-8 else total / norm


def backscatter_chroma_loss(backscatter, gaussian_alpha, water, alpha_threshold=0.5, margin_degrees=5.0):
    """Hue mismatch between backscatter and water on rays that hit opaque Gaussians.

    Per ray: sin^2 of the angle between the RGB backscatter vector and the
    water direction, minus the margin, clamped at 0. It is invariant to the
    backscatter's brightness, so darkening the haze cannot lower it; rays are
    weighted by their detached brightness so near-black, ill-defined hues do
    not dominate. Returns (loss, mean angle in degrees over selected rays).
    """
    backscatter = backscatter.reshape(-1, 3)
    selected = gaussian_alpha.reshape(-1).detach() > alpha_threshold
    zero = backscatter.sum() * 0
    if water is None or not bool(selected.any()):
        return zero, float('nan')
    values = backscatter[selected]
    norm = values.norm(dim=-1).clamp_min(1e-8)
    cosine = ((values @ water.to(values)) / norm).clamp(-1, 1)
    sine2 = (1 - cosine * cosine).clamp_min(0)
    excess = (sine2 - math.sin(math.radians(margin_degrees)) ** 2).clamp_min(0)
    weight = norm.detach()
    weight = weight / weight.mean().clamp_min(1e-8)
    angle = float(torch.rad2deg(torch.acos(cosine.detach())).mul(weight).mean())
    return (weight * excess).mean(), angle


def surface_coverage_loss(alpha, threshold=0.5):
    """Mean (1 - A) over rays whose Gaussian-only alpha A already exceeds ``threshold``.

    Pushes partly covered seabed rays to full coverage; rays below the
    threshold (open water, silhouettes) get no gradient. Returns (loss, fraction).
    """
    alpha = alpha.reshape(-1)
    selected = alpha.detach() > threshold
    if not bool(selected.any()):
        return alpha.sum() * 0, 0.0
    return (1 - alpha[selected]).mean(), float(selected.float().mean())


def water_clear_loss(alpha, threshold=0.3):
    """Mean A over rays whose Gaussian-only alpha A is still below ``threshold``.

    Mirror of surface_coverage_loss: pushes Gaussians off open-water rays;
    rays at or above the threshold (seabed, its silhouette) get no gradient.
    Returns (loss, fraction of rays selected).
    """
    alpha = alpha.reshape(-1)
    selected = alpha.detach() < threshold
    if not bool(selected.any()):
        return alpha.sum() * 0, 0.0
    return alpha[selected].mean(), float(selected.float().mean())


def under_surface_loss(backscatter):
    """Mean backscatter intensity produced behind the opaque Gaussian surface."""
    backscatter = backscatter.reshape(-1, 3)
    if backscatter.shape[0] == 0:
        return backscatter.sum() * 0
    return backscatter.mean()
