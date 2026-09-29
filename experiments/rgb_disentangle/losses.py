"""Differentiable counterparts of the CPU ray heatmaps; no trainer side effects."""
from dataclasses import dataclass, asdict
import math
import torch


@dataclass(frozen=True)
class ConflictConfig:
    overlap_weight: float = 0.
    opacity_weight: float = 0.
    overlap_start: int = 1000
    opacity_start: int = 1000
    overlap_ramp: int = 1000
    opacity_ramp: int = 1000
    opacity_threshold: float = .1
    rays: int = 128
    samples: int = 1024
    range_multiplier: float = 8.
    band_fraction: float = .02
    strength_scale: float = .05
    nearest_workers: int = 4

    def __post_init__(self):
        for name in ('overlap_weight','opacity_weight'):
            x=getattr(self,name)
            if not math.isfinite(x) or x<0:raise ValueError(f'Invalid {name}')
        for name in ('overlap_start','opacity_start','overlap_ramp','opacity_ramp'):
            x=getattr(self,name)
            if type(x) is not int or x<0:raise ValueError(f'Invalid {name}')
        for name in ('rays','samples','nearest_workers'):
            x=getattr(self,name)
            if type(x) is not int or x<1:raise ValueError(f'Invalid {name}')
        if self.samples<2:raise ValueError('At least two depth samples required')
        for name in ('range_multiplier','band_fraction','strength_scale'):
            x=getattr(self,name)
            if not math.isfinite(x) or x<=0:raise ValueError(f'Invalid {name}')
        if not 0<self.band_fraction<=1:raise ValueError('Invalid band_fraction')
        if not 0<=self.opacity_threshold<1:raise ValueError('Invalid opacity_threshold')

    def as_dict(self):return asdict(self)

    def weights(self,iteration):
        return (self.overlap_weight*ramp(iteration,self.overlap_start,self.overlap_ramp),
                self.opacity_weight*ramp(iteration,self.opacity_start,self.opacity_ramp))


def ramp(iteration,start,duration):
    if iteration<start:return 0.
    if duration==0:return 1.
    return min(max((iteration-start)/duration,0.),1.)


def opacity_loss(alpha,threshold=.1):
    """Conditional mean of opacity VALUES, not of the threshold predicate."""
    mask=(alpha.detach()>threshold).to(alpha.dtype)
    return ((1-alpha)*mask).sum()/mask.sum().clamp_min(1),mask.mean()


def concentration(mass):
    total=mass.sum(-1)
    distribution=mass/total.clamp_min(1e-12).unsqueeze(-1)
    value=1-1/(mass.shape[-1]*distribution.square().sum(-1).clamp_min(1e-12))
    return torch.where(total>1e-12,value.clamp(0,1),torch.zeros_like(value))


def overlap_profile(backscatter,colors,gaussian_tau,dt,band_fraction=.02,strength_scale=.05):
    """Mean suspect_overlap, with ALL Gaussian-derived terms detached.

    backscatter/colors: [R,S,3], gaussian_tau: [R,S]. Uniform depth bins.
    Only medium b and B differentiate; not a, sites or hard cell assignment.
    """
    if backscatter.ndim!=3 or backscatter.shape[-1]!=3 or colors.shape!=backscatter.shape:
        raise ValueError('Expected b and B [rays,samples,3]')
    if gaussian_tau.shape!=backscatter.shape[:2]:raise ValueError('Invalid Gaussian profile')
    if dt<=0 or band_fraction<=0 or strength_scale<=0:raise ValueError('Invalid profile scale')
    gtau=gaussian_tau.detach()
    gsum=gtau.cumsum(-1)
    tg=torch.exp(-(gsum-.5*gtau))
    btau=backscatter*dt
    tb=torch.exp(-(btau.cumsum(1)-btau))
    mass=(tg.unsqueeze(-1)*tb*colors*(-torch.expm1(-btau))).mean(-1)
    gm=torch.exp(-(gsum-gtau))*(-torch.expm1(-gtau))
    confidence=gm.sum(-1)
    n=mass.shape[1]
    depth=(torch.arange(n,device=mass.device,dtype=mass.dtype)+.5)*dt
    gd=(gm*depth).sum(-1)/confidence.clamp_min(1e-20)
    variance=(gm*(depth-gd.unsqueeze(-1)).square()).sum(-1)/confidence.clamp_min(1e-20)
    band=band_fraction*n*dt
    surface_confidence=confidence*torch.exp(-variance/(band*band))
    near=((depth-gd.unsqueeze(-1)).abs()<=band).to(mass.dtype)
    total=mass.sum(-1)
    local_mass=(mass*near).sum(-1)
    fraction=local_mass/total.clamp_min(1e-20)
    c=concentration(mass)
    strength=total/(total+strength_scale)
    score=c*strength*fraction*surface_confidence
    return score.mean(),dict(score=score,concentration=c,strength=strength,
        near_fraction=fraction,near_mass=local_mass,backscatter=total,
        gaussian_confidence=surface_confidence,gaussian_opacity=confidence,
        gaussian_depth=gd,mass=mass)
