"""Opt-in smoothing, isolated from the source fingerprint of the running sweep."""
from dataclasses import dataclass, asdict
import math
import torch
import torch.nn.functional as F


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
