"""Opt-in regularizer renderer. Original CUDA/OptiX renderer remains untouched.

Native Gaussian-only opacity is differentiable. Medium profiles use differentiable
quadrature of the actual nearest-site field, with detached cell assignments.
"""
from dataclasses import fields
import time
import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F
from .losses import opacity_loss,overlap_profile


@torch.no_grad()
def gaussian_tau(scene,rays,hit_depths,hit_indices,samples,length):
    """Bin native OptiX planar hits using the same activated alpha as CUDA."""
    ids=hit_indices.t().long()
    depth=hit_depths.t().detach()
    valid=(ids>=0)&torch.isfinite(depth)&(depth>=0)&(depth<length)
    # Missing hits may contain arbitrary depths; sanitize BEFORE arithmetic.
    depth=torch.where(valid,depth,torch.zeros_like(depth))
    safe=ids.clamp_min(0)
    q=F.normalize(scene.quaternions.detach()[safe],dim=-1)
    w,x,y,z=q.unbind(-1)
    u=torch.stack((1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y)),-1)
    v=torch.stack((2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)),-1)
    hit=rays[:,None,:3]+depth.unsqueeze(-1)*F.normalize(rays[:,3:],dim=-1)[:,None,:]
    local=hit-scene.means.detach()[safe]
    scales=scene.log_scales.detach()[safe].exp()
    r2=((local*u).sum(-1)/scales[...,0]).square()+((local*v).sum(-1)/scales[...,1]).square()
    kappa=1+F.softplus(scene.raw_kappa.detach()[safe].squeeze(-1))
    response=torch.exp(-.5*r2.clamp_min(1e-8).pow(kappa)/kappa)
    alpha=torch.sigmoid(scene.opacity_logits.detach()[safe].squeeze(-1))*response
    tau=-torch.log1p(-alpha.clamp(0,1-1e-7))
    tau=torch.where(valid,tau,torch.zeros_like(tau))
    bins=(depth/length*samples).long().clamp(0,samples-1)
    result=rays.new_zeros((len(rays),samples))
    result.scatter_add_(1,bins,tau)
    return result


@torch.no_grad()
def nearest_ids(sites,rays,samples,length,workers=4):
    """Rebuild on EVERY query: sites move even when topology revision is constant."""
    points=sites.detach().cpu().numpy()
    r=rays.detach().cpu().double().numpy()
    direction=r[:,3:]/np.linalg.norm(r[:,3:],axis=-1,keepdims=True)
    t=(np.arange(samples)+.5)*length/samples
    queries=r[:,None,:3]+direction[:,None,:]*t[None,:,None]
    ids=cKDTree(points).query(queries.reshape(-1,3),workers=workers)[1]
    return torch.as_tensor(ids.reshape(len(rays),samples),device=sites.device,dtype=torch.long)


def compute_regularizers(render_native,rays,scene,medium,config,iteration,extent,far,seed):
    """render_native is the original bound method, not the capture wrapper."""
    ow,gw=config.weights(iteration)
    zero=medium.raw_density.new_zeros(())
    logs=dict(overlap_raw=0.,opacity_raw=0.,overlap_weight=ow,opacity_weight=gw,
              overlap_weighted=0.,opacity_weighted=0.,mask_fraction=0.,rays=0,
              concentration=0.,strength=0.,near_fraction=0.,near_mass=0.,backscatter=0.,
              gaussian_confidence=0.,gaussian_opacity=0.,hit_full_fraction=0.,seconds=0.)
    if ow==0 and gw==0:return zero,logs
    started=time.perf_counter()
    # Local deterministic RNG: does not perturb camera sampling or densification.
    generator=torch.Generator(device=rays.device).manual_seed(seed*1000003+iteration)
    count=min(config.rays,len(rays))
    pick=torch.randint(len(rays),(count,),device=rays.device,generator=generator)
    chosen=rays[pick].detach().contiguous()
    if gw:
        rendered=render_native(chosen,gaussians=scene,background=(0.,0.,0.),collect_diagnostics=False)
        gloss,mask=opacity_loss(rendered.alpha,config.opacity_threshold)
    else:
        # No Gaussian graph is even built for the overlap-only variant.
        with torch.no_grad():
            rendered=render_native(chosen,gaussians=scene,background=(0.,0.,0.),collect_diagnostics=False)
        gloss=zero;mask=(rendered.alpha>config.opacity_threshold).float().mean()
    oloss=zero
    if ow:
        length=min(far,config.range_multiplier*extent)
        if not np.isfinite(length) or length<=0:raise ValueError('Invalid regularizer ray range')
        gtau=gaussian_tau(scene,chosen,rendered.hit_depths,rendered.hit_indices,config.samples,length)
        ids=nearest_ids(medium.sites,chosen,config.samples,length,config.nearest_workers)
        oloss,parts=overlap_profile(medium.optical_coefficients[ids,3:],medium.colors[ids],
            gtau,length/config.samples,config.band_fraction,config.strength_scale)
        for key in ('concentration','strength','near_fraction','near_mass','backscatter',
                    'gaussian_confidence','gaussian_opacity'):
            logs[key]=float(parts[key].detach().mean())
    logs.update(overlap_raw=float(oloss.detach()),opacity_raw=float(gloss.detach()),
        overlap_weighted=float(ow*oloss.detach()),opacity_weighted=float(gw*gloss.detach()),
        mask_fraction=float(mask.detach()),rays=count,
        hit_full_fraction=float((rendered.hit_indices>=0).all(dim=0).float().mean()),
        seconds=time.perf_counter()-started)
    return ow*oloss+gw*gloss,logs
