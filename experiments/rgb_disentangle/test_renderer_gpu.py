"""Required real CUDA tests: never skip these on CPU; invoke through the gate."""
from dataclasses import fields
import numpy as np
import pytest
import torch
from foamsurf import NativeHybridRenderer,GaussianScene
from foamsurf.types import RGBVoronoiScene
from experiments.rgb_disentangle.losses import ConflictConfig,overlap_profile
from experiments.rgb_disentangle.renderer import compute_regularizers,gaussian_tau


class Medium:
    def __init__(self):
        self.sites=torch.tensor([[0.,0,1],[0,0,3]],device='cuda',requires_grad=True)
        self.raw_density=torch.tensor([[-1.]*6,[-.4]*6],device='cuda',requires_grad=True)
        self.raw_color=torch.tensor([[.2,.3,.4],[.5,.1,.3]],device='cuda',requires_grad=True)
    @property
    def optical_coefficients(self):return torch.nn.functional.softplus(self.raw_density)
    @property
    def colors(self):return (.5+.28209479177387814*self.raw_color).clamp_min(0)
    def scene(self):
        a,b=self.optical_coefficients.split(3,-1)
        return RGBVoronoiScene(self.sites,a.mean(-1,keepdim=True),self.colors,a,b)


@pytest.fixture(scope='module')
def renderer():
    assert torch.cuda.is_available(),'Required test: submit test.sbatch'
    return NativeHybridRenderer(max_batch_size=16,far=5.,transmittance_threshold=0.)


def scene(renderer):
    g=GaussianScene(*[torch.tensor(x,device='cuda',dtype=torch.float32,requires_grad=True) for x in
        ([[.04,-.03,2.4]],[[-.9,-1.1]],[[.97,.08,-.12,.05]],[[.8,.2,.1]],[[1.2]],[[.2]])])
    renderer.set_gaussian_geometry(g)
    m=Medium();ms=m.scene()
    renderer.set_voronoi_geometry(ms,torch.tensor([1,0],device='cuda',dtype=torch.uint32),
                                 torch.tensor([0,1,2],device='cuda',dtype=torch.uint32))
    rays=torch.tensor([[0.,0,0,.02,.01,1],[.1,-.08,0,-.01,.02,1]],device='cuda')
    return rays,g,m


@pytest.mark.parametrize('kind',['overlap','opacity','both'])
def test_routed_gradients(renderer,kind):
    rays,g,m=scene(renderer)
    cfg=ConflictConfig(overlap_weight=1. if kind!='opacity' else 0.,
        opacity_weight=1. if kind!='overlap' else 0.,overlap_start=0,opacity_start=0,
        overlap_ramp=0,opacity_ramp=0,rays=2,samples=256,range_multiplier=5.,band_fraction=.1)
    loss,logs=compute_regularizers(renderer.render,rays,g,m,cfg,1,1.,5.,42)
    assert loss>0 and torch.isfinite(loss)
    loss.backward()
    params=[getattr(g,f.name) for f in fields(g)]
    if kind=='overlap':assert all(p.grad is None or not p.grad.any() for p in params)
    else:assert g.opacity_logits.grad is not None and g.opacity_logits.grad.abs().sum()>0
    if kind=='opacity':assert m.raw_density.grad is None and m.raw_color.grad is None
    else:
        assert m.raw_density.grad[:,3:].abs().sum()>0 and m.raw_color.grad.abs().sum()>0
        assert not m.raw_density.grad[:,:3].any()  # a does not enter this backscatter loss
    assert m.sites.grad is None  # nearest-cell assignment is intentionally detached
    assert logs['hit_full_fraction']==0


def test_hit_profile_agrees_with_native_opacity(renderer):
    rays,g,m=scene(renderer)
    result=renderer.render(rays,gaussians=g,background=(0.,0.,0.))
    tau=gaussian_tau(g,rays,result.hit_depths,result.hit_indices,1024,5.)
    torch.testing.assert_close(-torch.expm1(-tau.sum(-1)),result.alpha.flatten(),atol=3e-6,rtol=3e-5)


def test_additional_render_does_not_corrupt_primary_backward(renderer):
    rays,g,m=scene(renderer)
    params=[g.means,g.opacity_logits,m.raw_density,m.raw_color,m.sites]
    original=renderer.render(rays,g,m.scene(),background=(.1,.2,.3))
    expected=torch.autograd.grad(original.image.sum(),params)
    actual=renderer.render(rays,g,m.scene(),background=(.1,.2,.3))
    cfg=ConflictConfig(overlap_weight=.1,opacity_weight=.1,overlap_start=0,opacity_start=0,
                       overlap_ramp=0,opacity_ramp=0,rays=2,samples=64)
    extra,_=compute_regularizers(renderer.render,rays,g,m,cfg,2,1.,5.,42)
    observed=torch.autograd.grad(actual.image.sum(),params)
    for x,y in zip(expected,observed):torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-6)


def test_cpu_cuda_profile_and_gradients_match():
    assert torch.cuda.is_available()
    torch.manual_seed(1)
    cpu=[torch.rand(3,128,3,dtype=torch.double,requires_grad=True) for _ in range(2)]
    gt=torch.zeros(3,128,dtype=torch.double);gt[:,60]=3
    loss,_=overlap_profile(*cpu,gt,.1,.1)
    gradients=torch.autograd.grad(loss,cpu)
    gpu=[x.detach().cuda().requires_grad_() for x in cpu]
    gl,_=overlap_profile(*gpu,gt.cuda(),.1,.1)
    gg=torch.autograd.grad(gl,gpu)
    torch.testing.assert_close(gl.cpu(),loss)
    for x,y in zip(gradients,gg):torch.testing.assert_close(x,y.cpu())
