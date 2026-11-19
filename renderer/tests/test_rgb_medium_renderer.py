"""RGB medium (per-channel attenuation/backscatter) vs the independent autograd oracle.

The six-pass CUDA composition (foamsurf.rgb_medium.compose_cuda) must match
reference_segments built from the PyTorch HybridRenderer geometry, in values
and in gradients for every Voronoi and Gaussian tensor.
"""
from dataclasses import replace
import pytest
import torch
from foamsurf import NativeHybridRenderer, GaussianScene, VoronoiScene, HybridRenderer
from foamsurf.types import RGBVoronoiScene
from foamsurf.rgb_medium import reference_segments

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="RGB medium renderer requires CUDA")


def reference(rays, medium, gaussian=None, far=5.):
    oracle=HybridRenderer(far=far)
    result=[]
    for ray in rays:
        o,d=ray[:3],torch.nn.functional.normalize(ray[3:],dim=0)
        cuts=[ray.new_tensor(0.),ray.new_tensor(far)]
        tags=[-1,-1]
        cross=oracle._voronoi_crossings(o,d,medium.sites)
        cuts.extend(cross.unbind());tags.extend([-1]*len(cross))
        if gaussian is not None:
            depths,alphas,valid=oracle._gaussian_hits(o,d,gaussian)
            ids=valid.nonzero().flatten();cuts.extend(depths[ids].unbind());tags.extend(ids.tolist())
        depths=torch.stack(cuts);order=depths.argsort(stable=True);depths=depths[order]
        lengths=depths[1:]-depths[:-1]
        mid=o+(depths[1:]+depths[:-1])[:,None]*.5*d
        cells=(mid[:,None]-medium.sites[None]).square().sum(-1).argmin(-1)
        a=[];c=[]
        for k in order[1:].tolist():
            tag=tags[k]
            a.append(alphas[tag] if tag>=0 else ray.new_zeros(()))
            c.append(gaussian.colors[tag] if tag>=0 else ray.new_zeros(3))
        result.append(reference_segments(lengths,medium.attenuation[cells],medium.backscatter[cells],
            medium.colors[cells],torch.stack(a),torch.stack(c),ray.new_tensor([.1,.2,.3]))[0])
    return torch.stack(result)


@pytest.fixture(scope='module')
def renderer():
    return NativeHybridRenderer(max_batch_size=16,far=5.,transmittance_threshold=0.)


def scene():
    torch.manual_seed(21)
    sites=(torch.randn(8,3,device='cuda')*2).requires_grad_()
    a=(torch.rand(8,3,device='cuda')*.4+.01).requires_grad_()
    b=(torch.rand(8,3,device='cuda')*.3+.02).requires_grad_()
    B=torch.rand(8,3,device='cuda').requires_grad_()
    return RGBVoronoiScene(sites,a.mean(-1,keepdim=True),B,a,b)


def synchronize(renderer,m,g=None):
    n=len(m.sites);ids=torch.arange(n,device='cuda');adj=ids.repeat(n,1)
    adj=adj[adj!=ids[:,None]].to(torch.uint32).contiguous()
    offsets=(torch.arange(n+1,device='cuda')*(n-1)).to(torch.uint32)
    renderer.set_voronoi_geometry(m,adj,offsets)
    if g is not None:renderer.set_gaussian_geometry(g)


def gaussian():
    return GaussianScene(*[torch.tensor(x,device='cuda',dtype=torch.float32).requires_grad_() for x in
        ([[.04,-.03,2.4]],[[-.9,-1.1]],[[.97,.08,-.12,.05]],[[.8,.2,.1]],[[1.2]],[[.2]])])


@requires_cuda
@pytest.mark.parametrize('with_gaussian',[False,True])
def test_cuda_against_independent_reference_and_gradients(renderer,with_gaussian):
    m=scene();g=gaussian() if with_gaussian else None;synchronize(renderer,m,g)
    rays=torch.tensor([[0,0,0,.03,.02,1],[.1,-.08,0,-.02,.01,1]],device='cuda',dtype=torch.float32)
    actual=renderer.render(rays,g,m,background=(.1,.2,.3),collect_diagnostics=True)
    expected=reference(rays,m,g)
    torch.testing.assert_close(actual.image,expected,atol=3e-6,rtol=3e-5)
    torch.testing.assert_close(actual.image,actual.direct+actual.backscatter)
    assert actual.diagnostics.semantics=='max_rgb_coefficient_proxy'
    params=[m.sites,m.attenuation,m.backscatter,m.colors]
    if g is not None:params += [g.means,g.log_scales,g.quaternions,g.colors,g.opacity_logits,g.raw_kappa]
    weights=torch.tensor([[.4,-.2,.7],[-.3,.8,.1]],device='cuda')
    ag=torch.autograd.grad((actual.image*weights).sum(),params)
    eg=torch.autograd.grad((expected*weights).sum(),params)
    for a,e in zip(ag,eg):torch.testing.assert_close(a,e,atol=2e-5,rtol=3e-3)


@requires_cuda
def test_tied_coefficients_recover_legacy(renderer):
    m=scene();g=gaussian();sigma=m.attenuation[:,:1].contiguous()
    tied=replace(m,densities=sigma,attenuation=sigma.expand(-1,3),backscatter=sigma.expand(-1,3))
    synchronize(renderer,tied,g)
    rays=torch.tensor([[0,0,0,0,0,1]],device='cuda',dtype=torch.float32)
    old=renderer.render(rays,g,VoronoiScene(m.sites,sigma,m.colors),background=(.1,.2,.3))
    new=renderer.render(rays,g,tied,background=(.1,.2,.3))
    torch.testing.assert_close(old.image,new.image,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(old.alpha,new.alpha,atol=2e-6,rtol=2e-5)


@requires_cuda
@pytest.mark.parametrize('coefficient',[0.,1e-7,1.,1000.])
def test_uniform_analytic_and_extremes(renderer,coefficient):
    m=scene();a=torch.full_like(m.attenuation,coefficient);b=a*.7
    B=torch.full_like(m.colors,.4);m=replace(m,densities=a.mean(-1,keepdim=True),attenuation=a,backscatter=b,colors=B)
    synchronize(renderer,m)
    rays=torch.tensor([[0,0,0,0,0,1]],device='cuda',dtype=torch.float32)
    actual=renderer.render(rays,voronoi=m,background=(.1,.2,.3))
    expected=torch.tensor([.1,.2,.3],device='cuda')*torch.exp(-a[0]*5)+B[0]*(-torch.expm1(-b[0]*5))
    torch.testing.assert_close(actual.image[0],expected,atol=2e-6,rtol=2e-5)
    assert torch.isfinite(actual.image).all()


def random_gaussians(count, seed):
    torch.manual_seed(seed)
    means=torch.randn(count,3,device='cuda')*.6+torch.tensor([0.,0.,2.5],device='cuda')
    values=(means,torch.rand(count,2,device='cuda')*.6-1.2,torch.randn(count,4,device='cuda'),
            torch.rand(count,3,device='cuda'),torch.randn(count,1,device='cuda')+1.,torch.randn(count,1,device='cuda')*.3)
    return GaussianScene(*[v.contiguous().requires_grad_() for v in values])


@requires_cuda
@pytest.mark.parametrize('threshold',[0.,1e-4,1e-2,.3])
def test_fused_matches_six_pass_with_early_termination(threshold):
    """Every pass must stop where its own scalar launch stops; values are exact."""
    renderer=NativeHybridRenderer(max_batch_size=4096,far=5.,transmittance_threshold=threshold)
    m=scene();m=replace(m,attenuation=(m.attenuation.detach()*6).requires_grad_(),
                        backscatter=(m.backscatter.detach()*4).requires_grad_())
    g=random_gaussians(40,7);synchronize(renderer,m,g)
    torch.manual_seed(3)
    directions=torch.randn(1024,3,device='cuda')*torch.tensor([.35,.35,1.],device='cuda')+torch.tensor([0.,0.,1.],device='cuda')
    rays=torch.cat((torch.zeros_like(directions),directions),-1)
    params=[m.sites,m.attenuation,m.backscatter,m.colors,g.means,g.log_scales,g.quaternions,g.colors,g.opacity_logits,g.raw_kappa]
    weights=torch.randn(3,3,device='cuda')
    results={}
    for backend in ('six_pass','fused'):
        renderer.rgb_medium_backend=backend
        out=renderer.render(rays,g,m,background=(.1,.2,.3))
        loss=(out.image*weights[0]).sum()+(out.backscatter*weights[1]).sum()+(out.alpha*weights[2,0]).sum()
        results[backend]=(out,torch.autograd.grad(loss,params))
    (old,old_grads),(new,new_grads)=results['six_pass'],results['fused']
    for name in ('image','alpha','direct','backscatter','direct_transmittance','num_intersections'):
        assert torch.equal(getattr(old,name),getattr(new,name)),name
    for a,e in zip(new_grads,old_grads):torch.testing.assert_close(a,e,atol=1e-5,rtol=1e-4)


def test_reference_gradcheck_and_opaque_occlusion():
    torch.manual_seed(4)
    args=[torch.rand(shape,dtype=torch.float64,requires_grad=True) for shape in [(3,),(3,3),(3,3),(3,3),(3,),(3,3),(3,)]]
    assert torch.autograd.gradcheck(lambda *a:reference_segments(*a)[0],tuple(args))
    args[4]=torch.tensor([1.,.2,.3],dtype=torch.float64)
    full=reference_segments(*args)[0]
    truncated=reference_segments(*(a[:1] for a in args[:-1]),args[-1])[0]
    torch.testing.assert_close(full,truncated)


@requires_cuda
@torch.no_grad()
@pytest.mark.parametrize('interval', [(0., 5.), (0., 0.), (1., 3.), (7., 8.)])
def test_draining_uniform_medium_matches_analytic_optical_length(interval):
    renderer = NativeHybridRenderer(max_batch_size=16, far=5., transmittance_threshold=0.)
    renderer.rgb_medium_backend = 'fused'
    medium = scene()
    attenuation = torch.full_like(medium.attenuation, .3)
    backscatter = torch.full_like(medium.backscatter, .2)
    colors = torch.full_like(medium.colors, .4)
    medium = replace(medium, densities=attenuation.mean(-1, keepdim=True),
                     attenuation=attenuation, backscatter=backscatter, colors=colors)
    synchronize(renderer, medium)
    rays = torch.tensor([[0., 0., 0., 0., 0., 1.]], device='cuda')
    renderer.water_intervals = torch.tensor([interval], device='cuda')
    actual = renderer.render(rays, voronoi=medium, background=(.1, .2, .3))
    length = max(0., min(5., interval[1]) - max(0., interval[0]))
    expected = rays.new_tensor([.1, .2, .3]) * torch.exp(-attenuation[0] * length)
    expected += colors[0] * (-torch.expm1(-backscatter[0] * length))
    torch.testing.assert_close(actual.image[0], expected, atol=2e-6, rtol=2e-5)


@requires_cuda
@torch.no_grad()
def test_draining_full_interval_preserves_surface_and_dry_interval_preserves_gaussians():
    renderer = NativeHybridRenderer(max_batch_size=16, far=5., transmittance_threshold=0.)
    renderer.rgb_medium_backend = 'fused'
    medium, surface = scene(), gaussian()
    synchronize(renderer, medium, surface)
    rays = torch.tensor([[0., 0., 0., .03, .02, 1.]], device='cuda')
    wet = renderer.render(rays, surface, medium, background=(.1, .2, .3)).image
    renderer.water_intervals = rays.new_tensor([[0., 5.]])
    clipped = renderer.render(rays, surface, medium, background=(.1, .2, .3)).image
    torch.testing.assert_close(clipped, wet, atol=0., rtol=0.)
    renderer.water_intervals = rays.new_zeros((1, 2))
    dry = renderer.render(rays, surface, medium, background=(.1, .2, .3)).image
    renderer.water_intervals = None
    clear = replace(medium, attenuation=torch.zeros_like(medium.attenuation),
                    backscatter=torch.zeros_like(medium.backscatter))
    expected = renderer.render(rays, surface, clear, background=(.1, .2, .3)).image
    torch.testing.assert_close(dry, expected, atol=2e-6, rtol=2e-5)
