"""Required GPU tests: no skip decorator, missing GPU is a test failure."""
import math
from pathlib import Path
import pytest
import torch
from foamsurf import NativeHybridRenderer, VoronoiScene
from training.warmup_quantiles import render_quantiles
from training.warmup_profile import quantile_loss


def fixture(sigma=(.2,.8),near=0.,far=5.):
    assert torch.cuda.is_available()
    points=torch.tensor([[0.,0.,1.],[0.,0.,3.]],device='cuda',requires_grad=True)
    density=torch.tensor(sigma,device='cuda').reshape(-1,1).requires_grad_()
    colors=torch.full((2,3),.7,device='cuda',requires_grad=True)
    scene=VoronoiScene(points,density,colors)
    adj=torch.tensor([1,0],dtype=torch.uint32,device='cuda')
    off=torch.tensor([0,1,2],dtype=torch.uint32,device='cuda')
    renderer=NativeHybridRenderer(max_batch_size=8,near=near,far=far,transmittance_threshold=0.)
    renderer.set_voronoi_geometry(scene,adj,off)
    rays=torch.tensor([[0.,0.,0.,0.,0.,1.]],device='cuda')
    return renderer,scene,rays


def test_analytic_depth_and_all_density_site_gradients():
    renderer,scene,rays=fixture()
    q=torch.tensor([[.9,.3]],device='cuda')
    depth,owners=render_quantiles(renderer,scene,rays,q)
    # Independent differentiable oracle for the two-site z-axis geometry.
    points=scene.sites.detach().double().requires_grad_()
    sigma=scene.densities.detach().double().requires_grad_()
    boundary=(points[0,2]+points[1,2])/2
    expected=torch.stack((-torch.log(q.double()[0,0])/sigma[0,0],
        boundary+(-torch.log(q.double()[0,1])-sigma[0,0]*boundary)/sigma[1,0]))[None]
    torch.testing.assert_close(depth.double(),expected,atol=2e-6,rtol=2e-6)
    assert owners.tolist()==[[0,1]]
    upstream=torch.tensor([[.3,-.7]],device='cuda')
    (depth*upstream).sum().backward()
    (expected*upstream.double()).sum().backward()
    torch.testing.assert_close(scene.sites.grad.double(),points.grad,atol=3e-6,rtol=3e-6)
    torch.testing.assert_close(scene.densities.grad.double(),sigma.grad,atol=3e-6,rtol=3e-6)
    assert scene.colors.grad is None


def test_finite_difference_density_and_sites():
    renderer,scene,rays=fixture()
    q=torch.tensor([[.85,.2]],device='cuda')
    depth,_=render_quantiles(renderer,scene,rays,q)
    depth.sum().backward()
    for parameter in (scene.sites,scene.densities):
        analytic=parameter.grad.clone()
        for i in range(parameter.numel()):
            eps=1e-3
            with torch.no_grad():
                original=parameter.flatten()[i].item()
                parameter.flatten()[i]=original+eps
            plus=render_quantiles(renderer,scene,rays,q)[0].sum().item()
            with torch.no_grad():parameter.flatten()[i]=original-eps
            minus=render_quantiles(renderer,scene,rays,q)[0].sum().item()
            with torch.no_grad():parameter.flatten()[i]=original
            assert analytic.flatten()[i].item()==pytest.approx((plus-minus)/(2*eps),abs=5e-4,rel=3e-3)


def test_invalid_transparent_zero_weight_and_api():
    renderer,scene,rays=fixture(sigma=(.001,.001))
    q=torch.tensor([[.9,.1]],device='cuda')
    depths,owners=render_quantiles(renderer,scene,rays,q)
    assert depths.tolist()==[[-1.,-1.]] and owners.tolist()==[[-1,-1]]
    quantile_loss(depths)[0].backward()
    assert torch.count_nonzero(scene.densities.grad)==0
    for bad in ([[.1,.9]], [[1.,.5]], [[.5,0.]]):
        with pytest.raises(ValueError):render_quantiles(renderer,scene,rays,torch.tensor(bad,device='cuda'))
    with pytest.raises(ValueError,match='Voronoi-only'):
        render_quantiles(renderer,scene,rays,q,gaussians=object())
    renderer,scene,rays=fixture()
    depth,_=render_quantiles(renderer,scene,rays,q)
    (0*depth.sum()).backward()
    assert torch.count_nonzero(scene.densities.grad)==0


def test_clipped_near_plane():
    renderer,scene,rays=fixture(near=2.5)
    q=torch.tensor([[.9,.5]],device='cuda')
    depth,_=render_quantiles(renderer,scene,rays,q)
    torch.testing.assert_close(depth,2.5-torch.log(q)/.8,atol=2e-6,rtol=2e-6)
    depth.sum().backward()
    assert torch.count_nonzero(scene.sites.grad)==0


def test_diffuse_vs_concentrated_and_rgb_gradient_invariance():
    renderer,scene,rays=fixture(sigma=(.2,.2))
    q=torch.tensor([[.95,.75]],device='cuda')
    wide=render_quantiles(renderer,scene,rays,q)[0]
    with torch.no_grad():scene.densities.copy_(torch.tensor([[.5],[0.]],device='cuda'))
    narrow=render_quantiles(renderer,scene,rays,q)[0]
    assert wide[0,1]-wide[0,0]>narrow[0,1]-narrow[0,0]
    a=renderer.render_voronoi(rays,scene)
    ga=torch.autograd.grad(a.image.sum()+a.alpha.sum(),(scene.sites,scene.densities,scene.colors))
    render_quantiles(renderer,scene,rays,q)
    b=renderer.render_voronoi(rays,scene)
    gb=torch.autograd.grad(b.image.sum()+b.alpha.sum(),(scene.sites,scene.densities,scene.colors))
    assert torch.equal(a.image,b.image) and torch.equal(a.alpha,b.alpha)
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_radfoam_forward_and_density_backward_reference(tmp_path):
    # exec a fresh interpreter (not fork): RadFoam configures the CUDA heap at
    # import and must do so before either renderer launches any CUDA kernels.
    import os
    import subprocess
    import sys
    worker = Path(__file__).with_name("radfoam_reference_worker.py")
    env = dict(os.environ, CUDA_LAUNCH_BLOCKING="1")
    result = subprocess.run(
        [sys.executable, str(worker)], cwd=Path(__file__).resolve().parents[2],
        env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=600,
    )
    log_dir = (
        Path(__file__).resolve().parents[2] / "outputs" / "warmup_tests" / os.environ["SLURM_JOB_ID"]
        if "SLURM_JOB_ID" in os.environ else tmp_path
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / "radfoam_reference.log"
    log.write_text(result.stdout)
    assert result.returncode == 0, (
        f"Isolated RadFoam reference failed (exit {result.returncode}); log: {log}\n"
        + result.stdout
    )
    assert "RADFOAM_REFERENCE_PASSED" in result.stdout, result.stdout
