"""Required RTX4090 tests: native optical + boundary gradients and isolation."""
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from foamsurf import NativeHybridRenderer, HybridRenderer, VoronoiScene
from experiments.rgb_disentangle.test_renderer_gpu import scene
from experiments.rgb_surface.loss import SurfaceConfig, optical_density, local_rays, occupancy_loss
from experiments.rgb_surface.renderer import LocalOverlap, gaussian_moments


@pytest.fixture
def setup():
    assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(), 'Submit test.sbatch'
    renderer = NativeHybridRenderer(max_batch_size=16, far=5., transmittance_threshold=0.)
    rays, g, m = scene(renderer)
    # Boundary z=2.3 is inside the local band around z~2.4.
    with torch.no_grad():
        m.sites[:] = torch.tensor([[0., 0., 1.3], [0., 0., 3.3]], device='cuda')
    topology = SimpleNamespace(revision=1, adjacency=torch.tensor([1, 0], device='cuda', dtype=torch.uint32),
                               offsets=torch.tensor([0, 1, 2], device='cuda', dtype=torch.uint32))
    cfg = SurfaceConfig(weight=1., start=0, ramp_steps=0, rays=2, range_multiplier=5., band_fraction=.1)
    return renderer, rays, g, m, topology, cfg


def test_full_gradient_matches_independent_oracle(setup):
    renderer, rays, g, m, topology, cfg = setup
    native = LocalOverlap()
    observed, logs = native.compute(renderer.render, rays, g, m, topology, cfg, 1, 1., 5., 0., 42)
    rng = torch.Generator(device='cuda').manual_seed(42*1000003+1)
    chosen = rays[torch.randint(len(rays), (2,), device='cuda', generator=rng)]
    with torch.no_grad():
        hits = renderer.render(chosen, gaussians=g)
        depth, confidence, _ = gaussian_moments(g, chosen, hits.hit_depths, hits.hit_indices, 0., 5., .5)
        windows = local_rays(chosen, depth, .5, 5.)
    proxy = VoronoiScene(m.sites, optical_density(m.optical_coefficients, m.colors), torch.zeros_like(m.colors))
    expected = occupancy_loss(HybridRenderer(far=1.)(windows, voronoi=proxy).alpha, confidence)
    torch.testing.assert_close(observed, expected, atol=3e-6, rtol=3e-5)
    params = (m.sites, m.raw_density, m.raw_color)
    actual_grads = torch.autograd.grad(observed, params, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, params)
    for a, b in zip(actual_grads, expected_grads):
        assert torch.isfinite(a).all() and a.abs().sum() > 0
        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-3)
    assert (actual_grads[1] > 0).all() and (actual_grads[2] > 0).all()
    ggrads = torch.autograd.grad(observed, [getattr(g, f.name) for f in fields(g)], allow_unused=True)
    assert all(x is None for x in ggrads)
    assert logs['traversal_truncated_fraction'] == 0
    # Central finite difference independently checks native site backward.
    eps = .001
    with torch.no_grad():
        m.sites[0, 2] += eps
    plus = native.compute(renderer.render, rays, g, m, topology, cfg, 1, 1., 5., 0., 42)[0]
    with torch.no_grad():
        m.sites[0, 2] -= 2*eps
    minus = native.compute(renderer.render, rays, g, m, topology, cfg, 1, 1., 5., 0., 42)[0]
    with torch.no_grad():
        m.sites[0, 2] += eps
    torch.testing.assert_close((plus-minus)/(2*eps), actual_grads[0][0, 2], atol=3e-5, rtol=.02)


def test_auxiliary_renderer_does_not_corrupt_primary_backward(setup):
    renderer, rays, g, m, topology, cfg = setup
    params = (g.means, g.opacity_logits, m.raw_density, m.raw_color, m.sites)
    baseline = renderer.render(rays, g, m.scene(), background=(0., 0., 0.))
    expected = torch.autograd.grad(baseline.image.sum(), params)
    primary = renderer.render(rays, g, m.scene(), background=(0., 0., 0.))
    LocalOverlap().compute(renderer.render, rays, g, m, topology, cfg, 2, 1., 5., 0., 42)
    actual = torch.autograd.grad(primary.image.sum(), params)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


def test_native_black_direct_and_comparable_g_only(setup):
    renderer, rays, g, m, topology, cfg = setup
    water = renderer.render(rays, voronoi=m.scene(), background=(0., 0., 0.))
    assert not water.direct.any() and water.backscatter.sum() > 0
    hybrid = renderer.render(rays, g, m.scene(), background=(0., 0., 0.))
    assert hybrid.direct.sum() > 0
    black = renderer.render(rays, gaussians=g, background=(0., 0., 0.))
    white = renderer.render(rays, gaussians=g, background=(1., 1., 1.))
    torch.testing.assert_close(black.image, white.image-white.transmittance, atol=2e-6, rtol=2e-5)


def test_auxiliary_sync_after_site_replacement(setup):
    renderer, rays, g, m, topology, cfg = setup
    local = LocalOverlap()
    first, _ = local.compute(renderer.render, rays, g, m, topology, cfg, 1, 1., 5., 0., 42)
    old_key = local.key
    m.sites = m.sites.detach().clone().requires_grad_()
    topology.revision += 1
    second, _ = local.compute(renderer.render, rays, g, m, topology, cfg, 1, 1., 5., 0., 42)
    assert local.key != old_key
    torch.testing.assert_close(first, second)
