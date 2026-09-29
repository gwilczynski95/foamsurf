from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from foamsurf import HybridRenderer, VoronoiScene
from experiments.rgb_surface.loss import (SurfaceConfig, optical_density, event_moments,
                                          local_rays, occupancy_loss)
from experiments.rgb_surface.coverage_artifacts import black_gaussians, coverage_metrics
from experiments.rgb_surface.renderer import LocalOverlap, gaussian_moments
from experiments.rgb_surface.train import choices, resolve, settings


def test_exact_two_cell_integral_and_all_parameter_gradients():
    sites = torch.tensor([[0., 0., 1.], [0., 0., 3.]], dtype=torch.double, requires_grad=True)
    coeff = torch.tensor([[.1]*3+[.2]*3, [.4]*3+[.5]*3], dtype=torch.double, requires_grad=True)
    color = torch.tensor([[.3]*3, [.8]*3], dtype=torch.double, requires_grad=True)
    rays = torch.tensor([[0., 0., 1.5, 0., 0., 1.]], dtype=torch.double)
    sigma = optical_density(coeff, color)
    result = HybridRenderer(far=1.)(rays, voronoi=VoronoiScene(sites, sigma, torch.zeros_like(color)))
    expected = -torch.expm1(-.5*sigma.sum())
    torch.testing.assert_close(result.alpha.squeeze(), expected)
    grads = torch.autograd.grad(result.alpha.sum(), (sites, coeff, color))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    assert (grads[1] > 0).all() and (grads[2] > 0).all()
    expected_site_grad = torch.exp(-.5*sigma.sum())*.5*(sigma[0, 0]-sigma[1, 0])
    torch.testing.assert_close(grads[0][:, 2], expected_site_grad.expand(2))
    assert torch.autograd.gradcheck(lambda s, c, b: HybridRenderer(far=1.)(rays,
        voronoi=VoronoiScene(s, optical_density(c, b), torch.zeros_like(b))).alpha,
        (sites, coeff, color), eps=1e-6, atol=1e-5)


def test_black_background_removes_direct_without_gaussians():
    from foamsurf.rgb_medium import reference_segments
    lengths = torch.tensor([1., 2.], dtype=torch.double)
    a = torch.full((2, 3), .2, dtype=torch.double, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    c = torch.full_like(a, .4, requires_grad=True)
    alpha = torch.zeros(2, dtype=torch.double)
    out = reference_segments(lengths, a, b, c, alpha, torch.ones_like(c), torch.zeros(3, dtype=torch.double))
    assert torch.equal(out[1], torch.zeros(3, dtype=torch.double))
    assert (out[2] > 0).all()
    white = reference_segments(lengths, a, b, c, alpha, torch.ones_like(c), torch.ones(3, dtype=torch.double))
    torch.testing.assert_close(white[1], torch.exp(-a[0]*3))


def test_event_order_weights_and_detachment():
    depth = torch.tensor([[3., 1., float('nan')]], requires_grad=True)
    alpha = torch.tensor([[.5, .8, .9]], requires_grad=True)
    mean, confidence, opacity = event_moments(depth, alpha, torch.tensor([[True, True, False]]), 2.)
    assert opacity.item() == pytest.approx(.9)
    assert mean.item() == pytest.approx((.8+3*.1)/.9)
    assert 0 < confidence <= opacity
    assert not mean.requires_grad and not confidence.requires_grad
    loss = occupancy_loss(torch.tensor([[.3]], requires_grad=True), confidence)
    loss.backward()
    assert alpha.grad is None and depth.grad is None


def test_empty_and_spread_out_gaussian_profiles():
    depths = torch.tensor([[1., 4.], [1., 4.]])
    alpha = torch.tensor([[0., 0.], [.5, 1.]])
    mean, confidence, opacity = event_moments(depths, alpha, torch.ones_like(alpha, dtype=torch.bool), .1)
    assert confidence[0] == 0 and mean[0] == 0
    assert confidence[1] < .001 and opacity[1] > .99
    x = torch.tensor([[.3], [.7]], requires_grad=True)
    loss = occupancy_loss(x, torch.zeros(2))
    loss.backward()
    assert loss == 0 and not x.grad.any()


def test_windows_stay_inside_range_and_gaussian_empty_scene():
    rays = torch.tensor([[0., 0., 0., 0., 0., 2.]]).repeat(3, 1)
    result = local_rays(rays, torch.tensor([0., 2., 5.]), .5, 5.)
    torch.testing.assert_close(result[:, 2], torch.tensor([0., 1.5, 4.]))
    assert not result.requires_grad
    scene = SimpleNamespace(means=torch.empty((0, 3)))
    args = (scene, rays, torch.empty((0, 3)), torch.empty((0, 3), dtype=torch.long), 0., 5., .5)
    assert all(not x.any() for x in gaussian_moments(*args))


def test_color_cannot_hide_attenuation_or_backscatter():
    coefficients = torch.ones(1, 6, requires_grad=True)
    colors = torch.zeros(1, 3, requires_grad=True)
    loss = optical_density(coefficients, colors).sum()
    loss.backward()
    assert loss > 0 and (coefficients.grad > 0).all() and (colors.grad > 0).all()


def test_off_branch_no_renderer_and_schedule():
    c = SurfaceConfig(weight=.1)
    assert [c.weight_at(t) for t in (0, 1000, 1500, 2000)] == [0, 0, .05, .1]
    def forbidden(*a, **kw):
        raise AssertionError('Off path rendered')
    loss, logs = LocalOverlap().compute(forbidden, None, None, SimpleNamespace(raw_density=torch.zeros(1)),
                                       None, c, 1000, 1, 5, 0, 42)
    assert loss == 0 and logs['rays'] == 0


@pytest.mark.parametrize('kwargs', [{'weight': -1}, {'weight': float('nan')}, {'start': -1},
    {'rays': 0}, {'ramp_steps': .5}, {'band_fraction': .6}, {'optical_scale': 0}, {'range_multiplier': float('inf')}])
def test_bad_config(kwargs):
    with pytest.raises(ValueError):
        SurfaceConfig(**kwargs)


def test_coverage_independent_of_background():
    alpha = torch.tensor([[[0.], [.2], [.95]]])
    rgb = torch.tensor([[[0., 0., 0.], [.1, .15, .2], [.2, .3, .4]]])
    white_image = rgb + (1-alpha)
    torch.testing.assert_close(black_gaussians(white_image, alpha, [1, 1, 1]), rgb)
    torch.testing.assert_close(black_gaussians(rgb, alpha, [0, 0, 0]), rgb)
    assert coverage_metrics(alpha)['coverage_gt_0.9'] == pytest.approx(1/3)


def test_factorial_manifests_and_historical_control():
    path = Path(__file__).with_name('sweep.yaml')
    assert len(choices(settings(path))) == 72
    names = set()
    for i in range(72):
        m = resolve(path, i, f'/tmp/surface_manifest_{i}')
        v = m['variant']
        names.add(v['name'])
        assert m['warmup_steps'] == 0
        assert m['joint']['background'] == ([0., 0., 0.] if v['background'] == 'black' else [1., 1., 1.])
        assert m['rgb_surface']['weight'] == v['overlap_weight']
        assert m['rgb_smoothing']['color'] == v['tv_multiplier']
        assert m['joint']['optimization']['iterations'] == 30000
        assert m['joint']['wandb']['mode'] == 'online'
        assert m['joint']['wandb']['project'] == 'foamsurf_iui3_rgb_surface'
        assert m['joint']['gaussian_adaptation']['densify_until'] == 15000
        assert m['joint']['voronoi_adaptation']['densify_until'] == 30000
        assert 'rgb_conflict' not in m
    assert len(names) == 72
    with pytest.raises(ValueError):
        resolve(path, 72, '/tmp/surface_invalid')


def test_analysis_fixed_iteration_paired_comparison_and_floor_mask(tmp_path):
    import json
    import numpy as np
    from PIL import Image
    from experiments.rgb_surface.analyze import write_csv, write_report
    mask_dir = tmp_path/'masks'
    mask_dir.mkdir()
    Image.fromarray(np.array([[255, 0], [255, 0]], dtype=np.uint8)).save(mask_dir/'camera.png')
    for i, bg, value in ((0, 'white', .4), (36, 'black', .95)):
        output = tmp_path/f'run_{i}'
        at = output/'joint/evaluation/iteration_000500'
        at.mkdir(parents=True)
        (output/'pipeline_config.json').write_text(json.dumps(dict(index=i, variant=dict(
            background=bg, overlap_weight=0., tv_multiplier=0., seed=42))))
        write_csv(at/'per_view.csv', [dict(hybrid_psnr=20+i/36)])
        write_csv(at/'coverage.csv', [dict(alpha_mean=value, **{'coverage_gt_0.9': float(value > .9)}, gaussian_black_psnr=10)])
        np.savez_compressed(at/'0000_camera_gaussian_alpha.npz', alpha=np.full((2, 2, 1), value, dtype=np.float32))
        Image.new('RGB', (120, 20), 'gray').save(at/'0000_camera_coverage_panel.png')
    result = write_report(tmp_path, tmp_path/'report', 500, mask_dir)
    assert len(result) == 2 and result[0]['floor_coverage_09'] == 0 and result[1]['floor_coverage_09'] == 1
    assert all(r['floor_views'] == 1 for r in result)
    assert (tmp_path/'report/factorial.png').is_file()
    assert (tmp_path/'report/paired_deltas.csv').is_file()
    assert (tmp_path/'report/black_O0_view0.png').is_file()
    with pytest.raises(ValueError, match='No coverage'):
        write_report(tmp_path, tmp_path/'missing', 30000)


def test_evaluation_artifacts_and_background_normalization(tmp_path):
    import numpy as np
    from PIL import Image
    from experiments.rgb_surface.coverage_artifacts import install_coverage
    alpha = torch.tensor([[[0.], [.2]], [[.9], [1.]]])
    black = torch.ones(2, 2, 3)*alpha*.3
    camera = SimpleNamespace(name='test', ground_truth=lambda bg: torch.ones(2, 2, 3)*.3)
    original = lambda *a, **kw: SimpleNamespace(image=black+1-alpha, alpha=alpha)
    renderer = SimpleNamespace(render_gaussians=original)
    at = tmp_path/'evaluation/iteration_000005'
    at.mkdir(parents=True)
    for label in ('hybrid', 'voronoi_only'):
        Image.new('RGB', (2, 2)).save(at/f'0000_test_{label}.png')
    def evaluate(iteration):
        renderer.render_gaussians()
        return {}, {}, {}
    evaluator = SimpleNamespace(evaluate=evaluate, renderer=renderer, output_dir=tmp_path,
        dataset=SimpleNamespace(test=[camera]), config=SimpleNamespace(background=[1., 1., 1.],
        validation=SimpleNamespace(image_views=1)))
    install_coverage(evaluator)
    summary, images, _ = evaluator.evaluate(5)
    assert renderer.render_gaussians is original
    assert summary['eval/gaussian_coverage/coverage_gt_0.9'] == .25
    assert 'coverage/view_00' in images
    np.testing.assert_allclose(np.load(at/'0000_test_gaussian_alpha.npz')['alpha'], alpha)
    saved = np.asarray(Image.open(at/'0000_test_gaussian_only_black.png'))
    np.testing.assert_allclose(saved/255., black, atol=1/255.)
