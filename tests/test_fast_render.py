"""Faster rendering without changing a pixel: SH colours in one CUDA kernel when no gradients are needed,
render(surface=False, check_inputs=False) for viewing a trained model."""
from pathlib import Path

import pytest
import torch

from foamsurf_model.sh import evaluate_sh

ROOT = Path(__file__).resolve().parents[1]
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


@gpu
@pytest.mark.parametrize("degree", [0, 1, 2, 3])
@pytest.mark.parametrize("sigmoid", [True, False])
def test_sh_kernel_matches_python(degree, sigmoid):
    from foamsurf_model.sh_cuda import sh_colors
    g = torch.Generator(device="cuda").manual_seed(degree)
    n = 5000
    means = torch.randn(n, 3, device="cuda", generator=g) * 3
    dc = torch.randn(n, 3, 1, device="cuda", generator=g)
    rest = torch.randn(n, 3, 15, device="cuda", generator=g) * 0.3
    origin = torch.tensor([0.5, -1.0, 2.0], device="cuda")
    directions = torch.nn.functional.normalize(origin[None] - means, dim=-1)
    value = evaluate_sh(degree, torch.cat((dc, rest), -1)[..., :(degree + 1) ** 2], directions)
    expected = torch.sigmoid(value) if sigmoid else 0.5 + value
    torch.testing.assert_close(sh_colors(means, dc, rest, degree, origin, sigmoid), expected, atol=2e-6, rtol=1e-5)


@gpu
def test_model_colours_fast_only_without_gradients():
    import numpy as np
    from foamsurf_model.gaussians import SurfaceGaussianModel
    from training.config import ModelConfig
    config = ModelConfig(appearance="sh", sh_degree=3, gaussian_rgb_activation="sigmoid")
    rng = np.random.default_rng(0)
    model = SurfaceGaussianModel(rng.normal(size=(300, 3)).astype(np.float32), rng.random((300, 3)).astype(np.float32),
                                 config, "cuda")
    with torch.no_grad():
        model.appearance_rest.normal_(0, 0.2)
    origin = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    with torch.no_grad():
        fast = model.colors(origin)
    slow = model.colors(origin)  # grad enabled: the Python path
    assert slow.requires_grad and not fast.requires_grad
    torch.testing.assert_close(fast, slow.detach(), atol=2e-6, rtol=1e-5)


@gpu
def test_fast_render_identical_on_a_trained_model():
    """On the current default (551590 IUI3 s42): colours from the kernel and render(surface=False,
    check_inputs=False) give the same image, direct and backscatter as the regular path."""
    run = ROOT / "outputs/seathru/551590/IUI3-RedSea-Undistorted_clamp05_minit_rs_rel_oceane_isp"
    if not (run / "joint/checkpoints/last.pt").exists():
        pytest.skip("needs the 551590 checkpoint")
    from evaluation.evaluator import _rays
    from training.config import ExperimentConfig
    from training.trainer import Trainer
    config = ExperimentConfig.load(run / "joint/config.yaml")
    config.output_dir = str(ROOT / "outputs/analysis_2026-10-27/_trainer/test_fast_render"); config.wandb.mode = "disabled"
    trainer = Trainer(config, resume=run / "joint/checkpoints/last.pt")
    model, renderer = trainer.model, trainer.renderer
    renderer.transmittance_threshold = config.renderer.validation_transmittance_threshold
    trainer.topology.synchronize_renderer(renderer, model.medium)
    cam = trainer.dataset.test[1]
    rays = _rays(cam)
    medium = model.medium.render_scene()
    bg = tuple(config.background)
    with torch.enable_grad():
        slow_scene = model.gaussians.render_scene(cam.origin)
    with torch.no_grad():
        slow_scene = type(slow_scene)(**{k: v.detach() for k, v in slow_scene.__dict__.items()})
        fast_scene = model.gaussians.render_scene(cam.origin)
        torch.testing.assert_close(fast_scene.colors, slow_scene.colors, atol=2e-6, rtol=1e-5)
        renderer.set_gaussian_geometry(slow_scene)
        regular = renderer.render(rays, gaussians=slow_scene, voronoi=medium, background=bg)
        renderer.set_gaussian_geometry(fast_scene)
        fast = renderer.render(rays, gaussians=fast_scene, voronoi=medium, background=bg, surface=False, check_inputs=False)
    assert fast.surface_depth is None and regular.surface_depth is not None
    for name in ("image", "direct", "backscatter", "alpha"):
        torch.testing.assert_close(getattr(fast, name), getattr(regular, name), atol=1e-5, rtol=0)
    assert float((fast.image - regular.image).abs().max()) * 255 < 0.01


@gpu
def test_geometry_upload_retries_once_after_freeing_the_cache(monkeypatch):
    """OptiX allocates outside PyTorch: a failed SetGeometry frees PyTorch's cache and is tried once more."""
    from renderer.foamsurf.cuda import NativeHybridRenderer
    from renderer.foamsurf.types import GaussianScene
    renderer = NativeHybridRenderer.__new__(NativeHybridRenderer)
    calls, freed = [], []

    class FlakyOptix:
        def SetGeometry(self, *tensors):
            calls.append(len(tensors))
            if len(calls) == 1:
                raise RuntimeError("Caught an unknown exception!")

    renderer._optix = FlakyOptix()
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: freed.append(True))
    n = 4
    scene = GaussianScene(means=torch.zeros(n, 3, device="cuda"), log_scales=torch.zeros(n, 2, device="cuda"),
                          quaternions=torch.tensor([[1.0, 0, 0, 0]] * n, device="cuda"),
                          colors=torch.zeros(n, 3, device="cuda"), opacity_logits=torch.zeros(n, 1, device="cuda"),
                          raw_kappa=torch.zeros(n, 1, device="cuda"))
    renderer.set_gaussian_geometry(scene)
    assert calls == [5, 5] and freed == [True] and renderer._gaussian_geometry_is_set
    calls.clear()
    renderer._optix = type("Broken", (), {"SetGeometry": lambda self, *t: (_ for _ in ()).throw(RuntimeError("x"))})()
    with pytest.raises(RuntimeError):
        renderer.set_gaussian_geometry(scene)  # a second failure is not hidden


@gpu
def test_fps_bench_threshold_option(tmp_path):
    """scripts/fps_bench.py --threshold renders at the given transmittance threshold and reports it."""
    import json
    import subprocess
    import sys
    run = ROOT / "outputs/seathru/553596/IUI3-RedSea-Undistorted_wfit"
    if not (run / "joint/checkpoints/last.pt").exists():
        pytest.skip("needs the 553596 checkpoint")
    out = tmp_path / "fps.json"
    result = subprocess.run([sys.executable, "scripts/fps_bench.py", str(run), "--threshold", "1e-4", "--repeats", "1",
                             "--json", str(out)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    row = json.loads(out.read_text())[0]
    assert row["threshold"] == 1e-4 and row["view"]["fps"] > 0 and row["eval"]["psnr"] > 20
