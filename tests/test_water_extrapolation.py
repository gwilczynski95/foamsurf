"""Water light continued beyond the training directions (optimization.water_extrapolation, the _wext suffix),
evaluation only: an HG trend in the angle to the vertical, fitted on the training photos' open water, scales the
backscatter only above the highest training direction."""
import math
from types import SimpleNamespace

import pytest
import torch

from test_oceansplat import overrides
from training.config import ExperimentConfig
from training.trainer import Trainer
from training.water_extrapolation import WaterExtrapolation, henyey_greenstein

UP = torch.tensor([0.0, 0.0, 1.0])


def directions_at(elevations_deg):
    e = torch.deg2rad(torch.tensor(elevations_deg, dtype=torch.float32))
    return torch.stack((torch.cos(e), torch.zeros_like(e), torch.sin(e)), -1)


def test_identity_inside_the_training_range_and_the_trend_beyond():
    cos_max = math.sin(math.radians(20.0))
    ext = WaterExtrapolation(UP * 2.0, torch.tensor([0.4, 0.8, 0.85]), cos_max)
    inside = directions_at([-30.0, 0.0, 10.0, 19.9])
    torch.testing.assert_close(ext.ratio(inside), torch.ones(4, 3))
    above = directions_at([25.0, 30.0])
    r = ext.ratio(above)
    assert bool((r > 1.0).all()) and bool((r[1] > r[0]).all())  # brighter the higher, g > 0
    assert bool((r[:, 1] > r[:, 0]).all())  # the more forward-scattering channel grows faster
    cos = torch.sin(torch.deg2rad(torch.tensor(30.0)))
    expected = henyey_greenstein(cos, ext.anisotropy) / henyey_greenstein(torch.tensor(cos_max), ext.anisotropy)
    torch.testing.assert_close(r[1], expected)


def test_apply_scales_only_the_backscatter():
    ext = WaterExtrapolation(UP, torch.tensor([0.5, 0.5, 0.5]), 0.3)
    image, backscatter = torch.rand(2, 3), torch.rand(2, 3) * 0.3
    dirs = directions_at([0.0, 40.0])
    out = ext.apply(image, backscatter, dirs)
    torch.testing.assert_close(out[0], image[0])
    torch.testing.assert_close(out[1] - (image[1] - backscatter[1]), backscatter[1] * ext.ratio(dirs[1:])[0])


def test_fit_recovers_the_anisotropy_despite_per_photo_exposure():
    torch.manual_seed(0)
    g = torch.tensor([0.3, 0.6, 0.8])
    photos = 6
    elevation = torch.rand(6000) * 40.0 - 10.0  # the photos see -10..30 deg
    dirs = directions_at(elevation.tolist())
    photo = torch.randint(0, photos, (6000,))
    scale = torch.exp(torch.randn(photos, 3) * 0.2) * 0.2  # exposure / level per photo and channel
    colours = scale[photo] * henyey_greenstein(dirs @ UP, g)
    ext = WaterExtrapolation.fit(UP, dirs * 3.0, colours, photo)  # directions need not be unit
    torch.testing.assert_close(ext.anisotropy, g, atol=0.02, rtol=0)
    assert abs(ext.cos_max - float((dirs @ UP).max())) < 1e-6


def test_evaluation_light_composes_with_the_trained_terms():
    ext = WaterExtrapolation(UP, torch.tensor([0.5, 0.5, 0.5]), 0.0)
    fake = SimpleNamespace(water_extrapolation=ext)
    gain = torch.tensor([1.5, 1.0, 0.5])
    fake._water_modules = lambda: [object()]  # one trained term with a constant gain
    fake._water_light_parts = lambda image, backscatter, rays: image + (gain - 1.0) * backscatter
    image, backscatter = torch.full((1, 3), 0.4), torch.full((1, 3), 0.2)
    rays = torch.cat((torch.zeros(1, 3), directions_at([30.0])), -1)
    out = Trainer._evaluation_water_light(fake, image, backscatter, rays)
    direct = image - backscatter
    torch.testing.assert_close(out, direct + gain * backscatter * ext.ratio(rays[:, 3:]))
    fake.water_extrapolation = None  # without _wext: the trained terms only
    torch.testing.assert_close(Trainer._evaluation_water_light(fake, image, backscatter, rays), direct + gain * backscatter)
    fake._water_modules = lambda: []
    fake.water_extrapolation = ext  # _wext alone
    torch.testing.assert_close(Trainer._evaluation_water_light(fake, image, backscatter, rays),
                               direct + backscatter * ext.ratio(rays[:, 3:]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_short_real_training_with_the_new_default_terms_and_resume(tmp_path):
    """_wphase + _rgbs15 + _wext together: the evaluation fits and saves the extension, the colour scale is in the
    checkpoint, and a resumed trainer evaluates to finite numbers."""
    import json
    from test_medium_fix import _short_manifest
    from train import fit_iui3
    m = _short_manifest(tmp_path, "new", overrides("_wphase_rgbs15_wext"))
    fit_iui3(m)
    root = tmp_path / "new/joint"
    checkpoint = torch.load(root / "checkpoints/last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["model_metadata"]["gaussians"]["rgb_scale"] == 1.5
    config = ExperimentConfig.load(root / "config.yaml")
    config.output_dir = str(tmp_path / "resumed")
    trainer = Trainer(config, resume=root / "checkpoints/last.pt")
    assert trainer.model.gaussians.rgb_scale == 1.5
    # the evaluation wrote the fit exactly when there was open water to fit (4 steps at 1/16 may have none)
    saved = root / "water_extrapolation.json"
    assert saved.exists() == (trainer.fit_water_extrapolation() is not None)
    if saved.exists():
        assert json.loads(saved.read_text())["anisotropy"] == trainer.water_extrapolation.state_dict()["anisotropy"]
    fitted = trainer.fit_water_extrapolation(step=1)  # every pixel: the IUI3 open water is enough
    assert fitted is not None and len(fitted.anisotropy) == 3 and -1.0 < fitted.cos_max <= 1.0
    summary, _, _ = trainer.evaluator.evaluate(4)  # with the extension in the evaluator's image
    assert math.isfinite(summary["eval/primary_psnr"])


def test_suffix_and_config():
    assert overrides("_wext") == ["joint.optimization.water_extrapolation=true"]
    assert ExperimentConfig().optimization.water_extrapolation is False
    assert overrides("_cull05r_bin01_wphase_wext")[-1] == "joint.optimization.water_extrapolation=true"
