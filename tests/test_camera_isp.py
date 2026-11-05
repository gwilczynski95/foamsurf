"""PPISP camera model: per-photo exposure and colour, novel views from the nearest training photos."""
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from train import fit_iui3, resolve_iui3
from training import oceansplat as oc
from training.config import ExperimentConfig
from helpers import ROOT, requires_gpu  # noqa: F401
from test_oceansplat import PlaneRenderer, plane_camera, plane_scene


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


@gpu
def test_identity_at_start_then_per_photo_exposure_and_neutral_test_views():
    from training.camera_isp import CameraISP, pixel_grid
    isp = CameraISP(["a", "b"], iterations=100)
    rgb = torch.rand(6 * 8, 3, device="cuda") * 0.8 + 0.1
    coords = pixel_grid(6, 8, "cuda")
    for frame in ("a", "b", None):  # PPISP starts at the identity
        torch.testing.assert_close(isp.apply(rgb, coords, (6, 8), frame), rgb, atol=1e-4, rtol=0)
    with torch.no_grad():
        isp.module.exposure_params[0] = 1.0  # photo "a": +1 stop
    torch.testing.assert_close(isp.apply(rgb, coords, (6, 8), "a"), (2 * rgb).clamp(max=1), atol=1e-3, rtol=0)
    torch.testing.assert_close(isp.apply(rgb, coords, (6, 8), None), rgb, atol=1e-4, rtol=0)  # test view: neutral
    with torch.no_grad():
        isp.module.exposure_params.zero_()
        isp.module.vignetting_params[0, :, 2] = -0.5  # shared vignetting reaches test views too
    neutral = isp.apply(rgb, coords, (6, 8), None)
    assert bool((neutral <= rgb + 1e-5).all()) and float((rgb - neutral).max()) > 0.01
    rgb.requires_grad_(True)
    isp.zero_grad()
    (isp.apply(rgb, coords, (6, 8), "b").sum() + isp.regularization()).backward()
    assert rgb.grad.abs().sum() > 0 and isp.module.vignetting_params.grad.abs().sum() > 0


@gpu
def test_novel_views_take_the_exposure_of_their_nearest_training_photos():
    from training.camera_isp import CameraISP, pixel_grid
    centres = torch.tensor([[0.0, 0, 0], [1, 0, 0], [3, 0, 0], [10, 0, 0]])
    isp = CameraISP(["a", "b", "c", "d"], iterations=100, centres=centres, neighbours=2)
    with torch.no_grad():
        isp.module.exposure_params.copy_(torch.tensor([1.0, 0.0, -1.0, 5.0]))
    rgb = torch.rand(6 * 8, 3, device="cuda") * 0.4 + 0.1
    coords = pixel_grid(6, 8, "cuda")
    centre = torch.tensor([0.25, 0, 0])  # 0.25 from "a", 0.75 from "b": weights 3/4 and 1/4
    torch.testing.assert_close(isp.novel_view_weights(centre).cpu(), torch.tensor([0.75, 0.25, 0.0, 0.0]))
    torch.testing.assert_close(isp.apply(rgb, coords, (6, 8), None, centre), rgb * 2 ** 0.75, atol=1e-3, rtol=0)
    torch.testing.assert_close(isp.apply(rgb, coords, (6, 8), "c"), rgb / 2, atol=1e-3, rtol=0)  # training photo
    with torch.no_grad():  # colour too: the novel view equals the photo it sits on
        isp.module.color_params[1] = torch.linspace(-0.05, 0.05, 8)
    torch.testing.assert_close(isp.apply(rgb, coords, (6, 8), None, torch.tensor([1.0, 0, 0])),
                               isp.apply(rgb, coords, (6, 8), "b"), atol=1e-4, rtol=0)
    neutral = CameraISP(["a", "b", "c", "d"], iterations=100, centres=centres, neighbours=0)
    neutral.load_state_dict(isp.state_dict())
    torch.testing.assert_close(neutral.apply(rgb, coords, (6, 8), None, centre), rgb, atol=1e-4, rtol=0)
    with pytest.raises(ValueError, match="camera centre"):
        CameraISP(["a", "b"], iterations=10, neighbours=2)


@gpu
def test_state_round_trip_and_frame_check():
    from training.camera_isp import CameraISP
    a = CameraISP(["x", "y", "z"], iterations=50)
    with torch.no_grad():
        a.module.exposure_params.copy_(torch.tensor([0.1, -0.2, 0.3]))
    b = CameraISP(["x", "y", "z"], iterations=50)
    b.load_state_dict(a.state_dict())
    torch.testing.assert_close(b.module.exposure_params, a.module.exposure_params)
    with pytest.raises(ValueError, match="other photos"):
        CameraISP(["x", "y"], iterations=50).load_state_dict(a.state_dict())


def test_training_batch_pixel_coordinates():
    from training.trainer import Trainer
    seen = {}
    fake = SimpleNamespace(camera_isp=SimpleNamespace(apply=lambda image, coords, size, frame: seen.update(
        coords=coords, size=size, frame=frame) or image))
    camera = SimpleNamespace(height=3, width=4, name="p")
    index = torch.tensor([0, 5, 11])  # (x, y) = (0, 0), (1, 1), (3, 2)
    batch = SimpleNamespace(camera=camera, image_size=None, rays=SimpleNamespace(pixel_indices=index))
    Trainer._apply_camera_isp(fake, torch.zeros(3, 3), batch)
    torch.testing.assert_close(seen["coords"], torch.tensor([[0.5, 0.5], [1.5, 1.5], [3.5, 2.5]]))
    assert seen["size"] == (3, 4) and seen["frame"] == "p"
    batch = SimpleNamespace(camera=camera, image_size=(2, 2), rays=SimpleNamespace(pixel_indices=None))
    Trainer._apply_camera_isp(fake, torch.zeros(4, 3), batch)  # resolution schedule: whole image at 1/2
    torch.testing.assert_close(seen["coords"], torch.tensor([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]]))
    assert seen["size"] == (2, 2)
    assert Trainer._apply_camera_isp(SimpleNamespace(camera_isp=None), torch.ones(2, 3), batch).sum() == 6


def test_ocean_full_term_goes_through_the_camera_model():
    scene = plane_scene()
    camera = plane_camera()
    renderer = PlaneRenderer()
    target = renderer.render(oc.image_rays(camera, 1)).direct.reshape(48, 64, 3) + 0.05
    run = lambda model: oc.ocean_terms(renderer, lambda origin: scene, None, (0, 0, 0), camera, target, 0.6, 1, 1e-3,
                                       1.0, 1000, generator=torch.Generator().manual_seed(1), camera_model=model)
    plain, darker = run(None), run(lambda image: 0.5 * image)
    assert darker["full"].item() > plain["full"].item() + 0.1  # the photo is compared with the camera's image
    torch.testing.assert_close(darker["obj"], plain["obj"])
