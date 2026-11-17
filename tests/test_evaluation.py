"""Paper evaluation conventions, including missing water pixels and camera/medium composition."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch
from PIL import Image

from evaluation.metrics import check_psnr, depth_errors, view_opacity
from evaluation.evaluator import Evaluator
from evaluate import separation


def test_separation_weights_views_then_classes(monkeypatch, tmp_path):
    # The second view has no water; its seabed still contributes equally to coverage.
    masks = tmp_path / "masks.npz"
    np.savez(masks, a=np.array([[1, 0]], bool), b=np.array([[0, 0]], bool))
    images = {"a": torch.tensor([[[0.2], [0.8]]]), "b": torch.tensor([[[0.4], [0.4]]])}
    monkeypatch.setattr("evaluate.render_view", lambda trainer, camera, modes: {"alpha": images[camera.name]})
    trainer = SimpleNamespace(dataset=SimpleNamespace(test=[SimpleNamespace(name=n) for n in images]))
    result = separation(trainer, masks)
    assert result["mean"]["op_seabed"] == pytest.approx(0.6)
    assert result["mean"]["op_water"] == pytest.approx(0.2)
    assert result["mean"]["separation"] == pytest.approx(0.7)
    trainer.dataset.test.pop()
    with pytest.raises(ValueError, match="differ"):
        separation(trainer, masks)


def test_opacity_requires_a_seabed_and_matching_pixels():
    with pytest.raises(ValueError, match="seabed"):
        view_opacity(np.ones(2), np.ones(2, bool))
    with pytest.raises(ValueError, match="sizes"):
        view_opacity(np.ones(3), np.ones(2, bool))


def test_camera_response_follows_water_light_and_clamping_is_last():
    evaluator = SimpleNamespace(water_light=lambda image, scatter, rays: image + 2 * scatter,
                                camera_isp=lambda image, camera: image * 0.5)
    result = SimpleNamespace(image=torch.tensor([0.4, 0.8]), backscatter=torch.tensor([0.6, 1.0]))
    raw, full = Evaluator.finish_image(evaluator, result, None, None)
    torch.testing.assert_close(raw, torch.tensor([1.6, 2.8]))
    torch.testing.assert_close(full, torch.tensor([0.8, 1.0]))


def test_depth_scale_is_per_view_and_invalid_pixels_are_excluded():
    result = depth_errors(torch.tensor([2., 4., 0., float("nan")]), torch.tensor([1., 2., 3., 4.]))
    assert result == {"depth_absrel": 1., "depth_absrel_median_scaled": 0., "depth_ratio_median": 2.}
    with pytest.raises(ValueError, match="valid"):
        depth_errors(torch.zeros(3), torch.ones(3))


def test_saved_psnr_must_match_every_view():
    assert check_psnr({"a": 30.0001}, {"a": 30.0}) < 0.001
    with pytest.raises(ValueError, match="differs"):
        check_psnr({"a": 31.0}, {"a": 30.0})
    with pytest.raises(ValueError, match="views"):
        check_psnr({"b": 30.0}, {"a": 30.0})


def test_fog_equation_and_ray_depth_convention():
    import scripts.make_foggy_garden as fog
    z = np.ones((1, 1))
    np.testing.assert_array_equal(fog.ray_distance(z, 1, 1, 0.5, 0.5), z)
    clear = np.ones((1, 1, 3))
    np.testing.assert_allclose(fog.apply_fog(clear, z, fog.FOG_PRESETS["easy"]),
                               np.exp(-0.6) + 0.5 * (1 - np.exp(-0.6)))
    np.testing.assert_array_equal(fog.to_uint8(np.array([0., 0.5, 1.])), [0, 128, 255])


def test_transmittance_exports_float_data_and_checks_view_names(monkeypatch, tmp_path):
    from evaluate import transmittance
    monkeypatch.setattr("evaluation.evaluator._rays", lambda camera: torch.zeros(1, 2, 6))
    camera = SimpleNamespace(name="a", height=1, width=2)
    result = SimpleNamespace(transmittance=torch.tensor([[0., 1.]]), direct_transmittance=None)
    trainer = SimpleNamespace(dataset=SimpleNamespace(test=[camera]),
                              model=SimpleNamespace(medium=SimpleNamespace(render_scene=lambda: None)),
                              renderer=SimpleNamespace(render_voronoi=lambda *a, **kw: result),
                              config=SimpleNamespace(background=(0, 0, 0)))
    output = transmittance(trainer, tmp_path)
    assert output["mean"]["transmittance"] == 0.5
    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path / "a_medium_transmittance.png")), [[0, 255]])
    with np.load(tmp_path / "a_medium_transmittance.npz") as archive:
        np.testing.assert_array_equal(archive["transmittance"], [[0., 1.]])
    with pytest.raises(ValueError, match="not test views"):
        transmittance(trainer, tmp_path, ["missing"])


@pytest.mark.parametrize("level,preset", [("easy", "easy"), ("hard", "hard_fig5")])
def test_fog_generation_matches_paper_image(level, preset):
    import os
    import scripts.make_foggy_garden as fog
    required = ("FS_FOG_DEPTHS", "FS_GARDEN_CLEAR", "FS_GARDEN_COLMAP", "FS_FOGGY")
    if not all(os.environ.get(name) for name in required):
        pytest.skip("set FS_FOG_DEPTHS, FS_GARDEN_CLEAR, FS_GARDEN_COLMAP and FS_FOGGY for dataset regression")
    depths, clear, colmap, reference = [Path(os.environ[name]) for name in required]
    data = reference / f"garden_{level}"
    name = (data / "test_list.txt").read_text().splitlines()[0]
    size = Image.open(clear / (Path(name).stem + ".JPG")).size
    views = sorted(json.loads((depths / "depth_summary.json").read_text())["per_view"])
    model = fog.scaled_model(colmap, views, size)
    image = next(image for image in model.images.values() if image.name == name)
    z = fog.ray_distance(np.load(depths / "depth_ns" / Path(name).with_suffix(".npy")),
                        *model.cameras[image.camera_id].params)
    photo = np.asarray(Image.open(clear / (Path(name).stem + ".JPG")).convert("RGB"))
    generated = fog.to_uint8(fog.apply_fog(fog.to_float(photo), z, fog.FOG_PRESETS[preset]))
    np.testing.assert_array_equal(generated, np.asarray(Image.open(data / "images" / name)))


def test_sam_proposal_selection_prioritizes_prompt_agreement():
    from scripts.sam_water_mask import choose, clean, segment
    masks = np.array([[[1, 1], [1, 1]], [[1, 0], [1, 0]], [[1, 0], [1, 0]]], bool)
    assert choose(masks, [0.99, 0.5, 0.7], np.array([[0, 0], [1, 1]]), np.array([1, 0])) == 2
    empty, _ = segment(None, np.zeros((10, 10, 3)), {"water": [], "seabed": [(0.5, 0.5)]})
    assert not empty.any()
    mask = np.zeros((20, 20), bool)
    mask[1, 1] = True
    assert not clean(mask, min_region=0.01).any()
