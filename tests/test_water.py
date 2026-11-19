"""Water: initial fit on the COLMAP tracks, the phase function, the extension beyond the training directions, the anchor."""
import json
import math
import os
import struct
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from foamsurf_model.medium import VoronoiMediumModel
from train import fit_iui3, resolve_iui3
from training import water_fit
from training.config import ExperimentConfig, ModelConfig
from training.config import ExperimentConfig
from training.water_phase import WaterPhase
from types import SimpleNamespace
from training.trainer import Trainer
from training.water_extrapolation import WaterExtrapolation, henyey_greenstein
import csv
from training.losses import MediumFixConfig, water_anchor_loss
from helpers import ROOT, requires_gpu  # noqa: F401


def synthetic(beta_d, beta_b, b_inf, points=400, seed=0):
    rng = np.random.default_rng(seed)
    index, colours, distances = [], [], []
    for k in range(points):
        j = rng.uniform(0.2, 0.9, 3)
        for d in rng.uniform(3.0, 20.0, 4):
            colours.append(np.asarray(b_inf) * (1 - np.exp(-np.asarray(beta_b) * d)) + j * np.exp(-np.asarray(beta_d) * d))
            index.append(k)
            distances.append(d)
    return np.array(index), np.array(colours), np.array(distances)


def test_recovers_known_water_per_channel():
    grid = water_fit.GRID
    beta_d, beta_b, b_inf = [grid[15], grid[12], grid[9]], [grid[10], grid[13], grid[14]], [0.2, 0.35, 0.45]
    data = synthetic(beta_d, beta_b, b_inf)
    fit = water_fit.fit_water(*data)
    for c in range(3):
        assert math.isclose(fit["beta_d"][c], beta_d[c], rel_tol=1e-6)
        assert math.isclose(fit["beta_b"][c], beta_b[c], rel_tol=1e-6)
        assert abs(fit["b_inf"][c] - b_inf[c]) < 1e-3
    fixed = water_fit.fit_water(*data, b_inf=b_inf)  # B_inf known: the same betas, B_inf kept
    assert fixed["beta_d"] == fit["beta_d"] and fixed["beta_b"] == fit["beta_b"]
    assert fixed["b_inf"] == pytest.approx(b_inf)
    brighter = water_fit.fit_water(*data, b_inf=[0.4, 0.7, 0.9])  # brighter water: weaker haze for it
    assert all(brighter["beta_b"][c] < beta_b[c] for c in range(3))


def write_model(folder, centres, points, keypoints, width=40):
    """Minimal binary COLMAP model: one PINHOLE camera, identity rotations."""
    folder.mkdir(parents=True)
    with open(folder / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", 1) + struct.pack("<iiQQ", 1, 1, width, width) + struct.pack("<dddd", 20, 20, 20, 20))
    with open(folder / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(centres)))
        for i, (centre, kps) in enumerate(zip(centres, keypoints), start=1):
            t = -np.asarray(centre, float)  # R = I, centre = -t
            f.write(struct.pack("<idddddddi", i, 1, 0, 0, 0, *t, 1) + f"v{i}.png".encode() + b"\x00")
            f.write(struct.pack("<Q", len(kps)) + b"".join(struct.pack("<ddq", x, y, p) for x, y, p in kps))
    with open(folder / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for pid, xyz in points.items():
            f.write(struct.pack("<QdddBBBd", pid, *xyz, 0, 0, 0, 0.0) + struct.pack("<Q", 0))


def test_tracks_and_observations(tmp_path):
    points = {7: (0.0, 0.0, 10.0), 8: (1.0, 0.0, 10.0)}
    centres = [(0, 0, 0), (0, 0, 4), (0, 0, -6)]
    keypoints = [[(20, 20, 7), (30, 20, 8)], [(20, 20, 7), (35, 20, 8)], [(20, 20, 7), (-1, 20, -1)]]
    write_model(tmp_path / "sparse/0", centres, points, keypoints)
    images, pts, widths = water_fit.read_tracks(tmp_path / "sparse/0")
    assert len(images) == 3 and set(pts) == {7, 8} and widths == {1: 40}
    photos = {f"v{i}": np.full((20, 20, 3), 0.1 * i) for i in (1, 2, 3)}  # half resolution: keypoints scaled
    index, colours, distances = water_fit.observations(images, pts, widths, photos, min_views=3, min_spread=1.3)
    assert index.tolist() == [0, 0, 0]  # only point 7 has three views (point 8: two, one out of frame)
    assert sorted(np.round(distances, 3).tolist()) == [6.0, 10.0, 16.0]
    assert sorted(np.round(colours[:, 0], 3).tolist()) == [0.1, 0.2, 0.3]


def test_uniform_medium_coefficients():
    medium = VoronoiMediumModel(torch.zeros(3), torch.ones(3), ModelConfig(medium_type="rgb_backscatter", medium_points=8), "cpu")
    medium.set_uniform_coefficients_(torch.tensor([0.17, 0.13, 0.1]), torch.tensor([0.06, 0.1, 0.13]))
    coefficients = medium.optical_coefficients
    torch.testing.assert_close(coefficients, torch.tensor([0.17, 0.13, 0.1, 0.06, 0.1, 0.13]).expand(8, 6), rtol=1e-5, atol=1e-6)


UP = torch.tensor([0.0, 0.0, 1.0])


def test_starts_as_the_plain_model_and_scales_only_the_backscatter():
    phase = WaterPhase(UP * 3.0, "cpu")  # any length: normalised
    image, backscatter = torch.rand(6, 3), torch.rand(6, 3) * 0.2
    directions = torch.randn(6, 3)
    torch.testing.assert_close(phase.apply(image, backscatter, directions), image)  # g = 0: isotropic
    torch.testing.assert_close(phase.toward_light.detach(), UP)
    with torch.no_grad():
        phase.raw_anisotropy.fill_(math.atanh(0.5 / 0.95))  # g = 0.5, forward scattering
    looking_up, sideways, down = torch.tensor([[0.0, 0.0, 2.0]]), torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[0.0, 0.0, -1.0]])
    torch.testing.assert_close(phase.gain(looking_up), torch.full((1, 3), (1 - 0.25) / (1 - 0.5) ** 3))  # = 6
    torch.testing.assert_close(phase.gain(sideways), torch.full((1, 3), 0.75 / 1.25 ** 1.5))
    assert bool((phase.gain(looking_up) > phase.gain(sideways)).all()) and bool((phase.gain(sideways) > phase.gain(down)).all())
    direct = image[:1] - backscatter[:1]
    torch.testing.assert_close(phase.apply(image[:1], backscatter[:1], looking_up) - direct, backscatter[:1] * 6.0)


def test_phase_integrates_to_the_isotropic_one():
    # the HG phase over the isotropic one averages to 1 over the sphere: brightening toward the light is
    # paid for by darkening away from it (no free gain in total scattered light)
    phase = WaterPhase(UP, "cpu")
    with torch.no_grad():
        phase.raw_anisotropy.copy_(torch.tensor([0.2, 0.6, -0.4]))
    torch.manual_seed(0)
    directions = torch.nn.functional.normalize(torch.randn(400_000, 3), dim=-1)
    torch.testing.assert_close(phase.gain(directions).mean(0), torch.ones(3), atol=0.02, rtol=0)


def test_learns_the_light_direction_and_anisotropy():
    torch.manual_seed(0)
    light = torch.nn.functional.normalize(torch.tensor([0.3, 0.1, 1.0]), dim=0)
    truth = WaterPhase(light, "cpu")
    with torch.no_grad():
        truth.raw_anisotropy.fill_(math.atanh(0.4 / 0.95))
    directions = torch.nn.functional.normalize(torch.randn(2048, 3), dim=-1)
    directions = directions[directions[:, 2] < 0.6]  # like the training photos: never looking far up
    backscatter = torch.full((len(directions), 3), 0.3)
    target = truth.apply(0.1 + backscatter, backscatter, directions).detach()
    phase = WaterPhase(UP, "cpu", lr=0.02)
    for _ in range(800):
        phase.zero_grad()
        ((phase.apply(0.1 + backscatter, backscatter, directions) - target) ** 2).mean().backward()
        phase.step()
    torch.testing.assert_close(phase.anisotropy.detach(), torch.full((3,), 0.4), atol=0.03, rtol=0)
    found = torch.nn.functional.normalize(phase.toward_light.detach(), dim=0)
    assert float(found @ light) > 0.99
    # and it extrapolates: looking straight up (never in the data) gets the true gain
    up = torch.tensor([[0.0, 0.0, 1.0]])
    torch.testing.assert_close(phase.gain(up).detach(), truth.gain(up).detach(), rtol=0.1, atol=0)
    copy = WaterPhase(UP, "cpu")
    copy.load_state_dict(phase.state_dict())
    torch.testing.assert_close(copy.gain(up), phase.gain(up))


def test_fixed_light_learns_only_the_anisotropy():
    torch.manual_seed(0)
    truth = WaterPhase(UP, "cpu")
    with torch.no_grad():
        truth.raw_anisotropy.fill_(math.atanh(0.6 / 0.95))
    directions = torch.nn.functional.normalize(torch.randn(2048, 3), dim=-1)
    backscatter = torch.full((len(directions), 3), 0.3)
    target = truth.apply(0.1 + backscatter, backscatter, directions).detach()
    start = torch.nn.functional.normalize(torch.tensor([0.2, 0.0, 1.0]), dim=0)  # a tilted "up"
    phase = WaterPhase(start, "cpu", lr=0.05, fixed_light=True)
    for _ in range(300):
        phase.zero_grad()
        ((phase.apply(0.1 + backscatter, backscatter, directions) - target) ** 2).mean().backward()
        phase.step()
    torch.testing.assert_close(phase.toward_light, start)  # never moved
    assert not phase.toward_light.requires_grad
    assert float(phase.anisotropy.mean()) > 0.4  # g still found, as well as a tilted light allows
    copy = WaterPhase(start, "cpu", fixed_light=True)
    copy.load_state_dict(phase.state_dict())
    torch.testing.assert_close(copy.raw_anisotropy, phase.raw_anisotropy.detach())


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


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


def test_anchor_moves_the_level_not_the_differences():
    coefficients = torch.tensor([[0.1, 0.1, 0.1, 1, 1, 1], [0.4, 0.4, 0.4, 1, 1, 1], [9.0, 9, 9, 1, 1, 1]],
                                requires_grad=True)
    reference = torch.tensor([True, True, False])
    target = torch.tensor([0.4, 0.4, 0.4])
    loss = water_anchor_loss(coefficients, reference, target)
    level = (torch.log(torch.tensor(0.1)) + torch.log(torch.tensor(0.4))) / 2  # mean log = log 0.2
    torch.testing.assert_close(loss.detach(), 3 * (level - torch.log(torch.tensor(0.4))) ** 2)
    loss.backward()
    grad = coefficients.grad
    # d/d log sigma is the same for every reference cell: the level moves, the ratio 0.1 : 0.4 stays
    torch.testing.assert_close(grad[0, :3] * 0.1, grad[1, :3] * 0.4)
    assert bool((grad[:2, :3] < 0).all())  # towards the larger fitted attenuation
    assert grad[2].abs().sum() == 0 and grad[:, 3:].abs().sum() == 0  # other cells, backscatter untouched
    assert water_anchor_loss(coefficients, torch.zeros(3, dtype=torch.bool), target) is None
    assert MediumFixConfig().water_anchor_weight == 0.0
    assert MediumFixConfig(water_anchor_weight=0.1).water_anchor_weight_at(0) == 0.1
