"""edit_scene.py on the CPU: the surfel transform under a deformation and the medium edits."""
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from examples import edit as E
from foamsurf_model.gaussian_adaptation import _quaternion_rotation


def frame(seed=0, dtype=torch.float32):
    q = F.normalize(torch.randn(1, 4, generator=torch.Generator().manual_seed(seed), dtype=dtype), dim=-1)
    basis = _quaternion_rotation(q)[0]
    return dict(c=torch.tensor([0.3, -0.2, 1.0], dtype=dtype), e1=basis[:, 0], e2=basis[:, 1], n=basis[:, 2], L=2.0,
                H=0.5, h_lo=-0.1)


def gaussians(count=64, seed=1, dtype=torch.float64):
    """float64: the Jacobian is a central difference, in float32 its round-off is ~1e-4 relative."""
    g = torch.Generator().manual_seed(seed)
    return SimpleNamespace(means=torch.randn(count, 3, generator=g, dtype=dtype),
                           log_scales=torch.randn(count, 2, generator=g, dtype=dtype) * 0.3 - 2,
                           quaternions=F.normalize(torch.randn(count, 4, generator=g, dtype=dtype), dim=-1))


def covariance(g):
    rot = _quaternion_rotation(g.quaternions)
    axes = rot[:, :, :2] * torch.exp(g.log_scales)[:, None, :]
    return axes @ axes.transpose(1, 2)


def test_quaternion_round_trip():
    q = F.normalize(torch.randn(500, 4, generator=torch.Generator().manual_seed(2)), dim=-1)
    rot = _quaternion_rotation(q)
    assert torch.allclose(_quaternion_rotation(E.matrix_to_quaternion(rot)), rot, atol=1e-5)


def test_linear_deformation_maps_centres_covariances_and_normals():
    f = frame(dtype=torch.float64)
    M = torch.tensor([[1.3, 0.2, 0.0], [0.0, 0.8, 0.1], [0.3, 0.0, 1.6]], dtype=torch.float64)  # det > 0
    g = gaussians()
    before, normals, means = covariance(g), _quaternion_rotation(g.quaternions)[:, :, 2], g.means.clone()
    E.deform_gaussians(g, f, lambda p, f: p @ M.T)
    basis = torch.stack((f["e1"], f["e2"], f["n"]), 1)
    J = basis @ M @ basis.T  # the same map in world coordinates
    assert torch.allclose(g.means, f["c"] + (means - f["c"]) @ J.T, atol=1e-9)
    assert torch.allclose(covariance(g), J @ before @ J.T, atol=1e-9)
    cofactor = torch.linalg.solve(J.T, normals.T).T
    assert ((_quaternion_rotation(g.quaternions)[:, :, 2] * cofactor).sum(-1) > 0).all()


def test_identity_keeps_every_surfel():
    g = gaussians()
    before, means = covariance(g), g.means.clone()
    E.deform_gaussians(g, frame(dtype=torch.float64), lambda p, f: p)
    assert torch.allclose(g.means, means, atol=1e-9)
    assert torch.allclose(covariance(g), before, atol=1e-9)


def test_deformations_are_finite_and_keep_the_plane_centre():
    f, p = frame(), torch.zeros(1, 3)
    for name, phi in E.DEFORMATIONS.items():
        out = phi(torch.randn(100, 3) * f["L"], f)
        assert torch.isfinite(out).all(), name
    assert torch.allclose(E.def_bend(p, f), p) and torch.allclose(E.def_twist(p, f), p, atol=1e-6)


def medium(count=200, seed=3):
    g = torch.Generator().manual_seed(seed)
    return SimpleNamespace(raw_density=torch.randn(count, 6, generator=g) - 3, raw_color=torch.randn(count, 3, generator=g),
                           sites=torch.randn(count, 3, generator=g))


def test_scale_and_recolour():
    m = medium()
    density = F.softplus(m.raw_density)
    E.scale_coefficients(m, (0.25,) * 3, (2.5,) * 3)
    assert torch.allclose(F.softplus(m.raw_density), density * torch.tensor([0.25] * 3 + [2.5] * 3), rtol=1e-4)
    m = medium()
    weights = torch.tensor([0.299, 0.587, 0.114])
    luminance, backscatter = torch.sigmoid(m.raw_color) @ weights, F.softplus(m.raw_density[:, 3:])
    E.recolour(m, *E.WATER_TYPES["green"])
    colour = torch.sigmoid(m.raw_color)
    assert torch.allclose(colour @ weights, luminance, atol=1e-3)
    assert torch.allclose(F.normalize(colour, dim=-1), F.normalize(torch.tensor([0.34, 0.56, 0.30]), dim=0).expand_as(colour), atol=1e-3)
    assert torch.allclose(F.softplus(m.raw_density[:, 3:]), backscatter)


def test_transplant_matches_the_statistics(tmp_path):
    other = medium(seed=4)
    other.raw_density, other.raw_color = other.raw_density * 0.5 + 1.0, other.raw_color * 2 - 1
    (tmp_path / "joint" / "checkpoints").mkdir(parents=True)
    torch.save({"model": {"medium.raw_density": other.raw_density, "medium.raw_color": other.raw_color}},
               tmp_path / "joint" / "checkpoints" / "last.pt")
    m = medium()
    E.transplant(m, tmp_path)
    log_m, log_o = torch.log(F.softplus(m.raw_density)), torch.log(F.softplus(other.raw_density))
    assert torch.allclose(log_m.mean(0), log_o.mean(0), atol=1e-3) and torch.allclose(log_m.std(0), log_o.std(0), atol=1e-3)
    assert torch.allclose(m.raw_color.mean(0), other.raw_color.mean(0), atol=1e-4)


def test_haze_and_plume_only_add_density():
    f = frame()
    for edit in ("haze", "plume"):
        m = medium()
        density = F.softplus(m.raw_density)
        if edit == "haze":
            E.haze(m, f)
        else:
            E.plume(m, f, torch.zeros(3))
        after = F.softplus(m.raw_density)
        assert torch.isfinite(after).all() and (after >= density - 1e-5).all(), edit
        assert (after > density + 1e-6).any(), edit


def test_haze_patches3d_rainbow_patches():
    """patches3d: every layer cell gets the hue of its nearest seed (saturation 0.85), deterministic, density as RGB."""
    f = frame()
    a, b, rgb = medium(), medium(), medium()
    E.haze(a, f, strength=15.0, colour="patches3d", patch_size=0.25)
    E.haze(b, f, strength=15.0, colour="patches3d", patch_size=0.25)
    E.haze(rgb, f, strength=15.0)
    assert torch.equal(a.raw_color, b.raw_color) and torch.equal(a.raw_density, rgb.raw_density)
    sites = a.sites
    w = torch.ones(len(sites))
    hues = E.patch_hues(sites, w, 0.5)
    assert hues.shape == (len(sites), 3) and (hues.amax(-1) - 1).abs().max() < 1e-6
    assert ((hues.amax(-1) - hues.amin(-1)) - 0.85).abs().max() < 1e-6       # HSV saturation 0.85, value 1
    assert len(torch.unique(hues, dim=0)) > 1                                   # several patches
