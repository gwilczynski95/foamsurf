"""Inference-time edits of a trained FoamSurf run, separately for the surface (2D Gaussians) and the water (Voronoi
medium), without retraining. Each edit starts from the loaded model, is rendered from one camera and, with --save,
written as a run directory that render.py loads like any trained run.

    python edit.py --run RUN --scene DATA --view CAM --edits bulge,relief,clear,haze --out DIR [--save MODELS]

Frame. Edits are defined in a frame fitted to the reef visible from --view (reef_frame): c = mean of the visible
surface points (the 10% farthest from their median dropped), n = normal of their least-squares plane (towards the
camera), e2 = viewing direction projected onto the plane, e1 = e2 x n; local coordinates (a, b, h). L = 0.9 quantile
of the horizontal distance of the points from c, H = their height range (0.005-0.995 quantiles).

Surface edits move every Gaussian through a deformation phi given in local coordinates: mu' = phi(mu). The scaled
tangent axes A = [s_u t_u, s_v t_v] map through the Jacobian J of phi (central differences, step 1e-3 L) and are
refactored by a thin SVD, J A = U S V^T: new axes = columns of U, new scales = S, normal = u1 x u2 oriented like
J^-T n (cofactor). The surfel covariance becomes J A A^T J^T. Opacity, kernel exponent and SH colour (bands not
rotated) and the medium are kept; the Mip-Splatting 3D filter is recomputed from the training cameras.
    waves    h + 0.07 L sin(2 pi (0.8 b + 0.6 a) / 0.45 L)            sand dunes
    ripple   h + 0.06 L cos(2 pi r / 0.28 L) exp(-r / 0.7 L)           concentric ripple
    twist    (a, b) rotated by 150 deg exp(-r^2 / (0.45 L)^2)          swirl about the normal
    bulge    h + 0.22 L exp(-(a^2 + (b - 0.15 L)^2) / (2 (0.25 L)^2))  a hill under the reef
    relief   2.5 h                                                     relief exaggerated
    bend     the seabed rolled into a half-pipe along e2: rho = 0.6 L, theta = clip(a / rho, +-0.6 pi),
             a' = (rho - h) sin theta, h' = rho - (rho - h) cos theta
    clone    rigid copy (rotation about n, scale) of the Gaussians around the surface under one pixel to another (--clone)

Water edits change the per-cell attenuation sigma_a, backscatter sigma_b (after softplus) and colour c (after sigmoid);
the Gaussians are kept.
    clear / murky        sigma_a, sigma_b x 0.25 / x 2.5
    turquoise, green,    colour -> a target hue keeping each cell's luminance (Rec. 601 weights), sigma_a scaled per
    redtide              channel (see WATER_TYPES)
    transplant           per-channel mean / std of log sigma and of the colour logits matched to the medium of
                         another run (--transplant-from), spatial pattern kept
    haze                 cells split x8 (--medium-split), then a layer over the seabed:
                         w = exp(-max(h - h_lo, 0) / (0.25 H + 0.04 L)) [h > h_lo - 0.3 H] max(1 + 0.4 noise, 0),
                         sigma_a += 3 w m_a, sigma_b += 5 w m_b (m = mean of the per-channel medians), colour mixed
                         towards (0.55, 0.62, 0.55); --haze-colour patches3d paints random rainbow 3D patches
                         instead (paper: --haze-strength 15 --medium-split 24 --patch-size 0.25 on JG MTN_1098)
    plume                150k cells added in an ellipsoid above the surface under --plume-at (each with the parameters
                         of its nearest cell), then a rising, widening column: width 0.035 L + 0.35 h,
                         sigma_b += 2.5 w / 0.1 L, sigma_a += half of it, colour mixed towards (0.80, 0.74, 0.62)
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

import render as foamsurf_render
from foamsurf_model.gaussian_adaptation import _quaternion_rotation

WATER_TYPES = {  # target colour, attenuation factor per channel (R, G, B)
    "turquoise": ((0.10, 0.78, 0.72), (0.45, 0.6, 0.8)),
    "green": ((0.34, 0.56, 0.30), (1.0, 0.95, 1.5)),
    "redtide": ((0.55, 0.27, 0.20), (0.9, 1.35, 1.5)),
}
ALIASES = {"plume2": "plume"}  # name used for the paper package (2026-11-15)


# ------------------------------------------------------------------------------------------------------------ frame
@torch.no_grad()
def reef_frame(trainer, camera) -> dict:
    from evaluation.evaluator import _rays
    rays = _rays(camera)[::4, ::4].reshape(-1, 6).contiguous()
    g = trainer.model.gaussians.render_scene(camera.origin)
    trainer.renderer.set_gaussian_geometry(g)
    s = trainer.renderer.render_gaussians(rays, g, background=(0.0, 0.0, 0.0))
    depth, alpha = s.surface_depth.reshape(-1), s.alpha.reshape(-1)
    valid = (depth > 0) & (alpha > 0.7)
    points = rays[valid, :3] + rays[valid, 3:] * depth[valid, None]
    center = points.median(0).values
    distance = (points - center).norm(dim=-1)
    points = points[distance <= torch.quantile(distance, 0.9)]
    center = points.mean(0)
    _, vectors = torch.linalg.eigh((points - center).T @ (points - center) / len(points))
    n = vectors[:, 0]
    if torch.dot(n, camera.origin - center) < 0:
        n = -n
    e2 = F.normalize(camera.forward - (camera.forward @ n) * n, dim=0)
    e1 = torch.linalg.cross(e2, n)
    local = (points - center) @ torch.stack((e1, e2, n), 1)
    radius = local[:, :2].norm(dim=-1)
    return dict(c=center, e1=e1, e2=e2, n=n, L=float(torch.quantile(radius, 0.9)),
                H=float(torch.quantile(local[:, 2], 0.995) - torch.quantile(local[:, 2], 0.005)),
                b_lo=float(torch.quantile(local[:, 1], 0.05)), b_hi=float(torch.quantile(local[:, 1], 0.95)),
                h_lo=float(torch.quantile(local[:, 2], 0.02)), cam=camera.origin)


def to_local(x, f):
    return (x - f["c"]) @ torch.stack((f["e1"], f["e2"], f["n"]), 1)


def to_world(p, f):
    return f["c"] + p @ torch.stack((f["e1"], f["e2"], f["n"]), 0)


# ------------------------------------------------------------------------------------------ deformations (local)
def def_waves(p, f):
    L = f["L"]
    a, b, h = p.unbind(-1)
    return torch.stack((a, b, h + 0.07 * L * torch.sin(2 * math.pi * (0.8 * b + 0.6 * a) / (0.45 * L))), -1)


def def_ripple(p, f):
    L = f["L"]
    a, b, h = p.unbind(-1)
    r = torch.sqrt(a * a + b * b + 1e-12)
    return torch.stack((a, b, h + 0.06 * L * torch.cos(2 * math.pi * r / (0.28 * L)) * torch.exp(-r / (0.7 * L))), -1)


def def_twist(p, f):
    L = f["L"]
    a, b, h = p.unbind(-1)
    theta = math.radians(150.0) * torch.exp(-(a * a + b * b) / (0.45 * L) ** 2)
    cos, sin = torch.cos(theta), torch.sin(theta)
    return torch.stack((cos * a - sin * b, sin * a + cos * b, h), -1)


def def_bulge(p, f):
    L = f["L"]
    a, b, h = p.unbind(-1)
    r2 = a * a + (b - 0.15 * L) ** 2
    return torch.stack((a, b, h + 0.22 * L * torch.exp(-r2 / (2 * (0.25 * L) ** 2))), -1)


def def_relief(p, f):
    a, b, h = p.unbind(-1)
    return torch.stack((a, b, 2.5 * h), -1)


def def_bend(p, f):
    L = f["L"]
    a, b, h = p.unbind(-1)
    rho = 0.6 * L
    theta = (a / rho).clamp(-math.pi * 0.6, math.pi * 0.6)
    return torch.stack(((rho - h) * torch.sin(theta), b, rho - (rho - h) * torch.cos(theta)), -1)


DEFORMATIONS = {"waves": def_waves, "ripple": def_ripple, "twist": def_twist, "bulge": def_bulge,
                "relief": def_relief, "bend": def_bend}


# ---------------------------------------------------------------------------------------------- Gaussian transform
def matrix_to_quaternion(m: torch.Tensor) -> torch.Tensor:
    """(w, x, y, z) of rotation matrices [N,3,3] (Shepperd's method), the inverse of _quaternion_rotation."""
    m00, m11, m22 = m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]
    q = torch.stack((1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22), -1)
    choice = q.argmax(-1)
    out = torch.empty(m.shape[0], 4, device=m.device, dtype=m.dtype)
    for k in range(4):
        sel = choice == k
        if not sel.any():
            continue
        mm = m[sel]
        t = q[sel, k].clamp_min(1e-12).sqrt() * 2  # 4 |component k|
        if k == 0:
            v = (t / 4, (mm[:, 2, 1] - mm[:, 1, 2]) / t, (mm[:, 0, 2] - mm[:, 2, 0]) / t, (mm[:, 1, 0] - mm[:, 0, 1]) / t)
        elif k == 1:
            v = ((mm[:, 2, 1] - mm[:, 1, 2]) / t, t / 4, (mm[:, 0, 1] + mm[:, 1, 0]) / t, (mm[:, 0, 2] + mm[:, 2, 0]) / t)
        elif k == 2:
            v = ((mm[:, 0, 2] - mm[:, 2, 0]) / t, (mm[:, 0, 1] + mm[:, 1, 0]) / t, t / 4, (mm[:, 1, 2] + mm[:, 2, 1]) / t)
        else:
            v = ((mm[:, 1, 0] - mm[:, 0, 1]) / t, (mm[:, 0, 2] + mm[:, 2, 0]) / t, (mm[:, 1, 2] + mm[:, 2, 1]) / t, t / 4)
        out[sel] = torch.stack(v, -1)
    return F.normalize(out, dim=-1)


@torch.no_grad()
def deform_gaussians(g, f, phi_local, chunk: int = 200_000) -> None:
    """Means through phi, surfel axes and scales through its Jacobian (see the module docstring)."""
    def phi(x):
        return to_world(phi_local(to_local(x, f), f), f)
    eps = 1e-3 * f["L"]
    eye = torch.eye(3, device=g.means.device, dtype=g.means.dtype)
    new_means, new_q, new_s = [], [], []
    for start in range(0, g.means.shape[0], chunk):
        x = g.means[start:start + chunk]
        J = torch.stack([(phi(x + eps * eye[i]) - phi(x - eps * eye[i])) / (2 * eps) for i in range(3)], -1)
        rot = _quaternion_rotation(g.quaternions[start:start + chunk])
        axes = rot[:, :, :2] * torch.exp(g.log_scales[start:start + chunk])[:, None, :]
        U, S, _ = torch.linalg.svd(J @ axes, full_matrices=False)
        normal = torch.linalg.cross(U[..., 0], U[..., 1])
        old = torch.linalg.solve(J.transpose(1, 2), rot[:, :, 2])  # cofactor direction of the old normal
        flip = (normal * old).sum(-1) < 0
        U[flip, :, 1] = -U[flip, :, 1]
        normal[flip] = -normal[flip]
        new_means.append(phi(x))
        new_q.append(matrix_to_quaternion(torch.stack((U[..., 0], U[..., 1], normal), -1)))
        new_s.append(torch.log(S.clamp_min(1e-8)))
    g.means.copy_(torch.cat(new_means))
    g.quaternions.copy_(torch.cat(new_q))
    g.log_scales.copy_(torch.cat(new_s))


@torch.no_grad()
def surface_point(trainer, camera, u: float, v: float) -> torch.Tensor:
    """World point of the surface under the normalised pixel (u, v): median depth of the 5x5 block around it,
    along the ray of the pixel."""
    from evaluation.evaluator import _rays
    rays = _rays(camera)
    row, col = int(v * (camera.height - 1)), int(u * (camera.width - 1))
    block = rays[max(row - 2, 0):row + 3, max(col - 2, 0):col + 3].reshape(-1, 6).contiguous()
    g = trainer.model.gaussians.render_scene(camera.origin)
    trainer.renderer.set_gaussian_geometry(g)
    depth = trainer.renderer.render_gaussians(block, g, background=(0.0, 0.0, 0.0)).surface_depth.reshape(-1)
    depth = depth[depth > 0].median()
    ray = rays[row, col]
    return ray[:3] + F.normalize(ray[3:], dim=0) * depth * ray[3:].norm()


@torch.no_grad()
def clone(trainer, camera, f, src, dst, radius: float, scale: float = 1.0, angle: float = 0.0) -> int:
    """Copy the Gaussians within `radius` x L (horizontally) of the surface under pixel `src` to the surface under
    pixel `dst`, scaled by `scale` and rotated by `angle` degrees about n."""
    g = trainer.model.gaussians
    ls, ld = to_local(surface_point(trainer, camera, *src)[None], f)[0], to_local(surface_point(trainer, camera, *dst)[None], f)[0]
    local = to_local(g.means, f)
    sel = ((local[:, :2] - ls[:2]).norm(dim=-1) < radius * f["L"]) & (local[:, 2] > ls[2] - 0.6 * radius * f["L"])
    t = math.radians(angle)
    rot_local = torch.tensor([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]],
                             device=local.device)
    basis = torch.stack((f["e1"], f["e2"], f["n"]), 1)
    extra = {"means": to_world((local[sel] - ls) @ rot_local.T * scale + ld, f),
             "quaternions": matrix_to_quaternion(basis @ rot_local @ basis.T @ _quaternion_rotation(g.quaternions[sel])),
             "log_scales": g.log_scales[sel] + math.log(scale)}
    for name in g.PARAMETER_NAMES:
        value = getattr(g, name)
        setattr(g, name, torch.nn.Parameter(torch.cat((value, extra.get(name, value[sel])))))
    return int(sel.sum())


# --------------------------------------------------------------------------------------------------- medium edits
def inv_softplus(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp_min(1e-8)
    return x + torch.log(-torch.expm1(-x))


@torch.no_grad()
def scale_coefficients(m, factor_att, factor_bs) -> None:
    c = F.softplus(m.raw_density)
    c[:, :3] *= torch.as_tensor(factor_att, device=c.device, dtype=c.dtype)
    c[:, 3:] *= torch.as_tensor(factor_bs, device=c.device, dtype=c.dtype)
    m.raw_density.copy_(inv_softplus(c))


@torch.no_grad()
def recolour(m, target, att_factor=(1.0, 1.0, 1.0)) -> None:
    """Water colour -> target hue with each cell's luminance kept; attenuation scaled per channel."""
    colours = torch.sigmoid(m.raw_color)
    weights = torch.tensor([0.299, 0.587, 0.114], device=colours.device)
    target = torch.tensor(target, device=colours.device)
    new = target[None] * ((colours @ weights) / (target @ weights))[:, None]
    m.raw_color.copy_(torch.logit(new.clamp(1e-4, 1 - 1e-4)))
    scale_coefficients(m, att_factor, (1.0, 1.0, 1.0))


@torch.no_grad()
def transplant(m, other_run: Path) -> None:
    """Per-channel mean / std of log(sigma) (6 channels) and of the colour logits matched to another run's medium."""
    joint = other_run / "joint" if (other_run / "joint").is_dir() else other_run
    ck = torch.load(joint / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)
    od = ck["model"]["medium.raw_density"].to(m.raw_density.device)
    oc = ck["model"]["medium.raw_color"].to(m.raw_color.device)

    def match(x, y):
        return (x - x.mean(0)) / x.std(0).clamp_min(1e-6) * y.std(0) + y.mean(0)
    log_c = torch.log(F.softplus(m.raw_density))
    m.raw_density.copy_(inv_softplus(torch.exp(match(log_c, torch.log(F.softplus(od))))))
    m.raw_color.copy_(match(m.raw_color, oc))


def noise3(x: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Smooth pseudo-random field in [-1, 1]: 4 octaves of 6 random sinusoids (frequencies x 2, weights / 1.6)."""
    g = torch.Generator().manual_seed(seed)
    out = torch.zeros(x.shape[0], device=x.device)
    for octave in range(4):
        k = torch.randn(6, 3, generator=g).to(x.device) * (2.0 ** octave)
        phase = torch.rand(6, generator=g).to(x.device) * 2 * math.pi
        out += torch.sin(x @ k.T + phase).mean(-1) / (1.6 ** octave)
    return out / out.abs().max().clamp_min(1e-6)


def mix_colour(m, weight, colour) -> None:
    col = torch.sigmoid(m.raw_color)
    new = col * (1 - weight[:, None]) + torch.tensor(colour, device=col.device) * weight[:, None]
    m.raw_color.copy_(torch.logit(new.clamp(1e-4, 1 - 1e-4)))


@torch.no_grad()
def haze(m, f, strength=5.0, colour=(0.55, 0.62, 0.55), rise=1.0, patch_size=0.25):
    """A dense haze layer hugging the seabed, thinning upward. `colour` is an RGB triple, or "patches3d": irregular
    3D patches (Voronoi cells of random seeds of size patch_size * L inside the layer), each a random rainbow hue."""
    L, H = f["L"], f["H"]
    h = to_local(m.sites, f)[:, 2]
    w = torch.exp(-((h - f["h_lo"]).clamp_min(0)) / (rise * (0.25 * H + 0.04 * L))) * (h > f["h_lo"] - 0.3 * H)
    w = w * (1 + 0.4 * noise3(m.sites / (0.15 * L), seed=3)).clamp_min(0)
    c = F.softplus(m.raw_density)
    base = c.median(0).values
    c[:, :3] += strength * 0.6 * w[:, None] * base[:3].mean()
    c[:, 3:] += strength * w[:, None] * base[3:].mean()
    m.raw_density.copy_(inv_softplus(c))
    col = torch.sigmoid(m.raw_color)
    mix = (w / w.max()).clamp(0, 1)[:, None]
    if colour == "patches3d":
        target, mix = patch_hues(m.sites, w, patch_size * L), mix.sqrt()
    else:
        target = torch.tensor(colour, device=col.device)
    m.raw_color.copy_(torch.logit((col * (1 - mix) + target * mix).clamp(1e-4, 1 - 1e-4)))


def patch_hues(sites, w, size, seed=7):
    """RGB per cell: the hue of the nearest of `count` random seeds in the box of the layer's cells (w > 5% of max),
    hues in [0, 0.85) of the colour wheel, HSV saturation 0.85, value 1."""
    from scipy.spatial import cKDTree
    lit = w > 0.05 * w.max()
    lo, hi = sites[lit].amin(0), sites[lit].amax(0)
    count = int(((hi - lo) / size).clamp_min(1).prod().item())
    g = torch.Generator().manual_seed(seed)
    seeds = lo + torch.rand(count, 3, generator=g).to(lo.device) * (hi - lo)
    hues = torch.rand(count, generator=g).to(lo.device) * 0.85
    _, nearest = cKDTree(seeds.cpu().numpy()).query(sites.detach().cpu().numpy())
    hue = hues[torch.as_tensor(nearest, device=lo.device)]
    k = (torch.tensor([5.0, 3.0, 1.0], device=sites.device)[None] + hue[:, None] * 6.0) % 6.0
    print(f"haze patches3d: {count} patches of ~{size:.2f}", flush=True)
    return 1.0 - 0.85 * torch.minimum(k, 4.0 - k).clamp(0, 1)


@torch.no_grad()
def refine_region(trainer, f, centre, radii, count: int, seed: int = 0) -> None:
    """Add `count` cells uniformly inside an ellipsoid (local frame), each with the parameters of its nearest cell:
    the render is unchanged, with fine cells to paint in that region."""
    from scipy.spatial import cKDTree
    from foamsurf_model.medium import VoronoiMediumModel
    from foamsurf_model.topology import VoronoiTopologyController
    medium = trainer.model.medium
    u = torch.rand(count * 3, 3, generator=torch.Generator().manual_seed(seed)) * 2 - 1
    u = u[(u * u).sum(-1) <= 1][:count]
    new = to_world((u * torch.tensor(radii) + torch.tensor(centre)).to(medium.sites.device), f).cpu()
    new = torch.minimum(torch.maximum(new, medium.bbox_min.cpu()), medium.bbox_max.cpu())
    sites, guard = medium.sites.detach().cpu(), medium.guard_mask.cpu()
    _, nearest = cKDTree(sites.numpy()).query(new.numpy(), k=1)
    inner = torch.nonzero(~guard).flatten()
    parent = torch.cat((inner, torch.as_tensor(nearest), torch.nonzero(guard).flatten()))
    new_sites = torch.cat((sites[inner], new, sites[guard]))
    state = {"sites": new_sites, "raw_density": medium.raw_density.detach().cpu()[parent],
             "raw_color": medium.raw_color.detach().cpu()[parent]}
    metadata = dict(medium.checkpoint_metadata(), point_count=int(new_sites.shape[0]))
    trainer.model.medium = VoronoiMediumModel.from_checkpoint(state, metadata, medium.bbox_min, medium.bbox_max,
                                                              trainer.device)
    trainer.config.model.medium_points = int(new_sites.shape[0])
    trainer.topology = VoronoiTopologyController(trainer.model.medium)
    trainer.topology.synchronize_renderer(trainer.renderer, trainer.model.medium)
    print(f"medium refined: {sites.shape[0]} -> {new_sites.shape[0]} cells", flush=True)


@torch.no_grad()
def plume(m, f, base, colour=(0.80, 0.74, 0.62), tau: float = 2.5) -> None:
    """A sediment plume rising from the seabed point `base` (local coordinates), widening upward, billowing:
    backscatter optical depth ~tau over 0.1 L in its densest part."""
    L = f["L"]
    local = to_local(m.sites, f) - base
    h = local[:, 2]
    width = 0.035 * L + 0.35 * h.clamp_min(0)
    w = (torch.exp(-(local[:, :2].norm(dim=-1) / width) ** 2) * torch.sigmoid((h + 0.01 * L) / (0.01 * L))
         * torch.sigmoid((0.45 * L - h) / (0.05 * L)))
    w = w * (1 + 0.7 * noise3(local / (0.05 * L), seed=5)).clamp_min(0)
    c = F.softplus(m.raw_density)
    c[:, :3] += 0.5 * tau * w[:, None] / (0.1 * L)
    c[:, 3:] += tau * w[:, None] / (0.1 * L)
    m.raw_density.copy_(inv_softplus(c))
    mix_colour(m, (w / w.max()).clamp(0, 1) ** 0.5, colour)


# --------------------------------------------------------------------------------------------------------- driver
def save_edit(trainer, base_run: Path, out: Path, info: dict) -> None:
    """The edited model as a run directory (checkpoint format of stylize.py: fresh optimizer state, training extras
    of the base run, the edit metadata under payload["edit"] and in OUT/edit.json)."""
    from training.checkpoints import checkpoint_payload, load_checkpoint, save_checkpoint
    from training.optimizer import build_optimizers
    joint = out / "joint"
    (joint / "checkpoints").mkdir(parents=True, exist_ok=True)
    for name in ("pipeline_config.json", "water_fit.json"):
        if (base_run / name).is_file():
            shutil.copy(base_run / name, out / name)
    config = trainer.config
    cells = int(trainer.model.medium.sites.shape[0])
    config.model.medium_points = cells
    if config.voronoi_adaptation.max_points is not None:  # config validation: medium_points <= the budget
        config.voronoi_adaptation.max_points = max(config.voronoi_adaptation.max_points, cells)
    previous, config.output_dir = config.output_dir, str(joint)
    config.save(joint / "config.yaml")
    config.output_dir = previous
    gaussian_optimizer, medium_optimizer = build_optimizers(trainer.model, config.optimization, trainer.scene_extent)
    payload = checkpoint_payload(
        trainer.model, gaussian_optimizer, medium_optimizer, config, trainer.start_iteration - 1,
        trainer.training_seconds, trainer.best_psnr, trainer.topology.revision, None, None,
        trainer.medium_frozen_until, trainer.disentanglement.state_dict(), extras=trainer._checkpoint_extras())
    base_joint = base_run / "joint" if (base_run / "joint").is_dir() else base_run
    for key, value in load_checkpoint(base_joint / "checkpoints" / "last.pt").items():
        payload.setdefault(key, value)
    payload["edit"] = info
    save_checkpoint(joint / "checkpoints" / "last.pt", payload)
    (out / "edit.json").write_text(json.dumps(info, indent=1))
    print(f"saved {out}", flush=True)


def apply_edit(trainer, camera, f, edit: str, args) -> list[str]:
    """Apply one edit to the loaded model; returns the extra render modes that show the edited layer."""
    g, m = trainer.model.gaussians, trainer.model.medium
    if edit in DEFORMATIONS:
        deform_gaussians(g, f, DEFORMATIONS[edit])
    elif edit == "clone":
        v = [float(x) for x in args.clone.split(",")]
        print(f"clone: {clone(trainer, camera, f, v[0:2], v[2:4], v[4], v[5], v[6])} Gaussians", flush=True)
    elif edit in ("clear", "murky"):
        factor = (0.25 if edit == "clear" else 2.5,) * 3
        scale_coefficients(m, factor, factor)
    elif edit in WATER_TYPES:
        recolour(m, *WATER_TYPES[edit])
    elif edit == "transplant":
        if args.transplant_from is None:
            raise SystemExit("transplant needs --transplant-from RUN")
        transplant(m, args.transplant_from)
    elif edit == "haze":
        from apps.editing.medium import subdivide_medium
        subdivide_medium(trainer, args.medium_split, 0.25, 0)
        colour = args.haze_colour if args.haze_colour == "patches3d" else tuple(float(x) for x in args.haze_colour.split(","))
        haze(trainer.model.medium, f, strength=args.haze_strength, rise=args.haze_rise, colour=colour,
             patch_size=args.patch_size)
    elif edit == "plume":
        base = to_local(surface_point(trainer, camera, *[float(x) for x in args.plume_at.split(",")])[None], f)[0]
        centre = (base + torch.tensor([0.0, 0.0, 0.22 * f["L"]], device=base.device)).tolist()
        refine_region(trainer, f, centre, (0.25 * f["L"], 0.25 * f["L"], 0.26 * f["L"]), 150_000)
        plume(trainer.model.medium, f, base)
    else:
        raise SystemExit(f"unknown edit {edit}")
    if edit in DEFORMATIONS or edit == "clone":
        trainer._update_mip_filter()  # as on loading a run: the 3D filter of the moved Gaussians
        return ["gaussians"]
    return ["voronoi"]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="trained (or stylized / edited) run")
    parser.add_argument("--scene", type=Path, help="dataset directory of the run")
    parser.add_argument("--view", required=True, help="camera of the frame and of the renders")
    parser.add_argument("--edits", required=True, help="comma-separated: " + ", ".join(
        list(DEFORMATIONS) + ["clone", "clear", "murky", *WATER_TYPES, "transplant", "haze", "plume"]))
    parser.add_argument("--out", type=Path, required=True, help="renders: OUT/<edit>/<mode>.png")
    parser.add_argument("--save", type=Path, help="also write every edited model as a run: SAVE/<edit>")
    parser.add_argument("--width", type=int, default=0, help="render width (0: the photo's resolution)")
    parser.add_argument("--no-original", action="store_true", help="do not render the unedited model")
    parser.add_argument("--transplant-from", type=Path, help="run whose water statistics `transplant` uses")
    parser.add_argument("--clone", default="0.12,0.75,0.65,0.86,0.06,1.0,70",
                        help="u,v of the source, u,v of the target (normalised pixels), radius (x L), scale, angle")
    parser.add_argument("--plume-at", default="0.55,0.55", help="normalised pixel of the plume source")
    parser.add_argument("--haze-strength", type=float, default=5.0, help="density multiplier for seabed haze")
    parser.add_argument("--haze-rise", type=float, default=1.0, help="haze layer thickness multiplier")
    parser.add_argument("--haze-colour", default="0.55,0.62,0.55", help="haze colour r,g,b in [0, 1], or patches3d")
    parser.add_argument("--patch-size", type=float, default=0.25, help="patches3d: patch size as a fraction of L")
    parser.add_argument("--medium-split", type=int, default=8, help="haze: cells each water cell is split into")
    args = parser.parse_args(argv)

    trainer = foamsurf_render.load_run(args.run.resolve(), "last", args.scene)
    camera = next((c for c in trainer.dataset.train + trainer.dataset.test if c.name == args.view), None)
    if camera is None:
        raise SystemExit(f"unknown camera {args.view}")
    if args.width:
        record = foamsurf_render.camera_record(camera)
        s = args.width / record["width"]
        record.update(width=args.width, height=2 * round(record["height"] * s / 2), fx=record["fx"] * s,
                      fy=record["fy"] * s, cx=record["cx"] * s, cy=record["cy"] * s)
        camera = foamsurf_render.camera_from_record(record, trainer.device)
    f = reef_frame(trainer, camera)
    print(f"frame: L {f['L']:.3f} H {f['H']:.3f}", flush=True)
    frame_info = {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in f.items()}

    # every edit starts from the loaded model: parameters, medium structure and the config it changes are restored
    g = trainer.model.gaussians
    gaussians0 = {n: getattr(g, n).detach().clone() for n in g.PARAMETER_NAMES}
    medium0, topology0 = trainer.model.medium, trainer.topology
    water0 = {n: getattr(medium0, n).detach().clone() for n in ("raw_density", "raw_color")}
    config0 = (trainer.config.model.medium_points, trainer.config.voronoi_adaptation.max_points)

    def restore():
        for n, v in gaussians0.items():
            setattr(g, n, torch.nn.Parameter(v.clone()))
        trainer._update_mip_filter()
        trainer.model.medium, trainer.topology = medium0, topology0
        for n, v in water0.items():
            getattr(medium0, n).data.copy_(v)
        topology0.synchronize_renderer(trainer.renderer, medium0)
        trainer.config.model.medium_points, trainer.config.voronoi_adaptation.max_points = config0

    edits = [ALIASES.get(e, e) for e in args.edits.split(",") if e]
    for edit in ([] if args.no_original else ["original"]) + edits:
        modes = ["full", "gaussians", "voronoi"] if edit == "original" else ["full"] + apply_edit(trainer, camera, f, edit, args)
        if args.save and edit != "original":
            save_edit(trainer, args.run.resolve(), args.save / edit,
                      {"edit": edit, "base_run": str(args.run.resolve()), "frame_view": args.view,
                       "frame_width": camera.width, "frame": frame_info,
                       "args": {k: str(v) for k, v in vars(args).items()},
                       "medium_cells": int(trainer.model.medium.sites.shape[0]),
                       "gaussians": int(trainer.model.gaussians.means.shape[0])})
        for mode, image in foamsurf_render.render_view(trainer, camera, modes).items():
            path = args.out / edit / f"{mode}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(foamsurf_render.to_png(mode, image)).save(path)
        print(f"{edit}: {', '.join(modes)}", flush=True)
        restore()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "drain":
        from apps.draining.drain import main as drain_main
        del sys.argv[1]
        drain_main()
    else:
        main()
