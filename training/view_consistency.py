"""OceanSplat (Kweon & Park, AAAI 2026) trinocular view consistency for the hybrid ray tracer.

For the central training view and two virtual views shifted along the camera x axis (b_h) and y
axis (b_v), b_v ~ U[-b, b] in nerfstudio units and b_h = 1.5 b_v, the virtual object image warped to
the central view must match the central object image (L_obj-stereo), and with the central medium
added it must match the photo (L_full-stereo); both are L1 divided by the stop-gradient central
render (regularised L1, Eq. 12).

The paper warps the virtual image with the disparity f b / D_c of the central alpha-blended depth
D_c. A ray tracer does this exactly: the warped virtual pixel is the virtual camera's ray towards
the central ray's point at depth D_c, rendered directly (no resampling, no pixels outside the
frame). Here D_c is the alpha-blended depth of the Gaussians (medium ignored) under stop-gradient,
so the disparity smoothness term L_smooth (it acts on D_c through the disparity) and the depth
priors are not included.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

CHUNK = 32_768


def nerfstudio_scale(camera_origins: torch.Tensor) -> float:
    """Pose scale of the nerfstudio COLMAP parser (auto_scale_poses, orientation ignored, within ~3%):
    1 / largest absolute coordinate of the centred camera positions."""
    centred = camera_origins - camera_origins.mean(0)
    return float(1.0 / centred.abs().max())


def hit_alphas(scene, rays: torch.Tensor, hit_depths: torch.Tensor, hit_indices: torch.Tensor) -> torch.Tensor:
    """Alpha [K, R] of every buffered hit, as evaluate_gaussian_scalar in renderer/native/hybrid_cuda.cu
    (surfel plane coordinates u, v; response exp(-0.5 (u^2 + v^2)^kappa / kappa)); 0 for empty slots."""
    valid = hit_indices >= 0
    index = hit_indices.clamp_min(0).long()
    origins, directions = rays[None, :, :3], F.normalize(rays[None, :, 3:], dim=-1)
    local = origins + directions * hit_depths[..., None] - scene.means[index]
    q = scene.quaternions[index]
    q0, q1, q2, q3 = q.unbind(-1)
    factor = 2.0 / (q * q).sum(-1).clamp_min(1e-12)
    bb, cc, dd = q1 * q1 * factor, q2 * q2 * factor, q3 * q3 * factor
    ab, ac, ad = q0 * q1 * factor, q0 * q2 * factor, q0 * q3 * factor
    bc, bd, cd = q1 * q2 * factor, q1 * q3 * factor, q2 * q3 * factor
    lx, ly, lz = local.unbind(-1)
    inverse_scale = torch.exp(-scene.log_scales[index])
    u = inverse_scale[..., 0] * ((1 - cc - dd) * lx + (bc + ad) * ly + (bd - ac) * lz)
    v = inverse_scale[..., 1] * ((bc - ad) * lx + (1 - bb - dd) * ly + (cd + ab) * lz)
    kappa = 1.0 + F.softplus(scene.raw_kappa[index].squeeze(-1))
    response = torch.exp(-0.5 * (u * u + v * v).clamp_min(1e-12).pow(kappa) / kappa)
    alpha = torch.sigmoid(scene.opacity_logits[index].squeeze(-1)) * response
    return torch.where(valid, alpha, torch.zeros_like(alpha))


@torch.no_grad()
def blended_depth(scene, rays: torch.Tensor, hit_depths: torch.Tensor, hit_indices: torch.Tensor):
    """Alpha-blended Gaussian depth along each ray (ray distance) and its accumulated alpha, [R] each."""
    depths, accumulated = [], []
    for start in range(0, rays.shape[0], CHUNK):
        part = slice(start, start + CHUNK)
        t = hit_depths[:, part]
        alpha = hit_alphas(scene, rays[part], t, hit_indices[:, part])
        t = torch.where(hit_indices[:, part] >= 0, t, torch.full_like(t, float("inf")))
        t, order = torch.sort(t, dim=0)
        alpha = torch.gather(alpha, 0, order)
        transmittance = torch.cumprod(torch.cat((torch.ones_like(alpha[:1]), 1 - alpha[:-1])), dim=0)
        weight = alpha * transmittance
        total = weight.sum(0)
        depths.append((weight * torch.nan_to_num(t, posinf=0.0)).sum(0) / total.clamp_min(1e-12))
        accumulated.append(total)
    return torch.cat(depths), torch.cat(accumulated)


def virtual_rays(rays: torch.Tensor, depth: torch.Tensor, shift: torch.Tensor):
    """Rays of the camera moved by shift [3] towards the central rays' points at ray distance depth."""
    origins, directions = rays[:, :3], F.normalize(rays[:, 3:], dim=-1)
    points = origins + directions * depth[:, None]
    shifted = origins + shift
    return torch.cat((shifted, F.normalize(points - shifted, dim=-1)), dim=-1).contiguous()


def regularized_l1(prediction: torch.Tensor, target: torch.Tensor, reference: torch.Tensor, epsilon: float):
    return ((prediction - target).abs() / (reference.detach() + epsilon)).mean()


def trinocular_loss(renderer, gaussians, medium_scene, background, camera, rays, depth, direct, backscatter,
                    target, baseline: float, epsilon: float, generator: torch.Generator | None = None):
    """L_obj-stereo + L_full-stereo (horizontal and vertical) for central rays [R, 6] with depth [R]
    and their central object (direct) / medium (backscatter) parts and photo target [R, 3].
    baseline: b in scene units. Returns (loss, {"obj": ..., "full": ...})."""
    b_v = (torch.rand((), generator=generator, device="cpu").item() * 2.0 - 1.0) * baseline
    reference = (direct + backscatter).detach()
    obj = full = rays.new_zeros(())
    for axis, b in ((camera.right, 1.5 * b_v), (camera.down, b_v)):
        shift = axis.to(rays) * b
        scene = gaussians.render_scene(camera.origin.to(rays) + shift)
        renderer.set_gaussian_geometry(scene)
        warped = renderer.render(virtual_rays(rays, depth, shift), gaussians=scene, voronoi=medium_scene,
                                 background=background).direct
        obj = obj + regularized_l1(warped, direct, reference, epsilon)
        full = full + regularized_l1(warped + backscatter, target, reference, epsilon)
    return obj + full, {"obj": float(obj.detach()), "full": float(full.detach()), "baseline": b_v}
