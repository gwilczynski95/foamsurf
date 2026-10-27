"""OceanSplat (Kweon & Park, AAAI 2026) for the hybrid ray tracer: trinocular view consistency with
warping by the rendered depth (L_obj-stereo, L_full-stereo, L_smooth), the synthetic epipolar depth
prior (L_epi), the depth residual (L_res) and the depth-aware opacity (alpha^d).

The depth is the alpha-blended Gaussian depth over the K buffered hits of each ray (never the first
hit), with gradients: hit depth = ray / surfel-plane intersection and hit alpha exactly as
evaluate_gaussian_dual in renderer/native/hybrid_cuda.cu. The medium does not enter the depth.

Images for these terms are rendered at 1/downscale of the camera resolution; the virtual cameras are
the central one moved by b_h along its x axis and b_v along its y axis (b_v ~ U[-b, b], b_h = 1.5 b_v),
and I_{h->c}(x, y) = I_h(x - f b_h / D_c, y), I_{v->c}(x, y) = I_v(x, y - f b_v / D_c), D_c the central
depth along the optical axis (Eqs. 8-16). For Gaussians projected into the central view: L_res =
mean |D_c(x_i) - z_i| (all in the frustum, Eq. 20); L_epi = mean over k of log(1 + |D_c(x_i) - z_i|)
exp(-|d_k I_c(x_i)|) for Gaussians with opacity > 0.8 inside all three frusta (Eq. 19; the
triangulation of a Gaussian's centre from its two virtual projections is the centre itself, so
D_epi = z_i, used as a fixed prior).
"""
from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import nn

from foamsurf_data.rays import generate_camera_rays

CHUNK = 32_768


def rotation_columns(q: torch.Tensor):
    """Columns of the surfel rotation (u axis, v axis, normal) for quaternions (w, x, y, z), as the
    CUDA kernel builds them (works for unnormalised quaternions)."""
    q0, q1, q2, q3 = q.unbind(-1)
    factor = 2.0 / (q * q).sum(-1).clamp_min(1e-12)
    bb, cc, dd = q1 * q1 * factor, q2 * q2 * factor, q3 * q3 * factor
    ab, ac, ad = q0 * q1 * factor, q0 * q2 * factor, q0 * q3 * factor
    bc, bd, cd = q1 * q2 * factor, q1 * q3 * factor, q2 * q3 * factor
    u = torch.stack((1 - cc - dd, bc + ad, bd - ac), -1)
    v = torch.stack((bc - ad, 1 - bb - dd, cd + ab), -1)
    n = torch.stack((bd + ac, cd - ab, 1 - bb - cc), -1)
    return u, v, n


def gaussian_table(scene) -> torch.Tensor:
    """Per-Gaussian quantities of the hit evaluation, one row each [N, 16]: mean (3), u axis (3),
    v axis (3), normal (3), 1/scale_u, 1/scale_v, kappa, opacity. Computed once per scene and gathered
    per hit with a single index_select (whose backward is an atomic index_add_; advanced indexing
    sorts the indices of these heavily repeated Gaussians and is ~100x slower)."""
    u, v, n = rotation_columns(scene.quaternions)
    inverse_scale = torch.exp(-scene.log_scales)
    kappa = 1.0 + F.softplus(scene.raw_kappa)
    return torch.cat((scene.means, u, v, n, inverse_scale, kappa, torch.sigmoid(scene.opacity_logits)), -1)


def hit_terms(scene, rays: torch.Tensor, hit_depths: torch.Tensor, hit_indices: torch.Tensor, table=None):
    """(depth [K, R], alpha [K, R]) of the buffered hits, differentiable in the Gaussian parameters
    (0 for empty slots). The depth value is the renderer's; its gradient is the plane intersection's
    (as in CUDA). Only the occupied slots are evaluated."""
    table = gaussian_table(scene) if table is None else table
    valid = hit_indices >= 0
    slot, ray = torch.nonzero(valid, as_tuple=True)
    row = torch.index_select(table, 0, hit_indices[slot, ray].long())
    mean, u_axis, v_axis, normal = row[:, 0:3], row[:, 3:6], row[:, 6:9], row[:, 9:12]
    origin = rays[ray, :3]
    direction = F.normalize(rays[ray, 3:], dim=-1)
    saved = hit_depths[slot, ray]
    denominator = (normal * direction).sum(-1)
    denominator = torch.where(denominator.abs() > 1e-8, denominator, torch.ones_like(denominator))
    plane = (normal * (mean - origin)).sum(-1) / denominator
    depth = plane + (saved - plane).detach()
    local = origin + direction * depth[:, None] - mean
    u = row[:, 12] * (u_axis * local).sum(-1)
    v = row[:, 13] * (v_axis * local).sum(-1)
    kappa = row[:, 14]
    response = torch.exp(-0.5 * torch.exp(kappa * torch.log((u * u + v * v).clamp_min(1e-12))) / kappa)
    alpha = row[:, 15] * response
    dense_depth = torch.zeros_like(hit_depths).index_put((slot, ray), depth)
    dense_alpha = torch.zeros_like(hit_depths).index_put((slot, ray), alpha)
    return dense_depth, dense_alpha


def blended_depth(scene, rays: torch.Tensor, hit_depths: torch.Tensor, hit_indices: torch.Tensor):
    """Alpha-blended Gaussian ray distance sum(w t) / sum(w), w = alpha T, and accumulated alpha, [R] each;
    differentiable (not the first hit: faint Gaussians get small weights)."""
    table = gaussian_table(scene)
    depths, accumulated = [], []
    for start in range(0, rays.shape[0], CHUNK):
        part = slice(start, start + CHUNK)
        indices = hit_indices[:, part]
        with torch.no_grad():
            occupied = indices >= 0
            key = torch.where(occupied, hit_depths[:, part], torch.full_like(hit_depths[:, part], float("inf")))
            order = torch.argsort(key, dim=0)
            slots = max(int(occupied.sum(0).max()), 1)  # sorted: occupied slots first
            order = order[:slots]
        t, alpha = hit_terms(scene, rays[part], torch.gather(hit_depths[:, part], 0, order),
                             torch.gather(indices, 0, order), table)
        transmittance = torch.cumprod(torch.cat((torch.ones_like(alpha[:1]), 1 - alpha[:-1])), dim=0)
        weight = alpha * transmittance
        total = weight.sum(0)
        depths.append((weight * t).sum(0) / total.clamp_min(1e-6))
        accumulated.append(total)
    return torch.cat(depths), torch.cat(accumulated)


def camera_geometry(camera, downscale: int):
    """(height, width, fx, fy, cx, cy) at 1/downscale of the camera resolution (nerfstudio scaling)."""
    height, width = camera.height // downscale, camera.width // downscale
    if camera.intrinsics is not None:
        fx, fy, cx, cy = (value / downscale for value in camera.intrinsics)
    else:
        fx = width / (2.0 * torch.tan(torch.as_tensor(0.5 * camera.fov_x)).item())
        fy = height / (2.0 * torch.tan(torch.as_tensor(0.5 * camera.fov_y)).item())
        cx, cy = width / 2.0, height / 2.0
    return height, width, float(fx), float(fy), float(cx), float(cy)


def image_rays(camera, downscale: int, shift: torch.Tensor | None = None) -> torch.Tensor:
    """Rays [H*W, 6] of the camera (moved by shift [3]) at 1/downscale resolution."""
    height, width, fx, fy, cx, cy = camera_geometry(camera, downscale)
    origin = camera.origin if shift is None else camera.origin + shift.to(camera.origin)
    bundle = generate_camera_rays(origin, camera.right, camera.down, camera.forward, width, height,
                                  camera.fov_x, camera.fov_y, intrinsics=(fx, fy, cx, cy))
    return torch.cat((bundle.origins.reshape(-1, 3), bundle.directions.reshape(-1, 3)), -1).contiguous()


def downscaled_target(camera, downscale: int, background) -> torch.Tensor:
    """Photo [H, W, 3] at 1/downscale, bilinear with antialiasing (as the training sampler)."""
    height, width = camera.height // downscale, camera.width // downscale
    def resize(image):
        return F.interpolate(image.permute(2, 0, 1)[None], size=(height, width), mode="bilinear",
                             antialias=True, align_corners=False)[0].permute(1, 2, 0)
    alpha = resize(camera.alpha.reshape(camera.height, camera.width, 1))
    return resize(camera.foreground) + torch.as_tensor(background).to(alpha) * (1.0 - alpha)


def sample_image(image: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
    """Bilinear samples of image [H, W, C] at pixel coordinates (pixel centres at +0.5) x, y [...];
    returns (values [..., C], inside mask [...])."""
    height, width = image.shape[:2]
    grid = torch.stack((2.0 * x / width - 1.0, 2.0 * y / height - 1.0), -1).reshape(1, -1, 1, 2)
    values = F.grid_sample(image.permute(2, 0, 1)[None], grid, mode="bilinear", padding_mode="zeros",
                           align_corners=False)[0, :, :, 0].T.reshape(*x.shape, -1)
    inside = (x >= 0.5) & (x <= width - 0.5) & (y >= 0.5) & (y <= height - 0.5)
    return values, inside


def regularized_l1(prediction, target, reference, mask, epsilon):
    """Eq. 12: mean over valid pixels of |prediction - target| / (sg(reference) + epsilon)."""
    error = (prediction - target).abs() / (reference.detach() + epsilon)
    weight = mask[..., None].to(error).expand_as(error)
    return (error * weight).sum() / weight.sum().clamp_min(1.0)


def smoothness(disparity: torch.Tensor, image: torch.Tensor, mask: torch.Tensor, gamma: float):
    """Eq. 15 for one disparity map [H, W]: edge-aware |grad d| exp(-gamma |grad I|), x and y, on the
    disparity divided by its mean over the mask (as Monodepth2), so the term does not depend on the
    sampled baseline or the focal length."""
    scale = (disparity.abs() * mask).sum() / mask.sum().clamp_min(1)
    disparity = disparity / scale.detach().clamp_min(1e-6)
    total = disparity.new_zeros(())
    for axis in (0, 1):
        d = disparity.diff(dim=axis).abs()
        edge = torch.exp(-gamma * image.diff(dim=axis).abs().mean(-1))
        valid = (mask.narrow(axis, 1, mask.shape[axis] - 1) & mask.narrow(axis, 0, mask.shape[axis] - 1)).to(d)
        total = total + (d * edge * valid).sum() / valid.sum().clamp_min(1.0)
    return total


class DepthAwareOpacity(nn.Module):
    """alpha^d = phi(alpha, z, v) (Eq. 21): an MLP of the opacity, the camera depth (in scene extents)
    and the viewing direction; alpha' = (1 - w) alpha + w alpha^d."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, opacity: torch.Tensor, depth: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(torch.cat((opacity, depth, direction), -1)))


def adjusted_scene(scene, mlp: DepthAwareOpacity, weight: float, camera_origin: torch.Tensor,
                   camera_forward: torch.Tensor, extent: float):
    """The scene with opacity (1 - w) alpha + w phi(alpha, z, v) for this camera (w = 0: unchanged)."""
    if weight <= 0.0:
        return scene
    offset = scene.means - camera_origin
    depth = (offset @ camera_forward)[:, None] / extent
    direction = F.normalize(offset, dim=-1)
    opacity = torch.sigmoid(scene.opacity_logits)
    blended = ((1.0 - weight) * opacity + weight * mlp(opacity, depth, direction)).clamp(1e-6, 1 - 1e-6)
    return replace(scene, opacity_logits=torch.log(blended) - torch.log1p(-blended))


def ocean_terms(renderer, make_scene, medium_scene, background, camera, target, baseline: float,
                downscale: int, epsilon: float, gamma: float, gaussian_samples: int,
                epi_opacity: float = 0.8, depth_unit: float = 1.0, generator: torch.Generator | None = None):
    """L_obj-stereo + L_full-stereo, L_smooth, L_epi, L_res for one training camera.
    make_scene(origin) -> GaussianScene seen from origin. target [H, W, 3] photo at 1/downscale.
    depth_unit: scene units -> the paper's (nerfstudio) units for the depth errors of L_epi and L_res.
    Returns dict of losses (tensors) and floats for logging."""
    height, width, fx, fy, cx, cy = camera_geometry(camera, downscale)
    background_tuple = tuple(float(v) for v in background)

    built = []

    def render(shift):
        origin = camera.origin if shift is None else camera.origin + shift
        scene = make_scene(origin)
        if built != [scene.means.data_ptr()]:  # same Gaussians for all three views: one BVH build
            renderer.set_gaussian_geometry(scene)
            built[:] = [scene.means.data_ptr()]
        rays = image_rays(camera, downscale, shift)
        return scene, rays, renderer.render(rays, gaussians=scene, voronoi=medium_scene, background=background_tuple)

    scene_c, rays_c, central = render(None)
    distance, accumulated = blended_depth(scene_c, rays_c, central.hit_depths, central.hit_indices)
    axial = distance * (F.normalize(rays_c[:, 3:], dim=-1) @ camera.forward.to(rays_c))
    depth = axial.reshape(height, width)
    surface = (accumulated > 0.5).reshape(height, width) & (depth > 1e-3)
    safe_depth = torch.where(surface, depth, torch.ones_like(depth))
    obj_c = central.direct.reshape(height, width, 3)
    med_c = central.backscatter.reshape(height, width, 3)
    full_c = obj_c + med_c

    b_v = (torch.rand((), generator=generator).item() * 2.0 - 1.0) * baseline
    b_h = 1.5 * b_v
    ys, xs = torch.meshgrid(torch.arange(height, device=depth.device, dtype=depth.dtype) + 0.5,
                            torch.arange(width, device=depth.device, dtype=depth.dtype) + 0.5, indexing="ij")
    obj_loss = full_loss = smooth_loss = depth.new_zeros(())
    for axis, b, focal, horizontal in ((camera.right, b_h, fx, True), (camera.down, b_v, fy, False)):
        _, _, virtual = render(axis.to(depth) * b)
        image = virtual.direct.reshape(height, width, 3)
        disparity = focal * b / safe_depth
        x, y = (xs - disparity, ys) if horizontal else (xs, ys - disparity)
        warped, inside = sample_image(image, x, y)
        mask = surface & inside
        obj_loss = obj_loss + regularized_l1(warped, obj_c, full_c, mask, epsilon)
        full_loss = full_loss + regularized_l1(warped + med_c, target, full_c, mask, epsilon)
        smooth_loss = smooth_loss + smoothness(disparity, target, surface, gamma)

    # Gaussians projected into the central view (a random subset of those in front of the camera)
    means = scene_c.means
    offset = means - camera.origin.to(means)
    z = offset @ camera.forward.to(means)
    px = fx * (offset @ camera.right.to(means)) / z.clamp_min(1e-6) + cx
    py = fy * (offset @ camera.down.to(means)) / z.clamp_min(1e-6) + cy
    candidates = torch.nonzero((z > 1e-3) & (px >= 0.5) & (px <= width - 0.5) & (py >= 0.5) & (py <= height - 0.5)).squeeze(-1)
    if candidates.numel() > gaussian_samples:
        candidates = candidates[torch.randperm(candidates.numel(), device=candidates.device, generator=None)[:gaussian_samples]]
    px, py = torch.index_select(px, 0, candidates), torch.index_select(py, 0, candidates)
    rendered, _ = sample_image(depth[..., None], px, py)
    covered, _ = sample_image(surface[..., None].to(depth), px, py)
    rendered, on_surface = rendered[:, 0], covered[:, 0] > 0.5
    zi = torch.index_select(z, 0, candidates)
    res_loss = ((rendered - zi).abs() * depth_unit * on_surface).sum() / on_surface.sum().clamp_min(1)

    # L_epi: opaque Gaussians inside all three frusta; the prior depth is the Gaussian's own depth
    opaque = torch.sigmoid(torch.index_select(scene_c.opacity_logits[:, 0], 0, candidates)) > epi_opacity
    in_all = on_surface & opaque
    for b, focal, horizontal in ((b_h, fx, True), (b_v, fy, False)):
        shifted = (px - focal * b / zi) if horizontal else (py - focal * b / zi)
        limit = width if horizontal else height
        in_all = in_all & (shifted >= 0.5) & (shifted <= limit - 0.5)
    epi_loss = depth.new_zeros(())
    if bool(in_all.any()):
        error = torch.log1p((rendered - zi.detach()).abs() * depth_unit)
        for axis in (1, 0):  # horizontal (x) and vertical (y) image gradient of the central render
            grad = full_c.detach().diff(dim=axis, append=full_c.detach().narrow(axis, full_c.shape[axis] - 1, 1)).abs().mean(-1)
            edge, _ = sample_image(torch.exp(-grad)[..., None], px, py)
            epi_loss = epi_loss + (error * edge[:, 0] * in_all).sum() / in_all.sum()
    return {"obj": obj_loss, "full": full_loss, "smooth": smooth_loss, "epi": epi_loss, "res": res_loss,
            "baseline": b_v, "surface": float(surface.float().mean())}
