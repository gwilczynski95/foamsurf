"""Place spheres, meshes and lights relative to cameras and reconstructed surfaces."""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F

from .mesh import Mesh, load_manifest
from .shading import INF, sphere_directions


@torch.no_grad()
def surface_distance(tracer, center: torch.Tensor, samples: int = 1024) -> torch.Tensor:
    """[K] distance from a point to the first Gaussian surface along K directions (inf: none)."""
    directions = sphere_directions(samples, center.device)
    rays = torch.cat((center.expand_as(directions), directions), -1).contiguous()
    depth = tracer.renderer.render_gaussians(rays, tracer.gaussians, background=(0.0, 0.0, 0.0)).surface_depth
    depth = depth.reshape(-1)
    return torch.where(depth > 0, depth, torch.full_like(depth, INF))

@torch.no_grad()
def lift_clear(tracer, center: torch.Tensor, radius: float, margin: float = 0.03, steps: int = 40):
    """Move a sphere up (cameras' "up") until no surface lies within (1 + margin) radius of its centre: a sphere
    cutting into the reef shows the inside of the reconstruction (no surfaces there, only water)."""
    up = tracer.up.to(center)
    moved = 0.0
    for _ in range(steps):
        if float(surface_distance(tracer, center).min()) >= (1 + margin) * radius:
            break
        center = center + up * (0.1 * radius)
        moved += 0.1
    return center, moved

@torch.no_grad()
def place_on_seabed(tracer, camera, u: float, v: float, radius_fraction: float, lift: float = 1.0,
                    height: float = 0.0, depth_fraction: float | None = None, clear: bool = True,
                    distance: float | None = None):
    """Sphere at image fraction (u, v) of the camera; radius = radius_fraction of the distance to the surface seen
    there. Default: resting on that surface (centre lift radii above it; lift < 1 sinks it), plus height radii
    (floating). depth_fraction: centre on the pixel's ray at that fraction of the surface distance (in the water,
    closer to the camera). distance: centre on the pixel's ray at this distance from the camera (in the open water
    too; radius = radius_fraction x distance). clear: lifted out of any surface it would cut (see lift_clear)."""
    from evaluation.evaluator import _rays
    rays = _rays(camera)
    y, x = int(v * camera.height), int(u * camera.width)
    ray = rays[y, x][None].contiguous()
    only = tracer.renderer.render_gaussians(ray, tracer.gaussians, background=(0.0, 0.0, 0.0))
    depth = float(only.surface_depth.reshape(-1)[0])
    up = tracer.up.to(ray)
    if distance is not None:
        depth = distance
        radius = radius_fraction * depth
        center = ray[0, :3] + distance * ray[0, 3:] + up * radius * height
    elif depth_fraction is not None:
        radius = radius_fraction * depth
        center = ray[0, :3] + depth_fraction * depth * ray[0, 3:] + up * radius * height
    else:
        radius = radius_fraction * depth
        center = ray[0, :3] + depth * ray[0, 3:] + up * radius * (lift + height)
    moved = 0.0
    if clear:
        center, moved = lift_clear(tracer, center, radius)
    return center.tolist(), radius, depth, moved


def frame(up: torch.Tensor, reference: torch.Tensor, yaw_deg: float):
    """Right-handed world axes (x, y=up, z) for local mesh axes, x = reference made horizontal and turned by yaw."""
    up = F.normalize(up, dim=0)
    x = F.normalize(reference - (reference @ up) * up, dim=0)
    z = torch.linalg.cross(x, up)
    a = math.radians(yaw_deg)
    x, z = math.cos(a) * x + math.sin(a) * z, -math.sin(a) * x + math.cos(a) * z
    return x, up, z

@torch.no_grad()
def surface_heights(tracer, points: torch.Tensor, up: torch.Tensor, above: float) -> torch.Tensor:
    """Distance from points [N,3] (lifted by `above` along up) down to the Gaussian surface, minus `above`:
    the surface height relative to each point (nan: nothing below)."""
    starts = points + up * above
    rays = torch.cat((starts, (-up).expand_as(starts)), -1).contiguous()
    hits = tracer.renderer.render_gaussians(rays, tracer.gaussians, background=(0.0, 0.0, 0.0))
    depth = hits.surface_depth.reshape(-1)
    found = torch.isfinite(depth) & (depth > 0) & (depth < 4 * above)
    return torch.where(found, above - depth, torch.full_like(depth, float("nan")))

@torch.no_grad()
def tilt_local(vertices: np.ndarray, pitch_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    """Tilt a normalized mesh (y-up, foot at 0) before placement: pitch about x (positive lowers the +z end), roll
    about z (positive lowers the +x side); the foot is put back at y = 0."""
    if not pitch_deg and not roll_deg:
        return vertices
    p, r = math.radians(pitch_deg), math.radians(roll_deg)
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]], dtype=np.float32)
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]], dtype=np.float32)
    out = vertices @ (rz @ rx).T
    out[:, 1] -= out[:, 1].min()
    return out.astype(np.float32)

def place_mesh_on_seabed(tracer, camera, u: float, v: float, span_fraction: float, local_vertices: np.ndarray,
                         yaw_deg: float = 0.0, sink: float = 0.0, max_tilt_deg: float = 10.0):
    """World vertices of a normalized mesh (y-up, foot at 0, extent 1) resting on the seabed seen at image fraction
    (u, v) of a camera: size = span_fraction x the hit distance, x axis = the camera's right turned by yaw_deg, tilted
    to the plane fitted to the surface under its footprint (at most max_tilt_deg), lowered by `sink` x size."""
    from evaluation.evaluator import _rays
    rays = _rays(camera)
    y, x = int(v * camera.height), int(u * camera.width)
    ray = rays[y, x][None].contiguous()
    only = tracer.renderer.render_gaussians(ray, tracer.gaussians, background=(0.0, 0.0, 0.0))
    depth = float(only.surface_depth.reshape(-1)[0])
    anchor = ray[0, :3] + depth * ray[0, 3:]
    size = span_fraction * depth
    up = tracer.up.to(anchor)
    ax, ay, az = frame(up, camera.right.to(anchor), yaw_deg)
    local = torch.as_tensor(local_vertices, device=anchor.device)
    # Footprint samples: a 7x7 grid over the horizontal bounding box.
    lo, hi = local.min(0).values, local.max(0).values
    gx, gz = torch.meshgrid(torch.linspace(float(lo[0]), float(hi[0]), 7, device=anchor.device),
                            torch.linspace(float(lo[2]), float(hi[2]), 7, device=anchor.device), indexing="ij")
    foot = torch.stack((gx.reshape(-1), gz.reshape(-1)), -1)
    points = anchor + size * (foot[:, :1] * ax + foot[:, 1:] * az)
    height = surface_heights(tracer, points, up, above=size)
    ok = torch.isfinite(height)
    normal = up
    if int(ok.sum()) >= 6:  # least-squares plane h = a x + b z + c
        A = torch.cat((foot[ok] * size, torch.ones(int(ok.sum()), 1, device=anchor.device)), -1)
        coef = torch.linalg.lstsq(A, height[ok][:, None]).solution.reshape(-1)
        normal = F.normalize(up - coef[0] * ax - coef[1] * az, dim=0)
        tilt = math.degrees(math.acos(float((normal @ up).clamp(-1, 1))))
        if tilt > max_tilt_deg:
            w = max_tilt_deg / tilt
            normal = F.normalize(up * (1 - w) + normal * w, dim=0)
        base = anchor + up * float(coef[2])
    else:
        base = anchor
    bx, by, bz = frame(normal, ax, 0.0)
    world = base + size * (local[:, :1] * bx + local[:, 1:2] * by + local[:, 2:] * bz) - by * (sink * size)
    origin = base - by * (sink * size)
    return world, dict(anchor=anchor.tolist(), size=size, depth=depth, up=by.tolist(),
                       footprint_hits=int(ok.sum()), origin=origin.tolist(),
                       axes=[bx.tolist(), by.tolist(), bz.tolist()])  # world = origin + size * local @ axes


def anchor_point(anchor, u: float, v: float, distance: float) -> torch.Tensor:
    """The point at `distance` world units along the anchor camera's ray through image fraction (u, v) (cpu)."""
    from evaluation.evaluator import _rays
    grid = _rays(anchor)
    h, w = grid.shape[:2]
    ray = grid[min(int(v * h), h - 1), min(int(u * w), w - 1)].float().cpu()
    return ray[:3] + distance * F.normalize(ray[3:], dim=0)

def seabed_point(tracer, anchor, u: float, v: float) -> torch.Tensor:
    """The first surface point seen at image fraction (u, v) of the anchor camera (cpu)."""
    from evaluation.evaluator import _rays
    grid = _rays(anchor)
    h, w = grid.shape[:2]
    ray = grid[min(int(v * h), h - 1), min(int(u * w), w - 1)][None].contiguous()
    depth = tracer.renderer.render_gaussians(ray, tracer.gaussians, background=(0.0, 0.0, 0.0)).surface_depth
    return (ray[0, :3] + float(depth.reshape(-1)[0]) * ray[0, 3:]).float().cpu()

def build_school(tracer, anchor, spec: dict):
    """A school of fish (or any small mesh) as one Mesh: `count` copies of spec["manifest"], each `length` world
    units long (+- size_jitter), scattered in an ellipsoid of half-axes `spread` = [along the heading, up, across]
    (world units) around the point at `distance` along the anchor ray through (u, v), keeping `gap` lengths apart.
    `forward`: the mesh axis its head points along (x, -x, z, -z). All swim along the heading yaw_deg (from the anchor camera's right, about up) +- heading_jitter_deg, pitched by
    +- pitch_jitter_deg. Deterministic for a seed."""
    vertices, faces, part, materials, ids, uv = load_manifest(spec["manifest"], spec.get("materials"), with_uv=True)
    forward = spec.get("forward", "x")  # the mesh's head direction: x, -x, z or -z
    sign = -1.0 if forward.startswith("-") else 1.0
    if forward.endswith("z"):
        vertices = np.stack((sign * vertices[:, 2], vertices[:, 1], -sign * vertices[:, 0]), -1)
    elif sign < 0:
        vertices = vertices * np.array([-1.0, 1.0, -1.0], dtype=np.float32)
    vertices = vertices - 0.5 * (vertices.min(0) + vertices.max(0))
    device = tracer.trainer.device
    up = tracer.up.float().cpu()
    hx, hy, hz = frame(up, anchor.right.float().cpu(), spec.get("yaw_deg", 0.0))
    center = anchor_point(anchor, spec["u"], spec["v"], spec["distance"])
    rng = np.random.default_rng(spec.get("seed", 0))
    length, spread = spec.get("length", 0.25), np.array(spec.get("spread", [2.0, 0.6, 1.0]))
    gap = spec.get("gap", 1.1) * length
    spots = []
    for _ in range(50 * spec.get("count", 20)):
        if len(spots) == spec.get("count", 20):
            break
        p = rng.normal(size=3)
        p = p / max(1.0, float(np.linalg.norm(p))) * spread * rng.uniform() ** (1 / 3)
        if all(np.linalg.norm(p - q) > gap for q in spots):
            spots.append(p)
    all_v, all_f, all_p = [], [], []
    for k, p in enumerate(spots):
        yaw = math.radians(rng.normal() * spec.get("heading_jitter_deg", 10.0))
        pitch = math.radians(rng.normal() * spec.get("pitch_jitter_deg", 6.0))
        size = length * (1 + spec.get("size_jitter", 0.15) * rng.normal())
        fx = math.cos(pitch) * (math.cos(yaw) * hx + math.sin(yaw) * hz) + math.sin(pitch) * hy
        fz = -math.sin(yaw) * hx + math.cos(yaw) * hz
        fy = torch.linalg.cross(fz, fx)
        axes = torch.stack((fx, fy, fz))
        position = center + float(p[0]) * hx + float(p[1]) * hy + float(p[2]) * hz
        all_v.append(position + size * torch.as_tensor(vertices) @ axes)
        all_f.append(torch.as_tensor(faces) + k * len(vertices))
        all_p.append(torch.as_tensor(part))
    world = torch.cat(all_v).to(device)
    corner_uv = None if uv is None else torch.as_tensor(uv).repeat(len(spots), 1, 1).to(device)
    mesh = Mesh(world, torch.cat(all_f).to(device), torch.cat(all_p).to(device), materials,
                spec.get("name", "school"), ids, corner_uv)
    return mesh, dict(center=center.tolist(), count=len(spots), heading=hx.tolist())

def near_point(tracer, anchor, objects, spec: dict):
    """A point next to an object built earlier (spec["near"], by name): its centre plus spec["offset"] = [right,
    up, towards the anchor camera] in units of the object's radius. Returns the point (cpu) and the object."""
    target = next(o for o in objects if getattr(o, "name", None) == spec["near"])
    base = torch.as_tensor(target.center, dtype=torch.float32).cpu()
    up = tracer.up.float().cpu()
    right = F.normalize(anchor.right.float().cpu() - (anchor.right.float().cpu() @ up) * up, dim=0)
    back = torch.linalg.cross(right, up)
    if float(back @ (anchor.origin.float().cpu() - base)) < 0:
        back = -back
    off = spec.get("offset", [0.0, 2.0, 0.0])
    return base + target.radius * (off[0] * right + off[1] * up + off[2] * back), target

def place_mesh_near(tracer, anchor, objects, spec: dict, vertices):
    """World vertices of a normalized mesh floating (upright, turned by yaw_deg) with its bounding-box centre at
    near_point; its largest extent is spec["size_of_near"] x the object's radius. Same info as
    place_mesh_on_seabed."""
    center, target = near_point(tracer, anchor, objects, spec)
    device = tracer.trainer.device
    size = spec.get("size_of_near", 2.0) * target.radius
    up = tracer.up.to(device).float()
    bx, by, bz = frame(up, anchor.right.to(device).float(), spec.get("yaw_deg", 0.0))
    axes = torch.stack((bx, by, bz))
    local = torch.as_tensor(vertices, device=device)
    middle = 0.5 * (local.min(0).values + local.max(0).values)
    origin = center.to(device) - size * middle @ axes
    world = origin + size * local @ axes
    return world, dict(anchor=center.tolist(), size=size, origin=origin.tolist(), up=by.tolist(),
                       axes=axes.tolist())

def place_mesh_in_water(tracer, anchor, spec: dict, vertices):
    """World vertices of a normalized mesh floating upright (turned by yaw_deg) with its bounding-box centre at
    `distance` world units along the anchor ray through (u, v) and its largest extent `size` world units. Same
    info as place_mesh_on_seabed."""
    center = anchor_point(anchor, spec["u"], spec["v"], spec["distance"])
    device = tracer.trainer.device
    size = spec["size"]
    bx, by, bz = frame(tracer.up.to(device).float(), anchor.right.to(device).float(), spec.get("yaw_deg", 0.0))
    axes = torch.stack((bx, by, bz))
    local = torch.as_tensor(vertices, device=device)
    middle = 0.5 * (local.min(0).values + local.max(0).values)
    origin = center.to(device) - size * middle @ axes
    return origin + size * local @ axes, dict(anchor=center.tolist(), size=size, origin=origin.tolist(),
                                              up=by.tolist(), axes=axes.tolist())

def spot_axis(tracer, anchor, spec: dict) -> tuple:
    """Beam axis of a spot lamp: straight down tilted by down_deg towards the anchor camera's right turned by
    yaw_deg about up."""
    up = tracer.up.to(anchor.right.device).float()
    x, _, _ = frame(up, anchor.right.float(), spec.get("yaw_deg", 0.0))
    a = math.radians(spec.get("down_deg", 0.0))
    return tuple((-up * math.cos(a) + x * math.sin(a)).tolist())
