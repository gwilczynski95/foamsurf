"""Triangle meshes for vis/raytrace.py: loading (PLY parts + FALCON-style manifest), placement on the seabed and
ray queries on a GPU BVH (NVIDIA Warp, `pip install warp-lang`).

A manifest (as FALCON's source/fiat_bicycle/manifest.json) lists the parts of one object, each a PLY with a material
{"mode": metal | ceramic | glass | mirror | diffuse, "rgb": [...], "kd": ..., "eta": ...}. The parts share one
normalization: centred horizontally, the lowest point at y = 0 (Blender PLY export is y-up), the largest extent 1.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import warp as wp

from vis.raytrace import Material

wp.config.quiet = True
NO_HIT = 1e30


@wp.kernel
def _closest_hit(mesh: wp.uint64, origins: wp.array(dtype=wp.vec3), directions: wp.array(dtype=wp.vec3),
                 max_t: wp.array(dtype=float), t_out: wp.array(dtype=float), face_out: wp.array(dtype=int),
                 u_out: wp.array(dtype=float), v_out: wp.array(dtype=float)):
    i = wp.tid()
    query = wp.mesh_query_ray(mesh, origins[i], directions[i], max_t[i])
    if query.result:
        t_out[i] = query.t
        face_out[i] = query.face
        u_out[i] = query.u
        v_out[i] = query.v
    else:
        t_out[i] = 1.0e30
        face_out[i] = -1


def read_ply(path: Path):
    """Vertices [V,3] float32 and triangles [F,3] int64 (polygons fan-triangulated)."""
    from plyfile import PlyData
    ply = PlyData.read(str(path))
    v = ply["vertex"]
    vertices = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float32)
    faces = []
    for polygon in ply["face"].data["vertex_indices"]:
        polygon = np.asarray(polygon, dtype=np.int64)
        for k in range(1, len(polygon) - 1):
            faces.append((polygon[0], polygon[k], polygon[k + 1]))
    return vertices, np.asarray(faces, dtype=np.int64).reshape(-1, 3)


def load_manifest(path: str | Path, overrides: dict | None = None):
    """Parts of a manifest as one normalized mesh: vertices [V,3] (y-up, foot at 0, max extent 1), faces [F,3],
    (the part PLYs are taken as y-down, as FALCON's, unless the manifest says "flip_yz": false)
    per-face part index [F], materials (one per part), part ids. overrides: {part id: FALCON material dict}."""
    path = Path(os.path.expandvars(str(path)))
    manifest = json.loads(path.read_text())
    overrides = overrides or {}
    vertices, faces, part_of_face, materials, ids = [], [], [], [], []
    offset = 0
    for index, part in enumerate(manifest["parts"]):
        v, f = read_ply(path.parent / part["ply"])
        vertices.append(v)
        faces.append(f + offset)
        part_of_face.append(np.full(len(f), index))
        offset += len(v)
        spec = dict(part.get("material", {}), **overrides.get(part["id"], {}))
        materials.append(Material.from_falcon(spec))
        ids.append(part["id"])
    vertices = np.concatenate(vertices)
    if manifest.get("flip_yz", True):  # FALCON's part PLYs are y-down (its y/z flip of Blender's y-up export)
        vertices = vertices * np.array([1.0, -1.0, -1.0], dtype=np.float32)
    low, high = vertices.min(0), vertices.max(0)
    center = 0.5 * (low + high)
    vertices = vertices - center
    vertices[:, 1] -= vertices[:, 1].min()
    vertices /= float((high - low).max())
    return vertices, np.concatenate(faces), np.concatenate(part_of_face), materials, ids


def corner_normals(vertices: torch.Tensor, faces: torch.Tensor, crease_deg: float = 40.0) -> torch.Tensor:
    """Smooth normals per triangle corner [F,3,3]: area-weighted over faces sharing a vertex *position* (PLY exports
    split vertices at UV seams), falling back to the face normal across creases sharper than crease_deg."""
    p = vertices[faces]
    face_n = torch.linalg.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])  # length = 2 area
    key = torch.round(vertices / 1e-6).long()
    _, position = torch.unique(key, dim=0, return_inverse=True)
    corner_pos = position[faces]  # [F,3]
    accum = torch.zeros(int(position.max()) + 1, 3, device=vertices.device)
    accum.index_add_(0, corner_pos.reshape(-1), face_n.repeat_interleave(3, 0))
    smooth = F.normalize(accum[corner_pos], dim=-1)
    unit = F.normalize(face_n, dim=-1)[:, None].expand_as(smooth)
    keep = (smooth * unit).sum(-1, keepdim=True) > math.cos(math.radians(crease_deg))
    return torch.where(keep, smooth, unit)


class Mesh:
    """A triangle mesh in world coordinates with per-face materials and a Warp BVH."""

    def __init__(self, vertices: torch.Tensor, faces: torch.Tensor, face_material: torch.Tensor,
                 materials: list[Material], name: str = "mesh", part_ids: list[str] | None = None):
        self.name = name
        self.vertices = vertices.float().contiguous()
        self.faces = faces.long().contiguous()
        self.face_material = face_material.long().contiguous()
        self.materials = materials
        self.part_ids = part_ids or [str(i) for i in range(len(materials))]
        p = self.vertices[self.faces]
        self.face_normal = F.normalize(torch.linalg.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), dim=-1)
        self.corner_normal = corner_normals(self.vertices, self.faces)
        low, high = self.vertices.min(0).values, self.vertices.max(0).values
        self.center = 0.5 * (low + high)
        self.radius = float((self.vertices - self.center).norm(dim=-1).max())
        wp.init()
        device = str(self.vertices.device)
        self._points = wp.from_torch(self.vertices, dtype=wp.vec3)
        self._indices = wp.from_torch(self.faces.to(torch.int32).reshape(-1).contiguous(), dtype=wp.int32)
        self._mesh = wp.Mesh(points=self._points, indices=self._indices)
        self._device = device

    def intersect(self, rays: torch.Tensor, max_t: torch.Tensor | None = None):
        """Closest hit along rays [R,6] (unit directions): depth [R] (inf: none), face [R] (-1), barycentrics [R,2]."""
        count = rays.shape[0]
        origins = rays[:, :3].contiguous()
        directions = rays[:, 3:].contiguous()
        if max_t is None:
            max_t = torch.full((count,), NO_HIT, device=rays.device)
        max_t = max_t.float().nan_to_num(posinf=NO_HIT).clamp(max=NO_HIT).contiguous()
        t = torch.empty(count, device=rays.device)
        face = torch.empty(count, dtype=torch.int32, device=rays.device)
        u = torch.empty(count, device=rays.device)
        v = torch.empty(count, device=rays.device)
        if count:
            stream = wp.stream_from_torch(torch.cuda.current_stream(rays.device))
            wp.launch(_closest_hit, dim=count, stream=stream, device=self._device,
                      inputs=[self._mesh.id, wp.from_torch(origins, dtype=wp.vec3),
                              wp.from_torch(directions, dtype=wp.vec3), wp.from_torch(max_t),
                              wp.from_torch(t), wp.from_torch(face), wp.from_torch(u), wp.from_torch(v)])
        t = torch.where(t >= NO_HIT * 0.5, torch.full_like(t, float("inf")), t)
        return t, face.long(), torch.stack((u, v), -1)

    def surface(self, face: torch.Tensor, bary: torch.Tensor):
        """Shading normal (interpolated, unit) and geometric normal of hits (face >= 0)."""
        u, v = bary[:, :1], bary[:, 1:]
        corners = self.corner_normal[face]  # Warp: p = u p0 + v p1 + (1-u-v) p2
        shading = F.normalize(u * corners[:, 0] + v * corners[:, 1] + (1 - u - v) * corners[:, 2], dim=-1)
        return shading, self.face_normal[face]


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
    return world, dict(anchor=anchor.tolist(), size=size, depth=depth, up=by.tolist(),
                       footprint_hits=int(ok.sum()))
