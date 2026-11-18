"""Triangle meshes for the object tracer: loading (PLY parts + FALCON-style manifest), placement on the seabed and
ray queries on a GPU BVH (NVIDIA Warp, `pip install warp-lang`).

A manifest (as FALCON's source/fiat_bicycle/manifest.json) lists the parts of one object, each a PLY with a material
{"mode": metal | ceramic | glass | mirror | diffuse, "rgb": [...], "kd": ..., "eta": ...}. The parts share one
normalization: centred horizontally, the lowest point at y = 0 (Blender PLY export is y-up), the largest extent 1.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import warp as wp

from .shading import Material

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


def load_manifest(path: str | Path, overrides: dict | None = None, with_uv: bool = False):
    """Parts of a manifest as one normalized mesh: vertices [V,3] (y-up, foot at 0, max extent 1), faces [F,3],
    (the part PLYs are taken as y-down, as FALCON's, unless the manifest says "flip_yz": false)
    per-face part index [F], materials (one per part), part ids. overrides: {part id: FALCON material dict}.
    A manifest with "format": "npz" holds one y-up mesh with per-corner UVs and textured
    "pbr" materials (map paths relative to the manifest). with_uv: also return the corner UVs [F,3,2] (or None)."""
    path = Path(path)
    manifest = json.loads(path.read_text())
    overrides = overrides or {}
    if manifest.get("format") == "npz":
        data = np.load(path.parent / manifest["mesh"])
        vertices = data["vertices"].astype(np.float32)
        materials, ids = [], []
        for part in manifest["parts"]:
            spec = dict(part.get("material", {}), **overrides.get(part["id"], {}))
            if "maps" in spec:
                spec["maps"] = {k: str(path.parent / v) for k, v in spec["maps"].items()}
            materials.append(Material.from_falcon(spec))
            ids.append(part["id"])
        low, high = vertices.min(0), vertices.max(0)
        vertices = vertices - 0.5 * (low + high)
        vertices[:, 1] -= vertices[:, 1].min()
        vertices /= float((high - low).max())
        out = vertices, data["faces"].astype(np.int64), data["part"].astype(np.int64), materials, ids
        return (*out, data["corner_uv"].astype(np.float32)) if with_uv else out
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
    out = vertices, np.concatenate(faces), np.concatenate(part_of_face), materials, ids
    return (*out, None) if with_uv else out


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


def mirror_panel(aspect: float = 0.6, border: float = 0.06, thickness: float = 0.04, lean_deg: float = 15.0,
                 mirror: dict | None = None, frame_material: dict | None = None):
    """A standing framed mirror, as FALCON's procedural mirror quads: a box frame (width = aspect x height) with the
    mirror pane on its +z face (+z faces the anchor camera after place_mesh_on_seabed), the top leaned towards +z by
    lean_deg (the pane then reflects the seabed in front of it rather than the water behind the camera).
    Same normalization as load_manifest; returns vertices, faces, per-face part, materials, part ids."""
    w, h, t = aspect, 1.0, thickness
    box = np.array([[x, y, z] for x in (-w / 2, w / 2) for y in (0.0, h) for z in (-t / 2, t / 2)], dtype=np.float32)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    pane = np.array([[-w / 2 + border, border, t / 2 + 1e-3], [w / 2 - border, border, t / 2 + 1e-3],
                     [w / 2 - border, h - border, t / 2 + 1e-3], [-w / 2 + border, h - border, t / 2 + 1e-3]],
                    dtype=np.float32)
    vertices = np.concatenate((box, pane))
    faces = [f for a, b, c, d in quads for f in ((a, b, c), (a, c, d))] + [(8, 9, 10), (8, 10, 11)]
    part = np.array([1] * 12 + [0] * 2)
    a = math.radians(lean_deg)  # rotate about x: the top moves towards +z
    y, z = vertices[:, 1].copy(), vertices[:, 2].copy()
    vertices[:, 1], vertices[:, 2] = y * math.cos(a) - z * math.sin(a), y * math.sin(a) + z * math.cos(a)
    vertices[:, 1] -= vertices[:, 1].min()
    vertices /= float((vertices.max(0) - vertices.min(0)).max())
    materials = [Material.from_falcon(dict({"mode": "mirror", "rgb": [0.92, 0.92, 0.97]}, **(mirror or {}))),
                 Material.from_falcon(dict({"mode": "ceramic", "rgb": [0.96, 0.96, 0.98], "kd": 0.9, "eta": 1.025},
                                           **(frame_material or {})))]
    return vertices, np.array(faces, dtype=np.int64), part, materials, ["mirror", "frame"]


class Mesh:
    """A triangle mesh in world coordinates with per-face materials and a Warp BVH."""

    def __init__(self, vertices: torch.Tensor, faces: torch.Tensor, face_material: torch.Tensor,
                 materials: list[Material], name: str = "mesh", part_ids: list[str] | None = None,
                 corner_uv: torch.Tensor | None = None):
        self.name = name
        self.corner_uv = None if corner_uv is None else corner_uv.float().to(vertices.device).contiguous()
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

    def surface_uv(self, face: torch.Tensor, bary: torch.Tensor) -> torch.Tensor:
        """Texture coordinates [R,2] of hits (same barycentric convention as surface)."""
        u, v = bary[:, :1], bary[:, 1:]
        corners = self.corner_uv[face]
        return u * corners[:, 0] + v * corners[:, 1] + (1 - u - v) * corners[:, 2]








