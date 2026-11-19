"""Whitted-style ray tracing of inserted objects inside a trained FoamSurf scene (visualization only, no gradients).

The scene (Gaussian surface + Voronoi water) is rendered by the native RGB-medium kernel along ray *segments*:
`foamsurf._C.rgb_forward_limited` stops every ray at its own far depth (the first inserted object it hits).
A segment returns the direct light D, the water backscatter B and the per-channel direct transmittance T, so a
pixel is composed front to back as

    raw = D_1 + g(d_1) B_1 + T_1 * (light leaving the object towards the camera)

where g(d) is the trained, direction-dependent water light (phase function + extrapolation, linear in B; see
Trainer._evaluation_water_light) and the object's light is traced recursively: a mirror reflects, glass splits
into a Fresnel-weighted reflection and a refraction through the (water-free) solid, and every secondary ray
continues through the water from the object's surface. The camera model (PPISP) is applied once to the final
image, as in render.py. With no objects the result equals render.py's `raw` mode.

Objects are analytic spheres (`Sphere`) and triangle meshes (`examples.insertion.mesh.Mesh`, NVIDIA Warp BVH). Matte parts
(diffuse, the diffuse share of metal and ceramic) are lit by the scene itself: the radiance arriving at the
object's centre from a few hundred directions (traced through the trained scene without the objects), times an
ambient occlusion of the inserted objects.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

INF = float("inf")
RAY_CHUNK = 4_000_000  # shadow / occlusion rays per batch (GPU memory)


@dataclass
class Material:
    kind: str = "mirror"  # mirror | glass | diffuse | metal | ceramic | pbr
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0)  # reflectance / transmittance / albedo
    ior: float = 1.5  # glass and ceramic Fresnel, relative to water
    kd: float = 1.0  # metal: mirror fraction (rest diffuse); ceramic: diffuse weight
    maps: dict | None = None  # pbr: {"albedo" (sRGB), "metal", "rough": image paths}, sampled at the mesh's UVs

    @staticmethod
    def from_falcon(spec: dict) -> "Material":
        """FALCON material {"mode", "rgb", "kd", "eta"}."""
        return Material(kind=spec.get("mode", "ceramic"), tint=tuple(spec.get("rgb", (1, 1, 1))),
                        ior=float(spec.get("eta", 1.5)), kd=float(spec.get("kd", 1.0)), maps=spec.get("maps"))


_TEXTURES: dict = {}


def texture(path: str, device, srgb: bool) -> torch.Tensor:
    """[1,C,H,W] float texture (linear values; sRGB decoded), cached per path and device."""
    key = (path, str(device))
    if key not in _TEXTURES:
        import numpy as np
        from PIL import Image
        image = torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0)
        image = image ** 2.2 if srgb else image[..., :1]
        _TEXTURES[key] = image.permute(2, 0, 1)[None].contiguous().to(device)
    return _TEXTURES[key]


def sample(image: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Bilinear lookup [R,C] of a [1,C,H,W] texture at UVs [R,2] (Blender convention: v up), wrapping."""
    uv = torch.remainder(uv, 1.0)
    grid = torch.stack((uv[:, 0] * 2 - 1, 1 - uv[:, 1] * 2), -1)[None, None]
    return F.grid_sample(image, grid, mode="bilinear", align_corners=False)[0, :, 0].T


@dataclass
class Sphere:
    center: tuple[float, float, float]
    radius: float
    material: str = "mirror"  # mirror | glass | diffuse | metal | ceramic
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ior: float = 1.5  # glass, relative to water (< 1: a gas bubble, air/water = 1/1.333 = 0.75)
    kd: float = 1.0
    shadow: bool = True  # casts a shadow on the scene (gas bubbles: no)
    power: float = 0.0  # "emissive": a lamp of radiant intensity power x tint (raw units per steradian), see lamps
    spot: tuple[float, float, float] | None = None  # lamp: beam axis (unit), None: shines in all directions
    cone_deg: float = 30.0  # spot: half-angle of the beam (full intensity inside cone_deg - softness_deg)
    softness_deg: float = 8.0

    @property
    def materials(self) -> list[Material]:
        return [Material(self.material, tuple(self.tint), self.ior, self.kd)]

    def tensors(self, device):
        cache = self.__dict__.setdefault("_tensors", {})  # traced thousands of times per frame
        key = str(device)
        if key not in cache:
            cache[key] = (torch.tensor(self.center, dtype=torch.float32, device=device),
                          torch.tensor(self.tint, dtype=torch.float32, device=device))
        return cache[key]


@dataclass
class SphereBatch:
    """Spheres hit by a group of rays, one per ray: what the glass-sphere shading needs (centre, radius [R,1])."""
    center: torch.Tensor
    radius: torch.Tensor
    tint: torch.Tensor

    @staticmethod
    def of(objects: list, index: torch.Tensor) -> "SphereBatch":
        device = index.device
        centers = torch.tensor([o.center if not is_mesh(o) else (0.0, 0.0, 0.0) for o in objects], device=device)
        radii = torch.tensor([o.radius if not is_mesh(o) else 0.0 for o in objects], device=device)
        return SphereBatch(centers[index], radii[index][:, None], torch.ones(3, device=device))

    def tensors(self, device):
        return self.center, self.tint


@dataclass
class TraceSettings:
    daylight: float = 1.0  # scale scene illumination without dimming inserted lights
    max_depth: int = 4
    min_weight: float = 1e-3  # secondary rays carrying less (max channel) are dropped
    epsilon: float = 1e-4  # secondary ray offset, in units of the object's radius
    ambient: tuple[float, float, float] = (0.0, 0.0, 0.0)  # extra constant light on matte surfaces (raw space)
    # The trained water light extrapolation (HG phase, g up to ~0.93) explodes for rays looking up, which the
    # photos never saw but mirrors do (Curacao: x4500 straight up). Above this cosine with the cameras' "up" the
    # water light is the one at that elevation (same azimuth; near the zenith, where the azimuth is undefined, its
    # mean over azimuths) times the trained growth towards the zenith, softly limited per channel to
    # water_light_max_gain: the water stays brighter towards the surface without the blow-up. None: trained.
    water_light_max_up: float | None = 0.6
    water_light_max_gain: float = 2.0
    # Shadows of the objects on the scene's surfaces (only the direct light D of a segment is darkened, at the
    # first Gaussian surface the ray meets; the water's backscatter stays). Underwater light comes from above
    # through the surface and is soft: a cone of half-angle shadow_cone_deg around the cameras' "up", plus the
    # cosine-weighted solid angle an object takes from the sky (ambient occlusion, gives the contact darkening).
    shadows: bool = True
    shadow_strength: float = 0.6  # fraction of the direct light coming from the cone
    shadow_cone_deg: float = 25.0
    occlusion_strength: float = 1.0
    shadow_samples: int = 96  # meshes: rays per point for the cone and for the occlusion
    # The first surface a ray meets is ragged (needle-like surfels), and next to an object the occlusion changes
    # fast with height, so the shadow of the camera's own rays is smoothed over the image (normalized convolution
    # over the pixels that see a surface; Gaussian sigma in pixels, 0: off).
    shadow_blur_px: float = 3.0
    # Matte surfaces: environment directions (from the object's centre) and occlusion rays per hit.
    environment_samples: int = 256
    occlusion_samples: int = 16
    occlusion_distance: float = 0.5  # in units of the object's radius
    # The trained water holds no sunlight (rays looking up see only the dim backscatter), so matte objects would be
    # far darker than the seabed next to them. Light from above is calibrated on the seabed itself: the radiance
    # seen looking down from the object, divided by this reference albedo, is the light a horizontal surface gets;
    # a surface with normal n gets it times ((1 + n.up)/2)^sky_exponent (1: overcast sky, larger: more from above).
    # None: environment only.
    seabed_albedo: float | None = 0.7
    sky_exponent: float = 2.0
    # Pixels next to an object's outline (different object, or object / scene, in a 4-neighbour; dilated by one
    # pixel) and pixels of mirror, metal and glass spheres are averaged with antialias extra rays spread over the
    # pixel: no stair-stepped outlines, and the reef a sphere squeezes into a thin band along its horizon becomes a
    # texture instead of single-pixel speckles. 0: off.
    antialias: int = 4
    # Lamps (emissive spheres with power > 0): their light reaches surfaces (Gaussians: relit in proportion to
    # their colour over the scene's light from above; objects: added to the matte light) and is scattered by the
    # water towards the camera (single scattering, Henyey-Greenstein phase lamp_g, lamp_samples equiangular
    # samples per ray; the scattering coefficient is the attenuation times the water colour over the ambient
    # radiance, i.e. B_inf = b L / c). lamp_volume_shadows: the camera rays' samples test visibility of the lamp
    # (beams and shadows in the water).
    lamps: bool = True
    lamp_g: float = 0.9
    lamp_samples: int = 24
    lamp_volume_shadows: bool = True
    lamp_blur_px: float = 2.5  # camera rays: the lamp light on the scene smoothed over the image (glow: x1.5)


@dataclass
class TraceStats:
    rays_per_depth: list = field(default_factory=list)


# --- direction sets (deterministic, so videos do not shimmer) ---------------------------------------------------
GOLDEN = math.pi * (3 - math.sqrt(5))


def sphere_directions(count: int, device) -> torch.Tensor:
    i = torch.arange(count, device=device, dtype=torch.float32) + 0.5
    z = 1 - 2 * i / count
    r = (1 - z * z).clamp_min(0).sqrt()
    phi = i * GOLDEN
    return torch.stack((r * phi.cos(), r * phi.sin(), z), -1)


def cosine_directions(count: int, device) -> torch.Tensor:
    """Cosine-weighted unit hemisphere around +z [K,3]."""
    i = torch.arange(count, device=device, dtype=torch.float32) + 0.5
    r = (i / count).sqrt()
    phi = i * GOLDEN
    return torch.stack((r * phi.cos(), r * phi.sin(), (1 - r * r).clamp_min(0).sqrt()), -1)


def cone_directions(count: int, half_angle: float, device) -> torch.Tensor:
    """Uniform in a cone of half_angle (radians) around +z [K,3]."""
    i = torch.arange(count, device=device, dtype=torch.float32) + 0.5
    z = 1 - (i / count) * (1 - math.cos(half_angle))
    r = (1 - z * z).clamp_min(0).sqrt()
    phi = i * GOLDEN
    return torch.stack((r * phi.cos(), r * phi.sin(), z), -1)


def tangent_frame(n: torch.Tensor):
    helper = torch.where(n[:, 2:].abs() < 0.9, torch.tensor([0.0, 0.0, 1.0], device=n.device),
                         torch.tensor([1.0, 0.0, 0.0], device=n.device))
    t = F.normalize(torch.linalg.cross(helper, n), dim=-1)
    return t, torch.linalg.cross(n, t)


def around(local: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """Local directions [K,3] (+z = n) turned to every normal n [P,3]: [P,K,3]."""
    t, b = tangent_frame(n)
    return local[None, :, :1] * t[:, None] + local[None, :, 1:2] * b[:, None] + local[None, :, 2:] * n[:, None]


# --- intersections ---------------------------------------------------------------------------------------------
def lamps_of(objects) -> list:
    return [o for o in objects if not is_mesh(o) and o.material == "emissive" and o.power > 0]


def is_mesh(obj) -> bool:
    return hasattr(obj, "faces")


def sphere_hit(rays: torch.Tensor, sphere: Sphere) -> torch.Tensor:
    origin, direction = rays[:, :3], rays[:, 3:]
    center, _ = sphere.tensors(rays.device)
    oc = origin - center
    b = (oc * direction).sum(-1)
    c = (oc * oc).sum(-1) - sphere.radius ** 2
    disc = b * b - c
    root = disc.clamp_min(0).sqrt()
    near, far = -b - root, -b + root
    t = torch.where(near > 0, near, far)
    return torch.where((disc > 0) & (t > 0), t, torch.full_like(t, INF))


def spheres_hit(rays: torch.Tensor, centers: torch.Tensor, radii: torch.Tensor, chunk: int = 1 << 18):
    """Nearest of many spheres at once: (t [R], index into centers [R]); t = INF where none is hit."""
    t_out = torch.full((rays.shape[0],), INF, device=rays.device)
    i_out = torch.zeros(rays.shape[0], dtype=torch.long, device=rays.device)
    if centers.shape[0] > 4:  # only the rays through a sphere bounding them all (a bubble plume is narrow)
        middle = centers.mean(0)
        bound = Sphere(tuple(middle.tolist()), float(((centers - middle).norm(dim=-1) + radii).max()))
        near = (sphere_hit(rays, bound) < INF).nonzero().squeeze(-1)
        if near.numel() < rays.shape[0]:
            if near.numel():
                t_out[near], i_out[near] = spheres_hit(rays[near], centers, radii, chunk)
            return t_out, i_out
    for begin in range(0, rays.shape[0], chunk):
        r = rays[begin:begin + chunk]
        oc = r[:, None, :3] - centers[None]  # [C,S,3]
        b = (oc * r[:, None, 3:]).sum(-1)
        disc = b * b - ((oc * oc).sum(-1) - radii[None] ** 2)
        root = disc.clamp_min(0).sqrt()
        near, far = -b - root, -b + root
        t = torch.where(near > 0, near, far)
        t = torch.where((disc > 0) & (t > 0), t, torch.full_like(t, INF))
        t_out[begin:begin + chunk], i_out[begin:begin + chunk] = t.min(-1)
    return t_out, i_out


@dataclass
class Hits:
    t: torch.Tensor  # [R] inf: none
    obj: torch.Tensor  # [R] object index, -1
    material: torch.Tensor  # [R] index into the flat material table, -1
    point: torch.Tensor  # [R,3]
    normal: torch.Tensor  # [R,3] shading normal, facing the ray
    geometric: torch.Tensor  # [R,3] geometric normal, facing the ray
    uv: torch.Tensor | None = None  # [R,2] texture coordinates of mesh hits (meshes with UVs), else 0


def material_table(objects):
    """[(object index, Material)] over all objects; offsets[i] = first entry of object i."""
    table, offsets = [], []
    for i, obj in enumerate(objects):
        offsets.append(len(table))
        table.extend((i, m) for m in obj.materials)
    return table, offsets


def intersect(rays: torch.Tensor, objects: list, max_t: torch.Tensor | None = None) -> Hits:
    count, device = rays.shape[0], rays.device
    best = torch.full((count,), INF, device=device) if max_t is None else max_t.clone()
    index = torch.full((count,), -1, dtype=torch.long, device=device)
    faces = {}
    spheres = [i for i, obj in enumerate(objects) if not is_mesh(obj)]
    if spheres:  # all at once (bubble streams have dozens), before the meshes so they get a tighter max_t
        centers = torch.stack([objects[i].tensors(device)[0] for i in spheres])
        radii = torch.tensor([objects[i].radius for i in spheres], dtype=torch.float32, device=device)
        t, which = spheres_hit(rays, centers, radii)
        closer = t < best
        best = torch.where(closer, t, best)
        index = torch.where(closer, torch.tensor(spheres, device=device)[which], index)
    for i, obj in enumerate(objects):
        if not is_mesh(obj):
            continue
        t, face, bary = obj.intersect(rays, best)
        faces[i] = (face, bary)
        closer = t < best
        best = torch.where(closer, t, best)
        index = torch.where(closer, torch.full_like(index, i), index)
    best = torch.where(index >= 0, best, torch.full_like(best, INF))
    point = rays[:, :3] + best[:, None].nan_to_num(posinf=0.0) * rays[:, 3:]
    normal = torch.zeros_like(point)
    geometric = torch.zeros_like(point)
    material = torch.full_like(index, -1)
    uv = torch.zeros(count, 2, device=device) if any(getattr(o, "corner_uv", None) is not None for o in objects) \
        else None
    _, offsets = material_table(objects)
    for i, obj in enumerate(objects):
        mask = index == i
        if not mask.any():
            continue
        if is_mesh(obj):
            face, bary = faces[i]
            shading, geo = obj.surface(face[mask], bary[mask])
            normal[mask], geometric[mask] = shading, geo
            material[mask] = offsets[i] + obj.face_material[face[mask]]
            if uv is not None and obj.corner_uv is not None:
                uv[mask] = obj.surface_uv(face[mask], bary[mask])
        else:
            center, _ = obj.tensors(device)
            normal[mask] = geometric[mask] = F.normalize(point[mask] - center, dim=-1)
            material[mask] = offsets[i]
    # Face the ray (meshes may be open or inconsistently oriented); a shading normal turned away from the ray
    # (interpolation at silhouettes) falls back to the geometric one.
    d = rays[:, 3:]
    flip = (geometric * d).sum(-1, keepdim=True) > 0
    geometric = torch.where(flip, -geometric, geometric)
    normal = torch.where(flip, -normal, normal)
    normal = torch.where((normal * d).sum(-1, keepdim=True) >= 0, geometric, normal)
    return Hits(best, index, material, point, normal, geometric, uv)


def reflect(d: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return d - 2.0 * (d * n).sum(-1, keepdim=True) * n


def refract(d: torch.Tensor, n: torch.Tensor, eta: float | torch.Tensor):
    """Refraction of unit d at a surface with unit normal n facing against d, eta = n_from / n_to.
    Returns (direction, total_internal_reflection mask)."""
    cos_i = -(d * n).sum(-1, keepdim=True)
    k = 1.0 - eta * eta * (1.0 - cos_i * cos_i)
    tir = (k < 0).squeeze(-1)
    out = eta * d + (eta * cos_i - k.clamp_min(0).sqrt()) * n
    return F.normalize(out, dim=-1), tir


def fresnel_schlick(cos_i: torch.Tensor, n1: float, n2: float) -> torch.Tensor:
    r0 = ((n1 - n2) / (n1 + n2)) ** 2
    return r0 + (1 - r0) * (1 - cos_i.clamp(0, 1)) ** 5












