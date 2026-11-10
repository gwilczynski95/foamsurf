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

Objects are analytic spheres (`Sphere`) and triangle meshes (`vis.mesh.Mesh`, NVIDIA Warp BVH). Matte parts
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


class SceneTracer:
    """Holds a loaded trainer (render.load_run) and traces rays with inserted objects."""

    def __init__(self, trainer, color_origin: torch.Tensor | None = None):
        self.trainer = trainer
        self.renderer = trainer.renderer
        self.medium = trainer.model.medium.render_scene()
        self.color_origin = color_origin
        self.gaussians = None
        self._environment = {}
        self.image_shape = None  # (h, w) while render_camera traces a camera's rays
        self.shadow_map = None  # [H*W] the blurred shadow factor of the last camera (render_camera)
        if color_origin is not None:
            self.set_color_origin(color_origin)

    @torch.no_grad()
    def set_color_origin(self, origin: torch.Tensor) -> None:
        """Gaussian colours (SH) evaluated from this point; the OptiX geometry is rebuilt only once."""
        scene = self.trainer.model.gaussians.render_scene(origin)
        if self.gaussians is None:
            self.renderer.set_gaussian_geometry(scene)
        self.gaussians = scene

    # --- one segment of the scene, per-ray far -------------------------------------------------
    @torch.no_grad()
    def nearest_sites(self, queries: torch.Tensor, chunk: int = 32768) -> torch.Tensor:
        sites = self.medium.sites
        out = torch.empty(queries.shape[0], dtype=torch.long, device=queries.device)
        for begin in range(0, queries.shape[0], chunk):
            out[begin:begin + chunk] = torch.cdist(queries[begin:begin + chunk], sites).argmin(-1)
        return out

    @torch.no_grad()
    def segment(self, rays: torch.Tensor, far: torch.Tensor, start_points: torch.Tensor | None = None):
        """Scene light along rays [R,6] up to depth far [R]: direct D, backscatter B, direct transmittance T [R,3]."""
        r, g, v = self.renderer, self.gaussians, self.medium
        rays = torch.cat((rays[:, :3], F.normalize(rays[:, 3:], dim=-1)), -1).contiguous()
        count = rays.shape[0]
        if start_points is None:
            origins = rays[:, :3]
            if bool((origins == origins[:1]).all()):
                start_points = self.nearest_sites(origins[:1]).expand(count)
            else:
                start_points = self.nearest_sites(origins)
        start_points = start_points.to(torch.uint32).contiguous()
        hit_depths, hit_indices = r._optix.Render(rays[:, :3].contiguous(), rays[:, 3:].contiguous(),
                                                  torch.full((count,), r.near, device=rays.device), count)
        topology = r._voronoi_topology
        tensors = [v.sites, v.attenuation, v.backscatter, v.colors, topology.adjacency, topology.offsets, rays,
                   start_points, hit_depths, hit_indices, g.means, g.log_scales, g.quaternions, g.colors,
                   g.opacity_logits, g.raw_kappa, torch.zeros(3, device=rays.device)]
        far = far.clamp(max=r.far).float().contiguous()
        direct, scattered, transmission, _ = r._bindings.rgb_forward_limited(
            tensors, far, r.near, r.far, r.transmittance_threshold, r.max_intersections)
        self.last_surface = r._bindings.surface_hits(hit_depths, hit_indices, r.near, r.far)
        return direct, scattered, transmission

    @torch.no_grad()
    def water_light(self, direct, scattered, rays, max_up: float | None = None, max_gain: float = 2.0):
        light = self.trainer.evaluator.water_light
        if light is None:
            return direct + scattered
        out = light(direct + scattered, scattered, rays)
        if max_up is None:
            return out
        d = rays[:, 3:]
        up = self.up.to(d)
        cos = (d * up).sum(-1)
        high = cos > max_up
        if not high.any():
            return out
        d, cos = d[high], cos[high, None]
        image, back, origins = (direct + scattered)[high], scattered[high], rays[high, :3]
        at = lambda directions: light(image, back, torch.cat((origins, directions), -1))
        ring = lambda horizontal: horizontal * (1 - max_up ** 2) ** 0.5 + up * max_up
        side = self.side.to(d)
        other = torch.linalg.cross(up, side)
        mean = sum(at(ring((math.cos(a) * side + math.sin(a) * other).expand_as(d)))
                   for a in torch.linspace(0, 2 * math.pi, 9)[:-1].tolist()) / 8
        horizontal = d - cos * up
        length = horizontal.norm(dim=-1, keepdim=True)
        same_azimuth = at(ring(horizontal / length.clamp_min(1e-6)))
        w = ((length - 0.05) / 0.25).clamp(0, 1)  # 0 within ~3 deg of the zenith, 1 beyond ~17 deg
        w = w * w * (3 - 2 * w)
        reference = (w * same_azimuth + (1 - w) * mean).clamp_min(1e-6)
        ratio = out[high] / reference
        span = max_gain - 1
        limited = torch.where(ratio > 1, 1 + span * torch.tanh((ratio - 1) / span), ratio)
        out[high] = reference * limited
        return out
    @property
    def up(self) -> torch.Tensor:
        up = -torch.stack([c.down for c in self.trainer.dataset.train]).mean(0)
        return up / up.norm()

    @property
    def side(self) -> torch.Tensor:  # any unit vector orthogonal to up (straight-up rays)
        return F.normalize(torch.linalg.cross(self.up, torch.tensor([0.0, 0.0, 1.0], device=self.up.device)), dim=0)

    # --- light on matte surfaces ----------------------------------------------------------------
    @torch.no_grad()
    def environment(self, center: torch.Tensor, samples: int, settings: TraceSettings):
        """Directions [K,3] and the scene's radiance [K,3] arriving at a point (objects not included); cached."""
        key = (tuple(round(float(x), 5) for x in center), samples, settings.water_light_max_up,
               settings.water_light_max_gain)
        if key not in self._environment:
            directions = sphere_directions(samples, center.device)
            rays = torch.cat((center.expand_as(directions), directions), -1).contiguous()
            start = self.nearest_sites(center[None]).expand(samples)
            self._environment[key] = (directions, self.trace(rays, [], settings, start_points=start))
        return self._environment[key]

    @torch.no_grad()
    def irradiance(self, obj, normal: torch.Tensor, settings: TraceSettings) -> torch.Tensor:
        """Cosine-weighted incoming light / pi at normals [P,3] from the environment at the object's centre."""
        center = obj.center if is_mesh(obj) else obj.tensors(normal.device)[0]
        directions, radiance = self.environment(center, settings.environment_samples, settings)
        cos = (normal @ directions.T).clamp_min(0)  # [P,K]
        light = 4.0 / directions.shape[0] * cos @ radiance + torch.tensor(settings.ambient, device=normal.device)
        if settings.seabed_albedo is not None:
            up = self.up.to(normal)
            down = (-(directions @ up)).clamp_min(0)  # cosine-weighted mean over the lower hemisphere
            seabed = (down[:, None] * radiance).sum(0) / down.sum().clamp_min(1e-6)
            light = light + (seabed / settings.seabed_albedo) * ((1 + normal @ up) / 2).pow(settings.sky_exponent)[:, None]
        return light

    @torch.no_grad()
    def blocked(self, rays: torch.Tensor, objects: list, max_t: torch.Tensor) -> torch.Tensor:
        """[R] bool: some inserted object lies on the ray before max_t."""
        hit = torch.zeros(rays.shape[0], dtype=torch.bool, device=rays.device)
        for obj in objects:
            if not is_mesh(obj) and obj.material == "emissive":
                continue
            t = obj.intersect(rays, max_t)[0] if is_mesh(obj) else sphere_hit(rays, obj)
            hit |= t < max_t
        return hit

    @torch.no_grad()
    def occlusion(self, points, normal, objects, distance: float, samples: int, epsilon: float) -> torch.Tensor:
        """[P] cosine-weighted fraction of the hemisphere around normal hidden by objects within distance."""
        out = torch.zeros(points.shape[0], device=points.device)
        if points.shape[0] == 0 or not objects:
            return out
        local = cosine_directions(samples, points.device)
        chunk = max(1, RAY_CHUNK // samples)
        for begin in range(0, points.shape[0], chunk):
            q, m = points[begin:begin + chunk], normal[begin:begin + chunk]
            directions = around(local, m)  # [P,K,3]
            origins = (q + epsilon * m)[:, None].expand_as(directions)
            rays = torch.cat((origins, directions), -1).reshape(-1, 6).contiguous()
            far = torch.full((rays.shape[0],), distance, device=points.device)
            out[begin:begin + chunk] = self.blocked(rays, objects, far).reshape(-1, samples).float().mean(-1)
        return out

    # --- recursive tracing ------------------------------------------------------------------------
    @torch.no_grad()
    def trace(self, rays: torch.Tensor, objects: list, settings: TraceSettings = TraceSettings(),
              depth: int = 0, weight: torch.Tensor | None = None, start_points=None, stats: TraceStats | None = None,
              shadow: torch.Tensor | None = None):
        """Raw radiance [R,3] (water light applied, before the camera model). shadow: [R] factor of the direct light
        to use instead of computing it."""
        count = rays.shape[0]
        if stats is not None:
            while len(stats.rays_per_depth) <= depth:
                stats.rays_per_depth.append(0)
            stats.rays_per_depth[depth] += count
        if count == 0:
            return torch.zeros(0, 3, device=rays.device)
        if weight is None:
            weight = torch.ones(count, 3, device=rays.device)
        if not objects:
            direct, scattered, _ = self.segment(rays, torch.full((count,), INF, device=rays.device), start_points)
            return self.water_light(direct, scattered, rays, settings.water_light_max_up, settings.water_light_max_gain)
        lamps = lamps_of(objects) if settings.lamps else []
        if lamps:  # their caches trace the scene (segment), which would overwrite last_surface below
            self.water_coefficients()
            self.ambient_up()
        hits = intersect(rays, objects)
        direct, scattered, transmission = self.segment(rays, hits.t, start_points)
        image = depth == 0 and self.image_shape is not None and \
            count == self.image_shape[0] * self.image_shape[1]
        lamp_direct = None
        if lamps:
            ratio, valid = self.lamp_surface(rays, hits.t, objects, lamps)
            if image and settings.lamp_blur_px > 0:
                ratio = self._blur_image(ratio, valid.float(), settings.lamp_blur_px)
            lamp_direct = direct * ratio
        if settings.shadows and shadow is not None:
            direct = direct * shadow[:, None]
        elif settings.shadows:
            factor = self.shadow_factor(rays, hits.t, objects, settings)
            if depth == 0 and self.image_shape is not None and settings.shadow_blur_px > 0:
                factor = self._blur_shadow(factor, rays, hits.t, settings.shadow_blur_px)
            direct = direct * factor[:, None]
        out = self.water_light(direct, scattered, rays, settings.water_light_max_up, settings.water_light_max_gain)
        if lamps:
            glow = self.lamp_inscatter(rays, hits.t, objects, lamps, settings,
                                       settings.lamp_volume_shadows and depth == 0)
            if image and settings.lamp_blur_px > 0:
                glow = self._blur_image(glow, torch.ones(count, device=rays.device), 1.5 * settings.lamp_blur_px)
            out = out + lamp_direct + glow
        if not bool((hits.obj >= 0).any()) or depth >= settings.max_depth:  # past the last bounce: black
            return out
        # Light leaving the objects towards the ray origin, attenuated by the water in front of them.
        carried = transmission * weight
        table, _ = material_table(objects)
        alive = (hits.obj >= 0) & (carried.amax(-1) > settings.min_weight)
        # Glass spheres of one material are shaded together (a bubble stream is dozens of spheres; one by one,
        # every bounce would be traced per sphere).
        groups = {}
        for m in hits.material[alive].unique().tolist():
            obj_index, material = table[m]
            glass_sphere = material.kind == "glass" and not is_mesh(objects[obj_index])
            key = ("glass", material.tint, material.ior) if glass_sphere else m
            groups.setdefault(key, []).append(m)
        for key, ms in groups.items():
            mask = alive & torch.isin(hits.material, torch.tensor(ms, device=rays.device))
            obj_index, material = table[ms[0]]
            obj = objects[obj_index] if len(ms) == 1 else SphereBatch.of(objects, hits.obj[mask])
            sub = Hits(*(None if x is None else x[mask] for x in (hits.t, hits.obj, hits.material, hits.point,
                                                                   hits.normal, hits.geometric, hits.uv)))
            out[mask] += transmission[mask] * self.shade(obj, material, rays[mask], sub, settings,
                                                         depth, carried[mask], objects, stats)
        return out

    # --- lamps ---------------------------------------------------------------------------------------
    def water_coefficients(self):
        """Mean attenuation c [3] and scattering b [3] of the water (b = c B_inf / ambient radiance: single
        scattering of the scene's own light gives the trained water colour B_inf), cached."""
        if getattr(self, "_water", None) is None:
            v = self.medium
            c = v.attenuation.mean(0)
            _, radiance = self.environment(self.color_origin.to(c).float(), 256, TraceSettings())
            b = c * v.colors.mean(0) / radiance.mean(0).clamp_min(1e-6)
            self._water = (c, b)
        return self._water

    def gaussian_hit(self, rays: torch.Tensor, chunk: int = 1 << 21) -> torch.Tensor:
        """[R] depth of the first Gaussian surface along rays (inf: none)."""
        out = torch.empty(rays.shape[0], device=rays.device)
        for begin in range(0, rays.shape[0], chunk):
            depth = self.renderer.render_gaussians(rays[begin:begin + chunk].contiguous(), self.gaussians,
                                                   background=(0.0, 0.0, 0.0)).surface_depth.reshape(-1)
            out[begin:begin + chunk] = torch.where(torch.isfinite(depth) & (depth > 0), depth,
                                                   torch.full_like(depth, INF))
        return out

    def lamp_reach(self, points: torch.Tensor, lamp, objects, visibility: bool = True):
        """Light of a lamp arriving at points [P,3]: unit directions to the lamp [P,3] and irradiance on a surface
        facing it [P,3] (intensity / r^2, water attenuation over r, spot profile, shadows of objects and Gaussians)."""
        c, _ = self.water_coefficients()
        center, color = lamp.tensors(points.device)
        v = center - points
        r = v.norm(dim=-1, keepdim=True).clamp_min(lamp.radius)
        l = v / r
        e = lamp.power * color * torch.exp(-c * r) / (r * r)
        if lamp.spot is not None:
            axis = torch.tensor(lamp.spot, dtype=torch.float32, device=points.device)
            cos = (-l * axis).sum(-1, keepdim=True)
            inner, outer = (math.cos(math.radians(max(lamp.cone_deg - lamp.softness_deg, 0.0))),
                            math.cos(math.radians(lamp.cone_deg + lamp.softness_deg)))
            w = ((cos - outer) / max(inner - outer, 1e-6)).clamp(0, 1)
            e = e * (w * w * (3 - 2 * w))
        if visibility:
            live = e.amax(-1) > 1e-5
            if live.any():
                rays = torch.cat((points[live], l[live]), -1).contiguous()
                max_t = (r[live, 0] - lamp.radius).clamp_min(0)
                seen = (self.gaussian_hit(rays) >= max_t) & ~self.blocked(rays, objects, max_t)
                e[live] = e[live] * seen[:, None]
        return l, e

    def lamp_light(self, points, normals, objects, lamps) -> torch.Tensor:
        """[P,3] lamp light on matte surfaces, in the units of irradiance() (cosine-weighted)."""
        total = torch.zeros_like(points)
        for lamp in lamps:
            l, e = self.lamp_reach(points, lamp, objects)
            total = total + e * (normals * l).sum(-1, keepdim=True).clamp_min(0) / math.pi
        return total

    def ambient_up(self) -> torch.Tensor:
        """[3] the scene's light on a surface facing up (what the Gaussian colours were lit by), cached."""
        if getattr(self, "_ambient_up", None) is None:
            center = self.color_origin.cuda().float()
            directions, radiance = self.environment(center, 256, TraceSettings())
            up = self.up.to(center)
            light = 4.0 / directions.shape[0] * (directions @ up).clamp_min(0) @ radiance
            down = (-(directions @ up)).clamp_min(0)
            seabed = (down[:, None] * radiance).sum(0) / down.sum().clamp_min(1e-6)
            self._ambient_up = light + seabed / TraceSettings().seabed_albedo
        return self._ambient_up

    def lamp_surface(self, rays, t_hit, objects, lamps):
        """Lamp light on the first Gaussian surface of each ray over the scene's light from above, [R,3], and the
        rays that have one [R]: the surface's direct light D (which already holds the water from the surface to
        the camera) times this ratio is the lamp's light leaving it towards the camera."""
        from training.oceansplat import rotation_columns
        depth, index = (x.reshape(-1) for x in self.last_surface)
        valid = (index >= 0) & (depth > 0) & (depth < t_hit)
        out = torch.zeros_like(rays[:, :3])
        if not valid.any():
            return out, valid
        d = rays[valid, 3:]
        p = rays[valid, :3] + depth[valid, None] * d
        _, _, n = rotation_columns(self.gaussians.quaternions[index[valid].long()])
        n = F.normalize(n, dim=-1)
        n = torch.where((n * d).sum(-1, keepdim=True) > 0, -n, n)
        light = self.lamp_light(p - d * 1e-3 + n * 2e-2, n, objects, lamps)
        out[valid] = light / self.ambient_up().clamp_min(1e-6)
        return out, valid

    def _blur_image(self, values: torch.Tensor, valid: torch.Tensor, sigma: float) -> torch.Tensor:
        """[N,C] per-pixel values of the camera image smoothed over the valid pixels (normalized convolution)."""
        h, w = self.image_shape
        radius = int(3 * sigma)
        x = torch.arange(-radius, radius + 1, device=values.device, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()

        def blur(image):  # [C,H,W]
            c = image.shape[0]
            image = image[:, None]
            image = F.conv2d(F.pad(image, (radius, radius, 0, 0), mode="replicate"), kernel.view(1, 1, 1, -1))
            image = F.conv2d(F.pad(image, (0, 0, radius, radius), mode="replicate"), kernel.view(1, 1, -1, 1))
            return image[:, 0].reshape(c, -1)
        weight = blur(valid.reshape(1, h, w))[0]
        smooth = blur((values * valid[:, None]).T.reshape(-1, h, w)).T / weight.clamp_min(1e-6)[:, None]
        return torch.where(valid[:, None] > 0, smooth, values)

    def lamp_inscatter(self, rays, t_hit, objects, lamps, settings, volume_shadows: bool) -> torch.Tensor:
        """[R,3] lamp light scattered once by the water towards the ray origin, up to the ray's first surface
        (Gaussian or object): equiangular sampling about each lamp (Kulla & Fajardo 2012), stratified."""
        c, b = self.water_coefficients()
        depth, index = (x.reshape(-1) for x in self.last_surface)
        end = torch.where((index >= 0) & (depth > 0), depth, torch.full_like(depth, self.renderer.far))
        end = torch.minimum(end, t_hit.nan_to_num(posinf=self.renderer.far)).clamp(max=self.renderer.far)
        o, d = rays[:, :3], rays[:, 3:]
        g = settings.lamp_g
        out = torch.zeros_like(o)
        n = settings.lamp_samples
        for lamp in lamps:
            center, _ = lamp.tensors(o.device)
            along = ((center - o) * d).sum(-1)
            h = (o + along[:, None] * d - center).norm(dim=-1).clamp_min(0.5 * lamp.radius)
            theta_a, theta_b = torch.atan2(-along, h), torch.atan2(end - along, h)
            span = (theta_b - theta_a).clamp_min(0)
            jitter = torch.rand(o.shape[0], device=o.device)
            acc = torch.zeros_like(o)
            for k in range(n):
                theta = theta_a + span * (k + jitter) / n
                t = along + h * torch.tan(theta)
                x = o + t[:, None] * d
                l, e = self.lamp_reach(x, lamp, objects, visibility=volume_shadows)
                cos = (l * d).sum(-1, keepdim=True)  # light travels along -l, scattered into -d: cos = l.d
                phase = (1 - g * g) / (4 * math.pi * (1 + g * g - 2 * g * cos).clamp_min(1e-6) ** 1.5)
                r = (center - x).norm(dim=-1, keepdim=True).clamp_min(lamp.radius)
                # e = I exp(-c r) / r^2; the pdf of t is h / (span r^2): integrand / pdf = e r^2 span / h
                acc = acc + b * phase * e * torch.exp(-c * t[:, None]) * (r * r) * (span / h)[:, None]
            out = out + acc / n
        return out

    @torch.no_grad()
    def shadow_factor(self, rays, t_hit, objects, settings) -> torch.Tensor:
        """[R] multiplier of the direct light at the first Gaussian surface before t_hit (1 elsewhere)."""
        from training.oceansplat import rotation_columns
        depth, index = (x.reshape(-1) for x in self.last_surface)
        valid = (index >= 0) & (depth > 0) & (depth < t_hit)
        factor = torch.ones(rays.shape[0], device=rays.device)
        if not valid.any():
            return factor
        d = rays[valid, 3:]
        p = rays[valid, :3] + depth[valid, None] * d
        _, _, n = rotation_columns(self.gaussians.quaternions[index[valid].long()])
        n = F.normalize(n, dim=-1)
        n = torch.where((n * d).sum(-1, keepdim=True) > 0, -n, n)  # facing the ray
        up = self.up.to(d)
        cone = torch.tensor(settings.shadow_cone_deg, device=d.device).deg2rad()
        lit = torch.ones(p.shape[0], device=d.device)
        for obj in objects:
            if is_mesh(obj):
                lit = lit * self._mesh_shadow(obj, p, n, up, float(cone), settings)
                continue
            if not obj.shadow:
                continue
            center, _ = obj.tensors(d.device)
            v = center - p
            distance = v.norm(dim=-1).clamp_min(1e-6)
            v = v / distance[:, None]
            ratio = (obj.radius / distance).clamp(max=1.0)
            occlusion = (n * v).sum(-1).clamp_min(0) * ratio ** 2  # cosine-weighted solid angle / pi
            # Directional part: overlap of the light cone (half-angle cone) and the sphere's disc (half-angle beta)
            # seen from p, as a smooth step in their angular separation.
            beta = torch.asin(ratio)
            theta = torch.acos((v * up).sum(-1).clamp(-1, 1))
            overlap = ((beta + cone - theta) / (2 * torch.minimum(beta, cone)).clamp_min(1e-6)).clamp(0, 1)
            blocked = overlap * (beta / cone).clamp(max=1.0) ** 2
            lit = lit * (1 - settings.shadow_strength * blocked) * (1 - settings.occlusion_strength * occlusion)
        factor[valid] = lit
        return factor

    def _blur_shadow(self, factor, rays, t_hit, sigma: float) -> torch.Tensor:
        h, w = self.image_shape
        if factor.shape[0] != h * w:
            return factor
        depth, index = (x.reshape(-1) for x in self.last_surface)
        valid = ((index >= 0) & (depth > 0) & (depth < t_hit)).float()
        radius = int(3 * sigma)
        x = torch.arange(-radius, radius + 1, device=factor.device, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()

        def blur(image):
            image = image.reshape(1, 1, h, w)
            image = F.conv2d(F.pad(image, (radius, radius, 0, 0), mode="replicate"), kernel.view(1, 1, 1, -1))
            image = F.conv2d(F.pad(image, (0, 0, radius, radius), mode="replicate"), kernel.view(1, 1, -1, 1))
            return image.reshape(-1)
        weight = blur(valid)
        smooth = blur(factor * valid) / weight.clamp_min(1e-6)
        self.shadow_map = torch.where(weight > 1e-3, smooth, factor)  # also behind objects, for the antialias rays
        return torch.where(valid > 0, smooth, factor)

    def _mesh_shadow(self, mesh, p, n, up, cone: float, settings) -> torch.Tensor:
        """Sampled version of the sphere shadow for a mesh, on points within 3 radii of it."""
        lit = torch.ones(p.shape[0], device=p.device)
        near = (p - mesh.center).norm(dim=-1) < 3 * mesh.radius
        if not near.any():
            return lit
        # Surfel normals are ragged (elongated surfels on sand tilt every which way) and turn the occlusion into
        # spikes, surfel by surfel; the occlusion hemisphere is the one around "up" instead.
        q = p[near]
        m = up[None].expand_as(q)
        k = settings.shadow_samples
        eps = 1e-3 * mesh.radius
        cone_dirs = around(cone_directions(k, cone, p.device), up[None].expand(1, 3))[0]  # [K,3]
        in_cone = torch.empty(q.shape[0], device=p.device)
        chunk = max(1, RAY_CHUNK // k)
        for begin in range(0, q.shape[0], chunk):
            origins = (q[begin:begin + chunk] + eps * up)[:, None].expand(-1, k, -1)
            rays = torch.cat((origins, cone_dirs[None].expand_as(origins)), -1).reshape(-1, 6).contiguous()
            in_cone[begin:begin + chunk] = (mesh.intersect(rays)[0] < INF).reshape(-1, k).float().mean(-1)
        occluded = self.occlusion(q, m, [mesh], 2 * mesh.radius, k, eps)
        lit[near] = (1 - settings.shadow_strength * in_cone) * (1 - settings.occlusion_strength * occluded)
        return lit

    def _secondary(self, origins, directions, obj, settings):
        origins = origins + directions * (settings.epsilon * obj.radius)
        return torch.cat((origins, directions), -1).contiguous()

    def _branch(self, rays, objects, settings, depth, weight, stats):
        """trace() of the rays whose weight is worth it, zeros elsewhere."""
        out = torch.zeros(rays.shape[0], 3, device=rays.device)
        keep = weight.amax(-1) > settings.min_weight
        if keep.any():
            out[keep] = self.trace(rays[keep], objects, settings, depth + 1, weight[keep], stats=stats)
        return out

    def _matte(self, obj, hits: Hits, objects, settings) -> torch.Tensor:
        light = self.irradiance(obj, hits.normal, settings)
        if any(is_mesh(o) for o in objects):
            meshes = [o for o in objects if is_mesh(o)]
            hidden = self.occlusion(hits.point, hits.geometric, meshes, settings.occlusion_distance * obj.radius,
                                    settings.occlusion_samples, 1e-3 * obj.radius)
            light = light * (1 - hidden)[:, None]
        lamps = lamps_of(objects) if settings.lamps else []
        if lamps:
            light = light + self.lamp_light(hits.point + hits.geometric * (1e-3 * obj.radius), hits.normal,
                                            objects, lamps)
        return light

    @torch.no_grad()
    def shade(self, obj, material: Material, rays, hits: Hits, settings, depth, weight, objects, stats):
        device = rays.device
        tint = torch.tensor(material.tint, dtype=torch.float32, device=device)
        d = rays[:, 3:]
        n = hits.normal
        kind = material.kind
        if kind == "diffuse":
            return tint * self._matte(obj, hits, objects, settings)
        if kind == "emissive":  # a lamp's surface: radiance = intensity / (pi R^2)
            return (tint * obj.power / (math.pi * obj.radius ** 2)).expand(rays.shape[0], 3)
        if kind == "pbr":
            return self._pbr(obj, material, tint, rays, hits, settings, depth, weight, objects, stats)
        reflected = self._secondary(hits.point, reflect(d, n), obj, settings)
        if kind == "mirror":
            return tint * self._branch(reflected, objects, settings, depth, weight * tint, stats)
        if kind == "metal":
            mirror = self._branch(reflected, objects, settings, depth, weight * tint * material.kd, stats)
            return tint * (material.kd * mirror + (1 - material.kd) * self._matte(obj, hits, objects, settings))
        cos_i = -(d * n).sum(-1, keepdim=True)
        kr = fresnel_schlick(cos_i, 1.0, material.ior)
        if kind == "ceramic":
            mirror = self._branch(reflected, objects, settings, depth, weight * kr, stats)
            return kr * mirror + (1 - kr) * material.kd * tint * self._matte(obj, hits, objects, settings)
        if kind != "glass":
            raise ValueError(f"unknown material {kind}")
        if is_mesh(obj):
            return self._glass_mesh(obj, material, rays, hits, kr, tint, settings, depth, weight, objects, stats)
        # Glass sphere: Fresnel split; the refracted ray crosses the solid sphere (no water inside) and leaves it at
        # the far side. A sphere never totally reflects on the way out, so the path is analytic.
        inside, tir_in = refract(d, n, 1.0 / material.ior)
        if material.ior < 1.0:  # air bubble: entering the thinner medium can totally reflect (bright rim)
            cos_t = -(inside * n).sum(-1, keepdim=True)
            kr = torch.where(tir_in[:, None], torch.ones_like(cos_t), fresnel_schlick(cos_t, 1.0, material.ior))
        center, _ = obj.tensors(device)
        oc = hits.point - center
        b = (oc * inside).sum(-1, keepdim=True)
        chord = (-2.0 * b).clamp_min(0)  # second intersection of a ray starting on the sphere
        exit_point = hits.point + chord * inside
        exit_normal = F.normalize(exit_point - center, dim=-1)
        outgoing, tir = refract(inside, -exit_normal, material.ior)
        outgoing = torch.where(tir[:, None], reflect(inside, -exit_normal), outgoing)
        cos_o = (inside * exit_normal).sum(-1, keepdim=True)
        kt = (1 - kr) * (1 - fresnel_schlick(cos_o, material.ior, 1.0))
        reflected_light = self._branch(reflected, objects, settings, depth, weight * kr, stats)
        refracted = self._secondary(exit_point, outgoing, obj, settings)
        refracted_light = self._branch(refracted, objects, settings, depth, weight * kt * tint, stats)
        return kr * reflected_light + kt * tint * refracted_light

    def _pbr(self, obj, material, tint, rays, hits, settings, depth, weight, objects, stats):
        """Textured metal/roughness surface (Poly Haven style) with the tracer's lighting: a mirror reflection for
        the glossy share of the Fresnel term (one ray, no blur), the scene irradiance for the rough share and the
        diffuse (non-metal) base. F0 = 0.04 for dielectrics, the albedo for metals."""
        maps = material.maps or {}
        uv = hits.uv if hits.uv is not None else torch.zeros(rays.shape[0], 2, device=rays.device)
        albedo = tint * (sample(texture(maps["albedo"], rays.device, True), uv) if "albedo" in maps else 1.0)
        metal = sample(texture(maps["metal"], rays.device, False), uv) if "metal" in maps else \
            torch.zeros(rays.shape[0], 1, device=rays.device)
        rough = sample(texture(maps["rough"], rays.device, False), uv) if "rough" in maps else \
            torch.full((rays.shape[0], 1), 0.5, device=rays.device)
        d, n = rays[:, 3:], hits.normal
        cos_i = (-(d * n).sum(-1, keepdim=True)).clamp(0, 1)
        f0 = 0.04 * (1 - metal) + albedo * metal
        fresnel = f0 + (1 - f0) * (1 - cos_i) ** 5 * (1 - rough) ** 2  # rough surfaces lose the grazing boost
        gloss = (1 - rough).clamp(0, 1) ** 2
        mirror_weight = fresnel * gloss
        light = self._matte(obj, hits, objects, settings)
        reflected = self._secondary(hits.point, reflect(d, n), obj, settings)
        mirror = self._branch(reflected, objects, settings, depth, weight * mirror_weight, stats)
        diffuse = (1 - metal) * (1 - fresnel) * albedo
        return mirror_weight * mirror + (diffuse + fresnel * (1 - gloss)) * light

    def _glass_mesh(self, mesh, material, rays, hits, kr, tint, settings, depth, weight, objects, stats):
        """Glass part of a mesh, entered from the water: refract in, find the exit on the same mesh (no water
        inside), refract out (total internal reflection: leave along the reflection). An open surface with no exit
        is a thin sheet: the ray continues in its refracted direction."""
        d, n = rays[:, 3:], hits.normal
        inside, tir_in = refract(d, n, 1.0 / material.ior)
        kr = torch.where(tir_in[:, None], torch.ones_like(kr), kr)
        start = hits.point - n * (settings.epsilon * mesh.radius)
        inner = torch.cat((start, inside), -1).contiguous()
        t, face, bary = mesh.intersect(inner)
        exits = t < INF
        out_point, out_dir = start.clone(), inside.clone()
        kt = 1 - kr
        if exits.any():
            point = start[exits] + t[exits, None] * inside[exits]
            _, geo = mesh.surface(face[exits], bary[exits])
            geo = torch.where((geo * inside[exits]).sum(-1, keepdim=True) > 0, -geo, geo)  # facing the inner ray
            outgoing, tir = refract(inside[exits], geo, material.ior)
            outgoing = torch.where(tir[:, None], reflect(inside[exits], geo), outgoing)
            cos_o = -(inside[exits] * geo).sum(-1, keepdim=True)
            kt = kt.clone()
            kt[exits] = kt[exits] * (1 - fresnel_schlick(cos_o, material.ior, 1.0))
            out_point[exits] = point
            out_dir[exits] = outgoing
        reflected = self._secondary(hits.point, reflect(d, n), mesh, settings)
        refracted = self._secondary(out_point, out_dir, mesh, settings)
        reflected_light = self._branch(reflected, objects, settings, depth, weight * kr, stats)
        refracted_light = self._branch(refracted, objects, settings, depth, weight * kt * tint, stats)
        return kr * reflected_light + kt * tint * refracted_light


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


@torch.no_grad()
def render_camera(tracer: SceneTracer, camera, objects: list, settings: TraceSettings = TraceSettings(),
                  stats: TraceStats | None = None, isp_origin: torch.Tensor | None = None):
    """Raw and camera-model images [H,W,3] of one camera (render.py's `raw` / `full`). isp_origin: take the camera
    model's exposure and colour as at this point (default: the camera's own origin; a fixed point for videos, so
    the exposure does not jump when the nearest training photos change)."""
    from evaluation.evaluator import _rays
    from training.camera_isp import pixel_grid
    tracer.set_color_origin(camera.origin)
    grid = _rays(camera)
    rays = grid.reshape(-1, 6).contiguous()
    h, w = camera.height, camera.width
    tracer.image_shape = (h, w)
    tracer.shadow_map = None
    try:
        raw = tracer.trace(rays, objects, settings, stats=stats)
    finally:
        tracer.image_shape = None
    if objects and settings.antialias > 0:
        raw = _antialias(tracer, grid, raw, objects, settings, stats)
    isp = tracer.trainer.camera_isp
    full = raw if isp is None else isp.apply(raw, pixel_grid(h, w, raw.device).reshape(-1, 2), (h, w), None,
                                             camera.origin if isp_origin is None else isp_origin)
    return raw.reshape(h, w, 3), full.clamp(0, 1).reshape(h, w, 3)


@torch.no_grad()
def _antialias(tracer, grid: torch.Tensor, raw: torch.Tensor, objects, settings, stats):
    h, w = grid.shape[:2]
    hits = intersect(grid.reshape(-1, 6).contiguous(), objects)
    obj = hits.obj.reshape(h, w)
    table, _ = material_table(objects)
    shiny = [i for i, (o, m) in enumerate(table) if m.kind in ("mirror", "metal", "glass") and not is_mesh(objects[o])]
    shiny = torch.isin(hits.material, torch.tensor(shiny, device=raw.device, dtype=hits.material.dtype))
    padded = F.pad(obj[None, None].float(), (1, 1, 1, 1), mode="replicate")[0, 0]
    edge = torch.zeros(h, w, dtype=torch.bool, device=raw.device)
    for dy, dx in ((0, 1), (2, 1), (1, 0), (1, 2)):
        edge |= padded[dy:dy + h, dx:dx + w] != obj
    edge = (F.max_pool2d(edge[None, None].float(), 3, 1, 1)[0, 0] > 0) | shiny.reshape(h, w)
    ys, xs = edge.nonzero(as_tuple=True)
    if ys.numel() == 0:
        return raw
    d = grid[..., 3:]
    ddx = (d[:, torch.clamp(torch.arange(w, device=d.device) + 1, max=w - 1)]
           - d[:, torch.clamp(torch.arange(w, device=d.device) - 1, min=0)]) / 2
    ddy = (d[torch.clamp(torch.arange(h, device=d.device) + 1, max=h - 1)]
           - d[torch.clamp(torch.arange(h, device=d.device) - 1, min=0)]) / 2
    if settings.antialias <= 4:
        offsets = [(-0.375, -0.125), (0.125, -0.375), (0.375, 0.125), (-0.125, 0.375)][:settings.antialias]
    else:  # Fibonacci lattice over the pixel
        n = settings.antialias
        offsets = [(((i * 0.618034) % 1) - 0.5, ((i + 0.5) / n) - 0.5) for i in range(n)]
    index = ys * w + xs
    shadow = None if tracer.shadow_map is None else tracer.shadow_map[index]
    total = raw[index].clone()
    for ox, oy in offsets:
        directions = F.normalize(d[ys, xs] + ox * ddx[ys, xs] + oy * ddy[ys, xs], dim=-1)
        sub = torch.cat((grid[ys, xs, :3], directions), -1).contiguous()
        total += tracer.trace(sub, objects, settings, stats=stats, shadow=shadow)
    raw = raw.clone()
    raw[index] = total / (len(offsets) + 1)
    return raw
