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
    kind: str = "mirror"  # mirror | glass | diffuse | metal | ceramic
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0)  # reflectance / transmittance / albedo
    ior: float = 1.5  # glass and ceramic Fresnel, relative to water
    kd: float = 1.0  # metal: mirror fraction (rest diffuse); ceramic: diffuse weight

    @staticmethod
    def from_falcon(spec: dict) -> "Material":
        """FALCON material {"mode", "rgb", "kd", "eta"}."""
        return Material(kind=spec.get("mode", "ceramic"), tint=tuple(spec.get("rgb", (1, 1, 1))),
                        ior=float(spec.get("eta", 1.5)), kd=float(spec.get("kd", 1.0)))


@dataclass
class Sphere:
    center: tuple[float, float, float]
    radius: float
    material: str = "mirror"  # mirror | glass | diffuse | metal | ceramic
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ior: float = 1.5  # glass, relative to water
    kd: float = 1.0

    @property
    def materials(self) -> list[Material]:
        return [Material(self.material, tuple(self.tint), self.ior, self.kd)]

    def tensors(self, device):
        return (torch.tensor(self.center, dtype=torch.float32, device=device),
                torch.tensor(self.tint, dtype=torch.float32, device=device))


@dataclass
class TraceSettings:
    max_depth: int = 4
    min_weight: float = 1e-3  # secondary rays carrying less (max channel) are dropped
    epsilon: float = 1e-4  # secondary ray offset, in units of the object's radius
    ambient: tuple[float, float, float] = (0.0, 0.0, 0.0)  # extra constant light on matte surfaces (raw space)
    # The trained water light extrapolation (HG phase, g up to ~0.93) explodes for rays looking up, which the
    # photos never saw but mirrors do (Curacao: x4500 straight up). Directions are tilted down to this cosine
    # with the cameras' "up" before the water light is evaluated; None keeps the trained behaviour.
    water_light_max_up: float | None = 0.6
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


@dataclass
class Hits:
    t: torch.Tensor  # [R] inf: none
    obj: torch.Tensor  # [R] object index, -1
    material: torch.Tensor  # [R] index into the flat material table, -1
    point: torch.Tensor  # [R,3]
    normal: torch.Tensor  # [R,3] shading normal, facing the ray
    geometric: torch.Tensor  # [R,3] geometric normal, facing the ray


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
    for i, obj in enumerate(objects):
        if is_mesh(obj):
            t, face, bary = obj.intersect(rays, best)
            faces[i] = (face, bary)
        else:
            t = sphere_hit(rays, obj)
        closer = t < best
        best = torch.where(closer, t, best)
        index = torch.where(closer, torch.full_like(index, i), index)
    best = torch.where(index >= 0, best, torch.full_like(best, INF))
    point = rays[:, :3] + best[:, None].nan_to_num(posinf=0.0) * rays[:, 3:]
    normal = torch.zeros_like(point)
    geometric = torch.zeros_like(point)
    material = torch.full_like(index, -1)
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
    return Hits(best, index, material, point, normal, geometric)


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
    def water_light(self, direct, scattered, rays, max_up: float | None = None):
        light = self.trainer.evaluator.water_light
        if light is None:
            return direct + scattered
        if max_up is not None:
            d = rays[:, 3:]
            up = self.up.to(d)
            cos = (d * up).sum(-1, keepdim=True)
            horizontal = F.normalize(d - cos * up + 1e-8 * self.side.to(d), dim=-1)
            tilted = horizontal * (1 - max_up ** 2) ** 0.5 + up * max_up
            rays = torch.cat((rays[:, :3], torch.where(cos > max_up, tilted, d)), -1)
        return light(direct + scattered, scattered, rays)

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
        key = (tuple(round(float(x), 5) for x in center), samples, settings.water_light_max_up)
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
              depth: int = 0, weight: torch.Tensor | None = None, start_points=None, stats: TraceStats | None = None):
        """Raw radiance [R,3] (water light applied, before the camera model)."""
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
            return self.water_light(direct, scattered, rays, settings.water_light_max_up)
        hits = intersect(rays, objects)
        direct, scattered, transmission = self.segment(rays, hits.t, start_points)
        if settings.shadows:
            factor = self.shadow_factor(rays, hits.t, objects, settings)
            if depth == 0 and self.image_shape is not None and settings.shadow_blur_px > 0:
                factor = self._blur_shadow(factor, rays, hits.t, settings.shadow_blur_px)
            direct = direct * factor[:, None]
        out = self.water_light(direct, scattered, rays, settings.water_light_max_up)
        if not bool((hits.obj >= 0).any()) or depth >= settings.max_depth:  # past the last bounce: black
            return out
        # Light leaving the objects towards the ray origin, attenuated by the water in front of them.
        carried = transmission * weight
        table, _ = material_table(objects)
        alive = (hits.obj >= 0) & (carried.amax(-1) > settings.min_weight)
        for m in hits.material[alive].unique().tolist():
            mask = alive & (hits.material == m)
            obj_index, material = table[m]
            sub = Hits(*(x[mask] for x in (hits.t, hits.obj, hits.material, hits.point, hits.normal, hits.geometric)))
            out[mask] += transmission[mask] * self.shade(objects[obj_index], material, rays[mask], sub, settings,
                                                         depth, carried[mask], objects, stats)
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
        smooth = blur(factor * valid) / blur(valid).clamp_min(1e-6)
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
        inside, _ = refract(d, n, 1.0 / material.ior)
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
def place_on_seabed(tracer, camera, u: float, v: float, radius_fraction: float, lift: float = 1.0):
    """Sphere resting on the surface seen at image fraction (u, v): radius = fraction of the hit distance."""
    from evaluation.evaluator import _rays
    rays = _rays(camera)
    y, x = int(v * camera.height), int(u * camera.width)
    ray = rays[y, x][None].contiguous()
    only = tracer.renderer.render_gaussians(ray, tracer.gaussians, background=(0.0, 0.0, 0.0))
    depth = float(only.surface_depth.reshape(-1)[0])
    point = ray[0, :3] + depth * ray[0, 3:]
    up = tracer.up.to(point)
    radius = radius_fraction * depth
    center = point + up * radius * lift  # lift < 1 sinks the sphere into the surface
    return center.tolist(), radius, depth


@torch.no_grad()
def render_camera(tracer: SceneTracer, camera, objects: list, settings: TraceSettings = TraceSettings(),
                  stats: TraceStats | None = None, isp_origin: torch.Tensor | None = None):
    """Raw and camera-model images [H,W,3] of one camera (render.py's `raw` / `full`). isp_origin: take the camera
    model's exposure and colour as at this point (default: the camera's own origin; a fixed point for videos, so
    the exposure does not jump when the nearest training photos change)."""
    from evaluation.evaluator import _rays
    from training.camera_isp import pixel_grid
    tracer.set_color_origin(camera.origin)
    rays = _rays(camera).reshape(-1, 6).contiguous()
    tracer.image_shape = (camera.height, camera.width)
    try:
        raw = tracer.trace(rays, objects, settings, stats=stats)
    finally:
        tracer.image_shape = None
    isp = tracer.trainer.camera_isp
    h, w = camera.height, camera.width
    full = raw if isp is None else isp.apply(raw, pixel_grid(h, w, raw.device).reshape(-1, 2), (h, w), None,
                                             camera.origin if isp_origin is None else isp_origin)
    return raw.reshape(h, w, 3), full.clamp(0, 1).reshape(h, w, 3)
