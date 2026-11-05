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

Objects are analytic spheres for now (`Sphere`); meshes come later.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

INF = float("inf")


@dataclass
class Sphere:
    center: tuple[float, float, float]
    radius: float
    material: str = "mirror"  # mirror | glass | diffuse
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0)  # mirror/glass reflectance or transmittance, diffuse albedo
    ior: float = 1.5  # glass, relative to water

    def tensors(self, device):
        return (torch.tensor(self.center, dtype=torch.float32, device=device),
                torch.tensor(self.tint, dtype=torch.float32, device=device))


@dataclass
class TraceSettings:
    max_depth: int = 4
    min_weight: float = 1e-3  # secondary rays carrying less (max channel) are dropped
    epsilon: float = 1e-4  # secondary ray offset, in units of the object's radius
    ambient: tuple[float, float, float] = (0.0, 0.0, 0.0)  # diffuse objects: constant light (raw space)
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


@dataclass
class TraceStats:
    rays_per_depth: list = field(default_factory=list)


def intersect_spheres(rays: torch.Tensor, spheres: list[Sphere]):
    """Nearest positive hit: depth [R] (inf: none), object index [R] (-1), normal [R,3] (outward)."""
    origin, direction = rays[:, :3], rays[:, 3:]
    best = torch.full((rays.shape[0],), INF, device=rays.device)
    index = torch.full((rays.shape[0],), -1, dtype=torch.long, device=rays.device)
    for i, sphere in enumerate(spheres):
        center, _ = sphere.tensors(rays.device)
        oc = origin - center
        b = (oc * direction).sum(-1)
        c = (oc * oc).sum(-1) - sphere.radius ** 2
        disc = b * b - c
        root = disc.clamp_min(0).sqrt()
        near, far = -b - root, -b + root
        t = torch.where(near > 0, near, far)
        t = torch.where((disc > 0) & (t > 0), t, torch.full_like(t, INF))
        closer = t < best
        best = torch.where(closer, t, best)
        index = torch.where(closer, torch.full_like(index, i), index)
    points = rays[:, :3] + best[:, None].nan_to_num(posinf=0.0) * rays[:, 3:]
    normal = torch.zeros_like(points)
    for i, sphere in enumerate(spheres):
        mask = index == i
        if mask.any():
            center, _ = sphere.tensors(rays.device)
            normal[mask] = F.normalize(points[mask] - center, dim=-1)
    return best, index, points, normal


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

    # --- recursive tracing ------------------------------------------------------------------------
    @torch.no_grad()
    def trace(self, rays: torch.Tensor, objects: list[Sphere], settings: TraceSettings = TraceSettings(),
              depth: int = 0, weight: torch.Tensor | None = None, start_points=None, stats: TraceStats | None = None):
        """Raw radiance [R,3] (water light applied, before the camera model)."""
        count = rays.shape[0]
        if stats is not None:
            while len(stats.rays_per_depth) <= depth:
                stats.rays_per_depth.append(0)
            stats.rays_per_depth[depth] += count
        if weight is None:
            weight = torch.ones(count, 3, device=rays.device)
        if not objects:
            direct, scattered, _ = self.segment(rays, torch.full((count,), INF, device=rays.device), start_points)
            return self.water_light(direct, scattered, rays, settings.water_light_max_up)
        t_hit, index, points, normal = intersect_spheres(rays, objects)
        direct, scattered, transmission = self.segment(rays, t_hit, start_points)
        if settings.shadows:
            direct = direct * self.shadow_factor(rays, t_hit, objects, settings)[:, None]
        out = self.water_light(direct, scattered, rays, settings.water_light_max_up)
        hit = index >= 0
        if not hit.any() or depth >= settings.max_depth:  # past the last bounce an object is black
            return out
        # Light leaving the objects towards the ray origin, attenuated by the water in front of them.
        carried = transmission * weight
        for i, sphere in enumerate(objects):
            mask = hit & (index == i) & (carried.amax(-1) > settings.min_weight)
            if not mask.any():
                continue
            out[mask] += transmission[mask] * self.shade(sphere, rays[mask], points[mask], normal[mask],
                                                         settings, depth, carried[mask], objects, stats)
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
        for sphere in objects:
            center, _ = sphere.tensors(d.device)
            v = center - p
            distance = v.norm(dim=-1).clamp_min(1e-6)
            v = v / distance[:, None]
            ratio = (sphere.radius / distance).clamp(max=1.0)
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

    def _secondary(self, origins, directions, sphere, settings):
        origins = origins + directions * (settings.epsilon * sphere.radius)
        return torch.cat((origins, directions), -1).contiguous()

    @torch.no_grad()
    def shade(self, sphere: Sphere, rays, points, normal, settings, depth, weight, objects, stats):
        device = rays.device
        _, tint = sphere.tensors(device)
        d = rays[:, 3:]
        if sphere.material == "diffuse":
            return tint * torch.tensor(settings.ambient, device=device) * (-(d * normal).sum(-1, keepdim=True)).clamp_min(0.2)
        reflected = self._secondary(points, reflect(d, normal), sphere, settings)
        if sphere.material == "mirror":
            return tint * self.trace(reflected, objects, settings, depth + 1, weight * tint, stats=stats)
        if sphere.material != "glass":
            raise ValueError(f"unknown material {sphere.material}")
        # Glass: Fresnel split; the refracted ray crosses the solid sphere (no water inside) and leaves it at the
        # far side. A sphere never totally reflects on the way out, so the path is analytic.
        cos_i = -(d * normal).sum(-1, keepdim=True)
        kr = fresnel_schlick(cos_i, 1.0, sphere.ior)
        inside, _ = refract(d, normal, 1.0 / sphere.ior)
        center, _ = sphere.tensors(device)
        oc = points - center
        b = (oc * inside).sum(-1, keepdim=True)
        chord = (-2.0 * b).clamp_min(0)  # second intersection of a ray starting on the sphere
        exit_point = points + chord * inside
        exit_normal = F.normalize(exit_point - center, dim=-1)
        outgoing, tir = refract(inside, -exit_normal, sphere.ior)
        outgoing = torch.where(tir[:, None], reflect(inside, -exit_normal), outgoing)
        cos_o = (inside * exit_normal).sum(-1, keepdim=True)
        kt = (1 - kr) * (1 - fresnel_schlick(cos_o, sphere.ior, 1.0))
        reflected_light = self.trace(reflected, objects, settings, depth + 1, weight * kr, stats=stats)
        refracted = self._secondary(exit_point, outgoing, sphere, settings)
        refracted_light = self.trace(refracted, objects, settings, depth + 1, weight * kt * tint, stats=stats)
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
    up = -torch.stack([c.down for c in tracer.trainer.dataset.train]).mean(0)
    up = up / up.norm()
    radius = radius_fraction * depth
    center = point + up * radius * lift  # lift < 1 sinks the sphere into the surface
    return center.tolist(), radius, depth


@torch.no_grad()
def render_camera(tracer: SceneTracer, camera, objects: list[Sphere], settings: TraceSettings = TraceSettings(),
                  stats: TraceStats | None = None, isp_origin: torch.Tensor | None = None):
    """Raw and camera-model images [H,W,3] of one camera (render.py's `raw` / `full`). isp_origin: take the camera
    model's exposure and colour as at this point (default: the camera's own origin; a fixed point for videos, so
    the exposure does not jump when the nearest training photos change)."""
    from evaluation.evaluator import _rays
    from training.camera_isp import pixel_grid
    tracer.set_color_origin(camera.origin)
    rays = _rays(camera).reshape(-1, 6).contiguous()
    raw = tracer.trace(rays, objects, settings, stats=stats)
    isp = tracer.trainer.camera_isp
    h, w = camera.height, camera.width
    full = raw if isp is None else isp.apply(raw, pixel_grid(h, w, raw.device).reshape(-1, 2), (h, w), None,
                                             camera.origin if isp_origin is None else isp_origin)
    return raw.reshape(h, w, 3), full.clamp(0, 1).reshape(h, w, 3)
