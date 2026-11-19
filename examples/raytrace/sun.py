"""Sunlight through a wavy water surface for examples/raytrace/tracer.py: caustics on the scene and light shafts in the water.

The trained scene holds no sunlight (the photos were lit evenly), so the sun is an *inserted* light, like the lamps:

- Waves. The surface is a periodic height field over a square tile, a sum of sinusoids on the tile's wave-vector
  lattice (deep-water dispersion omega = sqrt(g k), so a clip animates it). A grid of parallel sun rays is refracted
  at the surface (air -> water, Snell) and followed down; their landing points are histogrammed on horizontal layers
  at increasing depth. A layer's histogram over the mean is the irradiance relative to a flat surface: the caustic
  pattern, focused at some depth and smeared deeper. Each layer is blurred by the sun's disc (its angular radius in
  the water times the depth) and by forward scattering (blur_per_m times the depth).
- Visibility. A shadow map seen along the mean refracted direction (Gaussians and inserted objects; lamps and gas
  bubbles let the light through) tells whether a point is lit.
- Light at a point x: E = E0 tint caustic(x) visible(x) exp(-c t_x), t_x the path from the surface to x in the water.
  E0 is relative to the scene's light from above (SceneTracer.ambient_up): power 1 adds, on a surface facing the
  sun just below the water, as much light again.
- Shafts: single scattering of E towards the camera, b phase(g) E exp(-c t), integrated along camera rays with
  samples drawn from exp(-c t) (stratified); the shadow map gives the shadows of the columns in the water. Shafts use
  a coarser, blurrier copy of the caustic field (sampling the sharp one along a ray only adds noise).
- Glass spheres focus the light: shadow-map rays that hit one are refracted through it (Fresnel-weighted, per colour
  channel with dispersion) and land on the scene; their energy is added where it lands, inside the shadow.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

WATER_IOR = 1.333
GRAVITY = 9.81


@dataclass
class Sun:
    elevation_deg: float = 60.0  # of the sun in the sky
    azimuth_deg: float = 0.0  # world azimuth of the sun (towards it), from frame[0] towards frame[1]
    surface: float = 20.0  # height of the water surface (along up)
    power: float = 1.0
    tint: tuple[float, float, float] = (1.0, 0.97, 0.9)
    up: tuple[float, float, float] | None = None  # None: the tracer's up
    # Waves: wavelengths (m) log-uniform in the range, steepness from the rms slope, directions around wind_deg.
    wavelength: tuple[float, float] = (0.8, 4.0)
    slope: float = 0.08
    waves: int = 96
    wind_deg: float = 30.0
    spread_deg: float = 70.0
    seed: int = 0
    time: float = 0.0
    tile: float = 12.0  # m, the caustic pattern repeats with this period
    tile_rotation_deg: float = 23.0  # of the tile's lattice against frame(): no wave crest along the scene's axes
    resolution: int = 1280  # per tile side
    photons: int = 2560  # per tile side
    layers: int = 48
    sun_radius_deg: float = 0.27
    blur_per_m: float = 0.002  # forward-scattering blur (m of sigma per m of depth)
    shaft_blur: float = 0.06  # m, extra blur of the field the shafts use
    g: float = 0.75  # Henyey-Greenstein phase of the shafts
    samples: int = 96  # per camera ray
    shaft_strength: float = 1.0
    shadow_resolution: int = 4096
    shadow_bias: float = 0.08  # m
    # Shadows in the water (the shafts' visibility). Single scattering alone gives them knife edges and no light
    # inside; in the sea, light scattered forward a few degrees widens a shadow's edge with the distance behind its
    # blocker (shadow_soft rad), and light scattered more than once fills it from the lit water around (a fraction
    # shadow_fill spread over fill_radius m). Shadows on surfaces stay sharp.
    shadow_soft: float = 0.04
    shadow_fill: float = 0.3
    fill_radius: float = 1.5
    volume_resolution: int = 1024
    volume_layers: int = 40
    glass_caustics: bool = True
    dispersion: float = 0.012  # glass: relative index offset of red / blue from green
    glass_photons: int = 1024  # per side of the shadow-map window around each glass sphere
    # The surface seen from below: rays that reach it are split by Fresnel (total reflection beyond the critical
    # angle): the reflection is traced on through the water, the refracted part sees the sky and the sun.
    show_surface: bool = True
    sky: float = 1.0  # radiance of the sky in Snell's window: looking straight up from the colour origin, the
    #                   window seen through the water is sky x the scene's light from above (ambient_up)
    sun_glow_deg: float = 1.0  # the sun seen through the surface: a Gaussian lobe of this width (in the air)
    pixel_angle: float = 1 / 1800  # rad, the camera's pixel: waves finer than its footprint are smoothed out
    _cache: dict = field(default_factory=dict, repr=False)

    # --- geometry ----------------------------------------------------------------------------------------
    def _memo(self, name, device, make):
        """Small geometry tensors kept per device and sun position (they are asked for millions of times)."""
        key = ("geometry", name, str(torch.device(device)), tuple(self.up or ()), self.elevation_deg, self.azimuth_deg)
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    def frame(self, device):
        """Unit up and two horizontal axes e1, e2 (world)."""
        return self._memo("frame", device, lambda: self._frame(device))

    def _frame(self, device):
        up = F.normalize(torch.tensor(self.up, dtype=torch.float32, device=device), dim=0)
        side = torch.tensor([1.0, 0.0, 0.0], device=device)
        if abs(float(side @ up)) > 0.9:
            side = torch.tensor([0.0, 1.0, 0.0], device=device)
        e1 = F.normalize(side - (side @ up) * up, dim=0)
        e2 = torch.linalg.cross(up, e1)
        return up, e1, e2

    def tile_axes(self, device):
        """The caustic tile's horizontal axes (frame() rotated by tile_rotation_deg)."""
        _, e1, e2 = self.frame(device)
        a = math.radians(self.tile_rotation_deg)
        return math.cos(a) * e1 + math.sin(a) * e2, -math.sin(a) * e1 + math.cos(a) * e2

    def air_direction(self, device):
        """Travel direction of the sunlight in the air (world, pointing down)."""
        return self._memo("air", device, lambda: self._air_direction(device))

    def _air_direction(self, device):
        up, e1, e2 = self.frame(device)
        el, az = math.radians(self.elevation_deg), math.radians(self.azimuth_deg)
        towards = math.cos(el) * (math.cos(az) * e1 + math.sin(az) * e2) + math.sin(el) * up
        return -towards

    def water_direction(self, device):
        """Mean travel direction in the water (refracted at a flat surface)."""
        up, _, _ = self.frame(device)
        return self._memo("water", device, lambda: refract(self.air_direction(device)[None], up[None], 1.0 / WATER_IOR)[0])

    def depth_below(self, x):
        up, _, _ = self.frame(x.device)
        return self.surface - x @ up

    def path_in_water(self, x):
        """[P] path length from the surface to x along the mean direction (<= 0: above the water)."""
        up, _, _ = self.frame(x.device)
        return self.depth_below(x) / (-(self.water_direction(x.device) @ up))

    def entry(self, x):
        """[P,2] horizontal (e1, e2) coordinates where the sun line through x crosses the surface."""
        _, e1, e2 = self.frame(x.device)
        s = x - self.path_in_water(x)[:, None] * self.water_direction(x.device)
        return torch.stack((s @ e1, s @ e2), -1)

    # --- caustic field -----------------------------------------------------------------------------------
    def _waves(self, device):
        g = torch.Generator().manual_seed(self.seed)
        lo, hi = self.wavelength
        lam = torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * torch.rand(self.waves, generator=g))
        angle = math.radians(self.wind_deg) + math.radians(self.spread_deg) * (torch.rand(self.waves, generator=g) - 0.5)
        # Snap to the tile's lattice: the height field is periodic.
        m = torch.round(self.tile / lam * angle.cos())
        n = torch.round(self.tile / lam * angle.sin())
        zero = (m == 0) & (n == 0)
        m[zero] = 1
        k = 2 * math.pi / self.tile * torch.stack((m, n), -1)
        knorm = k.norm(dim=-1)
        amplitude = 1.0 / knorm  # equal steepness a k
        rms = (0.5 * (amplitude * knorm) ** 2).sum().sqrt()
        amplitude = amplitude * self.slope / rms
        phase = 2 * math.pi * torch.rand(self.waves, generator=g)
        omega = (GRAVITY * knorm).sqrt()
        return k.to(device), amplitude.to(device), phase.to(device), omega.to(device)

    def max_depth(self, tracer) -> float:
        key = "max_depth"
        if key not in self._cache:
            means = tracer.trainer.model.gaussians.render_scene(tracer.color_origin.cuda()).means.detach()
            low = torch.quantile((means @ self.frame(means.device)[0])[::max(1, means.shape[0] // 500_000)], 0.001)
            self._cache[key] = float(self.surface - low) + 2.0
        return self._cache[key]

    @torch.no_grad()
    def caustics(self, tracer):
        """(sharp [L,M,M], coarse [L,M/4,M/4]) relative irradiance on horizontal layers 0 .. max_depth."""
        if "field" in self._cache:
            return self._cache["field"]
        device = torch.device("cuda")
        k, a, phase, omega = self._waves(device)
        up, _, _ = self.frame(device)
        e1, e2 = self.tile_axes(device)
        d_air = self.air_direction(device)
        air = torch.stack((d_air @ e1, d_air @ e2, d_air @ up))  # local (e1, e2, up)
        n_side, m_side, layers = self.photons, self.resolution, self.layers
        z_max = self.max_depth(tracer)
        depths = torch.linspace(0, z_max, layers, device=device)
        sharp = torch.zeros(layers, m_side * m_side, device=device)
        g = torch.Generator(device=device).manual_seed(self.seed + 1)
        rows = max(1, (1 << 22) // n_side)
        for begin in range(0, n_side, rows):
            i = torch.arange(begin, min(begin + rows, n_side), device=device, dtype=torch.float32)
            j = torch.arange(n_side, device=device, dtype=torch.float32)
            p = torch.stack(torch.meshgrid(i, j, indexing="ij"), -1).reshape(-1, 2)
            p = (p + torch.rand(p.shape, generator=g, device=device)) * (self.tile / n_side)
            arg = p @ k.T + phase - omega * self.time  # [P,W]
            grad = (a * arg.cos()) @ k  # dh/dx [P,2]
            normal = F.normalize(torch.cat((-grad, torch.ones_like(grad[:, :1])), -1), dim=-1)
            d = refract(air.expand(p.shape[0], 3), normal, 1.0 / WATER_IOR)
            cos_in = (-(air * normal).sum(-1)).clamp_min(0)
            weight = cos_in / float(-air[2])  # flux through the tilted facet per horizontal area
            weight = weight * (1 - fresnel(cos_in, 1.0, WATER_IOR))
            for layer, z in enumerate(depths.tolist()):
                q = p + d[:, :2] * (z / (-d[:, 2]))[:, None]
                splat(sharp[layer], q / self.tile * m_side - 0.5, weight, m_side)
        flat = (1 - fresnel(torch.tensor(float(-air[2])), 1.0, WATER_IOR)).item()
        sharp = sharp / (n_side * n_side / (m_side * m_side) * flat)
        sharp = sharp.reshape(layers, m_side, m_side)
        texel = self.tile / m_side
        sun_r = math.radians(self.sun_radius_deg) / WATER_IOR
        sigma = (depths * (sun_r / 1.5 + self.blur_per_m) / texel).clamp_min(0.7)  # texels
        sharp = gaussian_blur_periodic(sharp, sigma)
        coarse = F.avg_pool2d(sharp[:, None], 4)[:, 0]
        coarse = gaussian_blur_periodic(coarse, torch.full_like(sigma, self.shaft_blur / (4 * texel)))
        self._cache["field"] = (sharp.contiguous(), coarse.contiguous(), z_max)
        return self._cache["field"]

    def caustic_at(self, tracer, x, coarse: bool = False):
        sharp, small, z_max = self.caustics(tracer)
        volume = small if coarse else sharp
        e1, e2 = self.tile_axes(x.device)
        z = self.depth_below(x).clamp(0, z_max) / z_max * (volume.shape[0] - 1)
        uv = torch.stack((x @ e1, x @ e2), -1) / self.tile * volume.shape[1]
        return trilinear_periodic(volume, z, uv)

    # --- shadow map --------------------------------------------------------------------------------------
    @torch.no_grad()
    def shadow_map(self, tracer, objects):
        from .shading import intersect, is_mesh
        blockers = [o for o in objects if (is_mesh(o) and getattr(o, "shadow", True))
                    or (not is_mesh(o) and o.material not in ("emissive",) and o.shadow)]
        key = ("shadow", tuple(id(o) for o in blockers))
        if self._cache.get("shadow_key") == key:
            return self._cache["shadow"]
        device = torch.device("cuda")
        up, e1, e2 = self.frame(device)
        d = self.water_direction(device)
        if "window" not in self._cache:
            means = tracer.trainer.model.gaussians.render_scene(tracer.color_origin.cuda()).means.detach()
            means = means[::max(1, means.shape[0] // 500_000)]
            s = self.entry(means)
            lo = torch.quantile(s, 0.002, dim=0) - 5.0
            hi = torch.quantile(s, 0.998, dim=0) + 5.0
            self._cache["window"] = (lo, hi)
        lo, hi = self._cache["window"]
        res = self.shadow_resolution
        cell = float((hi - lo).max()) / res
        a = lo[0] + (torch.arange(res, device=device) + 0.5) * cell
        b = lo[1] + (torch.arange(res, device=device) + 0.5) * cell
        grid = torch.stack(torch.meshgrid(a, b, indexing="ij"), -1).reshape(-1, 2)
        origins = grid[:, :1] * e1 + grid[:, 1:] * e2 + self.surface * up
        rays = torch.cat((origins, d.expand_as(origins)), -1).contiguous()
        depth = tracer.gaussian_hit(rays)
        glass_rays = None
        if blockers:
            hits = intersect(rays, blockers, max_t=depth)
            depth = torch.minimum(depth, hits.t)
            glass = torch.zeros_like(depth, dtype=torch.bool)
            for i, o in enumerate(blockers):
                if not is_mesh(o) and o.material == "glass":
                    glass |= hits.obj == i
            if self.glass_caustics and glass.any():
                glass_rays = glass
        depth_map = depth.reshape(res, res)
        extra = None
        if glass_rays is not None:
            extra = self._glass_photons(tracer, objects, blockers, (lo, cell, res), depth_map)
        self._cache["shadow"] = (depth_map, lo, cell, extra)
        self._cache["shadow_key"] = key
        return self._cache["shadow"]

    def _glass_photons(self, tracer, objects, blockers, window, depth_map):
        """[3,res,res] irradiance (relative, like the caustics) brought by light refracted through glass spheres, on
        the shadow-map grid where it lands."""
        from .shading import intersect, is_mesh, refract as refract_ray
        lo, cell, res = window
        device = depth_map.device
        up, e1, e2 = self.frame(device)
        d = self.water_direction(device)
        out = torch.zeros(3, res * res, device=device)
        per_photon_area = None
        for sphere in [o for o in blockers if not is_mesh(o) and o.material == "glass"]:
            center, _ = sphere.tensors(device)
            c_entry = self.entry(center[None])[0]
            n = self.glass_photons
            span = 2.2 * sphere.radius
            u = (torch.arange(n, device=device) + 0.5) / n - 0.5
            grid = torch.stack(torch.meshgrid(u, u, indexing="ij"), -1).reshape(-1, 2) * span + c_entry
            # Spread the photons perpendicular to the beam: entry points on the surface plane map to the beam's
            # cross-section scaled by cos (the area weight below handles it).
            origins = grid[:, :1] * e1 + grid[:, 1:] * e2 + self.surface * up
            rays = torch.cat((origins, d.expand_as(origins)), -1).contiguous()
            hits = intersect(rays, blockers)
            mine = hits.obj == blockers.index(sphere)
            if not mine.any():
                continue
            area = (span / n) ** 2  # surface-plane area per photon
            p, nrm, t0 = hits.point[mine], hits.normal[mine], hits.t[mine]
            # Light arriving at the sphere: caustic pattern of the waves, water from the surface to it.
            arrive = self.caustic_at(tracer, p)[:, None] * self._attenuation(tracer, t0)  # [P,3]
            for channel, offset in enumerate((-self.dispersion, 0.0, self.dispersion)):
                eta = sphere.ior * (1 + offset)
                di = d.expand_as(p)
                cos_i = (-(di * nrm).sum(-1)).clamp(0, 1)
                t_in = 1 - fresnel(cos_i, 1.0, eta)
                inside = refract(di, nrm, 1.0 / eta)
                oc = p - center
                bq = (oc * inside).sum(-1)
                far = -bq + (bq * bq - ((oc * oc).sum(-1) - sphere.radius ** 2)).clamp_min(0).sqrt()
                q = p + far[:, None] * inside
                n_out = F.normalize(q - center, dim=-1)
                cos_o = (inside * n_out).sum(-1).clamp(0, 1)
                out_dir = refract(inside, -n_out, eta)
                ok = out_dir.norm(dim=-1) > 0.5  # total internal reflection: dropped
                t_out = 1 - fresnel(cos_o, eta, 1.0)
                start = q + out_dir * 1e-3 * sphere.radius
                rays2 = torch.cat((start, out_dir), -1).contiguous()
                land_t = tracer.gaussian_hit(rays2)
                others = [o for o in blockers if o is not sphere]
                if others:
                    land_t = torch.minimum(land_t, intersect(rays2, others, max_t=land_t).t)
                land = start + land_t.nan_to_num(posinf=0)[:, None] * out_dir
                keep = ok & torch.isfinite(land_t)
                travelled = land_t + far
                energy = arrive[:, channel] * t_in * t_out * torch.exp(-self._c(tracer)[channel] * travelled)
                s = self.entry(land[keep])
                ij = (s - lo) / cell - 0.5
                splat(out[channel], ij, energy[keep] * area / (cell * cell), res, periodic=False)
        return out.reshape(3, res, res)

    def visible(self, tracer, objects, x):
        """[P] fraction of the sun visible from x (bilinear comparison in the shadow map)."""
        depth_map, lo, cell, _ = self.shadow_map(tracer, objects)
        t = self.path_in_water(x)
        ij = (self.entry(x) - lo) / cell - 0.5
        res = depth_map.shape[0]
        i0, j0 = ij[:, 0].floor(), ij[:, 1].floor()
        fi, fj = ij[:, 0] - i0, ij[:, 1] - j0
        out = torch.zeros_like(t)
        for di, wi in ((0, 1 - fi), (1, fi)):
            for dj, wj in ((0, 1 - fj), (1, fj)):
                ii, jj = (i0 + di).long(), (j0 + dj).long()
                inside = (ii >= 0) & (ii < res) & (jj >= 0) & (jj < res)
                stored = torch.full_like(t, float("inf"))
                stored[inside] = depth_map[ii[inside], jj[inside]]
                out = out + wi * wj * (t <= stored + self.shadow_bias).float()
        return out

    @torch.no_grad()
    def volume_visibility(self, tracer, objects):
        """([L,M,M] visibility of the sun in the water on the shadow-map window, coarser, t_max): soft shadows
        (shadow_soft, shadow_fill) on layers of the path length t along the light."""
        depth_map, lo, cell, _ = self.shadow_map(tracer, objects)
        key = ("volume", self._cache["shadow_key"])
        if self._cache.get("volume_key") == key:
            return self._cache["volume"]
        res = depth_map.shape[0]
        m = self.volume_resolution
        factor = max(1, res // m)
        m = res // factor
        coarse_cell = cell * factor
        up, _, _ = self.frame(depth_map.device)
        cos = float(-(self.water_direction(depth_map.device) @ up))
        t_max = self.max_depth(tracer) / cos
        layers = torch.linspace(0, t_max, self.volume_layers).tolist()
        # Blocker search: the nearest blocker within ~1 m of each coarse texel.
        finite = torch.where(torch.isfinite(depth_map), depth_map, torch.full_like(depth_map, 1e6))
        blocker = -F.max_pool2d(-finite[None, None], factor, factor)[0, 0]
        k = 2 * max(1, round(1.0 / coarse_cell)) + 1
        blocker = -F.max_pool2d(-blocker[None, None], k, 1, k // 2)[0, 0]
        levels = [0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]  # m, blur levels interpolated per texel
        out = torch.empty(len(layers), m, m, device=depth_map.device)
        for i, t in enumerate(layers):
            lit = F.avg_pool2d((t <= depth_map + self.shadow_bias).float()[None, None], factor)[0, 0]
            if bool((lit > 0.999).all()):
                out[i] = 1.0
                continue
            want = (self.shadow_soft * (t - blocker).clamp_min(0)).clamp(max=levels[-1])
            blurred = [lit] + [gaussian_blur_clamped(lit, s / coarse_cell) for s in levels[1:]]
            soft = torch.zeros_like(lit)
            for (s0, b0), (s1, b1) in zip(zip(levels, blurred), zip(levels[1:], blurred[1:])):
                w = ((want - s0) / (s1 - s0)).clamp(0, 1)
                inside = (want >= s0) & (want <= s1)
                soft = torch.where(inside, (1 - w) * b0 + w * b1, soft)
            fill = gaussian_blur_clamped(soft, self.fill_radius / coarse_cell)
            out[i] = (1 - self.shadow_fill) * soft + self.shadow_fill * fill
        self._cache["volume"] = (out, coarse_cell, t_max)
        self._cache["volume_key"] = key
        return self._cache["volume"]

    def visible_in_water(self, tracer, objects, x):
        """[P] visibility of the sun at points x in the water, with the soft shadows of volume_visibility."""
        volume, cell, t_max = self.volume_visibility(tracer, objects)
        lo = self.shadow_map(tracer, objects)[1]
        layers, m = volume.shape[0], volume.shape[1]
        ij = (self.entry(x) - lo) / cell - 0.5
        z = (self.path_in_water(x).clamp(0, t_max) / t_max * (layers - 1))
        grid = torch.stack((ij[:, 1] / (m - 1) * 2 - 1, ij[:, 0] / (m - 1) * 2 - 1, z / (layers - 1) * 2 - 1), -1)
        v = F.grid_sample(volume[None, None], grid[None, :, None, None], mode="bilinear", padding_mode="border",
                          align_corners=True).reshape(-1)
        outside = (ij < 0).any(-1) | (ij > m - 1).any(-1)
        return torch.where(outside, torch.ones_like(v), v)

    def glass_light(self, tracer, objects, x):
        """[P,3] relative irradiance focused by glass spheres at x (0 where none, or where x is lit directly)."""
        depth_map, lo, cell, extra = self.shadow_map(tracer, objects)
        if extra is None:
            return torch.zeros(x.shape[0], 3, device=x.device)
        ij = (self.entry(x) - lo) / cell - 0.5
        res = depth_map.shape[0]
        i, j = ij[:, 0].round().long().clamp(0, res - 1), ij[:, 1].round().long().clamp(0, res - 1)
        behind = self.path_in_water(x) > depth_map[i, j] + self.shadow_bias  # x is in some shadow
        return extra[:, i, j].T * behind[:, None]

    # --- light -------------------------------------------------------------------------------------------
    def _c(self, tracer):
        return tracer.water_coefficients()[0]

    def _attenuation(self, tracer, t):
        return torch.exp(-self._c(tracer) * t.clamp_min(0)[:, None])

    def strength(self, tracer):
        """[3] E0: irradiance on a surface facing the sun just below the water, in lamp_light's units x pi."""
        return self.power * math.pi * tracer.ambient_up().mean() * torch.tensor(self.tint, device="cuda")

    def reach(self, tracer, points, objects):
        """Like SceneTracer.lamp_reach: unit directions towards the sun [P,3] and irradiance facing it [P,3]."""
        l = (-self.water_direction(points.device)).expand_as(points)
        t = self.path_in_water(points)
        e = self.strength(tracer) * self._attenuation(tracer, t)
        direct = self.caustic_at(tracer, points)[:, None] * self.visible(tracer, objects, points)[:, None]
        e = e * (direct + self.glass_light(tracer, objects, points))
        return l, torch.where((t > 0)[:, None], e, torch.zeros_like(e))

    @torch.no_grad()
    def inscatter(self, tracer, rays, end, objects, chunk: int = 1 << 18):
        """[R,3] sunlight scattered once by the water towards the ray origins, up to end [R]."""
        c, b = tracer.water_coefficients()
        c_mean = float(c.mean())
        up, _, _ = self.frame(rays.device)
        o, d = rays[:, :3], rays[:, 3:]
        # Stop at the water surface.
        rise = d @ up
        to_surface = torch.where(rise > 1e-6, (self.surface - o @ up) / rise.clamp_min(1e-6),
                                 torch.full_like(rise, float("inf")))
        end = torch.minimum(end, to_surface).clamp(min=0, max=8.0 / c_mean)
        g = self.g
        cos = -(d @ self.water_direction(rays.device))  # light travels along w, scattered into -d
        phase = (1 - g * g) / (4 * math.pi * (1 + g * g - 2 * g * cos).clamp_min(1e-6) ** 1.5)
        e0 = self.strength(tracer)
        out = torch.zeros_like(o)
        n = self.samples
        for begin in range(0, o.shape[0], chunk):
            sl = slice(begin, begin + chunk)
            oo, dd, ee = o[sl], d[sl], end[sl]
            # t ~ c exp(-c t) on [0, end]: stratified in the cdf.
            total = 1 - torch.exp(-c_mean * ee)
            jitter = torch.rand(oo.shape[0], device=o.device)
            acc = torch.zeros_like(oo)
            for k in range(n):
                u = (k + jitter) / n * total
                t = -torch.log1p(-u.clamp(max=1 - 1e-7)) / c_mean
                x = oo + t[:, None] * dd
                tx = self.path_in_water(x)
                light = self.caustic_at(tracer, x, coarse=True) * self.visible_in_water(tracer, objects, x)
                e = e0 * light[:, None] * torch.exp(-c * tx.clamp_min(0)[:, None]) * (tx > 0)[:, None]
                # integrand b phase e exp(-c t) over the pdf c_mean exp(-c_mean t) / total
                acc = acc + e * torch.exp(-(c - c_mean) * t[:, None])
            out[sl] = acc / n * total[:, None] / c_mean
        return self.shaft_strength * b * phase[:, None] * out


    # --- the surface seen from below ---------------------------------------------------------------------
    def surface_hit(self, rays):
        """[R] distance to the surface for rays from below going up (inf: none)."""
        up, _, _ = self.frame(rays.device)
        rise = rays[:, 3:] @ up
        below = self.surface - rays[:, :3] @ up
        ok = (rise > 1e-4) & (below > 1e-3)
        return torch.where(ok, below / rise.clamp_min(1e-4), torch.full_like(rise, float("inf")))

    def surface_normal(self, x, footprint, chunk: int = 1 << 18):
        """[P,3] wave normals (up side) at surface points x [P,3]; waves shorter than the footprint [P] (m) fade."""
        k, a, phase, omega = self._waves(x.device)
        up, _, _ = self.frame(x.device)
        e1, e2 = self.tile_axes(x.device)
        knorm = k.norm(dim=-1)
        out = torch.empty_like(x)
        for begin in range(0, x.shape[0], chunk):
            xx, fp = x[begin:begin + chunk], footprint[begin:begin + chunk]
            p = torch.stack((xx @ e1, xx @ e2), -1)
            arg = p @ k.T + phase - omega * self.time
            lod = torch.exp(-0.5 * (fp[:, None] * knorm) ** 2)
            grad = (a * lod * arg.cos()) @ k
            out[begin:begin + chunk] = F.normalize(up - grad[:, :1] * e1 - grad[:, 1:] * e2, dim=-1)
        return out

    @torch.no_grad()
    def surface_light(self, tracer, rays, t, objects, settings, depth: int, weight, stats=None):
        """[R,3] radiance leaving the surface (reached at distance t [R] from below) towards the ray origins."""
        device = rays.device
        up, _, _ = self.frame(device)
        o, d = rays[:, :3], rays[:, 3:]
        x = o + t[:, None] * d
        # A pixel's footprint on the surface, stretched along the view at grazing angles.
        footprint = t * self.pixel_angle * 1.5 / (d @ up).clamp_min(0.05)
        n = self.surface_normal(x, footprint)
        cos_i = (d * n).sum(-1).clamp(1e-4, 1)
        # Exact dielectric Fresnel from the water into the air (1 beyond the critical angle).
        sin_t2 = WATER_IOR ** 2 * (1 - cos_i ** 2)
        cos_t = (1 - sin_t2.clamp(max=1)).sqrt()
        rs = ((WATER_IOR * cos_i - cos_t) / (WATER_IOR * cos_i + cos_t)) ** 2
        rp = ((cos_i - WATER_IOR * cos_t) / (cos_i + WATER_IOR * cos_t)) ** 2
        kr = torch.where(sin_t2 >= 1, torch.ones_like(cos_i), 0.5 * (rs + rp))
        # Refracted: the sky and the sun's lobe (radiance n^2 times the air's).
        c = tracer.water_coefficients()[0]
        origin_height = float(tracer.color_origin.float().to(device) @ up)
        sky = self.sky * tracer.ambient_up().to(device) * torch.exp(c * (self.surface - origin_height))
        air = refract(d, -n, WATER_IOR)
        cos_sun = (air @ (-self.air_direction(device))).clamp(-1, 1)
        sigma = math.radians(self.sun_glow_deg)
        glow = torch.exp(-0.5 * (torch.acos(cos_sun) / sigma) ** 2) * WATER_IOR ** 2 / (2 * math.pi * sigma ** 2)
        refracted = (1 - kr)[:, None] * (sky + self.strength(tracer) * glow[:, None])
        # Reflected: back down into the water, traced on (the ruins, their caustics, the shafts).
        r = d - 2 * (d * n).sum(-1, keepdim=True) * n
        r = torch.where((r @ up < -1e-3)[:, None], r, d - 2 * (d @ up)[:, None] * up)
        if depth < settings.max_depth:
            down = torch.cat((x - 1e-3 * up, F.normalize(r, dim=-1)), -1).contiguous()
            water = (tracer.water_coefficients()[0].mean().expand(3), tracer.horizon_water())
            previous, tracer.uniform_water = tracer.uniform_water, water
            try:
                reflected = tracer.trace(down, objects, settings, depth + 1, weight * kr[:, None], stats=stats)
            finally:
                tracer.uniform_water = previous
        else:
            reflected = tracer.horizon_water().expand_as(x)
        return refracted + kr[:, None] * reflected


# --- helpers -------------------------------------------------------------------------------------------------
def refract(d, n, eta):
    """Directions d refracted at normals n (facing against d), index ratio eta = n1 / n2; zero on total reflection."""
    cos_i = -(d * n).sum(-1, keepdim=True)
    k = 1 - eta * eta * (1 - cos_i * cos_i)
    out = eta * d + (eta * cos_i - k.clamp_min(0).sqrt()) * n
    return torch.where(k >= 0, F.normalize(out, dim=-1), torch.zeros_like(out))


def fresnel(cos_i, n1: float, n2: float):
    r0 = ((n1 - n2) / (n1 + n2)) ** 2
    return r0 + (1 - r0) * (1 - cos_i.clamp(0, 1)) ** 5


def splat(target, ij, weight, size: int, periodic: bool = True):
    """Bilinear splat of weights at texel coordinates ij [P,2] (texel centres at integers) into target [size*size]:
    wrapped (the caustic tile) or dropped outside (the shadow-map grid)."""
    i0, j0 = ij[:, 0].floor(), ij[:, 1].floor()
    fi, fj = ij[:, 0] - i0, ij[:, 1] - j0
    i0, j0 = i0.long(), j0.long()
    for di, wi in ((0, 1 - fi), (1, fi)):
        for dj, wj in ((0, 1 - fj), (1, fj)):
            ii, jj = i0 + di, j0 + dj
            if periodic:
                ii, jj, w = ii % size, jj % size, weight * wi * wj
            else:
                inside = (ii >= 0) & (ii < size) & (jj >= 0) & (jj < size)
                ii, jj, w = ii[inside], jj[inside], (weight * wi * wj)[inside]
            target.index_add_(0, ii * size + jj, w)


def gaussian_blur_periodic(volume, sigma):
    """Per-layer periodic Gaussian blur (FFT) of [L,M,M] with sigma [L] in texels."""
    m = volume.shape[-1]
    f = torch.fft.fftfreq(m, device=volume.device)
    f2 = f[:, None] ** 2 + f[None, :] ** 2
    transfer = torch.exp(-2 * math.pi ** 2 * sigma[:, None, None] ** 2 * f2)
    return torch.fft.ifft2(torch.fft.fft2(volume) * transfer).real.clamp_min(0)


def gaussian_blur_clamped(image, sigma: float):
    """[M,M] image blurred by a Gaussian of sigma texels (separable; edges extended)."""
    if sigma < 0.3:
        return image
    radius = int(math.ceil(3 * sigma))
    x = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    out = F.pad(image[None, None], (radius, radius, 0, 0), mode="replicate")
    out = F.conv2d(out, kernel.view(1, 1, 1, -1))
    out = F.pad(out, (0, 0, radius, radius), mode="replicate")
    return F.conv2d(out, kernel.view(1, 1, -1, 1))[0, 0]


def trilinear_periodic(volume, z, uv):
    """volume [L,M,M] at layer coordinate z [P] (clamped) and texel coordinates uv [P,2] (periodic)."""
    layers, m = volume.shape[0], volume.shape[1]
    uv = uv - 0.5
    z0 = z.floor().clamp(0, layers - 1)
    fz = (z - z0).clamp(0, 1)
    z1 = (z0 + 1).clamp(max=layers - 1)
    i0, j0 = uv[:, 0].floor(), uv[:, 1].floor()
    fi, fj = uv[:, 0] - i0, uv[:, 1] - j0
    i0, j0 = i0.long() % m, j0.long() % m
    i1, j1 = (i0 + 1) % m, (j0 + 1) % m
    out = torch.zeros_like(z)
    flat = volume.reshape(layers, -1)
    for zz, wz in ((z0.long(), 1 - fz), (z1.long(), fz)):
        for ii, wi in ((i0, 1 - fi), (i1, fi)):
            for jj, wj in ((j0, 1 - fj), (j1, fj)):
                out = out + wz * wi * wj * flat[zz, ii * m + jj]
    return out
