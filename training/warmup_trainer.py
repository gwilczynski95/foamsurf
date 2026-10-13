"""Scene initialization only. The existing joint Trainer is not modified.

Inference/checkpoint model tensors use the original beta=1 format. The exact
warmup parameterization and optimizer state live in a separate payload field.
This also gives the existing evaluation/handoff code unchanged physical sigma.
"""
from dataclasses import replace, dataclass
from functools import lru_cache
import itertools
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from foamsurf_data import load_dataset
from foamsurf_model.medium import VoronoiMediumModel
from foamsurf_model.topology import VoronoiTopologyController
from foamsurf_model.voronoi_adaptation import VoronoiAdaptationController
from .checkpoints import checkpoint_payload, load_checkpoint, save_checkpoint
from .losses import b0_reconstruction_loss
from .optimizer import build_medium_optimizer, gradient_norm
from .sampler import RandomCameraRaySampler
from .trainer import Trainer
from .warmup_profile import WarmupProfile, inverse_density, opacity_loss, quantile_loss
from .warmup_population import WarmupPopulation
from foamsurf_data.rays import generate_indexed_rays


class WarmupMedium(VoronoiMediumModel):
    @property
    def densities(self):
        values = self.optical_coefficients
        return values[:, :3].mean(-1, keepdim=True) if self.is_rgb else values

    @property
    def optical_coefficients(self):
        return self.warmup_scale * F.softplus(self.raw_density, beta=self.warmup_beta)

    @property
    def colors(self):
        if getattr(self, "radfoam_dc", False):
            return (0.5 + 0.28209479177387814 * self.raw_color).clamp_min(0)
        return super().colors

    @property
    def outer_mask(self):
        # Outer provenance (protected only under the original population policy).
        # Reuse the established permutation-safe mask; no new buffers in hybrid.
        return self.guard_mask


class WarmupTopology(VoronoiTopologyController):
    @torch.no_grad()
    def update(self, medium, optimizer, incremental=True, max_attempts=25):
        import foamsurf_topology
        points = medium.sites.detach()
        if not bool(torch.isfinite(points).all()):
            raise RuntimeError("non-finite warmup positions")
        candidate = points
        for attempt in range(max_attempts + 1):
            try:
                permuted = bool(self.triangulation.rebuild(candidate, incremental=incremental))
                break
            except foamsurf_topology.TriangulationFailedError:
                if attempt == max_attempts:
                    raise RuntimeError("warmup topology rebuild exhausted retries")
                candidate = points.clone()
                inner = (torch.ones_like(medium.guard_mask) if getattr(self, "organic", False)
                         else ~medium.guard_mask)
                candidate[inner] += (2.0 ** (attempt + 1)) * 1e-6 * torch.randn_like(points[inner])
                incremental = False
        # Geometry used to build adjacency must match rendered geometry, also
        # on the FIRST successful perturbed retry. Protected outer sites never jitter.
        if candidate is not points:
            medium.sites.copy_(candidate)
        self.last_permutation = self.triangulation.permutation().long() if permuted else None
        if permuted:
            self._permute_with_optimizer(medium, optimizer, self.last_permutation)
        self._read_topology()
        self.revision += 1
        return permuted

    def replace_population(self, medium, optimizer, keep_indices, parent_indices, child_sites, **kwargs):
        organic = getattr(self, "organic", False)
        outer = medium.guard_mask.clone()
        if not organic:
            if bool(outer[parent_indices].any()):
                raise ValueError("outer sites cannot be densification parents")
            if int(outer[keep_indices].sum()) != int(outer.sum()):
                raise ValueError("outer sites cannot be pruned")
        lower, upper = medium.bbox_min, medium.bbox_max
        applied = False
        try:
            # Outer provenance survives spawning/permutation, but is not a
            # protection mask in the organic policy.
            if organic:
                medium.guard_mask = torch.zeros_like(outer)
            candidates = torch.cat((medium.sites.detach(), child_sites.detach()))
            medium.bbox_min = torch.minimum(lower, candidates.amin(0))
            medium.bbox_max = torch.maximum(upper, candidates.amax(0))
            applied = super().replace_population(medium, optimizer, keep_indices,
                                                  parent_indices, child_sites, **kwargs)
            return applied
        finally:
            if organic:
                medium.guard_mask = (torch.cat((outer[keep_indices], outer[parent_indices]))[
                    self.last_permutation] if applied else outer)
            medium.bbox_min, medium.bbox_max = lower, upper


class WarmupTrainer(Trainer):
    def __init__(self, config, profile: WarmupProfile, resume=None):
        profile.validate()
        if not config.voronoi_warmup.enabled or config.representation_mode != "voronoi_only":
            raise ValueError("WarmupTrainer requires enabled Voronoi-only initialization")
        if config.dataset.train_target != "both" or config.medium_handoff.prefix_iterations:
            raise ValueError("warmup requires combined RGB and no separate prefix")
        if any(getattr(config.disentanglement, key) != 0 for key in (
            "opacity_binarization_weight", "opaque_if_used_weight", "graph_tv_weight",
            "density_cap_weight", "optical_thickness_weight")):
            raise ValueError("joint disentanglement losses must be disabled for this warmup")
        if config.gaussian_adaptation.enabled:
            raise ValueError("Gaussian adaptation must be disabled during initialization")
        self.profile = profile
        self.warmup_metrics = {}
        self._training_resolution = None
        self._topology_period = 1
        self._topology_since_update = 1
        payload = load_checkpoint(resume) if resume else None
        if payload is not None and Path(resume).name.startswith("preadapt_"):
            raise ValueError("Pre-adaptation snapshots are diagnostic only; resume an iter/last checkpoint")
        if payload is not None and "warmup" not in payload:
            raise ValueError("resume requires a warmup checkpoint")
        if payload is not None and WarmupProfile.from_dict(payload.get("warmup", {}).get("profile", {})).as_dict() != profile.as_dict():
            raise ValueError("resume requires the identical warmup profile")
        if profile.site_initialization == "spatial":
            config.model.medium_points = profile.spatial_count
        if profile.site_initialization == "pcd":
            raise ValueError("PCD initialization is not part of the two supported configurations")
        super().__init__(config, resume=resume)
        medium = self.model.medium
        medium.__class__ = WarmupMedium
        medium.warmup_beta = profile.density_beta
        medium.warmup_scale = profile.activation_scale
        medium.radfoam_dc = profile.site_initialization in ('pcd', 'spatial')
        if payload is None:
            with torch.no_grad():
                if profile.site_initialization == "random":
                    medium.sites.copy_(medium.bbox_min + torch.rand_like(medium.sites) *
                                       (medium.bbox_max - medium.bbox_min))
                if profile.site_initialization == "spatial":
                    initialize_spatial_medium(medium, self.dataset, config, profile)
                if medium.is_rgb:
                    if profile.site_initialization != "spatial":
                        raise ValueError("RGB warmup currently requires spatial initialization")
                    # Matched scalar initialization, independent parameters thereafter.
                    medium.raw_density = nn.Parameter(medium.raw_density.expand(-1, 6).clone())
                if profile.site_initialization not in ("pcd", "spatial"):
                    medium.raw_density.fill_(float(inverse_density(torch.tensor(profile.initial_sigma, dtype=torch.float64),
                        profile.density_beta, profile.activation_scale)))
                n = 0 if profile.site_initialization in ("pcd", "spatial") else profile.outer_points
                if n:
                    center = (medium.bbox_min + medium.bbox_max) / 2
                    if profile.corner_guards:
                        signs = torch.tensor(list(itertools.product((-1., 1.), repeat=3)), device=self.device)
                        # Perfect cube corners are co-spherical and make 3D
                        # Delaunay non-unique. A deterministic sub-per-mille
                        # radial staggering keeps their guard role while
                        # guaranteeing general position.
                        radial = 1.0 + 1.0e-4 * torch.arange(8, device=self.device)[:, None]
                        outer = center + signs * (medium.bbox_max-medium.bbox_min) * 0.625 * radial
                    else:
                        # RadFoam uses an unconditioned Gaussian cloud; these
                        # scaffold points are identified by origin, not labels.
                        outer = center + torch.randn(n, 3, device=self.device) * profile.outer_scale
                    medium.sites = nn.Parameter(torch.cat((medium.sites, outer)))
                    density = inverse_density(torch.full((n, 1), profile.outer_sigma, device=self.device),
                                              profile.density_beta, profile.activation_scale)
                    medium.raw_density = nn.Parameter(torch.cat((medium.raw_density, density)))
                    medium.raw_color = nn.Parameter(torch.cat((medium.raw_color, medium.raw_color[:1].expand(n, 3))))
                    medium.guard_mask = torch.cat((medium.guard_mask, torch.ones(n, dtype=torch.bool, device=self.device)))
            self.medium_optimizer = build_medium_optimizer(self.model, config.optimization)
            for group in self.medium_optimizer.param_groups:
                group["eps"] = profile.adam_eps
            self.topology = WarmupTopology(medium, self.medium_optimizer)
            self.voronoi_adaptation = VoronoiAdaptationController(medium, self.medium_optimizer,
                self.topology, config.voronoi_adaptation)
        else:
            # Base resume rebuilt topology and permuted the stored optimizer
            # moments. Apply exactly the same permutation to the raw beta!=1 state.
            with torch.no_grad():
                medium.raw_density.copy_(payload["warmup"]["raw_density"].to(self.device)[self.topology.last_permutation])
                if medium.radfoam_dc:
                    medium.raw_color.copy_(payload["warmup"]["raw_color"].to(self.device)[self.topology.last_permutation])
            self.topology.__class__ = WarmupTopology
            # Base controller restored statistics before our profile was active;
            # topology's initial resume permutation must also apply to statistics.
            self.voronoi_adaptation.reorder_statistics(self.topology.last_permutation)
            for name, value in payload["warmup"].get("adaptation_accumulators", {}).items():
                setattr(self.voronoi_adaptation, name, value.to(self.device)[self.topology.last_permutation].contiguous())
        self.topology.organic = profile.population_policy == "radfoam"
        if self.topology.organic:
            controller = WarmupPopulation(medium, self.medium_optimizer, self.topology,
                config.voronoi_adaptation, profile, config.model.medium_points + profile.outer_points)
            if payload is not None:
                state = dict(payload["voronoi_adaptation"])
                state["low_activity_streak"] = state["low_activity_streak"].to(self.device)[self.topology.last_permutation]
                controller.load_state_dict(state)
            self.voronoi_adaptation = controller
        self.evaluator.topology = self.topology
        self.topology.synchronize_renderer(self.renderer, medium)
        self._select_resolution(self.start_iteration - 1)
        if payload is not None:
            state = payload["warmup"]
            self.sampler._camera_stack = list(state["camera_stack"])
            self._topology_period = state.get("topology_period", 1)
            self._topology_since_update = state.get("topology_since_update", 1)
            random.setstate(state["python_rng"])
            np.random.set_state(state["numpy_rng"])
            torch.set_rng_state(payload["rng"]["torch"])
            torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
        config.save(self.output_dir / "config.yaml")
        (self.output_dir / "warmup_profile.json").write_text(json.dumps(profile.as_dict(), indent=2)+"\n")

    def _select_resolution(self, step):
        factor = self.profile.resolution(step)
        if factor == self._training_resolution:
            return
        previous_stack = list(self.sampler._camera_stack) if self._training_resolution is not None else []
        data = self.dataset if factor == 1 else load_dataset(
            self.config.dataset.source_path, resolution=self.config.dataset.resolution * factor,
            device=self.device, llffhold=self.config.dataset.llffhold)
        if self.profile.ray_sampling == "mixed":
            self.sampler = MixedRaySampler(data, self.config.optimization.rays_per_batch,
                                          tuple(self.config.background))
        else:
            self.sampler = RandomCameraRaySampler(data, self.config.optimization.rays_per_batch,
                tuple(self.config.background), full_image=self.config.optimization.full_image_batch,
                cameras_without_replacement=self.config.optimization.cameras_without_replacement)
        self.sampler._camera_stack = previous_stack
        self._training_resolution = factor

    def _checkpoint(self, path, iteration):
        payload = checkpoint_payload(self.model, self.gaussian_optimizer, self.medium_optimizer,
            self.config, iteration, self.training_seconds, self.best_psnr, self.topology.revision,
            self.gaussian_adaptation.state_dict(), self.voronoi_adaptation.state_dict(),
            self.medium_frozen_until, self.disentanglement.state_dict(), self.prefix_training_seconds)
        medium = self.model.medium
        payload["warmup"] = {"version": 2, "profile": self.profile.as_dict(),
            "topology_period": self._topology_period, "topology_since_update": self._topology_since_update,
            "raw_density": medium.raw_density.detach().cpu(), "outer_mask": medium.outer_mask.detach().cpu(),
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "camera_stack": list(self.sampler._camera_stack), "resolution": self._training_resolution,
            "optimizer_parameterization": "warmup_raw_density; do not resume with joint Trainer",
            "adaptation_accumulators": {name: getattr(self.voronoi_adaptation, name).detach().cpu()
                for name in ("contribution", "visibility", "path_length", "weighted_path_length", "max_error", "gradient")}}
        # Legacy analysis and scene initialization see canonical beta=1 tensors.
        payload["model"]["medium.raw_density"] = inverse_density(medium.optical_coefficients.detach()).cpu()
        if medium.radfoam_dc:
            payload["warmup"]["raw_color"] = medium.raw_color.detach().cpu()
            payload["warmup"]["color_clipped_values"] = int((medium.colors.detach() > 1).sum())
            payload["model"]["medium.raw_color"] = torch.logit(medium.colors.detach().clamp(1e-6, 1-1e-6)).cpu()
        save_checkpoint(Path(path), payload)

    def _train_step(self, iteration):
        step = iteration - 1  # zero-based schedules applied BEFORE optimizer.step
        self._select_resolution(step)
        self.current_medium_lrs = self.profile.rates(step)
        self.current_gaussian_means_lr = 0.0
        for group in self.medium_optimizer.param_groups:
            group["lr"] = self.current_medium_lrs[group["name"]]
        torch.cuda.synchronize()
        started = time.perf_counter()
        batch = self.sampler.next()
        rays = torch.cat((batch.rays.origins, batch.rays.directions), -1).contiguous()
        self.medium_optimizer.zero_grad(set_to_none=True)
        collect = self.voronoi_adaptation.should_collect(iteration)
        scene = self.model.medium.render_scene()
        rendered = self.renderer.render(rays, voronoi=scene, background=tuple(self.config.background),
            collect_diagnostics=True, diagnostic_target=batch.target)
        loss = b0_reconstruction_loss(rendered.image, batch.target,
            mode="l1" if self.profile.ray_sampling == "mixed" else "l1_dssim",
            dssim_weight=self.config.optimization.dssim_weight,
            image_size=None if batch.camera is None else (batch.camera.height, batch.camera.width))
        if self.profile.loss == "smooth_l1":
            loss = replace(loss, total=F.smooth_l1_loss(rendered.image, batch.target,
                                                      beta=self.profile.smooth_l1_beta))
        rgb_loss = loss.total
        alpha_loss = opacity_loss(rendered.alpha, batch.alpha)
        loss = replace(loss, total=rgb_loss + self.profile.opacity_weight * alpha_loss)
        qloss = rgb_loss.new_zeros(())
        strength = self.profile.quantile_strength(step)
        valid_fraction = width_mean = width_p50 = width_p95 = qmean = 0.0
        if strength > 0:
            if self.profile.fixed_quantiles is None:
                q = torch.rand(len(rays), 2, device=self.device).clamp_min(torch.finfo(torch.float32).eps)
                q = q.sort(dim=-1, descending=True).values
            else:
                q = torch.tensor(self.profile.fixed_quantiles, device=self.device).expand(len(rays), 2).contiguous()
            depths, owners = render_quantiles(self.renderer, scene, rays, q)
            qloss, valid, width = quantile_loss(depths)
            valid_fraction = float(valid.float().mean())
            qmean = float(q.mean())
            if bool(valid.any()):
                values = width.detach()[valid]
                width_mean, width_p50, width_p95 = map(float, (values.mean(), values.median(), torch.quantile(values, .95)))
            loss = replace(loss, total=loss.total + strength * qloss)
        if not bool(torch.isfinite(loss.total)):
            raise FloatingPointError("non-finite warmup loss")
        loss.total.backward()
        medium = self.model.medium
        frozen = []
        for name, trainable in (("sites", self.profile.outer_positions_trainable),
                                ("raw_density", self.profile.outer_density_trainable),
                                ("raw_color", self.profile.outer_color_trainable)):
            parameter = getattr(medium, name)
            if not trainable:
                frozen.append((parameter, parameter.detach()[medium.outer_mask].clone()))
                if parameter.grad is not None:
                    parameter.grad[medium.outer_mask] = 0
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(f"non-finite {name} gradient")
        if collect:
            self.voronoi_adaptation.collect_statistics(rendered.diagnostics.medium)
        medium_norm = gradient_norm(medium.parameters())
        sites_norm = gradient_norm([medium.sites])
        density_norm = gradient_norm([medium.raw_density])
        self.medium_optimizer.step()
        with torch.no_grad():
            for parameter, values in frozen:
                parameter[medium.outer_mask] = values
                for key in ("exp_avg", "exp_avg_sq"):
                    state = self.medium_optimizer.state[parameter].get(key)
                    if state is not None:
                        state[medium.outer_mask] = 0
        self.last_voronoi_adaptation = {"densify": 0, "prune": 0, "skipped": 0}
        self.last_voronoi_adaptation_pre_psnr = float("nan")
        self.last_voronoi_adaptation_post_psnr = float("nan")
        self.last_voronoi_adaptation_psnr_delta = float("nan")
        self._error_map_seconds = 0.
        if self.profile.topology_schedule == "radfoam":
            if self._topology_since_update >= self._topology_period:
                self._refresh_topology()
                self._topology_since_update = 0
                if self._topology_period < 100:
                    self._topology_period += 2
            self._topology_since_update += 1
        changed = False
        if self.voronoi_adaptation.should_adapt(iteration):
            self._checkpoint(self.output_dir / "checkpoints" / f"preadapt_{iteration:08d}.pt", iteration)
            self.last_voronoi_adaptation_pre_psnr = self._same_batch_psnr(batch, rays)
            if self.profile.population_policy == "radfoam":
                self._collect_population_map()
            plan = self.voronoi_adaptation.propose_plan()
            self.last_voronoi_adaptation = self.voronoi_adaptation.apply_plan(plan)
            changed = bool(self.last_voronoi_adaptation["densify"] or self.last_voronoi_adaptation["prune"])
            self.topology.synchronize_renderer(self.renderer, medium)
            if self.profile.topology_schedule == "radfoam":
                self._topology_period = 1
            self.last_voronoi_adaptation_post_psnr = self._same_batch_psnr(batch, rays)
            self.last_voronoi_adaptation_psnr_delta = self.last_voronoi_adaptation_post_psnr-self.last_voronoi_adaptation_pre_psnr
            event = {"iteration": iteration, "policy": self.profile.population_policy,
                     "before_count": plan.original_count, "after_count": medium.point_count,
                     **self.last_voronoi_adaptation,
                     "density_reset": self.voronoi_adaptation.last_statistics.get("density_reset", 0),
                     "next_iteration": getattr(self.voronoi_adaptation, "next_iteration", None),
                     "pre_psnr": self.last_voronoi_adaptation_pre_psnr,
                     "post_psnr": self.last_voronoi_adaptation_post_psnr,
                     "error_map_seconds": self._error_map_seconds}
            with (self.output_dir / "metrics" / "population_events.jsonl").open("a") as output:
                output.write(json.dumps(event) + "\n")
            print("WARMUP_POPULATION=" + json.dumps(event), flush=True)
        if (self.profile.topology_schedule == "fixed" and not changed
                and iteration % self.config.optimization.topology_update_interval == 0):
            permuted = self.topology.update(medium, self.medium_optimizer,
                incremental=self.config.optimization.topology_incremental)
            if permuted:
                self.voronoi_adaptation.reorder_statistics(self.topology.last_permutation)
            self.topology.synchronize_renderer(self.renderer, medium)
        self.last_disentanglement = self.disentanglement.compute(iteration, self.model, self.topology)
        self.warmup_metrics = {
            "warmup/error_map_seconds": self._error_map_seconds,
            "warmup/total_density_reset": getattr(self.voronoi_adaptation, "total_density_reset", 0),
            "warmup/mixed_rays": int(self.profile.ray_sampling == "mixed"),
            "warmup/sites_gradient_norm": sites_norm, "warmup/density_gradient_norm": density_norm,
            "warmup/rgb_loss": float(rgb_loss.detach()), "warmup/opacity_loss": float(alpha_loss.detach()),
            "warmup/opacity_weight": self.profile.opacity_weight,
            "warmup/weighted_opacity_loss": self.profile.opacity_weight*float(alpha_loss.detach()),
            "warmup/target_alpha_mean": float(batch.alpha.mean()),
            "warmup/rendered_alpha_mean": float(rendered.alpha.detach().mean()),
            "warmup/quantile_loss": float(qloss.detach()),
            "warmup/quantile_weight": strength, "warmup/weighted_quantile_loss": strength*float(qloss.detach()),
            "warmup/valid_quantile_fraction": valid_fraction, "warmup/width_mean": width_mean,
            "warmup/width_p50": width_p50, "warmup/width_p95": width_p95, "warmup/quantile_mean": qmean,
            "warmup/resolution": self._training_resolution, "warmup/rays": len(rays),
            "warmup/inner_count": int((~medium.outer_mask).sum()), "warmup/outer_count": int(medium.outer_mask.sum()),
            "warmup/truncated_fraction": float(rendered.diagnostics.traversal_truncated.mean()),
            "warmup/intersections_p99": float(torch.quantile(rendered.num_intersections.float(), .99)),
        }
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        self.training_seconds += elapsed
        return loss, 0.0, medium_norm, elapsed


    def _refresh_topology(self):
        permuted = self.topology.update(self.model.medium, self.medium_optimizer, incremental=True)
        if permuted:
            self.voronoi_adaptation.reorder_statistics(self.topology.last_permutation)
        self.topology.synchronize_renderer(self.renderer, self.model.medium)

    def _collect_population_map(self):
        """Fresh L1 geometry gradients, summed per camera BEFORE taking norms.

        Contribution sums within each view, then max across views, as in RadFoam.
        Chunking only bounds memory; the per-view objective is a sum, not a mean.
        No optimizer step and no alpha/quantile terms in this map.
        """
        started = time.perf_counter()
        medium = self.model.medium
        error = torch.zeros(medium.point_count, device=self.device)
        contribution = torch.zeros_like(error)
        data = self.sampler.data
        d = self.profile.error_map_downsample
        chunk = min(self.profile.error_map_chunk, self.renderer.max_batch_size)
        for ci, camera in enumerate(self.sampler.dataset.train):
            self.medium_optimizer.zero_grad(set_to_none=True)
            offset = torch.randint(d, (2,), device=self.device)
            ys = torch.arange(int(offset[0]), camera.height, d, device=self.device)
            xs = torch.arange(int(offset[1]), camera.width, d, device=self.device)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            pixels = (yy * camera.width + xx).flatten()
            per_view = torch.zeros_like(contribution)
            for part in pixels.split(chunk):
                if part.numel() == 0:
                    continue
                bundle = generate_indexed_rays(part + ci * camera.width * camera.height,
                    data["origins"], data["rights"], data["downs"], data["forwards"],
                    data["fov_x"], data["fov_y"], camera.width, camera.height, intrinsics=data.get("intrinsics"))
                rays = torch.cat((bundle.origins, bundle.directions), -1).contiguous()
                target = camera.foreground.reshape(-1, 3)[part] + self.sampler.background * (
                    1 - camera.alpha.reshape(-1, 1)[part])
                render = self.renderer.render(rays, voronoi=medium.render_scene(),
                    background=tuple(self.config.background), collect_diagnostics=True)
                render.image.sub(target).abs().mean(dim=-1).sum().backward()
                per_view += render.diagnostics.medium[:, 0].detach()
            if medium.sites.grad is not None:
                error += medium.sites.grad.detach().norm(dim=-1)
            contribution = torch.maximum(contribution, per_view)
        self.medium_optimizer.zero_grad(set_to_none=True)
        self.voronoi_adaptation.set_error_map(error, contribution)
        torch.cuda.synchronize()
        self._error_map_seconds = time.perf_counter() - started

    def _training_row(self, *args):
        row = super()._training_row(*args)
        row.update(self.warmup_metrics)
        medium = self.model.medium
        for name, mask in (("inner", ~medium.outer_mask), ("outer", medium.outer_mask)):
            for field, values in (("sigma", medium.densities), ("raw", medium.raw_density)):
                selected = values.detach()[mask].flatten()
                for q in (.01, .05, .5, .95, .99):
                    row[f"warmup/{name}_{field}_p{round(100*q)}"] = float(torch.quantile(selected, q)) if selected.numel() else 0.0
        return row



@lru_cache(maxsize=1)
def bindings():
    from torch.utils.cpp_extension import load
    source = Path(__file__).resolve().parents[1] / "renderer/native/warmup_quantiles.cu"
    return load(name="foamsurf_warmup_quantiles", sources=[str(source)],
                extra_cuda_cflags=["-O3", "-lineinfo"], extra_cflags=["-O3"], verbose=True)

class _Quantiles(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, sigma, adjacency, offsets, rays, starts, q, settings):
        tensors = tuple(t.contiguous() for t in (points, sigma, adjacency, offsets, rays, starts, q))
        depth, owner = bindings().forward(list(tensors), *settings)
        ctx.save_for_backward(*tensors, depth, owner)
        ctx.settings = settings
        ctx.mark_non_differentiable(owner)
        return depth, owner

    @staticmethod
    def backward(ctx, grad_depth, grad_owner):
        *tensors, depth, owner = ctx.saved_tensors
        pg, dg = bindings().backward(tensors, depth, owner, grad_depth.contiguous(), *ctx.settings)
        return pg, dg, None, None, None, None, None, None

def render_quantiles(renderer, scene, rays, q, *, gaussians=None):
    if gaussians is not None:
        raise ValueError("depth quantiles support Voronoi-only initialization")
    if rays.requires_grad:
        raise ValueError("warmup requires fixed camera rays")
    if q.ndim != 2 or q.shape[0] != rays.reshape(-1, 6).shape[0] or q.shape[1] == 0:
        raise ValueError("quantiles must have shape [R,Q>0]")
    if not bool(torch.isfinite(q).all()) or bool(((q <= 0) | (q >= 1)).any()) or bool((q[:, 1:] > q[:, :-1]).any()):
        raise ValueError("quantiles must be descending and strictly inside (0,1)")
    if not rays.is_cuda or rays.dtype != torch.float32 or q.device != rays.device or q.dtype != torch.float32:
        raise ValueError("rays/quantiles must be CUDA float32 on the same device")
    scene.validate()
    topology = renderer._voronoi_topology
    if topology is None or topology.site_data_ptr != scene.sites.data_ptr():
        raise ValueError("synchronize Voronoi topology first")
    rays = rays.reshape(-1, 6).contiguous()
    if rays.shape[0] > renderer.max_batch_size:
        raise ValueError("too many rays")
    with torch.no_grad():
        origins, inverse = torch.unique(rays[:, :3], dim=0, return_inverse=True)
        starts = renderer._nearest_sites(scene.sites, origins)[inverse].to(torch.uint32)
    return _Quantiles.apply(scene.sites, scene.densities, topology.adjacency,
        topology.offsets, rays, starts, q,
        (renderer.near, renderer.far, renderer.transmittance_threshold, renderer.max_intersections))


@dataclass(frozen=True)
class MixedBatch:
    rays: object
    target: torch.Tensor
    alpha: torch.Tensor
    camera: object = None

class MixedRaySampler:
    """Uniform pixels with replacement across all TRAIN views.

    A single index tensor selects ray, RGB and alpha. Uses the checkpointed
    Torch RNG; no prefetch RNG state or independently shuffled target buffers.
    """
    def __init__(self, dataset, rays_per_batch, background):
        if rays_per_batch < 1 or not dataset.train:
            raise ValueError("mixed sampler requires rays and train cameras")
        self.dataset = dataset
        self.rays_per_batch = int(rays_per_batch)
        camera = dataset.train[0]
        self.width, self.height = camera.width, camera.height
        if any((c.width, c.height) != (self.width, self.height) for c in dataset.train):
            raise ValueError("mixed sampler requires equal image sizes")
        self.data = dataset.stacked_training_data()
        self.background = torch.as_tensor(background, device=dataset.device, dtype=torch.float32)
        self._camera_stack = []  # checkpoint compatibility; no stack used

    def batch_from_indices(self, indices):
        data = self.data
        rays = generate_indexed_rays(
            indices, data["origins"], data["rights"], data["downs"],
            data["forwards"], data["fov_x"], data["fov_y"], self.width, self.height, intrinsics=data.get("intrinsics"))
        alpha = data["alpha"][indices]
        target = data["foreground"][indices] + self.background * (1 - alpha)
        return MixedBatch(rays, target, alpha)

    def next(self):
        count = len(self.dataset.train) * self.width * self.height
        indices = torch.randint(count, (self.rays_per_batch,), device=self.dataset.device)
        return self.batch_from_indices(indices)


def stratified_box_sites(lo, hi, count):
    """Jittered grid with near-cubic cells: per-axis counts follow box extents.

    Grid cells are drawn without replacement and one site is placed uniformly
    inside each, so spacing is roughly (volume / count)^(1/3) on every axis.
    """
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    size = np.maximum(hi - lo, 1e-6)
    spacing = (np.prod(size) / count) ** (1 / 3)
    cells = np.maximum(np.ceil(size / spacing), 1).astype(np.int64)
    while np.prod(cells) < count:  # rounding can leave fewer cells than sites
        cells[np.argmax(size / cells)] += 1
    chosen = np.random.choice(int(np.prod(cells)), size=count, replace=False)
    index = np.stack(np.unravel_index(chosen, cells), axis=-1)
    unit = (index + np.random.random_sample((count, 3))) / cells
    return (lo + unit * size).astype(np.float32), cells.tolist(), float(spacing)


def outer_shell_sites(lo, hi, count, mode, radius, extent):
    """Sites strictly outside the box [lo, hi].

    radial: uniform directions from the box center, distance log-uniform from
    the box boundary to ``radius`` times that boundary distance, so cells grow
    with distance but cover view directions evenly.
    normal: the 'normal' Gaussian (std = extent * half box), rejecting samples
    that fall inside the box.
    """
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    center, half = (lo + hi) / 2, np.maximum((hi - lo) / 2, 1e-6)
    if mode == 'radial':
        direction = np.random.normal(size=(count, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        boundary = np.min(half / np.maximum(np.abs(direction), 1e-12), axis=1)
        scale = np.exp(np.random.random_sample(count) * np.log(radius))
        # Nudge off the boundary so every site is strictly outside the box.
        return (center + direction * (boundary * scale * (1 + 1e-6))[:, None]).astype(np.float32)
    kept = []
    total = 0
    for _ in range(1000):
        sample = center + np.random.normal(size=(4 * count, 3)) * half * extent
        sample = sample[np.any((sample < lo) | (sample > hi), axis=1)]
        kept.append(sample)
        total += len(sample)
        if total >= count:
            return np.concatenate(kept)[:count].astype(np.float32)
    raise ValueError('Normal outer sampling could not leave the box; increase spatial_extent')


def sample_spatial(points, cameras, count, fraction, extent, distribution='normal', margin=0.1,
                   inner_fraction=0.5, outer='radial', outer_radius=10.0):
    """Spatial Voronoi sites around the COLMAP (1-99%) and camera box.

    normal: axis-aligned Gaussian, std = extent * half box (unclipped).
    box: jittered grid filling the box enlarged by ``margin`` of its size per
    side; ``extent`` is unused.
    box_shell: ``inner_fraction`` of the sites as the box grid, the rest
    outside the enlarged box (see outer_shell_sites).
    """
    points = np.asarray(points, dtype=np.float32)
    cameras = np.asarray(cameras, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 5:
        raise ValueError('Need at least five COLMAP points')
    if cameras.ndim != 2 or cameras.shape[1] != 3 or len(cameras) == 0:
        raise ValueError('Need camera centers')
    if not np.isfinite(points).all() or not np.isfinite(cameras).all():
        raise ValueError('Nonfinite geometry')
    if count < 32 or not 0 <= fraction <= 1 or not np.isfinite(extent) or extent <= 0:
        raise ValueError('Invalid spatial initialization')
    lo = np.minimum(np.percentile(points,1,axis=0),cameras.min(axis=0))
    hi = np.maximum(np.percentile(points,99,axis=0),cameras.max(axis=0))
    if distribution not in ('normal','box','box_shell') or not np.isfinite(margin) or margin < 0:
        raise ValueError('Invalid spatial distribution')
    if distribution == 'box_shell' and (not 0 < inner_fraction < 1 or outer not in ('radial','normal')
                                        or not np.isfinite(outer_radius) or outer_radius <= 1):
        raise ValueError('Invalid box_shell settings')
    center = (lo+hi)/2
    n = int(count*fraction)
    selected = np.random.randint(len(points),size=n)
    jitter = max(float(np.linalg.norm(hi-lo))*1e-3,1e-5)
    near = points[selected] + np.random.normal(size=(n,3))*jitter
    report = dict(distribution=distribution,center=center.tolist(),colmap_count=n,
                  spatial_count=count-n,jitter=jitter,robust_lower=lo.tolist(),robust_upper=hi.tolist())
    if distribution == 'normal':
        # Axis-aligned Gaussian: one standard deviation is half the robust extent.
        # extent=1 and 2 give moderate/wide coverage, without clipping to this box.
        scale = np.maximum((hi-lo)/2,1e-3)*extent
        spatial = center + np.random.normal(size=(count-n,3))*scale
        report.update(std=scale.tolist())
    else:
        pad = (hi-lo)*margin
        inner = count-n if distribution == 'box' else int(round((count-n)*inner_fraction))
        spatial, cells, spacing = stratified_box_sites(lo-pad, hi+pad, inner)
        report.update(box_lower=(lo-pad).tolist(), box_upper=(hi+pad).tolist(), margin=margin,
                      grid_cells=cells, spacing=spacing, inner_count=inner)
        if distribution == 'box_shell':
            shell = outer_shell_sites(lo-pad, hi+pad, count-n-inner, outer, outer_radius, extent)
            spatial = np.concatenate((spatial, shell))
            report.update(outer=outer, outer_radius=outer_radius, outer_count=len(shell))
    sites = np.concatenate((near,spatial)).astype(np.float32)
    return sites, report

@torch.no_grad()
def initialize_spatial_medium(medium,dataset,config,profile):
    if dataset.__class__.__name__ != 'ColmapDataset':
        raise ValueError('Spatial initialization requires COLMAP')
    if config.model.medium_bbox_min is not None or config.model.medium_bbox_max is not None:
        raise ValueError('Spatial initialization must not have manual bounds')
    points,_ = dataset.initial_point_cloud()
    cameras = torch.stack([c.origin for c in dataset.train]).cpu().numpy()
    sites, report = sample_spatial(points,cameras,profile.spatial_count,
                                  profile.spatial_colmap_fraction,profile.spatial_extent,
                                  profile.spatial_distribution,profile.spatial_box_margin,
                                  profile.spatial_inner_fraction,profile.spatial_outer,
                                  profile.spatial_outer_radius)
    device=medium.sites.device
    medium.sites=nn.Parameter(torch.as_tensor(sites,device=device))
    # All are ordinary learnable medium cells, with identical density law.
    medium.raw_density=nn.Parameter(torch.rand(len(sites),1,device=device))
    medium.raw_color=nn.Parameter(torch.zeros(len(sites),3,device=device))
    medium.guard_mask=torch.zeros(len(sites),dtype=torch.bool,device=device)
    medium.bbox_min=medium.sites.detach().amin(0)
    medium.bbox_max=medium.sites.detach().amax(0)
    config.model.medium_points=len(sites)
    medium.initialization_report=report
