"""Scene initialization only. The existing joint Trainer is not modified.

Inference/checkpoint model tensors use the original beta=1 format. The exact
warmup parameterization and optimizer state live in a separate payload field.
This also gives the existing evaluation/handoff code unchanged physical sigma.
"""
from dataclasses import replace
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
from .warmup_quantiles import render_quantiles
from .warmup_sampling import MixedRaySampler
from .warmup_population import WarmupPopulation
from foamsurf_data.rays import generate_indexed_rays


class WarmupMedium(VoronoiMediumModel):
    @property
    def densities(self):
        return self.warmup_scale * F.softplus(self.raw_density, beta=self.warmup_beta)

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
        super().__init__(config, resume=resume)
        medium = self.model.medium
        medium.__class__ = WarmupMedium
        medium.warmup_beta = profile.density_beta
        medium.warmup_scale = profile.activation_scale
        if payload is None:
            with torch.no_grad():
                if profile.site_initialization == "random":
                    medium.sites.copy_(medium.bbox_min + torch.rand_like(medium.sites) *
                                       (medium.bbox_max - medium.bbox_min))
                medium.raw_density.fill_(float(inverse_density(torch.tensor(profile.initial_sigma, dtype=torch.float64),
                    profile.density_beta, profile.activation_scale)))
                n = profile.outer_points
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
        payload["model"]["medium.raw_density"] = inverse_density(medium.densities.detach()).cpu()
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
                    data["fov_x"], data["fov_y"], camera.width, camera.height)
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
            for field, values in (("sigma", medium.densities.flatten()), ("raw", medium.raw_density.flatten())):
                selected = values.detach()[mask]
                for q in (.01, .05, .5, .95, .99):
                    row[f"warmup/{name}_{field}_p{round(100*q)}"] = float(torch.quantile(selected, q)) if selected.numel() else 0.0
        return row
