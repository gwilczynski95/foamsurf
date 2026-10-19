"""Opt-in joint diagnostics and complete adaptation resume for COLMAP runs."""
import json
import random
import numpy as np
import torch
from scipy.spatial import cKDTree
from training.trainer import Trainer
from training.checkpoints import checkpoint_payload, save_checkpoint, load_checkpoint
from dataclasses import replace
from pathlib import Path
from training.end_to_end import atomic_json
from training.disentanglement import undirected_edges
from training.losses import (SmoothConfig, smooth_loss, SurfaceConfig,
    CapacityLossConfig, ramp_weight, repulsion_loss, masked_direct_consistency,
    MediumFixConfig, drop_legacy_medium_fix, water_direction, backscatter_chroma_loss,
    surface_coverage_loss, under_surface_loss, water_clear_loss)
from training.surface_depth import nearest_sites, opaque_depth, sorted_hit_alphas
from training.surface import LocalOverlap
from training.surface import install_coverage

ACCUMULATORS=('contribution','visibility','path_length','weighted_path_length','max_error','gradient')

def map_birth_steps(birth,plan,iteration):
    return torch.cat((birth[plan.keep_indices],birth.new_full((plan.clone_count+2*plan.split_count,),iteration)))


class ColmapJointTrainer(Trainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.birth_steps=torch.zeros(self.model.gaussians.point_count,dtype=torch.long,device=self.device)
        self.joint_iteration=0
        if self.resume_path is not None:
            payload=load_checkpoint(self.resume_path)
            extra=payload.get('colmap_joint')
            if extra is None: raise ValueError('Missing COLMAP joint resume state')
            self.birth_steps=extra['birth_steps'].to(self.device)
            if len(self.birth_steps)!=self.model.gaussians.point_count: raise ValueError('Invalid lineage')
            perm=self.topology.last_permutation
            self.voronoi_adaptation.reorder_statistics(perm)
            for name,value in extra['medium_accumulators'].items():
                setattr(self.voronoi_adaptation,name,value.to(self.device)[perm].contiguous())
            self.sampler._camera_stack=list(extra['camera_stack'])
            random.setstate(extra['python_rng']);np.random.set_state(extra['numpy_rng'])
        points,_=self.dataset.initial_point_cloud()
        self.colmap_tree=cKDTree(points)
        controller=self.gaussian_adaptation
        original=controller.apply_plan
        def apply(plan):
            birth=self.birth_steps if plan.is_noop else map_birth_steps(self.birth_steps,plan,self.joint_iteration)
            result=original(plan)
            self.birth_steps=birth
            return result
        controller.apply_plan=apply

    def _train_step(self,iteration):
        self.joint_iteration=iteration
        return super()._train_step(iteration)

    def _checkpoint(self,path,iteration):
        payload=checkpoint_payload(self.model,self.gaussian_optimizer,self.medium_optimizer,
            self.config,iteration,self.training_seconds,self.best_psnr,self.topology.revision,
            self.gaussian_adaptation.state_dict(),self.voronoi_adaptation.state_dict(),
            self.medium_frozen_until,self.disentanglement.state_dict())
        payload['colmap_joint']=dict(birth_steps=self.birth_steps.detach().cpu(),
            medium_accumulators={n:getattr(self.voronoi_adaptation,n).detach().cpu() for n in ACCUMULATORS},
            camera_stack=list(self.sampler._camera_stack),python_rng=random.getstate(),numpy_rng=np.random.get_state())
        save_checkpoint(path,payload)

    def _evaluate_and_checkpoint(self,iteration):
        super()._evaluate_and_checkpoint(iteration)
        with torch.no_grad():
            g=self.model.gaussians
            distance=self.colmap_tree.query(g.means.detach().cpu().numpy(),workers=4)[0]
            born=self.birth_steps.cpu().numpy()>0
            opacity=g.opacities.flatten().cpu().numpy()
            report=dict(iteration=iteration)
            for name,mask in [('initial',~born),('spawned',born)]:
                report[name]=dict(count=int(mask.sum()),
                    distance_quantiles=np.quantile(distance[mask],[.5,.9,.99]).tolist() if mask.any() else [],
                    opacity_quantiles=np.quantile(opacity[mask],[.1,.5,.9]).tolist() if mask.any() else [])
            folder=self.output_dir/'colmap_diagnostics';folder.mkdir(exist_ok=True)
            (folder/f'iteration_{iteration:06d}.json').write_text(json.dumps(report,indent=2))



class SmoothJointTrainer(ColmapJointTrainer):
    def __init__(self,config,*args,smoothing=None,**kwargs):
        if smoothing is None:
            manifest=json.loads((Path(config.output_dir).parent/'pipeline_config.json').read_text())
            smoothing=manifest['rgb_smoothing']
            if manifest['warmup_steps']!=0:raise ValueError('Smoothing sweep has no warmup')
        self.smoothing=SmoothConfig(**smoothing)
        self.smooth_metrics={}
        self._smooth_edge_key=None
        self._smooth_edges=None
        super().__init__(config,*args,**kwargs)
        if not self.model.medium.is_rgb:raise ValueError('Smoothing trainer requires RGB medium')
        if self.resume_path is not None:
            saved=json.loads(Path(self.resume_path).with_suffix('.smooth.json').read_text())
            if saved!=self.smoothing.as_dict():raise ValueError('Resume smoothing settings mismatch')
        original=self.disentanglement.compute
        def compute(iteration,model,topology,*args,**kwargs):
            result=original(iteration,model,topology,*args,**kwargs)
            key=(topology.revision,topology.adjacency.data_ptr(),topology.offsets.data_ptr(),model.medium.point_count)
            if key!=self._smooth_edge_key:
                self._smooth_edges=undirected_edges(topology.adjacency,topology.offsets,model.medium.point_count)
                self._smooth_edge_key=key
            extra,parts=smooth_loss(model.medium.optical_coefficients,model.medium.colors,
                                   *self._smooth_edges,self.smoothing,iteration)
            self.smooth_metrics={f'rgb_smooth/{k}':float(v.detach()) if hasattr(v,'detach') else v for k,v in parts.items()}
            self.smooth_metrics['rgb_smooth/weighted']=float(extra.detach())
            return replace(result,total=result.total+extra)
        self.disentanglement.compute=compute

    def _training_row(self,*args):
        return {**super()._training_row(*args),**self.smooth_metrics}

    def _checkpoint(self,path,iteration):
        super()._checkpoint(path,iteration)
        atomic_json(Path(path).with_suffix('.smooth.json'),self.smoothing.as_dict())


class SurfaceTrainer(SmoothJointTrainer):
    def __init__(self, config, *args, surface=None, **kwargs):
        if surface is None:
            manifest = json.loads((Path(config.output_dir).parent/'pipeline_config.json').read_text())
            surface = manifest['rgb_surface']
            if manifest['warmup_steps'] != 0:
                raise ValueError('Surface sweep requires zero warmup')
        self.surface = SurfaceConfig(**surface)
        self.surface_metrics = {}
        self._captured_surface = None
        self._capture_surface = False
        self._overlap = LocalOverlap()
        super().__init__(config, *args, **kwargs)
        if self.resume_path is not None:
            if json.loads(Path(self.resume_path).with_suffix('.surface.json').read_text()) != self.surface.as_dict():
                raise ValueError('Resume overlap settings mismatch')
        native_render = self.renderer.render

        def capture(rays, *a, **kw):
            result = native_render(rays, *a, **kw)
            if self._capture_surface and self._captured_surface is None and kw.get('gaussians') is not None and kw.get('voronoi') is not None:
                self._captured_surface = (rays, kw['gaussians'])
            return result

        self.renderer.render = capture
        compute_base = self.disentanglement.compute

        def compute(iteration, model, topology, *a, **kw):
            result = compute_base(iteration, model, topology, *a, **kw)
            if self._captured_surface is None:
                raise RuntimeError('No hybrid training render captured')
            extra, logs = self._overlap.compute(native_render, *self._captured_surface,
                model.medium, topology, self.surface, iteration, self.scene_extent,
                self.config.renderer.far, self.config.renderer.near, self.config.seed)
            self.surface_metrics = {f'surface_overlap/{k}': v for k, v in logs.items()}
            return replace(result, total=result.total+extra)

        self.disentanglement.compute = compute
        install_coverage(self.evaluator)

    def _train_step(self, iteration):
        self._captured_surface = None
        self._capture_surface = True
        try:
            return super()._train_step(iteration)
        finally:
            self._capture_surface = False
            self._captured_surface = None

    def _training_row(self, *args):
        return {**super()._training_row(*args), **self.surface_metrics}

    def _checkpoint(self, path, iteration):
        super()._checkpoint(path, iteration)
        atomic_json(Path(path).with_suffix('.surface.json'), self.surface.as_dict())


class CapacityTrainer(SurfaceTrainer):
    def __init__(self, config, *args, capacity=None, **kwargs):
        if capacity is None:
            manifest = json.loads((Path(config.output_dir).parent / "pipeline_config.json").read_text())
            capacity = manifest["capacity_loss"]
        self.capacity = CapacityLossConfig(**capacity)
        self.capacity_metrics = {}
        self._captured_color = None
        self._edge_key = None
        self._edges = None
        super().__init__(config, *args, **kwargs)
        if self.resume_path is not None:
            saved = json.loads(Path(self.resume_path).with_suffix(".capacity.json").read_text())
            if saved != self.capacity.as_dict():
                raise ValueError("Resume capacity-loss settings mismatch")

        previous_render = self.renderer.render

        def capture(rays, *args, **kwargs):
            result = previous_render(rays, *args, **kwargs)
            if (self._capture_surface and self._captured_color is None
                    and kwargs.get("gaussians") is not None
                    and kwargs.get("voronoi") is not None):
                self._captured_color = (rays, kwargs["gaussians"], result)
            return result

        self.renderer.render = capture
        previous_compute = self.disentanglement.compute

        def compute(iteration, model, topology, *args, **kwargs):
            result = previous_compute(iteration, model, topology, *args, **kwargs)
            rep_weight = ramp_weight(self.capacity.repulsion_weight, iteration,
                                     self.capacity.repulsion_start, self.capacity.repulsion_ramp)
            direct_weight = ramp_weight(self.capacity.direct_weight, iteration,
                                        self.capacity.direct_start, self.capacity.direct_ramp)
            zero = model.medium.sites.sum() * 0
            rep_raw, close_fraction = zero, zero.detach()
            direct_raw, selected_fraction = zero, zero.detach()
            if rep_weight:
                key = (topology.revision, topology.adjacency.data_ptr(),
                       topology.offsets.data_ptr(), model.medium.point_count)
                if key != self._edge_key:
                    self._edges = undirected_edges(topology.adjacency, topology.offsets,
                                                   model.medium.point_count)
                    self._edge_key = key
                rep_raw, close_fraction = repulsion_loss(
                    model.medium.sites, model.medium.guard_mask, *self._edges,
                    model.medium.bbox_min, model.medium.bbox_max,
                    self.capacity.repulsion_fraction)
            if direct_weight:
                if self._captured_color is None:
                    raise RuntimeError("No hybrid training render captured")
                rays, scene, hybrid = self._captured_color
                if hybrid.direct is None:
                    raise RuntimeError("Direct component unavailable for RGB medium")
                # Geometry and opacity are detached; the auxiliary renderer
                # can differentiate only through scene.colors.
                color_scene = replace(
                    scene, means=scene.means.detach(),
                    log_scales=scene.log_scales.detach(),
                    quaternions=scene.quaternions.detach(),
                    opacity_logits=scene.opacity_logits.detach(),
                    raw_kappa=scene.raw_kappa.detach())
                only = self.renderer.render_gaussians(
                    rays, color_scene, background=(0.0, 0.0, 0.0))
                direct_raw, selected_fraction = masked_direct_consistency(
                    only.image, hybrid.direct, only.alpha, self.capacity.alpha_threshold)
            self.capacity_metrics = {
                "capacity/repulsion_raw": float(rep_raw.detach()),
                "capacity/repulsion_weighted": float((rep_weight * rep_raw).detach()),
                "capacity/short_edge_fraction": float(close_fraction),
                "capacity/direct_raw": float(direct_raw.detach()),
                "capacity/direct_weighted": float((direct_weight * direct_raw).detach()),
                "capacity/direct_selected_fraction": float(selected_fraction),
            }
            return replace(result, total=result.total + rep_weight * rep_raw
                           + direct_weight * direct_raw)

        self.disentanglement.compute = compute

    def _train_step(self, iteration):
        self._captured_color = None
        try:
            return super()._train_step(iteration)
        finally:
            self._captured_color = None

    def _training_row(self, *args):
        return {**super()._training_row(*args), **self.capacity_metrics}

    def _checkpoint(self, path, iteration):
        super()._checkpoint(path, iteration)
        atomic_json(Path(path).with_suffix(".capacity.json"), self.capacity.as_dict())


# Present only when the matching prior is enabled, so other runs keep their CSV columns.
SURFACE_METRICS = tuple("medium_fix/" + name for name in (
    "surface_coverage_weight", "surface_coverage_raw", "surface_coverage_ray_fraction",
    "under_surface_weight", "under_surface_raw", "under_surface_ray_fraction"))
WATER_METRICS = tuple("medium_fix/" + name for name in (
    "water_clear_weight", "water_clear_raw", "water_clear_ray_fraction"))
CHROMA_METRICS = tuple("medium_fix/" + name for name in (
    "chroma_weight", "chroma_raw", "chroma_angle_degrees",
    "water_direction_r", "water_direction_g", "water_direction_b"))


class MediumFixTrainer(CapacityTrainer):
    """Adds the seabed/backscatter priors of MediumFixConfig."""

    def __init__(self, config, *args, medium_fix=None, **kwargs):
        if medium_fix is None:
            manifest = json.loads((Path(config.output_dir).parent / "pipeline_config.json").read_text())
            medium_fix = manifest.get("medium_fix", {})
        self.medium_fix = MediumFixConfig(**drop_legacy_medium_fix(medium_fix))
        self._captured_prior = None
        self.water_ema = None
        self.chroma_metrics = ({name: float("nan") for name in CHROMA_METRICS}
                               if self.medium_fix.chroma_weight > 0 else {})
        self.surface_metrics_enabled = (self.medium_fix.surface_coverage_weight > 0
                                        or self.medium_fix.under_surface_weight > 0)
        self.seabed_metrics = ({name: float("nan") for name in SURFACE_METRICS}
                               if self.surface_metrics_enabled else {})
        self.water_metrics = ({name: float("nan") for name in WATER_METRICS}
                              if self.medium_fix.water_clear_weight > 0 else {})
        super().__init__(config, *args, **kwargs)
        if self.medium_fix.chroma_weight > 0 and not self.model.medium.is_rgb:
            raise ValueError("Backscatter chroma prior requires the RGB medium")
        if self.resume_path is not None:
            sidecar = Path(self.resume_path).with_suffix(".medium_fix.json")
            saved = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
            saved = {**MediumFixConfig().as_dict(), **drop_legacy_medium_fix(saved)}
            if saved != self.medium_fix.as_dict():
                raise ValueError("Resume medium_fix settings mismatch")
            state = Path(self.resume_path).with_suffix(".medium_fix_state.json")
            if state.is_file():
                water = json.loads(state.read_text()).get("water_ema")
                self.water_ema = None if water is None else torch.tensor(water, device=self.device)

        previous_render = self.renderer.render

        def capture(rays, *args, **kwargs):
            result = previous_render(rays, *args, **kwargs)
            if (self._capture_surface and self._captured_prior is None
                    and kwargs.get("gaussians") is not None
                    and kwargs.get("voronoi") is not None):
                self._captured_prior = (rays, kwargs["gaussians"], result)
            return result

        self.renderer.render = capture
        previous_compute = self.disentanglement.compute

        def compute(iteration, model, topology, *args, **kwargs):
            result = previous_compute(iteration, model, topology, *args, **kwargs)
            chroma = self.medium_fix.chroma_weight_at(iteration)
            if self.chroma_metrics:
                self.chroma_metrics = {name: float("nan") for name in CHROMA_METRICS}
                self.chroma_metrics["medium_fix/chroma_weight"] = chroma
            cover = self.medium_fix.surface_coverage_weight_at(iteration)
            under = self.medium_fix.under_surface_weight_at(iteration)
            if self.surface_metrics_enabled:
                self.seabed_metrics = {name: float("nan") for name in SURFACE_METRICS}
                self.seabed_metrics.update({"medium_fix/surface_coverage_weight": cover,
                                            "medium_fix/under_surface_weight": under})
            water = self.medium_fix.water_clear_weight_at(iteration)
            if self.water_metrics:
                self.water_metrics = {name: float("nan") for name in WATER_METRICS}
                self.water_metrics["medium_fix/water_clear_weight"] = water
            if not (chroma or cover or under or water):
                return result
            if self._captured_prior is None:
                raise RuntimeError("No hybrid training render captured")
            rays, scene, hybrid = self._captured_prior
            rays = rays.reshape(-1, 6)
            count = min(self.medium_fix.prior_rays, rays.shape[0])
            generator = torch.Generator(device=rays.device)
            generator.manual_seed(self.config.seed * 1_000_003 + iteration)
            selected = torch.randperm(rays.shape[0], device=rays.device, generator=generator)[:count]
            only = self.renderer.render_gaussians(
                rays[selected].contiguous(), scene, background=(0.0, 0.0, 0.0))
            total = result.total
            if chroma:
                if hybrid.backscatter is None:
                    raise RuntimeError("Backscatter component unavailable")
                backscatter = hybrid.backscatter.reshape(-1, 3)[selected]
                current = water_direction(backscatter, only.alpha, self.medium_fix.chroma_water_alpha)
                if current is not None:
                    decay = self.medium_fix.chroma_ema
                    blended = current if self.water_ema is None else decay * self.water_ema + (1 - decay) * current
                    self.water_ema = blended / blended.norm().clamp_min(1e-8)
                chroma_raw, angle = backscatter_chroma_loss(
                    backscatter, only.alpha, self.water_ema,
                    self.medium_fix.chroma_alpha_threshold, self.medium_fix.chroma_margin_degrees)
                total = total + chroma * chroma_raw
                self.chroma_metrics.update({"medium_fix/chroma_raw": float(chroma_raw.detach()),
                                            "medium_fix/chroma_angle_degrees": angle})
                if self.water_ema is not None:
                    for channel, value in zip("rgb", self.water_ema.tolist()):
                        self.chroma_metrics[f"medium_fix/water_direction_{channel}"] = value
            if cover:
                cover_raw, fraction = surface_coverage_loss(only.alpha, self.medium_fix.surface_coverage_threshold)
                total = total + cover * cover_raw
                self.seabed_metrics.update({"medium_fix/surface_coverage_raw": float(cover_raw.detach()),
                                            "medium_fix/surface_coverage_ray_fraction": fraction})
            if water:
                water_raw, fraction = water_clear_loss(only.alpha, self.medium_fix.water_clear_threshold)
                total = total + water * water_raw
                self.water_metrics.update({"medium_fix/water_clear_raw": float(water_raw.detach()),
                                           "medium_fix/water_clear_ray_fraction": fraction})
            if under:
                under_raw, fraction = self._under_surface(rays[selected], only, model)
                total = total + under * under_raw
                self.seabed_metrics.update({"medium_fix/under_surface_raw": float(under_raw.detach()),
                                            "medium_fix/under_surface_ray_fraction": fraction})
            return replace(result, total=total)

        self.disentanglement.compute = compute

    def _under_surface(self, rays, only, model):
        """Backscatter of the medium alone behind the opaque Gaussian surface."""
        with torch.no_grad():
            origins = rays[:, :3]
            directions = torch.nn.functional.normalize(rays[:, 3:], dim=-1)
            depth, alpha, _ = sorted_hit_alphas(model.gaussians, only.hit_depths, only.hit_indices, origins,
                                                directions, self.config.renderer.gaussian_radius_squared)
            surface, found = opaque_depth(depth, alpha, self.medium_fix.under_surface_threshold)
            start = surface[found] + self.medium_fix.under_surface_offset
            under_rays = torch.cat((origins[found] + start[:, None] * directions[found], directions[found]),
                                   dim=-1).contiguous()
        fraction = float(found.float().mean())
        if under_rays.shape[0] == 0:
            return model.medium.raw_density.sum() * 0, fraction
        medium = model.medium.render_scene()
        # Every under-surface ray has its own origin; the renderer's own start-cell
        # search would build a [rays, sites, 3] tensor (several GB), so do it in chunks.
        start_points = nearest_sites(under_rays[:, :3].detach(), medium.sites.detach()).to(torch.uint32)
        behind = self.renderer.render(under_rays, voronoi=medium, background=(0.0, 0.0, 0.0),
                                      start_points=start_points)
        if behind.backscatter is None:
            raise RuntimeError("Under-surface prior requires the RGB medium")
        return under_surface_loss(behind.backscatter), fraction

    def _train_step(self, iteration):
        self._captured_prior = None
        try:
            return super()._train_step(iteration)
        finally:
            self._captured_prior = None

    def _training_row(self, *args):
        return {**super()._training_row(*args), **self.chroma_metrics, **self.seabed_metrics,
                **self.water_metrics}

    def _checkpoint(self, path, iteration):
        super()._checkpoint(path, iteration)
        atomic_json(Path(path).with_suffix(".medium_fix.json"), self.medium_fix.as_dict())
        atomic_json(Path(path).with_suffix(".medium_fix_state.json"),
                    dict(water_ema=None if self.water_ema is None else self.water_ema.tolist()))
