from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import random
import time

import numpy as np
import torch
from tqdm import trange

from evaluation import Evaluator
from evaluation.evaluator import _rays
from foamsurf import NativeHybridRenderer
from foamsurf_data import LayerGroundTruth, load_dataset
from foamsurf_data.camera_geometry import camera_normalization
from foamsurf_model import (
    GaussianAdaptationController,
    HybridSceneModel,
    SurfaceGaussianModel,
    VoronoiAdaptationController,
    VoronoiMediumModel,
    VoronoiTopologyController,
)
from foamsurf_model.gaussian_adaptation import scheduled_opacity_threshold
from .checkpoints import (
    checkpoint_payload,
    load_checkpoint,
    save_checkpoint,
)
from .config import ExperimentConfig
from .disentanglement import DisentanglementRegularizer
from .experiment_logging import ExperimentLogger
from .losses import b0_reconstruction_loss
from .optimizer import (
    build_optimizers,
    gradient_norm,
    update_gaussian_learning_rates,
    update_medium_learning_rates,
)
from .sampler import RandomCameraRaySampler
from .camera_isp import CameraISP, pixel_grid
from .water_extrapolation import WaterExtrapolation
from .water_phase import WaterPhase
from .oceansplat import DepthAwareOpacity, adjusted_scene, downscaled_target, ocean_terms
from .view_consistency import blended_depth, nerfstudio_scale, trinocular_loss


def _primitive_gradient_magnitude(module) -> torch.Tensor:
    squared = None
    for parameter in module.parameters():
        if parameter.grad is None or parameter.ndim == 0:
            continue
        value = parameter.grad.detach().reshape(parameter.shape[0], -1)
        value = value.square().sum(dim=1)
        squared = value if squared is None else squared + value
    if squared is None:
        reference = next(module.parameters())
        return reference.new_zeros((reference.shape[0],))
    return squared.sqrt()


def training_source(config: ExperimentConfig) -> str:
    paths = {
        "both": config.dataset.source_path,
        "lego": config.dataset.gaussian_ground_truth_path,
        "clouds": config.dataset.medium_ground_truth_path,
    }
    selected = paths[config.dataset.train_target]
    if selected is None:
        raise ValueError(f"missing path for train target {config.dataset.train_target}")
    return selected


def training_downscale(optimization, iteration: int) -> int:
    """1, or 2^k while coarse-to-fine training is still below full resolution."""
    levels = optimization.training_downscale_levels
    return 2 ** max(levels - iteration // optimization.training_downscale_interval, 0) if levels else 1


def gaussian_scale_cap(optimization, scene_extent: float) -> tuple[float, float] | None:
    """(min, max) Gaussian scale under gaussian_max_scale_scene_fraction, or None when it is off."""
    fraction = optimization.gaussian_max_scale_scene_fraction
    if fraction is None:
        return None
    return (optimization.gaussian_min_scale or 1e-7, fraction * scene_extent)


class Trainer:
    """End-to-end B0 trainer: joint optimization with RGB MSE only."""

    def __init__(
        self,
        config: ExperimentConfig,
        resume: str | Path | None = None,
        *,
        initial_model: HybridSceneModel | None = None,
    ) -> None:
        if resume is not None and initial_model is not None:
            raise ValueError("resume and initial_model are mutually exclusive")
        config.validate()
        if not torch.cuda.is_available():
            raise RuntimeError("FoamSurf B0 training requires CUDA")
        self.config = config
        self.uses_gaussians = config.representation_mode != "voronoi_only"
        self.uses_voronoi = config.representation_mode != "gaussian_only"
        self.device = torch.device(config.device)
        self.resume_path = Path(resume).resolve() if resume else None
        self._seed_everything(config.seed)

        self.output_dir = Path(config.output_dir).resolve()
        if self.resume_path is None:
            if self.output_dir.exists() and any(self.output_dir.iterdir()):
                raise FileExistsError(
                    f"output directory is not empty: {self.output_dir}"
                )
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        config.save(self.output_dir / "config.yaml")

        self.dataset = load_dataset(
            training_source(config),
            resolution=config.dataset.resolution,
            device=self.device,
            llffhold=config.dataset.llffhold,
            validation_hold=config.dataset.validation_hold,
        )
        self.layer_ground_truth = None
        if config.dataset.gaussian_ground_truth_path is not None:
            self.layer_ground_truth = LayerGroundTruth.load(
                self.dataset.test,
                config.dataset.gaussian_ground_truth_path,
                config.dataset.medium_ground_truth_path,
                resolution=config.dataset.resolution,
            )
            print(
                "Loaded aligned lego_lego and lego_clouds test ground truth "
                "on CPU for validation diagnostics.",
                flush=True,
            )
        self.camera_center, self.scene_extent = camera_normalization(
            self.dataset.train
        )
        self.model = initial_model if initial_model is not None else HybridSceneModel.initialize(
            self.dataset, config, self.device
        )
        self.start_iteration = 1
        self.training_seconds = 0.0
        self.best_psnr = -float("inf")
        payload = None

        if (
            self.resume_path is None
            and config.optimization.gaussian_sh_degree_interval > 0
            and self.model.gaussians.appearance_mode == "sh"
        ):
            self.model.gaussians.active_sh_degree = 0

        if self.resume_path is None:
            self.topology = VoronoiTopologyController(
                self.model.medium
            )
            (
                self.gaussian_optimizer,
                self.medium_optimizer,
            ) = build_optimizers(
                self.model, config.optimization, self.scene_extent
            )
        else:
            payload = load_checkpoint(
                self.resume_path
            )
            saved_mode = payload.get("config", {}).get("representation_mode", "hybrid")
            saved_target = payload.get("config", {}).get("dataset", {}).get("train_target", "both")
            if saved_mode != config.representation_mode or saved_target != config.dataset.train_target:
                raise ValueError(
                    "resume representation_mode and dataset.train_target must match the checkpoint"
                )
            gaussian_state = {
                name: payload["model"][f"gaussians.{name}"]
                for name in SurfaceGaussianModel.PARAMETER_NAMES
            }
            self.model.gaussians = SurfaceGaussianModel.from_checkpoint(
                gaussian_state,
                payload["model_metadata"]["gaussians"],
                self.device,
            )
            medium_state = {
                name: payload["model"][f"medium.{name}"]
                for name in VoronoiMediumModel.PARAMETER_NAMES
            }
            if "medium.guard_mask" not in payload["model"]:
                count = int(payload["model_metadata"]["medium"]["point_count"])
                guard_count = int(payload["model_metadata"]["medium"].get("guard_count", config.model.medium_guard_points))
                guard_mask = torch.zeros(count, dtype=torch.bool)
                if guard_count:
                    guard_mask[-guard_count:] = True
                payload["model"]["medium.guard_mask"] = guard_mask
            self.model.medium = VoronoiMediumModel.from_checkpoint(
                medium_state,
                payload["model_metadata"]["medium"],
                payload["model"]["medium.bbox_min"],
                payload["model"]["medium.bbox_max"],
                self.device,
            )
            if self.model.medium.medium_type != config.model.medium_type:
                raise ValueError("Cannot resume with a different medium_type")
            self.model.load_state_dict(payload["model"], strict=True)
            (
                self.gaussian_optimizer,
                self.medium_optimizer,
            ) = build_optimizers(
                self.model, config.optimization, self.scene_extent
            )
            self.gaussian_optimizer.load_state_dict(
                payload["gaussian_optimizer"]
            )
            self.medium_optimizer.load_state_dict(
                payload["medium_optimizer"]
            )
            self.topology = VoronoiTopologyController(
                self.model.medium, self.medium_optimizer
            )
            self.topology.revision = int(
                payload.get("topology_revision", 0)
            )
            self.start_iteration = int(payload["iteration"]) + 1
            self.training_seconds = float(
                payload["training_seconds"]
            )
            self.best_psnr = float(payload["best_psnr"])
            torch.set_rng_state(payload["rng"]["torch"])
            torch.cuda.set_rng_state_all(payload["rng"]["cuda"])

        self.model.medium.homogeneous = bool(config.model.medium_homogeneous)
        self.gaussian_point_extent = self._gaussian_point_extent()
        self._update_mip_filter()
        if self.resume_path is None and config.optimization.gaussian_max_scale_scene_fraction is not None:
            self._apply_gaussian_scale_guard()  # the cap holds from iteration 0, initial scales too
        if not self.uses_gaussians:
            self.model.gaussians.requires_grad_(False)
        if not self.uses_voronoi:
            self.model.medium.requires_grad_(False)

        self.gaussian_adaptation = GaussianAdaptationController(
            self.model.gaussians,
            self.gaussian_optimizer,
            config.gaussian_adaptation,
            self.scene_extent,
            camera_centers=torch.stack([camera.origin for camera in self.dataset.train]),
            image_width=self.dataset.train[0].width,
        )
        if payload is not None and payload.get("gaussian_adaptation") is not None:
            self.gaussian_adaptation.load_state_dict(
                payload["gaussian_adaptation"]
            )
        self.voronoi_adaptation = VoronoiAdaptationController(
            self.model.medium, self.medium_optimizer, self.topology,
            config.voronoi_adaptation,
        )
        if payload is not None and payload.get("voronoi_adaptation") is not None:
            self.voronoi_adaptation.load_state_dict(payload["voronoi_adaptation"])
        self.disentanglement = DisentanglementRegularizer(
            config.disentanglement
        )
        if payload is not None and payload.get("disentanglement") is not None:
            self.disentanglement.load_state_dict(payload["disentanglement"])
        self.last_voronoi_adaptation = {
            "densify": 0, "prune": 0, "skipped": 0
        }
        self.last_adaptation = {
            "clone": 0,
            "split": 0,
            "prune": 0,
            "budget_skip": 0,
        }
        self.last_opacity_reset = {"applied": 0, "changed": 0}
        self.medium_frozen_until = int(
            payload.get("medium_frozen_until", 0)
            if payload is not None
            else 0
        )
        self.population_runaway = False
        self.last_adaptation_pre_psnr = float("nan")
        self.last_adaptation_post_psnr = float("nan")
        self.last_adaptation_psnr_delta = float("nan")
        self.last_voronoi_adaptation_pre_psnr = float("nan")
        self.last_voronoi_adaptation_post_psnr = float("nan")
        self.last_voronoi_adaptation_psnr_delta = float("nan")
        self.last_reset_post_psnr = float("nan")
        self.last_reset_psnr_delta = float("nan")
        torch.cuda.reset_peak_memory_stats(self.device)

        maximum_rays = max(
            config.optimization.rays_per_batch,
            self.dataset.width * self.dataset.height,
        )
        self.renderer = NativeHybridRenderer(
            max_batch_size=maximum_rays,
            near=config.renderer.near,
            far=config.renderer.far,
            transmittance_threshold=(
                config.renderer.training_transmittance_threshold
            ),
            max_intersections=config.renderer.max_intersections,
            number_of_sides=config.renderer.number_of_sides,
            gaussian_radius_squared=(
                config.renderer.gaussian_radius_squared
            ),
            gaussian_hit_buffer_size=(
                config.renderer.gaussian_hit_buffer_size
            ),
        )
        first_origin = self.dataset.train[0].origin
        if self.uses_gaussians:
            self.renderer.set_gaussian_geometry(
                self.model.gaussians.render_scene(first_origin)
            )
        if self.uses_voronoi:
            self.topology.synchronize_renderer(
                self.renderer, self.model.medium
            )
        self.sampler = RandomCameraRaySampler(
            self.dataset,
            config.optimization.rays_per_batch,
            tuple(config.background),
            full_image=config.optimization.full_image_batch,
            cameras_without_replacement=(
                config.optimization.cameras_without_replacement
            ),
            jitter=config.optimization.ray_jitter,
        )
        self.evaluator = Evaluator(
            self.renderer,
            self.model,
            self.topology,
            self.dataset,
            config,
            self.output_dir,
            layer_ground_truth=self.layer_ground_truth,
        )
        self.logger = ExperimentLogger(
            config, self.output_dir, resume=self.resume_path is not None
        )
        self.last_trinocular = {"obj": float("nan"), "full": float("nan")}
        self.last_ocean = dict.fromkeys(("obj", "full", "smooth", "epi", "res", "alpha_weight"), float("nan"))
        # OceanSplat depth-aware opacity alpha^d: a small MLP with its own optimizer (checkpointed).
        self.camera_isp = None
        if config.optimization.camera_isp:
            self.camera_isp = CameraISP([camera.name for camera in self.dataset.train], config.optimization.iterations,
                                        self.device, torch.stack([camera.origin for camera in self.dataset.train]),
                                        config.optimization.camera_isp_neighbours)
            if payload is not None and payload.get("camera_isp") is not None:
                self.camera_isp.load_state_dict(payload["camera_isp"])
            self.evaluator.camera_isp = lambda image, camera: self.camera_isp.apply(
                image, pixel_grid(camera.height, camera.width, image.device).reshape(*image.shape[:-1], 2),
                (camera.height, camera.width), None, camera.origin)
        self.water_phase = None
        if config.optimization.water_phase:
            up = -torch.stack([camera.down for camera in self.dataset.train]).mean(0)  # the cameras' mean "up"
            self.water_phase = WaterPhase(up, self.device, config.optimization.water_phase_lr,
                                          config.optimization.water_phase_fixed_light)
            if payload is not None and payload.get("water_phase") is not None:
                self.water_phase.load_state_dict(payload["water_phase"])
        self.water_extrapolation = None  # fitted from the training photos before each evaluation (_wext)
        if self._water_modules() or config.optimization.water_extrapolation:
            self.evaluator.water_light = self._evaluation_water_light
        self.alpha_mlp = self.alpha_optimizer = None
        if config.optimization.ocean_alpha_until > 0:
            self.alpha_mlp = DepthAwareOpacity().to(self.device)
            self.alpha_optimizer = torch.optim.Adam(self.alpha_mlp.parameters(), lr=config.optimization.ocean_alpha_lr)
            if payload is not None and payload.get("alpha_mlp") is not None:
                self.alpha_mlp.load_state_dict(payload["alpha_mlp"])
                self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        # OceanSplat baseline in scene units: b (nerfstudio units) / the nerfstudio pose scale.
        self.nerfstudio_scale = nerfstudio_scale(
            torch.stack([camera.origin for camera in self.dataset.train + self.dataset.test]))
        self.trinocular_baseline = config.optimization.trinocular_baseline / self.nerfstudio_scale
        self.last_hit_statistics = {
            "renderer/hit_buffer_size": config.renderer.gaussian_hit_buffer_size,
            "renderer/hit_count_mean": 0.0,
            "renderer/hit_count_p90": 0.0,
            "renderer/hit_count_p99": 0.0,
            "renderer/hit_buffer_full_fraction": 0.0,
        }

    @torch.no_grad()
    def _water_attenuation(self) -> torch.Tensor:
        """Median attenuation per channel over the non-guard Voronoi cells."""
        medium = self.model.medium
        coefficients = medium.optical_coefficients
        attenuation = coefficients[:, :3] if medium.is_rgb else coefficients
        return attenuation[~medium.guard_mask].median(dim=0).values

    def _seed_everything(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _checkpoint(self, path: Path, iteration: int) -> None:
        save_checkpoint(
            path,
            checkpoint_payload(
                self.model,
                self.gaussian_optimizer,
                self.medium_optimizer,
                self.config,
                iteration,
                self.training_seconds,
                self.best_psnr,
                self.topology.revision,
                self.gaussian_adaptation.state_dict(),
                self.voronoi_adaptation.state_dict(),
                self.medium_frozen_until,
                self.disentanglement.state_dict(),
                extras=self._checkpoint_extras(),
            ),
        )

    def _checkpoint_extras(self) -> dict | None:
        """Training state outside the model and its two optimizers (OceanSplat alpha^d, camera ISP)."""
        extras = {}
        if self.alpha_mlp is not None:
            extras.update(alpha_mlp=self.alpha_mlp.state_dict(), alpha_optimizer=self.alpha_optimizer.state_dict())
        if self.camera_isp is not None:
            extras["camera_isp"] = self.camera_isp.state_dict()
        if self.water_phase is not None:
            extras["water_phase"] = self.water_phase.state_dict()
        return extras or None

    def _water_modules(self) -> list:
        """The direction-dependent water light terms in use (the phase function)."""
        return [m for m in (self.water_phase,) if m is not None]

    def _water_light_parts(self, image: torch.Tensor, backscatter: torch.Tensor, rays: torch.Tensor) -> torch.Tensor:
        """image + (gain - 1) backscatter with the product of the gains of every water light term."""
        gain = 1.0
        for module in self._water_modules():
            gain = gain * module.gain(rays[..., 3:])
        return image + (gain - 1.0) * backscatter

    def _evaluation_water_light(self, image: torch.Tensor, backscatter: torch.Tensor, rays: torch.Tensor) -> torch.Tensor:
        """The evaluator's image: the trained water light terms, then (with _wext) the water light continued beyond
        the training directions, applied to the backscatter as those terms left it."""
        modules = bool(self._water_modules())
        out = self._water_light_parts(image, backscatter, rays) if modules else image
        if self.water_extrapolation is not None:
            effective = self._water_light_parts(backscatter, backscatter, rays) if modules else backscatter
            out = self.water_extrapolation.apply(out, effective, rays[..., 3:])
        return out

    def fit_water_extrapolation(self, step: int = 4) -> WaterExtrapolation | None:
        """Fit the water light trend on the open water of the training photos (rays where the trained Gaussians
        cover < 2%, every `step`-th pixel), as training/water_extrapolation.py describes."""
        background = tuple(self.config.background)
        directions, colours, photo = [], [], []
        with torch.no_grad():
            for index, camera in enumerate(self.dataset.train):
                rays = _rays(camera)[::step, ::step].reshape(-1, 6).contiguous()
                scene = self.model.gaussians.render_scene(camera.origin)
                self.renderer.set_gaussian_geometry(scene)
                alpha = self.renderer.render_gaussians(rays, scene, background=background).alpha.reshape(-1)
                target = camera.ground_truth(background).reshape(camera.height, camera.width, 3)[::step, ::step].reshape(-1, 3)
                water = (alpha < 0.02) & (target.min(-1).values > 0.01)
                directions.append(rays[water, 3:]); colours.append(target[water])
                photo.append(torch.full((int(water.sum()),), index, device=rays.device, dtype=torch.long))
        directions = torch.cat(directions)
        if directions.shape[0] < 100:  # no open water: nothing to continue
            self.water_extrapolation = None
            return None
        up = -torch.stack([camera.down for camera in self.dataset.train]).mean(0)
        self.water_extrapolation = WaterExtrapolation.fit(up.to(directions), directions, torch.cat(colours), torch.cat(photo))
        return self.water_extrapolation

    def _water_light(self, render, rays: torch.Tensor) -> torch.Tensor:
        """The rendered image with direction-dependent water light (unchanged without these terms)."""
        if not self._water_modules() or render.backscatter is None:
            return render.image
        return self._water_light_parts(render.image, render.backscatter, rays)

    def _apply_camera_isp(self, image: torch.Tensor, batch) -> torch.Tensor:
        """The training image through the camera model of its photo (identity without camera_isp)."""
        if self.camera_isp is None:
            return image
        camera = batch.camera
        if batch.image_size is not None:  # whole image at 1/downscale (resolution schedule)
            height, width = batch.image_size
            coords = pixel_grid(height, width, image.device)
        else:
            height, width = camera.height, camera.width
            index = batch.rays.pixel_indices
            coords = torch.stack((index % width, torch.div(index, width, rounding_mode="floor")), -1).float() + 0.5
        return self.camera_isp.apply(image, coords, (height, width), camera.name)

    def _training_row(
        self,
        iteration: int,
        loss,
        gaussian_gradient_norm: float,
        medium_gradient_norm: float,
        step_seconds: float,
    ) -> dict[str, float | int]:
        adaptation_statistics = (
            self.gaussian_adaptation.last_plan_statistics
        )
        row = {
            "iteration": iteration,
            "training_seconds": self.training_seconds,
            "step_seconds": step_seconds,
            "train/loss": float(loss.total.item()),
            "train/rgb_mse": float(loss.rgb_mse.item()),
            "train/rgb_l1": float(loss.rgb_l1.item()),
            "train/ssim": float(loss.ssim.item()),
            "train/psnr": float(loss.psnr.item()),
            "train/gaussian_scale_regularization": float(
                loss.gaussian_scale_regularization.item()
            ),
            "gradient/gaussian_norm": gaussian_gradient_norm,
            "gradient/medium_norm": medium_gradient_norm,
            "optimization/medium_sites_lr": self.current_medium_lrs[
                "medium_sites"
            ],
            "optimization/medium_density_lr": self.current_medium_lrs[
                "medium_density"
            ],
            "optimization/medium_color_lr": self.current_medium_lrs[
                "medium_color"
            ],
            "gaussian/opacity_mean": float(
                self.model.gaussians.opacities.mean().item()
            ),
            "gaussian/opacity_min": float(
                self.model.gaussians.opacities.min().item()
            ),
            "gaussian/opacity_max": float(
                self.model.gaussians.opacities.max().item()
            ),
            "gaussian/scale_mean": float(
                self.model.gaussians.scales.mean().item()
            ),
            "medium/density_mean": float(
                self.model.medium.densities.mean().item()
            ),
            "medium/density_min": float(
                self.model.medium.densities.min().item()
            ),
            "medium/density_max": float(
                self.model.medium.densities.max().item()
            ),
            "medium/color_mean": float(
                self.model.medium.colors.mean().item()
            ),
            "model/gaussian_count": self.model.gaussians.point_count,
            "model/voronoi_count": self.model.medium.point_count,
            "topology/revision": self.topology.revision,
            "voronoi_adaptation/enabled": int(self.config.voronoi_adaptation.enabled),
            "voronoi_adaptation/densified": self.last_voronoi_adaptation["densify"],
            "voronoi_adaptation/pruned": self.last_voronoi_adaptation["prune"],
            "voronoi_adaptation/skipped": self.last_voronoi_adaptation["skipped"],
            "voronoi_adaptation/total_failed_events": (
                self.voronoi_adaptation.total_failed_events
            ),
            "voronoi_adaptation/total_densified": self.voronoi_adaptation.total_densified,
            "voronoi_adaptation/total_pruned": self.voronoi_adaptation.total_pruned,
            "voronoi_adaptation/revision": self.voronoi_adaptation.revision,
            "voronoi_adaptation/pre_psnr": self.last_voronoi_adaptation_pre_psnr,
            "voronoi_adaptation/post_psnr": self.last_voronoi_adaptation_post_psnr,
            "voronoi_adaptation/psnr_delta": self.last_voronoi_adaptation_psnr_delta,
            "adaptation/enabled": int(
                self.config.gaussian_adaptation.enabled
            ),
            "adaptation/cloned": self.last_adaptation["clone"],
            "adaptation/split": self.last_adaptation["split"],
            "adaptation/pruned": self.last_adaptation["prune"],
            "adaptation/total_cloned": self.gaussian_adaptation.total_cloned,
            "adaptation/total_split": self.gaussian_adaptation.total_split,
            "adaptation/total_pruned": self.gaussian_adaptation.total_pruned,
            "adaptation/budget_skip": self.last_adaptation["budget_skip"],
            "adaptation/total_budget_skips": (
                self.gaussian_adaptation.total_budget_skips
            ),
            "adaptation/revision": self.gaussian_adaptation.revision,
            "adaptation/pre_psnr": self.last_adaptation_pre_psnr,
            "adaptation/post_psnr": self.last_adaptation_post_psnr,
            "adaptation/psnr_delta": self.last_adaptation_psnr_delta,
            "opacity_reset/applied": self.last_opacity_reset["applied"],
            "opacity_reset/changed": self.last_opacity_reset["changed"],
            "opacity_reset/total": (
                self.gaussian_adaptation.total_opacity_resets
            ),
            "opacity_reset/post_psnr": self.last_reset_post_psnr,
            "opacity_reset/psnr_delta": self.last_reset_psnr_delta,
            "gaussian/frozen": int(
                iteration <= self.config.optimization.gaussian_freeze_until
            ),
            "gaussian/frozen_until": (
                self.config.optimization.gaussian_freeze_until
            ),
            "medium/frozen": int(iteration < self.medium_frozen_until),
            "medium/frozen_until": self.medium_frozen_until,
            "safety/population_runaway": int(self.population_runaway),
            "adaptation/gradient_p50": adaptation_statistics[
                "gradient_p50"
            ],
            "adaptation/gradient_p95": adaptation_statistics[
                "gradient_p95"
            ],
            "adaptation/gradient_max": adaptation_statistics[
                "gradient_max"
            ],
            "adaptation/gradient_candidates": adaptation_statistics[
                "gradient_candidates"
            ],
            "adaptation/clone_candidates": adaptation_statistics[
                "clone_candidates"
            ],
            "adaptation/split_candidates": adaptation_statistics[
                "split_candidates"
            ],
            "adaptation/prune_candidates": adaptation_statistics[
                "prune_candidates"
            ],
            "adaptation/opacity_prune_candidates": adaptation_statistics[
                "opacity_prune"
            ],
            "adaptation/opacity_prune_raw": adaptation_statistics[
                "opacity_prune_raw"
            ],
            "adaptation/opacity_prune_rejected": adaptation_statistics[
                "opacity_prune_rejected"
            ],
            "adaptation/opacity_threshold": (
                self.gaussian_adaptation.config.min_opacity
                if self.gaussian_adaptation.config.policy != "graphdeco"
                else scheduled_opacity_threshold(
                    self.gaussian_adaptation.config, iteration
                )
            ),
            "adaptation/densification_due": int(
                self.gaussian_adaptation.should_densify(iteration)
            ),
            "adaptation/opacity_pruning_due": int(
                self.gaussian_adaptation.should_prune(iteration)
            ),
            "adaptation/screen_prune_candidates": adaptation_statistics[
                "screen_prune"
            ],
            "adaptation/world_prune_candidates": adaptation_statistics[
                "world_prune"
            ],
            "adaptation/gradient_opacity_overlap": adaptation_statistics[
                "gradient_opacity_overlap"
            ],
            "adaptation/gradient_screen_overlap": adaptation_statistics[
                "gradient_screen_overlap"
            ],
            "adaptation/gradient_world_overlap": adaptation_statistics[
                "gradient_world_overlap"
            ],
            "adaptation/budget_rejected": adaptation_statistics[
                "budget_rejected"
            ],
            "adaptation/screen_replacements": adaptation_statistics[
                "screen_replacements"
            ],
            "adaptation/observed_fraction": float(
                (self.gaussian_adaptation.gradient_denom > 0)
                .float()
                .mean()
                .item()
            ),
            "adaptation/radius_p50": adaptation_statistics["radius_p50"],
            "adaptation/radius_p95": adaptation_statistics["radius_p95"],
            "adaptation/radius_p99": adaptation_statistics["radius_p99"],
            "adaptation/max_radius2d": adaptation_statistics["radius_max"],
            "scene/cameras_extent": self.gaussian_adaptation.scene_extent,
            "optimization/gaussian_means_lr": self.current_gaussian_means_lr,
            "model/active_sh_degree": self.model.gaussians.active_sh_degree,
            "train/trinocular_obj": self.last_trinocular["obj"],
            "train/trinocular_full": self.last_trinocular["full"],
            **{f"ocean/{name}": value for name, value in self.last_ocean.items()},
            "memory/allocated_mib": torch.cuda.memory_allocated(
                self.device
            ) / (1024.0**2),
            "memory/reserved_mib": torch.cuda.memory_reserved(
                self.device
            ) / (1024.0**2),
            "memory/peak_allocated_mib": torch.cuda.max_memory_allocated(
                self.device
            ) / (1024.0**2),
        }
        for name, value in self.voronoi_adaptation.last_statistics.items():
            row[f"voronoi_adaptation/{name}"] = value
        gaussian_opacity = self.model.gaussians.opacities.detach().flatten()
        gaussian_scale = self.model.gaussians.scales.detach().flatten()
        medium_density = self.model.medium.densities.detach().flatten()
        gaussian_grad = _primitive_gradient_magnitude(self.model.gaussians)
        medium_grad = _primitive_gradient_magnitude(self.model.medium)
        row.update(
            {
                "gaussian/opacity_p50": float(torch.quantile(gaussian_opacity, 0.5).item()),
                "gaussian/opacity_p95": float(torch.quantile(gaussian_opacity, 0.95).item()),
                "gaussian/opacity_lt_0_01_fraction": float((gaussian_opacity < 0.01).float().mean().item()),
                "gaussian/opacity_lt_0_1_fraction": float((gaussian_opacity < 0.1).float().mean().item()),
                "gaussian/opacity_gt_0_1_fraction": float((gaussian_opacity > 0.1).float().mean().item()),
                "gaussian/opacity_gt_0_5_fraction": float((gaussian_opacity > 0.5).float().mean().item()),
                "gaussian/opacity_gt_0_9_fraction": float((gaussian_opacity > 0.9).float().mean().item()),
                "gaussian/scale_p50": float(torch.quantile(gaussian_scale, 0.5).item()),
                "gaussian/scale_p95": float(torch.quantile(gaussian_scale, 0.95).item()),
                "gaussian/scale_max": float(gaussian_scale.max().item()),
                "medium/density_p50": float(torch.quantile(medium_density, 0.5).item()),
                "medium/density_p95": float(torch.quantile(medium_density, 0.95).item()),
                "medium/density_p99": float(torch.quantile(medium_density, 0.99).item()),
                "medium/density_gt_1_fraction": float((medium_density > 1.0).float().mean().item()),
                "gradient/gaussian_primitive_p50": float(torch.quantile(gaussian_grad, 0.5).item()),
                "gradient/gaussian_primitive_p95": float(torch.quantile(gaussian_grad, 0.95).item()),
                "gradient/gaussian_primitive_max": float(gaussian_grad.max().item()),
                "gradient/gaussian_active_fraction": float((gaussian_grad > 0).float().mean().item()),
                "gradient/medium_primitive_p50": float(torch.quantile(medium_grad, 0.5).item()),
                "gradient/medium_primitive_p95": float(torch.quantile(medium_grad, 0.95).item()),
                "gradient/medium_primitive_max": float(medium_grad.max().item()),
                "gradient/medium_active_fraction": float((medium_grad > 0).float().mean().item()),
            }
        )
        row.update(self.last_hit_statistics)
        row.update(self.last_disentanglement.metrics())
        if self.config.gaussian_adaptation.far_prune_distance is not None:
            # Only when enabled: CSV columns of resumed runs must not change.
            row["adaptation/far_prune_candidates"] = adaptation_statistics["far_prune"]
        return row

    @torch.no_grad()
    def _alpha_weight(self, iteration: int) -> float:
        """OceanSplat alpha^d weight: ocean_alpha_weight decaying linearly to 0 at ocean_alpha_until."""
        until = self.config.optimization.ocean_alpha_until
        if self.alpha_mlp is None or until <= 0:
            return 0.0
        return self.config.optimization.ocean_alpha_weight * max(0.0, 1.0 - iteration / until)

    def _depth_aware(self, scene, origin, forward, iteration: int):
        return adjusted_scene(scene, self.alpha_mlp, self._alpha_weight(iteration), origin.to(scene.means),
                              forward.to(scene.means), self.scene_extent)

    def _ocean_term(self, iteration, batch, medium_scene):
        """Full OceanSplat losses for this training camera, or None when inactive."""
        optimization = self.config.optimization
        self.last_ocean = dict.fromkeys(self.last_ocean, float("nan"))
        self.last_ocean["alpha_weight"] = self._alpha_weight(iteration)
        if not optimization.ocean_from <= iteration < optimization.ocean_until or medium_scene is None \
                or not self.uses_gaussians or (iteration - optimization.ocean_from) % optimization.ocean_every:
            return None
        span = max(optimization.ocean_until - optimization.ocean_from, 1)
        progress = (iteration - optimization.ocean_from) / span
        epi_weight = optimization.ocean_epi_weight_start + progress * (
            optimization.ocean_epi_weight_end - optimization.ocean_epi_weight_start)
        if optimization.ocean_tri_weight <= 0 and optimization.ocean_res_weight <= 0 and epi_weight <= 0:
            return None
        camera = batch.camera
        terms = ocean_terms(
            self.renderer,
            lambda origin: self._depth_aware(self.model.gaussians.render_scene(origin), origin, camera.forward, iteration),
            medium_scene, self.config.background, camera,
            downscaled_target(camera, optimization.ocean_downscale, self.config.background),
            self.trinocular_baseline, optimization.ocean_downscale, optimization.trinocular_epsilon,
            optimization.ocean_smooth_gamma, optimization.ocean_gaussian_samples,
            depth_unit=self.nerfstudio_scale, visible=optimization.ocean_visible,
            camera_model=None if self.camera_isp is None else (
                lambda image: self.camera_isp.apply(
                    image, pixel_grid(*image.shape[:2], image.device).reshape(*image.shape[:2], 2),
                    image.shape[:2], camera.name)))
        for name in ("obj", "full", "smooth", "epi", "res"):
            self.last_ocean[name] = float(terms[name].detach())
        return optimization.ocean_every * (
            optimization.ocean_tri_weight * (terms["obj"] + terms["full"] + terms["smooth"])
            + epi_weight * terms["epi"] + optimization.ocean_res_weight * terms["res"])

    def _trinocular_term(self, iteration, batch, rays, render, gaussian_scene, medium_scene):
        """lambda_tri x OceanSplat L_tri on a random subset of the batch rays, or None when inactive."""
        optimization = self.config.optimization
        self.last_trinocular = {"obj": float("nan"), "full": float("nan")}
        if (optimization.trinocular_weight <= 0.0 or not optimization.trinocular_from <= iteration < optimization.trinocular_until
                or gaussian_scene is None or medium_scene is None or render.direct is None):
            return None
        count = rays.shape[0]
        subset = torch.randperm(count, device=rays.device)[: optimization.trinocular_rays]
        depth, accumulated = blended_depth(gaussian_scene, rays[subset], render.hit_depths[:, subset],
                                           render.hit_indices[:, subset])
        keep = accumulated > 0.5  # rays that end on a surface
        if not bool(keep.any()):
            return None
        subset, depth = subset[keep], depth[keep]
        direct = render.direct.reshape(-1, 3)[subset]
        backscatter = render.backscatter.reshape(-1, 3)[subset]
        term, values = trinocular_loss(
            self.renderer, self.model.gaussians, medium_scene, tuple(self.config.background), batch.camera,
            rays[subset], depth, direct, backscatter, batch.target.reshape(-1, 3)[subset],
            self.trinocular_baseline, optimization.trinocular_epsilon)
        self.last_trinocular = values
        return optimization.trinocular_weight * term

    def _same_batch_psnr(self, batch, rays: torch.Tensor) -> float:
        gaussian_scene = None
        medium_scene = None
        if self.uses_gaussians:
            gaussian_scene = self.model.gaussians.render_scene(batch.camera_origin)
            self.renderer.set_gaussian_geometry(gaussian_scene)
        if self.uses_voronoi:
            medium_scene = self.model.medium.render_scene()
        render = self.renderer.render(
            rays, gaussians=gaussian_scene, voronoi=medium_scene,
            background=tuple(self.config.background),
            collect_diagnostics=False,
        )
        mse = torch.mean((render.image - batch.target) ** 2)
        return float((-10.0 * torch.log10(mse.clamp_min(1.0e-12))).item())

    def _schedule_medium_freeze(self, iteration: int, steps: int) -> None:
        if steps > 0:
            self.medium_frozen_until = max(
                self.medium_frozen_until, iteration + steps
            )

    def _independent_medium_freeze_event(self, iteration: int) -> bool:
        config = self.config.gaussian_adaptation
        return (
            config.medium_freeze_interval > 0
            and iteration < config.densify_until
            and iteration >= config.medium_freeze_from
            and (iteration - config.medium_freeze_from)
            % config.medium_freeze_interval
            == 0
        )

    def _population_exceeds_safety_threshold(self) -> bool:
        threshold = self.config.gaussian_adaptation.runaway_max_points
        return (
            threshold is not None
            and self.model.gaussians.point_count > threshold
        )

    def _update_mip_filter(self) -> None:
        """Recompute the Mip-Splatting 3D filter from the training cameras (after every change of the
        Gaussian population and every 100 iterations, as the paper)."""
        if self.uses_gaussians and self.config.model.gaussian_mip_filter > 0:
            self.model.gaussians.mip_compensate = self.config.model.gaussian_mip_compensate
            self.model.gaussians.update_mip_filter(self.dataset.train, self.config.model.gaussian_mip_filter,
                                                   self.config.renderer.near)

    def _train_step(self, iteration: int):
        if iteration % 100 == 0:
            self._update_mip_filter()
        gaussian_trainable = (
            self.uses_gaussians
            and iteration > self.config.optimization.gaussian_freeze_until
        )
        if self.uses_gaussians:
            self.current_gaussian_means_lr = update_gaussian_learning_rates(
                self.gaussian_optimizer, self.config.optimization,
                self.scene_extent, iteration,
            )
            sh_interval = self.config.optimization.gaussian_sh_degree_interval
            if sh_interval > 0 and iteration % sh_interval == 0:
                self.model.gaussians.one_up_sh_degree()
        else:
            self.current_gaussian_means_lr = 0.0

        if self.uses_voronoi:
            self.current_medium_lrs = update_medium_learning_rates(
                self.medium_optimizer, self.config.optimization, iteration
            )
        else:
            self.current_medium_lrs = {
                "medium_sites": 0.0,
                "medium_density": 0.0,
                "medium_color": 0.0,
            }

        torch.cuda.synchronize()
        started = time.perf_counter()
        self.sampler.downscale = training_downscale(self.config.optimization, iteration)
        batch = self.sampler.next()
        collect_gaussian = (
            gaussian_trainable
            and self.gaussian_adaptation.should_collect(iteration)
        )
        collect_voronoi = (
            self.uses_voronoi
            and self.voronoi_adaptation.should_collect(iteration)
        )
        collect_any = collect_gaussian or collect_voronoi

        gaussian_scene = None
        screen_points = None
        if self.uses_gaussians:
            if collect_gaussian:
                gaussian_scene, screen_points = self.model.gaussians.training_scene(batch.camera)
            else:
                gaussian_scene = self.model.gaussians.render_scene(batch.camera_origin)
            gaussian_scene = self._depth_aware(gaussian_scene, batch.camera.origin, batch.camera.forward, iteration)
            self.renderer.set_gaussian_geometry(gaussian_scene)
        medium_scene = self.model.medium.render_scene() if self.uses_voronoi else None
        rays = torch.cat((batch.rays.origins, batch.rays.directions), dim=-1).contiguous()

        screen_absolute = None
        if collect_gaussian and self.config.gaussian_adaptation.absolute_gradient:
            camera = batch.camera
            screen_absolute = (
                torch.stack((camera.right, camera.down)).to(rays),
                rays.new_zeros((self.model.gaussians.point_count, 2)),
            )

        self.gaussian_optimizer.zero_grad(set_to_none=True)
        self.medium_optimizer.zero_grad(set_to_none=True)
        render = self.renderer.render(
            rays, gaussians=gaussian_scene, voronoi=medium_scene,
            background=tuple(self.config.background),
            collect_diagnostics=collect_any,
            diagnostic_target=batch.target if collect_any else None,
            screen_absolute=screen_absolute,
        )
        if (
            render.hit_indices.numel()
            and (
                iteration == self.start_iteration
                or iteration % self.config.validation.log_interval == 0
                or iteration == self.config.optimization.iterations
            )
        ):
            with torch.no_grad():
                hit_counts = (render.hit_indices >= 0).sum(dim=0).float()
                self.last_hit_statistics = {
                    "renderer/hit_buffer_size": (
                        self.config.renderer.gaussian_hit_buffer_size
                    ),
                    "renderer/hit_count_mean": float(hit_counts.mean().item()),
                    "renderer/hit_count_p90": float(
                        torch.quantile(hit_counts, 0.90).item()
                    ),
                    "renderer/hit_count_p99": float(
                        torch.quantile(hit_counts, 0.99).item()
                    ),
                    "renderer/hit_buffer_full_fraction": float(
                        (hit_counts >= render.hit_indices.size(0))
                        .float()
                        .mean()
                        .item()
                    ),
                }
        loss = b0_reconstruction_loss(
            self._apply_camera_isp(self._water_light(render, rays), batch), batch.target,
            mode=self.config.optimization.reconstruction_loss,
            dssim_weight=self.config.optimization.dssim_weight,
            image_size=batch.image_size or (batch.camera.height, batch.camera.width),
        )
        scale_weight = self.config.optimization.gaussian_scale_regularization
        if self.uses_gaussians and scale_weight > 0.0:
            scale_regularization = (
                scale_weight
                / self.model.gaussians.point_count
                * torch.linalg.vector_norm(
                    self.model.gaussians.scales, dim=1
                ).sum()
            )
            loss = replace(
                loss,
                total=loss.total + scale_regularization,
                gaussian_scale_regularization=scale_regularization,
            )

        self.last_disentanglement = self.disentanglement.compute(
            iteration, self.model, self.topology
        )
        loss = replace(
            loss,
            total=loss.total + self.last_disentanglement.total,
        )
        trinocular = self._trinocular_term(iteration, batch, rays, render, gaussian_scene, medium_scene)
        if trinocular is not None:
            loss = replace(loss, total=loss.total + trinocular)
        ocean = self._ocean_term(iteration, batch, medium_scene)
        if ocean is not None:
            loss = replace(loss, total=loss.total + ocean)
        if self.camera_isp is not None:
            loss = replace(loss, total=loss.total + self.camera_isp.regularization())
            self.camera_isp.zero_grad()
        for module in self._water_modules():
            module.zero_grad()
        if self.alpha_optimizer is not None:
            self.alpha_optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        if self.alpha_optimizer is not None and self._alpha_weight(iteration) > 0.0:
            self.alpha_optimizer.step()
        if self.camera_isp is not None:
            self.camera_isp.step()
        for module in self._water_modules():
            module.step()
        if collect_any and render.diagnostics is None:
            raise RuntimeError("adaptation statistics were not returned")
        if collect_gaussian:
            if self.config.gaussian_adaptation.gradient_attenuation_compensation:
                self.gaussian_adaptation.medium_attenuation = self._water_attenuation()
            screen_gradient = screen_points.grad
            if screen_absolute is not None:
                # Same chain rule as screen_points: d offset / d screen = scale * axis.
                screen_gradient = screen_absolute[1] * self.model.gaussians.world_per_viewport(batch.camera).abs()
            self.gaussian_adaptation.collect_statistics(
                screen_gradient, render.diagnostics.gaussian, batch.camera, iteration
            )
        if collect_voronoi:
            self.voronoi_adaptation.collect_statistics(render.diagnostics.medium)

        gaussian_gradient_norm = gradient_norm(self.model.gaussians.parameters())
        medium_gradient_norm = gradient_norm(self.model.medium.parameters())
        self.last_adaptation = {"clone": 0, "split": 0, "prune": 0, "budget_skip": 0}
        self.last_voronoi_adaptation = {
            "densify": 0, "prune": 0, "skipped": 0
        }
        self.last_opacity_reset = {"applied": 0, "changed": 0}
        self.last_adaptation_pre_psnr = float("nan")
        self.last_adaptation_post_psnr = float("nan")
        self.last_adaptation_psnr_delta = float("nan")
        self.last_voronoi_adaptation_pre_psnr = float("nan")
        self.last_voronoi_adaptation_post_psnr = float("nan")
        self.last_voronoi_adaptation_psnr_delta = float("nan")
        self.last_reset_post_psnr = float("nan")
        self.last_reset_psnr_delta = float("nan")

        if self.uses_gaussians and self._independent_medium_freeze_event(iteration):
            self._schedule_medium_freeze(iteration, self.config.gaussian_adaptation.medium_freeze_steps)
        if (
            self.config.gaussian_adaptation.policy == "graphdeco"
            and self.gaussian_adaptation.should_adapt(iteration)
            and (gaussian_trainable or self.gaussian_adaptation.should_prune(iteration))
        ):
            plan = self.gaussian_adaptation.propose_plan(
                iteration, allow_densification=gaussian_trainable
            )
            self.last_adaptation_pre_psnr = float(loss.psnr.item())
            self.last_adaptation = self.gaussian_adaptation.apply_plan(plan)
            self._update_mip_filter()
            self.population_runaway = self._population_exceeds_safety_threshold()
            if not self.population_runaway:
                self.last_adaptation_post_psnr = self._same_batch_psnr(batch, rays)
                self.last_adaptation_psnr_delta = self.last_adaptation_post_psnr - self.last_adaptation_pre_psnr
            if any(self.last_adaptation[name] for name in ("clone", "split", "prune")):
                torch.cuda.empty_cache()
            statistics = self.gaussian_adaptation.last_plan_statistics
            print(
                f"iteration={iteration} Gaussian adaptation "
                f"clone={self.last_adaptation['clone']} "
                f"split={self.last_adaptation['split']} "
                f"prune={self.last_adaptation['prune']} "
                f"far_prune={statistics['far_prune']} "
                f"budget_skip={self.last_adaptation['budget_skip']} "
                f"count={self.model.gaussians.point_count} "
                f"grad_p95={statistics['gradient_p95']:.3e} "
                f"grad_candidates={statistics['gradient_candidates']}",
                flush=True,
            )

        if (
            gaussian_trainable
            and self.gaussian_adaptation.should_reset_opacity(iteration)
        ):
            reset_reference_psnr = (
                self.last_adaptation_post_psnr
                if self.last_adaptation_post_psnr == self.last_adaptation_post_psnr
                else float(loss.psnr.item())
            )
            changed = self.gaussian_adaptation.reset_opacity()
            self.last_opacity_reset = {"applied": 1, "changed": changed}
            self._schedule_medium_freeze(
                iteration, self.config.gaussian_adaptation.medium_freeze_after_reset
            )
            if not self.population_runaway:
                self.last_reset_post_psnr = self._same_batch_psnr(batch, rays)
                self.last_reset_psnr_delta = self.last_reset_post_psnr - reset_reference_psnr
            print(
                f"iteration={iteration} opacity reset "
                f"value={self.config.gaussian_adaptation.opacity_reset_value} "
                f"changed={changed} medium_frozen_until={self.medium_frozen_until}",
                flush=True,
            )

        voronoi_changed = False
        if self.uses_voronoi and self.voronoi_adaptation.should_adapt(iteration):
            self.last_voronoi_adaptation_pre_psnr = self._same_batch_psnr(
                batch, rays
            )
            plan = self.voronoi_adaptation.propose_plan()
            self.last_voronoi_adaptation = (
                self.voronoi_adaptation.apply_plan(plan)
            )
            voronoi_changed = bool(
                self.last_voronoi_adaptation["densify"]
                or self.last_voronoi_adaptation["prune"]
            )
            if voronoi_changed:
                self.topology.synchronize_renderer(
                    self.renderer, self.model.medium
                )
                torch.cuda.empty_cache()
            self.last_voronoi_adaptation_post_psnr = self._same_batch_psnr(
                batch, rays
            )
            self.last_voronoi_adaptation_psnr_delta = (
                self.last_voronoi_adaptation_post_psnr
                - self.last_voronoi_adaptation_pre_psnr
            )
            print(
                f"iteration={iteration} Voronoi adaptation "
                f"densify={self.last_voronoi_adaptation['densify']} "
                f"prune={self.last_voronoi_adaptation['prune']} "
                f"skipped={self.last_voronoi_adaptation['skipped']} "
                f"count={self.model.medium.point_count} "
                f"psnr_delta={self.last_voronoi_adaptation_psnr_delta:.6f}",
                flush=True,
            )

        if not self.population_runaway:
            if gaussian_trainable:
                self.gaussian_optimizer.step()
                self._apply_gaussian_scale_guard()
            if self.uses_voronoi and iteration >= self.medium_frozen_until and not voronoi_changed:
                self.medium_optimizer.step()

        if (
            self.uses_voronoi and not self.population_runaway and not voronoi_changed
            and iteration % self.config.optimization.topology_update_interval == 0
        ):
            changed = self.topology.update(
                self.model.medium, self.medium_optimizer,
                incremental=self.config.optimization.topology_incremental,
            )
            if changed:
                self.voronoi_adaptation.reorder_statistics(self.topology.last_permutation)
            self.topology.synchronize_renderer(self.renderer, self.model.medium)

        torch.cuda.synchronize()
        step_seconds = time.perf_counter() - started
        self.training_seconds += step_seconds
        return loss, gaussian_gradient_norm, medium_gradient_norm, step_seconds

    @torch.no_grad()
    def _gaussian_point_extent(self) -> float:
        means = self.model.gaussians.means
        span = means.amax(dim=0) - means.amin(dim=0)
        return float(torch.linalg.vector_norm(span).item())

    @torch.no_grad()
    def _apply_gaussian_scale_guard(self) -> None:
        optimization = self.config.optimization
        cap = gaussian_scale_cap(optimization, self.scene_extent)
        if cap is not None:
            self.model.gaussians.clamp_scales_(*cap)
            return
        if (
            optimization.gaussian_min_scale is None
            or optimization.gaussian_max_scale_fraction is None
        ):
            return
        maximum = (
            optimization.gaussian_max_scale_fraction
            * (self.gaussian_point_extent / 2.0)
        )
        self.model.gaussians.clamp_scales_(
            optimization.gaussian_min_scale, maximum
        )
        self.gaussian_point_extent = self._gaussian_point_extent()

    def _evaluate_and_checkpoint(self, iteration: int) -> None:
        if self.config.optimization.water_extrapolation:
            fitted = self.fit_water_extrapolation()
            if fitted is not None:
                (self.output_dir / "water_extrapolation.json").write_text(json.dumps(fitted.state_dict()))
        summary, images, histograms = self.evaluator.evaluate(
            iteration
        )
        self.logger.log_eval(summary, images, histograms)
        current_psnr = float(summary["eval/primary_psnr"])
        if current_psnr > self.best_psnr:
            self.best_psnr = current_psnr
        self._checkpoint(
            self.output_dir / "checkpoints" / "last.pt",
            iteration,
        )
        print(
            f"iteration={iteration} "
            f"primary_psnr={current_psnr:.4f} "
            f"gaussian_only_psnr="
            f"{summary['eval/gaussian_only_psnr']:.4f} "
            f"voronoi_only_psnr="
            f"{summary['eval/voronoi_only_psnr']:.4f}",
            flush=True,
        )

    def save_voronoi_warmup_milestone(
        self, iteration: int
    ) -> Path | None:
        """Persist an immutable checkpoint used to inspect native pretraining."""

        warmup = self.config.voronoi_warmup
        if not warmup.enabled or iteration not in warmup.milestone_iterations:
            return None
        path = (
            self.output_dir
            / "checkpoints"
            / f"iter_{iteration:08d}.pt"
        )
        self._checkpoint(path, iteration)
        return path

    def _write_summary(
        self, iteration: int, status: str
    ) -> dict[str, float | int | str]:
        summary = {
            "status": status,
            "final_iteration": iteration,
            "requested_iterations": self.config.optimization.iterations,
            "training_seconds": self.training_seconds,
            "best_psnr": self.best_psnr,
            "gaussian_count": self.model.gaussians.point_count,
            "voronoi_count": self.model.medium.point_count,
            "representation_mode": self.config.representation_mode,
            "voronoi_adaptation_revision": self.voronoi_adaptation.revision,
            "total_voronoi_densified": self.voronoi_adaptation.total_densified,
            "total_voronoi_pruned": self.voronoi_adaptation.total_pruned,
            "total_voronoi_failed_events": (
                self.voronoi_adaptation.total_failed_events
            ),
            "adaptation_revision": self.gaussian_adaptation.revision,
            "total_cloned": self.gaussian_adaptation.total_cloned,
            "total_split": self.gaussian_adaptation.total_split,
            "total_pruned": self.gaussian_adaptation.total_pruned,
            "total_opacity_resets": (
                self.gaussian_adaptation.total_opacity_resets
            ),
            "total_budget_skips": (
                self.gaussian_adaptation.total_budget_skips
            ),
            "peak_allocated_mib": torch.cuda.max_memory_allocated(
                self.device
            )
            / (1024.0**2),
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        return summary

    def _new_sampler(self) -> RandomCameraRaySampler:
        return RandomCameraRaySampler(
            self.dataset,
            self.config.optimization.rays_per_batch,
            tuple(self.config.background),
            full_image=self.config.optimization.full_image_batch,
            cameras_without_replacement=(
                self.config.optimization.cameras_without_replacement
            ),
            jitter=self.config.optimization.ray_jitter,
        )

    def fit(self) -> dict[str, float | int]:
        final_iteration = self.config.optimization.iterations
        if self.start_iteration > final_iteration:
            raise ValueError(
                "checkpoint iteration already exceeds configured iterations"
            )
        last_eval_iteration = 0
        try:
            progress = trange(
                self.start_iteration, final_iteration + 1
            )
            for iteration in progress:
                (
                    loss,
                    gaussian_gradient_norm,
                    medium_gradient_norm,
                    step_seconds,
                ) = self._train_step(iteration)
                progress.set_postfix(
                    psnr=f"{loss.psnr.item():.3f}",
                    medium=f"{self.model.medium.densities.mean().item():.4f}",
                )
                should_evaluate = (
                    iteration % self.config.validation.eval_interval == 0
                    or iteration == final_iteration
                )
                should_log = (
                    iteration == self.start_iteration
                    or iteration
                    % self.config.validation.log_interval
                    == 0
                    or iteration == final_iteration
                )
                if should_log:
                    self.logger.log_train(
                        self._training_row(
                            iteration,
                            loss,
                            gaussian_gradient_norm,
                            medium_gradient_norm,
                            step_seconds,
                        ),
                        commit=(
                            self.population_runaway or not should_evaluate
                        ),
                    )

                if self.population_runaway:
                    self._checkpoint(
                        self.output_dir / "checkpoints" / "last.pt",
                        iteration,
                    )
                    summary = self._write_summary(
                        iteration, "runaway_population"
                    )
                    print(
                        "FOAMSURF_RUNAWAY_POPULATION="
                        + json.dumps(summary),
                        flush=True,
                    )
                    return summary

                if should_evaluate:
                    self._evaluate_and_checkpoint(iteration)
                    last_eval_iteration = iteration
                elif (
                    iteration
                    % self.config.validation.checkpoint_interval
                    == 0
                ):
                    self._checkpoint(
                        self.output_dir
                        / "checkpoints"
                        / "last.pt",
                        iteration,
                    )
                self.save_voronoi_warmup_milestone(iteration)

            if last_eval_iteration != final_iteration:
                self._evaluate_and_checkpoint(final_iteration)
            return self._write_summary(final_iteration, "completed")
        finally:
            self.logger.close()
