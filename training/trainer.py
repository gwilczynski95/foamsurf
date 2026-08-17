from __future__ import annotations

import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from tqdm import trange

from evaluation import Evaluator
from foamsurf import NativeHybridRenderer
from foamsurf_data import LayerGroundTruth, load_dataset
from foamsurf_data.camera_geometry import camera_normalization
from foamsurf_model import (
    GaussianAdaptationController,
    HybridSceneModel,
    SurfaceGaussianModel,
    VoronoiTopologyController,
)
from .checkpoints import (
    checkpoint_payload,
    load_checkpoint,
    save_checkpoint,
)
from .config import ExperimentConfig
from .experiment_logging import ExperimentLogger
from .losses import b0_reconstruction_loss
from .optimizer import (
    build_optimizers,
    gradient_norm,
    update_gaussian_learning_rate,
)
from .sampler import RandomCameraRaySampler


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


class Trainer:
    """End-to-end B0 trainer: joint optimization with RGB MSE only."""

    def __init__(
        self,
        config: ExperimentConfig,
        resume: str | Path | None = None,
    ) -> None:
        config.validate()
        if not torch.cuda.is_available():
            raise RuntimeError("FoamSurf B0 training requires CUDA")
        self.config = config
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
            config.dataset.source_path,
            resolution=config.dataset.resolution,
            device=self.device,
            llffhold=config.dataset.llffhold,
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
                "on CPU (evaluation only).",
                flush=True,
            )
        self.camera_center, self.scene_extent = camera_normalization(
            self.dataset.train
        )
        self.model = HybridSceneModel.initialize(
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
            gaussian_state = {
                name: payload["model"][f"gaussians.{name}"]
                for name in SurfaceGaussianModel.PARAMETER_NAMES
            }
            self.model.gaussians = SurfaceGaussianModel.from_checkpoint(
                gaussian_state,
                payload["model_metadata"]["gaussians"],
                self.device,
            )
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

        self.gaussian_adaptation = GaussianAdaptationController(
            self.model.gaussians,
            self.gaussian_optimizer,
            config.gaussian_adaptation,
            self.scene_extent,
        )
        if payload is not None and payload.get("gaussian_adaptation") is not None:
            self.gaussian_adaptation.load_state_dict(
                payload["gaussian_adaptation"]
            )
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
        )
        first_origin = self.dataset.train[0].origin
        self.renderer.set_gaussian_geometry(
            self.model.gaussians.render_scene(first_origin)
        )
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
                self.medium_frozen_until,
            ),
        )

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
            "gradient/gaussian_norm": gaussian_gradient_norm,
            "gradient/medium_norm": medium_gradient_norm,
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
        gaussian_opacity = self.model.gaussians.opacities.detach().flatten()
        gaussian_scale = self.model.gaussians.scales.detach().flatten()
        medium_density = self.model.medium.densities.detach().flatten()
        gaussian_grad = _primitive_gradient_magnitude(self.model.gaussians)
        medium_grad = _primitive_gradient_magnitude(self.model.medium)
        row.update(
            {
                "gaussian/opacity_p50": float(torch.quantile(gaussian_opacity, 0.5).item()),
                "gaussian/opacity_p95": float(torch.quantile(gaussian_opacity, 0.95).item()),
                "gaussian/opacity_gt_0_1_fraction": float((gaussian_opacity > 0.1).float().mean().item()),
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
        return row

    @torch.no_grad()
    def _same_batch_psnr(self, batch, rays: torch.Tensor) -> float:
        gaussian_scene = self.model.gaussians.render_scene(
            batch.camera_origin
        )
        self.renderer.set_gaussian_geometry(gaussian_scene)
        render = self.renderer.render(
            rays,
            gaussians=gaussian_scene,
            voronoi=self.model.medium.render_scene(),
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

    def _train_step(self, iteration: int):
        self.current_gaussian_means_lr = update_gaussian_learning_rate(
            self.gaussian_optimizer,
            self.config.optimization,
            self.scene_extent,
            iteration,
        )
        sh_interval = self.config.optimization.gaussian_sh_degree_interval
        if sh_interval > 0 and iteration % sh_interval == 0:
            self.model.gaussians.one_up_sh_degree()
        torch.cuda.synchronize()
        started = time.perf_counter()
        batch = self.sampler.next()
        collect_adaptation = self.gaussian_adaptation.should_collect(
            iteration
        )
        if collect_adaptation:
            gaussian_scene, screen_points = (
                self.model.gaussians.training_scene(batch.camera)
            )
        else:
            gaussian_scene = self.model.gaussians.render_scene(batch.camera_origin)
            screen_points = None
        self.renderer.set_gaussian_geometry(gaussian_scene)
        medium_scene = self.model.medium.render_scene()
        rays = torch.cat(
            (batch.rays.origins, batch.rays.directions), dim=-1
        ).contiguous()

        self.gaussian_optimizer.zero_grad(set_to_none=True)
        self.medium_optimizer.zero_grad(set_to_none=True)
        render = self.renderer.render(
            rays,
            gaussians=gaussian_scene,
            voronoi=medium_scene,
            background=tuple(self.config.background),
            collect_diagnostics=collect_adaptation,
            diagnostic_target=(batch.target if collect_adaptation else None),
        )
        loss = b0_reconstruction_loss(
            render.image,
            batch.target,
            mode=self.config.optimization.reconstruction_loss,
            dssim_weight=self.config.optimization.dssim_weight,
            image_size=(batch.camera.height, batch.camera.width),
        )
        loss.total.backward()
        if collect_adaptation:
            if render.diagnostics is None:
                raise RuntimeError("B1 statistics were not returned")
            self.gaussian_adaptation.collect_statistics(
                screen_points.grad,
                render.diagnostics.gaussian,
                batch.camera,
            )
        gaussian_gradient_norm = gradient_norm(
            self.model.gaussians.parameters()
        )
        medium_gradient_norm = gradient_norm(
            self.model.medium.parameters()
        )
        self.last_adaptation = {
            "clone": 0,
            "split": 0,
            "prune": 0,
            "budget_skip": 0,
        }
        self.last_opacity_reset = {"applied": 0, "changed": 0}
        self.last_adaptation_pre_psnr = float("nan")
        self.last_adaptation_post_psnr = float("nan")
        self.last_adaptation_psnr_delta = float("nan")
        self.last_reset_post_psnr = float("nan")
        self.last_reset_psnr_delta = float("nan")
        if self._independent_medium_freeze_event(iteration):
            self._schedule_medium_freeze(
                iteration,
                self.config.gaussian_adaptation.medium_freeze_steps,
            )
        if self.gaussian_adaptation.should_adapt(iteration):
            plan = self.gaussian_adaptation.propose_plan(iteration)
            self.last_adaptation_pre_psnr = float(loss.psnr.item())
            self.last_adaptation = self.gaussian_adaptation.apply_plan(plan)
            self.population_runaway = (
                self._population_exceeds_safety_threshold()
            )
            if not self.population_runaway:
                self.last_adaptation_post_psnr = self._same_batch_psnr(
                    batch, rays
                )
                self.last_adaptation_psnr_delta = (
                    self.last_adaptation_post_psnr
                    - self.last_adaptation_pre_psnr
                )
            if any(
                self.last_adaptation[name]
                for name in ("clone", "split", "prune")
            ):
                torch.cuda.empty_cache()
            statistics = self.gaussian_adaptation.last_plan_statistics
            print(
                f"iteration={iteration} Gaussian adaptation "
                f"clone={self.last_adaptation['clone']} "
                f"split={self.last_adaptation['split']} "
                f"prune={self.last_adaptation['prune']} "
                f"budget_skip={self.last_adaptation['budget_skip']} "
                f"count={self.model.gaussians.point_count} "
                f"grad_p95={statistics['gradient_p95']:.3e} "
                f"grad_max={statistics['gradient_max']:.3e} "
                f"grad_candidates={statistics['gradient_candidates']} "
                f"clone_candidates={statistics['clone_candidates']} "
                f"split_candidates={statistics['split_candidates']} "
                f"opacity_prune={statistics['opacity_prune']} "
                f"screen_prune={statistics['screen_prune']} "
                f"world_prune={statistics['world_prune']} "
                f"budget_rejected="
                f"{statistics['budget_rejected']} "
                f"grad_screen_overlap="
                f"{statistics['gradient_screen_overlap']} "
                f"screen_replacements="
                f"{statistics['screen_replacements']} "
                f"radius_p95={statistics['radius_p95']:.3f} "
                f"radius_max={statistics['radius_max']:.3f}",
                flush=True,
            )

        if self.gaussian_adaptation.should_reset_opacity(iteration):
            reset_reference_psnr = (
                self.last_adaptation_post_psnr
                if self.last_adaptation_post_psnr
                == self.last_adaptation_post_psnr
                else float(loss.psnr.item())
            )
            changed = self.gaussian_adaptation.reset_opacity()
            self.last_opacity_reset = {"applied": 1, "changed": changed}
            self._schedule_medium_freeze(
                iteration,
                self.config.gaussian_adaptation.medium_freeze_after_reset,
            )
            if not self.population_runaway:
                self.last_reset_post_psnr = self._same_batch_psnr(
                    batch, rays
                )
                self.last_reset_psnr_delta = (
                    self.last_reset_post_psnr - reset_reference_psnr
                )
            print(
                f"iteration={iteration} opacity reset "
                f"value={self.config.gaussian_adaptation.opacity_reset_value} "
                f"changed={changed} "
                f"medium_frozen_until={self.medium_frozen_until}",
                flush=True,
            )

        if not self.population_runaway:
            self.gaussian_optimizer.step()
            if iteration >= self.medium_frozen_until:
                self.medium_optimizer.step()

        if (
            not self.population_runaway
            and iteration
            % self.config.optimization.topology_update_interval
            == 0
        ):
            self.topology.update(
                self.model.medium,
                self.medium_optimizer,
                incremental=(
                    self.config.optimization.topology_incremental
                ),
            )
            self.topology.synchronize_renderer(
                self.renderer, self.model.medium
            )

        torch.cuda.synchronize()
        step_seconds = time.perf_counter() - started
        self.training_seconds += step_seconds
        return (
            loss,
            gaussian_gradient_norm,
            medium_gradient_norm,
            step_seconds,
        )

    def _evaluate_and_checkpoint(self, iteration: int) -> None:
        summary, images, histograms = self.evaluator.evaluate(
            iteration
        )
        self.logger.log_eval(summary, images, histograms)
        current_psnr = float(summary["eval/hybrid_psnr"])
        if current_psnr > self.best_psnr:
            self.best_psnr = current_psnr
            self._checkpoint(
                self.output_dir / "checkpoints" / "best.pt",
                iteration,
            )
        self._checkpoint(
            self.output_dir / "checkpoints" / "last.pt",
            iteration,
        )
        print(
            f"iteration={iteration} "
            f"hybrid_psnr={current_psnr:.4f} "
            f"gaussian_only_psnr="
            f"{summary['eval/gaussian_only_psnr']:.4f} "
            f"voronoi_only_psnr="
            f"{summary['eval/voronoi_only_psnr']:.4f}",
            flush=True,
        )

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

    def fit(self) -> dict[str, float | int]:
        final_iteration = self.config.optimization.iterations
        if self.start_iteration > final_iteration:
            raise ValueError(
                "checkpoint iteration already exceeds configured iterations"
            )
        progress = trange(
            self.start_iteration, final_iteration + 1
        )
        last_eval_iteration = 0
        try:
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

            if last_eval_iteration != final_iteration:
                self._evaluate_and_checkpoint(final_iteration)
            return self._write_summary(final_iteration, "completed")
        finally:
            self.logger.close()
