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
from foamsurf_model import (
    HybridSceneModel,
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
from .optimizer import build_optimizers, gradient_norm
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
        self.model = HybridSceneModel.initialize(
            self.dataset, config, self.device
        )
        self.start_iteration = 1
        self.training_seconds = 0.0
        self.best_psnr = -float("inf")

        if self.resume_path is None:
            self.topology = VoronoiTopologyController(
                self.model.medium
            )
            (
                self.gaussian_optimizer,
                self.medium_optimizer,
            ) = build_optimizers(self.model, config.optimization)
        else:
            payload = load_checkpoint(
                self.resume_path
            )
            self.model.load_state_dict(payload["model"], strict=True)
            (
                self.gaussian_optimizer,
                self.medium_optimizer,
            ) = build_optimizers(self.model, config.optimization)
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
        row = {
            "iteration": iteration,
            "training_seconds": self.training_seconds,
            "step_seconds": step_seconds,
            "train/loss": float(loss.total.item()),
            "train/rgb_mse": float(loss.rgb_mse.item()),
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

    def _train_step(self, iteration: int):
        torch.cuda.synchronize()
        started = time.perf_counter()
        batch = self.sampler.next()
        gaussian_scene = self.model.gaussians.render_scene(
            batch.camera_origin
        )
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
        )
        loss = b0_reconstruction_loss(
            render.image, batch.target
        )
        loss.total.backward()
        gaussian_gradient_norm = gradient_norm(
            self.model.gaussians.parameters()
        )
        medium_gradient_norm = gradient_norm(
            self.model.medium.parameters()
        )
        self.gaussian_optimizer.step()
        self.medium_optimizer.step()

        if (
            iteration
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

        current_gaussian_scene = (
            self.model.gaussians.render_scene(batch.camera_origin)
        )
        self.renderer.set_gaussian_geometry(
            current_gaussian_scene
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
                        )
                    )

                should_evaluate = (
                    iteration
                    % self.config.validation.eval_interval
                    == 0
                    or iteration == final_iteration
                )
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
            summary = {
                "final_iteration": final_iteration,
                "training_seconds": self.training_seconds,
                "best_psnr": self.best_psnr,
                "gaussian_count": self.model.gaussians.point_count,
                "voronoi_count": self.model.medium.point_count,
            }
            (self.output_dir / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            return summary
        finally:
            self.logger.close()
