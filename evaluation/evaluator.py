from __future__ import annotations

import csv
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from foamsurf_model.scene import HybridSceneModel
from foamsurf_model.topology import VoronoiTopologyController
from training.config import ExperimentConfig
from .metrics import psnr, ssim


def _uint8(image: torch.Tensor) -> np.ndarray:
    return (
        image.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0
    ).round().astype(np.uint8)


def _save(path: Path, image: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_uint8(image)).save(path)


def _rays(camera) -> torch.Tensor:
    bundle = camera.rays()
    return torch.cat(
        (bundle.origins, bundle.directions), dim=-1
    ).contiguous()


class Evaluator:
    """Validate hybrid and both isolated branches on identical camera rays."""

    def __init__(
        self,
        renderer,
        model: HybridSceneModel,
        topology: VoronoiTopologyController,
        dataset,
        config: ExperimentConfig,
        output_dir: Path,
    ) -> None:
        self.renderer = renderer
        self.model = model
        self.topology = topology
        self.dataset = dataset
        self.config = config
        self.output_dir = output_dir

    @torch.no_grad()
    def _timed(self, operation):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = operation()
        torch.cuda.synchronize()
        return result, time.perf_counter() - started

    @torch.no_grad()
    def evaluate(
        self,
        iteration: int,
    ) -> tuple[
        dict[str, float],
        dict[str, np.ndarray],
        dict[str, torch.Tensor],
    ]:
        original_threshold = self.renderer.transmittance_threshold
        self.renderer.transmittance_threshold = (
            self.config.renderer.validation_transmittance_threshold
        )
        directory = (
            self.output_dir
            / "evaluation"
            / f"iteration_{iteration:06d}"
        )
        directory.mkdir(parents=True, exist_ok=True)

        medium_scene = self.model.medium.render_scene()
        self.topology.synchronize_renderer(
            self.renderer, self.model.medium
        )
        views = self.dataset.test
        if self.config.validation.max_eval_views > 0:
            views = views[: self.config.validation.max_eval_views]

        rows = []
        wandb_images: dict[str, np.ndarray] = {}
        try:
            for index, camera in enumerate(views):
                gaussian_scene = self.model.gaussians.render_scene(
                    camera.origin
                )
                self.renderer.set_gaussian_geometry(gaussian_scene)
                rays = _rays(camera)
                target = camera.ground_truth(
                    tuple(self.config.background)
                )

                hybrid, hybrid_seconds = self._timed(
                    lambda: self.renderer.render(
                        rays,
                        gaussians=gaussian_scene,
                        voronoi=medium_scene,
                        background=tuple(self.config.background),
                    )
                )
                gaussian, gaussian_seconds = self._timed(
                    lambda: self.renderer.render_gaussians(
                        rays,
                        gaussian_scene,
                        background=tuple(self.config.background),
                    )
                )
                medium, medium_seconds = self._timed(
                    lambda: self.renderer.render_voronoi(
                        rays,
                        medium_scene,
                        background=tuple(self.config.background),
                    )
                )

                hybrid_image = hybrid.image.clamp(0.0, 1.0)
                gaussian_image = gaussian.image.clamp(0.0, 1.0)
                medium_image = medium.image.clamp(0.0, 1.0)
                error = (hybrid_image - target).abs()
                transmittance = hybrid.transmittance.expand_as(
                    hybrid_image
                )
                row = {
                    "iteration": iteration,
                    "view_index": index,
                    "view_name": camera.name,
                    "hybrid_psnr": float(
                        psnr(hybrid_image, target).item()
                    ),
                    "hybrid_ssim": float(
                        ssim(hybrid_image, target).item()
                    ),
                    "gaussian_only_psnr": float(
                        psnr(gaussian_image, target).item()
                    ),
                    "gaussian_only_ssim": float(
                        ssim(gaussian_image, target).item()
                    ),
                    "voronoi_only_psnr": float(
                        psnr(medium_image, target).item()
                    ),
                    "voronoi_only_ssim": float(
                        ssim(medium_image, target).item()
                    ),
                    "hybrid_alpha_mean": float(
                        hybrid.alpha.mean().item()
                    ),
                    "gaussian_alpha_mean": float(
                        gaussian.alpha.mean().item()
                    ),
                    "voronoi_alpha_mean": float(
                        medium.alpha.mean().item()
                    ),
                    "hybrid_render_seconds": hybrid_seconds,
                    "gaussian_render_seconds": gaussian_seconds,
                    "voronoi_render_seconds": medium_seconds,
                }
                rows.append(row)

                if index < self.config.validation.image_views:
                    panel = torch.cat(
                        (
                            target,
                            hybrid_image,
                            error,
                            gaussian_image,
                            medium_image,
                            transmittance,
                        ),
                        dim=1,
                    )
                    prefix = f"{index:04d}_{camera.name}"
                    _save(directory / f"{prefix}_panel.png", panel)
                    _save(
                        directory / f"{prefix}_hybrid.png",
                        hybrid_image,
                    )
                    _save(
                        directory / f"{prefix}_gaussian_only.png",
                        gaussian_image,
                    )
                    _save(
                        directory / f"{prefix}_voronoi_only.png",
                        medium_image,
                    )
                    _save(
                        directory / f"{prefix}_transmittance.png",
                        transmittance,
                    )
                    wandb_images[f"view_{index:02d}"] = _uint8(panel)
        finally:
            self.renderer.transmittance_threshold = original_threshold

        if not rows:
            raise RuntimeError("validation split is empty")
        per_view_path = directory / "per_view.csv"
        with per_view_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        mean_keys = [
            "hybrid_psnr",
            "hybrid_ssim",
            "gaussian_only_psnr",
            "gaussian_only_ssim",
            "voronoi_only_psnr",
            "voronoi_only_ssim",
            "hybrid_alpha_mean",
            "gaussian_alpha_mean",
            "voronoi_alpha_mean",
            "hybrid_render_seconds",
            "gaussian_render_seconds",
            "voronoi_render_seconds",
        ]
        summary = {
            f"eval/{key}": float(
                np.mean([float(row[key]) for row in rows])
            )
            for key in mean_keys
        }
        summary.update(
            {
                "iteration": iteration,
                "eval/hybrid_fps": 1.0
                / summary["eval/hybrid_render_seconds"],
                "eval/gaussian_fps": 1.0
                / summary["eval/gaussian_render_seconds"],
                "eval/voronoi_fps": 1.0
                / summary["eval/voronoi_render_seconds"],
                "model/gaussian_count": self.model.gaussians.point_count,
                "model/voronoi_count": self.model.medium.point_count,
            }
        )
        histograms = {
            "gaussian_opacity": self.model.gaussians.opacities,
            "gaussian_scale": self.model.gaussians.scales,
            "medium_density": self.model.medium.densities,
            "medium_color": self.model.medium.colors,
        }
        return summary, wandb_images, histograms
