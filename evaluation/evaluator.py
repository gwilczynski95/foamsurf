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
from .layer_metrics import decomposition_metrics
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


def _heatmap(value: torch.Tensor) -> torch.Tensor:
    value = value.squeeze(-1).clamp(0.0, 1.0)
    return torch.stack(
        (
            value,
            1.0 - (2.0 * value - 1.0).abs(),
            1.0 - value,
        ),
        dim=-1,
    )


def _gradient_magnitude(module) -> torch.Tensor:
    squared = None
    for parameter in module.parameters():
        if parameter.grad is None or parameter.ndim == 0:
            continue
        per_primitive = parameter.grad.detach().reshape(
            parameter.shape[0], -1
        ).square().sum(dim=1)
        squared = (
            per_primitive
            if squared is None
            else squared + per_primitive
        )
    if squared is None:
        reference = next(module.parameters())
        return reference.new_zeros((reference.shape[0],))
    return squared.sqrt()


def _quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def _masked_scalar(value: torch.Tensor, mask: torch.Tensor) -> float:
    expanded = mask.expand_as(value)
    if not bool(expanded.any()):
        return float("nan")
    return float(value[expanded].mean().item())


def _summary_name(name: str) -> str:
    direct = ("lego_lego/", "lego_clouds/", "entanglement/", "regions/")
    return name if name.startswith(direct) else f"eval/{name}"


def _write_primitive_csv(
    path: Path,
    statistics: torch.Tensor,
    names: tuple[str, ...],
) -> None:
    values = statistics.detach().cpu().numpy()
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("primitive_index", *names))
        for index, row in enumerate(values):
            writer.writerow((index, *(float(value) for value in row)))


class Evaluator:
    """B0.1 validation with isolated branches and responsibility diagnostics."""

    def __init__(
        self,
        renderer,
        model: HybridSceneModel,
        topology: VoronoiTopologyController,
        dataset,
        config: ExperimentConfig,
        output_dir: Path,
        layer_ground_truth=None,
    ) -> None:
        self.renderer = renderer
        self.model = model
        self.topology = topology
        self.dataset = dataset
        self.config = config
        self.layer_ground_truth = layer_ground_truth
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
        # gaussian_only never trains the medium: the main ("hybrid") image is the Gaussians alone.
        hybrid_medium = None if self.config.representation_mode == "gaussian_only" else medium_scene
        self.topology.synchronize_renderer(
            self.renderer, self.model.medium
        )
        views = self.dataset.test
        if self.config.validation.max_eval_views > 0:
            views = views[: self.config.validation.max_eval_views]

        diagnostics_enabled = self.config.diagnostics.enabled
        gaussian_statistics = torch.zeros(
            (self.model.gaussians.point_count, 4),
            device=self.model.gaussians.means.device,
        )
        medium_statistics = torch.zeros(
            (self.model.medium.point_count, 5),
            device=self.model.medium.sites.device,
        )
        rows: list[dict[str, float | int | str]] = []
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
                        voronoi=hybrid_medium,
                        background=tuple(self.config.background),
                    )
                )
                diagnostic_result = hybrid
                diagnostic_seconds = 0.0
                if diagnostics_enabled:
                    diagnostic_result, diagnostic_seconds = self._timed(
                        lambda: self.renderer.render(
                            rays,
                            gaussians=gaussian_scene,
                            voronoi=hybrid_medium,
                            background=tuple(self.config.background),
                            collect_diagnostics=True,
                            diagnostic_target=target,
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
                ray_error = error.mean(dim=-1, keepdim=True)
                transmittance = hybrid.transmittance.expand_as(
                    hybrid_image
                )
                hybrid_psnr = float(psnr(hybrid_image, target).item())
                gaussian_psnr = float(
                    psnr(gaussian_image, target).item()
                )
                medium_psnr = float(
                    psnr(medium_image, target).item()
                )
                row: dict[str, float | int | str] = {
                    "iteration": iteration,
                    "view_index": index,
                    "view_name": camera.name,
                    "hybrid_psnr": hybrid_psnr,
                    "hybrid_ssim": float(
                        ssim(hybrid_image, target).item()
                    ),
                    "gaussian_only_psnr": gaussian_psnr,
                    "gaussian_only_ssim": float(
                        ssim(gaussian_image, target).item()
                    ),
                    "gaussian_raw_over_1_fraction": float(
                        (gaussian.image > 1.0).any(dim=-1).float().mean().item()
                    ),
                    "gaussian_raw_max": float(gaussian.image.max().item()),
                    "gaussian_near_white_on_covered": float(
                        (
                            (gaussian_image >= 0.95).all(dim=-1)
                            & (gaussian.alpha[..., 0] > 0.5)
                        ).float().sum().div(
                            (gaussian.alpha[..., 0] > 0.5).float().sum().clamp_min(1)
                        ).item()
                    ),
                    "voronoi_only_psnr": medium_psnr,
                    "voronoi_only_ssim": float(
                        ssim(medium_image, target).item()
                    ),
                    "hybrid_minus_gaussian_psnr": (
                        hybrid_psnr - gaussian_psnr
                    ),
                    "hybrid_minus_voronoi_psnr": (
                        hybrid_psnr - medium_psnr
                    ),
                    "hybrid_worse_than_gaussian": int(
                        hybrid_psnr < gaussian_psnr
                    ),
                    "hybrid_worse_than_voronoi": int(
                        hybrid_psnr < medium_psnr
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
                    "ray_error_mean": float(ray_error.mean().item()),
                    "ray_error_p95": float(
                        torch.quantile(ray_error, 0.95).item()
                    ),
                    "ray_error_max": float(ray_error.max().item()),
                    "hybrid_render_seconds": hybrid_seconds,
                    "diagnostics_render_seconds": diagnostic_seconds,
                    "gaussian_render_seconds": gaussian_seconds,
                    "voronoi_render_seconds": medium_seconds,
                }

                layer_masks = None
                lego_target = None
                cloud_target = None
                if self.layer_ground_truth is not None:
                    lego_camera = self.layer_ground_truth.lego_lego[index]
                    cloud_camera = self.layer_ground_truth.lego_clouds[index]
                    lego_target = lego_camera.ground_truth(
                        tuple(self.config.background)
                    ).to(target.device)
                    cloud_target = cloud_camera.ground_truth(
                        tuple(self.config.background)
                    ).to(target.device)
                    lego_alpha = lego_camera.alpha.to(target.device)
                    cloud_alpha = cloud_camera.alpha.to(target.device)
                    layer_values, layer_masks = decomposition_metrics(
                        gaussian_image,
                        gaussian.alpha,
                        medium_image,
                        medium.alpha,
                        lego_target,
                        lego_alpha,
                        cloud_target,
                        cloud_alpha,
                        self.config.diagnostics.layer_alpha_threshold,
                    )
                    row.update(layer_values)

                contribution_panel = None
                if diagnostics_enabled:
                    diagnostics = diagnostic_result.diagnostics
                    if diagnostics is None:
                        raise RuntimeError(
                            "renderer did not return requested diagnostics"
                        )
                    gaussian_statistics[:, :3] += (
                        diagnostics.gaussian[:, :3]
                    )
                    gaussian_statistics[:, 3] = torch.maximum(
                        gaussian_statistics[:, 3],
                        diagnostics.gaussian[:, 3],
                    )
                    if hybrid_medium is not None:  # gaussian_only renders no cells
                        medium_statistics[:, :4] += diagnostics.medium[:, :4]
                        medium_statistics[:, 4] = torch.maximum(
                            medium_statistics[:, 4],
                            diagnostics.medium[:, 4],
                        )
                    gaussian_weight = diagnostics.gaussian_weight
                    medium_weight = diagnostics.medium_weight
                    total_weight = gaussian_weight + medium_weight
                    medium_fraction = medium_weight / total_weight.clamp_min(
                        self.config.diagnostics.contribution_epsilon
                    )
                    active_gaussian = (
                        gaussian_weight
                        > self.config.diagnostics.contribution_epsilon
                    )
                    active_medium = (
                        medium_weight
                        > self.config.diagnostics.contribution_epsilon
                    )
                    conservation_error = (
                        total_weight
                        + diagnostics.final_transmittance
                        - 1.0
                    ).abs()
                    row.update(
                        {
                            "responsibility/gaussian_weight_mean": float(
                                gaussian_weight.mean().item()
                            ),
                            "responsibility/medium_weight_mean": float(
                                medium_weight.mean().item()
                            ),
                            "responsibility/medium_fraction_mean": float(
                                medium_fraction.mean().item()
                            ),
                            "responsibility/overlap_fraction": float(
                                (active_gaussian & active_medium)
                                .float()
                                .mean()
                                .item()
                            ),
                            "responsibility/gaussian_event_mean": float(
                                diagnostics.gaussian_event_count.mean().item()
                            ),
                            "responsibility/medium_segment_mean": float(
                                diagnostics.medium_segment_count.mean().item()
                            ),
                            "responsibility/medium_path_mean": float(
                                diagnostics.medium_path_length.mean().item()
                            ),
                            "responsibility/weighted_medium_path_mean": float(
                                diagnostics.weighted_medium_path_length
                                .mean()
                                .item()
                            ),
                            "responsibility/conservation_error_max": float(
                                conservation_error.max().item()
                            ),
                            "responsibility/truncated_ray_fraction": float(
                                diagnostics.traversal_truncated.mean().item()
                            ),
                        }
                    )
                    if layer_masks is not None:
                        row.update(
                            {
                                "entanglement/medium_weight_on_lego_only": _masked_scalar(
                                    medium_weight, layer_masks["lego_only"]
                                ),
                                "entanglement/gaussian_weight_on_cloud_only": _masked_scalar(
                                    gaussian_weight, layer_masks["cloud_only"]
                                ),
                                "entanglement/gaussian_weight_on_background": _masked_scalar(
                                    gaussian_weight, layer_masks["background"]
                                ),
                                "entanglement/medium_weight_on_background": _masked_scalar(
                                    medium_weight, layer_masks["background"]
                                ),
                            }
                        )
                    contribution_panel = torch.cat(
                        (
                            _heatmap(gaussian_weight),
                            _heatmap(medium_weight),
                            _heatmap(medium_fraction),
                        ),
                        dim=1,
                    )
                if self.model.medium.is_rgb:
                    row = {k.replace("responsibility/", "proxy_responsibility/"): v for k,v in row.items()}
                    if hybrid.direct is not None:
                        row["medium_rgb/direct_mean"] = float(hybrid.direct.mean())
                        row["medium_rgb/backscatter_mean"] = float(hybrid.backscatter.mean())
                        row["medium_rgb/mean_direct_transmittance"] = float(hybrid.direct_transmittance.mean())
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
                    if hybrid.direct is not None:
                        _save(directory / f"{prefix}_direct_with_background.png", hybrid.direct)
                        _save(directory / f"{prefix}_backscatter.png", hybrid.backscatter)
                        _save(directory / f"{prefix}_direct_transmittance_rgb.png", hybrid.direct_transmittance)
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
                    _save(
                        directory / f"{prefix}_error.png",
                        error,
                    )
                    wandb_images[f"comparison/view_{index:02d}"] = (
                        _uint8(panel)
                    )
                    if contribution_panel is not None:
                        _save(
                            directory
                            / f"{prefix}_{'proxy_responsibility' if self.model.medium.is_rgb else 'responsibility'}.png",
                            contribution_panel,
                        )
                        wandb_images[
                            f"responsibility/view_{index:02d}"
                        ] = _uint8(contribution_panel)
                    if lego_target is not None and cloud_target is not None:
                        lego_alpha_rgb = lego_alpha.expand_as(lego_target)
                        cloud_alpha_rgb = cloud_alpha.expand_as(cloud_target)
                        gaussian_alpha_rgb = gaussian.alpha.expand_as(lego_target)
                        medium_alpha_rgb = medium.alpha.expand_as(cloud_target)
                        lego_panel = torch.cat(
                            (
                                lego_target,
                                gaussian_image,
                                (gaussian_image - lego_target).abs(),
                                lego_alpha_rgb,
                                gaussian_alpha_rgb,
                                (gaussian_alpha_rgb - lego_alpha_rgb).abs(),
                            ),
                            dim=1,
                        )
                        cloud_panel = torch.cat(
                            (
                                cloud_target,
                                medium_image,
                                (medium_image - cloud_target).abs(),
                                cloud_alpha_rgb,
                                medium_alpha_rgb,
                                (medium_alpha_rgb - cloud_alpha_rgb).abs(),
                            ),
                            dim=1,
                        )
                        mask_panel = torch.cat(
                            tuple(
                                layer_masks[name].float().expand_as(target)
                                for name in (
                                    "lego_only",
                                    "cloud_only",
                                    "overlap",
                                    "background",
                                )
                            ),
                            dim=1,
                        )
                        _save(
                            directory / f"{prefix}_lego_lego.png",
                            lego_panel,
                        )
                        _save(
                            directory / f"{prefix}_lego_clouds.png",
                            cloud_panel,
                        )
                        _save(
                            directory / f"{prefix}_layer_masks.png",
                            mask_panel,
                        )
                        wandb_images[f"lego_lego/view_{index:02d}"] = _uint8(
                            lego_panel
                        )
                        wandb_images[f"lego_clouds/view_{index:02d}"] = _uint8(
                            cloud_panel
                        )
                        wandb_images[f"entanglement/masks_{index:02d}"] = _uint8(
                            mask_panel
                        )
        finally:
            self.renderer.transmittance_threshold = original_threshold

        if not rows:
            raise RuntimeError("validation split is empty")
        per_view_path = directory / "per_view.csv"
        with per_view_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        numeric_keys = [
            key
            for key, value in rows[0].items()
            if key not in {"iteration", "view_index", "view_name"}
            and isinstance(value, (float, int))
        ]
        summary = {
            _summary_name(key): float(
                np.mean([float(row[key]) for row in rows])
            )
            for key in numeric_keys
        }
        hybrid_psnrs = [float(row["hybrid_psnr"]) for row in rows]
        primary_name = {
            "hybrid": "hybrid",
            "gaussian_only": "gaussian_only",
            "voronoi_only": "voronoi_only",
        }[self.config.representation_mode]
        summary["eval/primary_psnr"] = summary[f"eval/{primary_name}_psnr"]
        summary["eval/primary_ssim"] = summary[f"eval/{primary_name}_ssim"]
        summary.update(
            {
                "iteration": iteration,
                "eval/hybrid_psnr_min": min(hybrid_psnrs),
                "eval/hybrid_psnr_p05": _quantile(
                    hybrid_psnrs, 0.05
                ),
                "eval/hybrid_psnr_median": _quantile(
                    hybrid_psnrs, 0.5
                ),
                "eval/hybrid_psnr_max": max(hybrid_psnrs),
                "eval/hybrid_fps": 1.0
                / summary["eval/hybrid_render_seconds"],
                "eval/diagnostics_fps": (
                    1.0 / summary["eval/diagnostics_render_seconds"]
                    if diagnostics_enabled
                    else 0.0
                ),
                "eval/gaussian_fps": 1.0
                / summary["eval/gaussian_render_seconds"],
                "eval/voronoi_fps": 1.0
                / summary["eval/voronoi_render_seconds"],
                "model/gaussian_count": self.model.gaussians.point_count,
                "model/voronoi_count": self.model.medium.point_count,
                "model/medium_color_max": float(
                    self.model.medium.colors.max().item()
                ),
                "model/medium_color_over_1_fraction": float(
                    (self.model.medium.colors > 1.0).any(dim=-1).float().mean().item()
                ),
            }
        )

        gaussian_grad = _gradient_magnitude(self.model.gaussians)
        medium_grad = _gradient_magnitude(self.model.medium)
        histograms = {
            "gaussian_opacity": self.model.gaussians.opacities,
            "gaussian_scale": self.model.gaussians.scales,
            "medium_density": self.model.medium.densities,
            "medium_color": self.model.medium.colors,
            "gaussian_gradient_magnitude": gaussian_grad,
            "medium_gradient_magnitude": medium_grad,
        }
        if diagnostics_enabled:
            epsilon = self.config.diagnostics.contribution_epsilon
            gaussian_active = gaussian_statistics[:, 0] > epsilon
            medium_active = medium_statistics[:, 0] > epsilon
            summary.update(
                {
                    "responsibility/gaussian_active_fraction": float(
                        gaussian_active.float().mean().item()
                    ),
                    "responsibility/medium_active_fraction": float(
                        medium_active.float().mean().item()
                    ),
                    "responsibility/gaussian_max_ray_error": float(
                        gaussian_statistics[:, 3].max().item()
                    ),
                    "responsibility/medium_max_ray_error": float(
                        medium_statistics[:, 4].max().item()
                    ),
                    "responsibility/gaussian_contribution_total": float(
                        gaussian_statistics[:, 0].sum().item()
                    ),
                    "responsibility/medium_contribution_total": float(
                        medium_statistics[:, 0].sum().item()
                    ),
                }
            )
            gaussian_total = gaussian_statistics[:, 0].sum().clamp_min(
                epsilon
            )
            medium_total = medium_statistics[:, 0].sum().clamp_min(
                epsilon
            )
            summary[
                "responsibility/gaussian_top1_contribution_share"
            ] = float(
                gaussian_statistics[:, 0].max().div(gaussian_total).item()
            )
            summary[
                "responsibility/medium_top1_contribution_share"
            ] = float(
                medium_statistics[:, 0].max().div(medium_total).item()
            )
            histograms.update(
                {
                    "gaussian_contribution": gaussian_statistics[:, 0],
                    "gaussian_visibility": gaussian_statistics[:, 1],
                    "gaussian_max_ray_error": gaussian_statistics[:, 3],
                    "medium_contribution": medium_statistics[:, 0],
                    "medium_visibility": medium_statistics[:, 1],
                    "medium_path_length": medium_statistics[:, 2],
                    "medium_weighted_path_length": medium_statistics[:, 3],
                    "medium_max_ray_error": medium_statistics[:, 4],
                }
            )
            if self.config.diagnostics.save_per_primitive:
                _write_primitive_csv(
                    directory / "gaussian_primitives.csv",
                    gaussian_statistics,
                    (
                        "contribution",
                        "visibility",
                        "alpha_sum",
                        "max_ray_error",
                    ),
                )
                _write_primitive_csv(
                    directory / "medium_primitives.csv",
                    medium_statistics,
                    (
                        "contribution",
                        "visibility",
                        "path_length",
                        "weighted_path_length",
                        "max_ray_error",
                    ),
                )
        if self.model.medium.is_rgb:
            summary = {(k if "proxy_responsibility/" in k else k.replace("responsibility/", "proxy_responsibility/")): v for k,v in summary.items()}
            histograms["medium_attenuation_rgb"] = self.model.medium.optical_coefficients[:, :3]
            histograms["medium_backscatter_rgb"] = self.model.medium.optical_coefficients[:, 3:]
            wandb_images = {k.replace("responsibility/", "proxy_responsibility/"): v for k,v in wandb_images.items()}
        return summary, wandb_images, histograms
