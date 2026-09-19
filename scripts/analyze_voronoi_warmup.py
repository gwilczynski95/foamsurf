#!/usr/bin/env python3
"""Analyze native Voronoi warmup checkpoints and render their separation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.metrics import psnr, ssim
from foamsurf import VoronoiScene
from training import ExperimentConfig
from training.trainer import Trainer


EMPTY, SURFACE, MEDIUM = 0, 1, 2
CLASS_NAMES = {EMPTY: "empty", SURFACE: "surface", MEDIUM: "medium"}


def _otsu(values: torch.Tensor, bins: int = 512) -> float:
    flat = values.detach().float().reshape(-1).cpu()
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        raise ValueError("Otsu received no finite values")
    low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return low
    counts = torch.histc(finite, bins=bins, min=low, max=high)
    weights = counts / counts.sum()
    edges = torch.linspace(low, high, bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    below = torch.cumsum(weights, 0)
    above = 1.0 - below
    mass = weights * centres
    below_mean = torch.cumsum(mass, 0) / below.clamp_min(1.0e-12)
    above_mean = (mass.sum() - torch.cumsum(mass, 0)) / above.clamp_min(1.0e-12)
    variance = below * above * (below_mean - above_mean).square()
    variance = torch.nan_to_num(variance)
    return float(centres[int(torch.argmax(variance))])


def _local_density_tv(
    density: torch.Tensor,
    adjacency: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    density = density.reshape(-1)
    counts = (offsets[1:].long() - offsets[:-1].long())
    owners = torch.repeat_interleave(
        torch.arange(density.numel(), device=density.device), counts
    )
    differences = (density[adjacency.long()] - density[owners]).abs()
    total = torch.zeros_like(density).index_add_(0, owners, differences)
    return total / counts.clamp_min(1).to(density.dtype)


def _rays(camera) -> torch.Tensor:
    bundle = camera.rays()
    return torch.cat((bundle.origins, bundle.directions), dim=-1).contiguous()


def _to_image(tensor: torch.Tensor) -> Image.Image:
    array = (
        tensor.detach().float().cpu().clamp(0.0, 1.0).numpy() * 255.0
    ).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _titled(image: Image.Image, title: str, height: int = 32) -> Image.Image:
    result = Image.new("RGB", (image.width, image.height + height), "black")
    result.paste(image, (0, height))
    draw = ImageDraw.Draw(result)
    box = draw.textbbox((0, 0), title)
    draw.text(((image.width - box[2]) // 2, 9), title, fill="white")
    return result


def _grid(rows: list[list[tuple[str, Image.Image]]]) -> Image.Image:
    titled = [[_titled(image, title) for title, image in row] for row in rows]
    width = max(sum(image.width for image in row) for row in titled)
    height = sum(max(image.height for image in row) for row in titled)
    result = Image.new("RGB", (width, height), "black")
    y = 0
    for row in titled:
        x = 0
        row_height = max(image.height for image in row)
        for image in row:
            result.paste(image, (x, y))
            x += image.width
        y += row_height
    return result


def _masked_scene(model, labels: torch.Tensor, label: int) -> VoronoiScene:
    mask = (labels == label).to(model.medium.densities.dtype)[:, None]
    return VoronoiScene(
        model.medium.sites,
        model.medium.densities * mask,
        model.medium.colors,
    )


def _occupancy_scene(
    model, labels: torch.Tensor, label: int, density: float = 0.5
) -> VoronoiScene:
    mask = (labels == label).to(model.medium.densities.dtype)[:, None]
    return VoronoiScene(
        model.medium.sites,
        mask * density,
        torch.ones_like(model.medium.colors),
    )


def _label_scene(model, labels: torch.Tensor) -> VoronoiScene:
    colors = torch.zeros_like(model.medium.colors)
    colors[labels == SURFACE] = colors.new_tensor((1.0, 0.05, 0.05))
    colors[labels == MEDIUM] = colors.new_tensor((0.05, 0.25, 1.0))
    colors[labels == EMPTY] = colors.new_tensor((0.15, 0.15, 0.15))
    return VoronoiScene(model.medium.sites, model.medium.densities, colors)


def _mask_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return float("nan")
    return float(value[mask].mean())


@torch.no_grad()
def _render_separation(
    trainer: Trainer,
    labels: torch.Tensor,
    output: Path,
    iteration: int,
    image_views: int,
) -> dict[str, float]:
    renderer = trainer.renderer
    renderer.transmittance_threshold = (
        trainer.config.renderer.validation_transmittance_threshold
    )
    trainer.topology.synchronize_renderer(renderer, trainer.model.medium)
    scenes = {
        "surface": _masked_scene(trainer.model, labels, SURFACE),
        "empty": _masked_scene(trainer.model, labels, EMPTY),
        "medium": _masked_scene(trainer.model, labels, MEDIUM),
        "surface occupancy": _occupancy_scene(
            trainer.model, labels, SURFACE
        ),
        "empty occupancy": _occupancy_scene(trainer.model, labels, EMPTY),
        "medium occupancy": _occupancy_scene(trainer.model, labels, MEDIUM),
        "class overlay": _label_scene(trainer.model, labels),
    }
    full_scene = trainer.model.medium.render_scene()
    totals: dict[str, list[float]] = {
        "full_psnr": [],
        "surface_psnr_vs_lego": [],
        "surface_ssim_vs_lego": [],
        "medium_psnr_vs_cloud": [],
        "medium_ssim_vs_cloud": [],
        "surface_alpha_on_cloud_only": [],
        "medium_alpha_on_lego_only": [],
        "empty_alpha_mean": [],
    }
    for index, camera in enumerate(trainer.dataset.test):
        rays = _rays(camera)
        target = camera.ground_truth(tuple(trainer.config.background))
        full = renderer.render_voronoi(
            rays, full_scene, background=tuple(trainer.config.background)
        )
        renders = {
            name: renderer.render_voronoi(
                rays, scene, background=tuple(trainer.config.background)
            )
            for name, scene in scenes.items()
        }
        lego_camera = trainer.layer_ground_truth.lego_lego[index]
        cloud_camera = trainer.layer_ground_truth.lego_clouds[index]
        lego = lego_camera.ground_truth(tuple(trainer.config.background)).to(
            target.device
        )
        cloud = cloud_camera.ground_truth(tuple(trainer.config.background)).to(
            target.device
        )
        lego_alpha = lego_camera.alpha.to(target.device).squeeze(-1)
        cloud_alpha = cloud_camera.alpha.to(target.device).squeeze(-1)
        lego_only = (lego_alpha > 0.01) & (cloud_alpha <= 0.01)
        cloud_only = (cloud_alpha > 0.01) & (lego_alpha <= 0.01)
        surface = renders["surface"]
        medium = renders["medium"]
        empty = renders["empty"]
        totals["full_psnr"].append(float(psnr(full.image, target)))
        totals["surface_psnr_vs_lego"].append(float(psnr(surface.image, lego)))
        totals["surface_ssim_vs_lego"].append(float(ssim(surface.image, lego)))
        totals["medium_psnr_vs_cloud"].append(float(psnr(medium.image, cloud)))
        totals["medium_ssim_vs_cloud"].append(float(ssim(medium.image, cloud)))
        totals["surface_alpha_on_cloud_only"].append(
            _mask_mean(surface.alpha.squeeze(-1), cloud_only)
        )
        totals["medium_alpha_on_lego_only"].append(
            _mask_mean(medium.alpha.squeeze(-1), lego_only)
        )
        totals["empty_alpha_mean"].append(float(empty.alpha.mean()))

        if index < image_views:
            target_image = _to_image(target)
            full_image = _to_image(full.image)
            grid = _grid(
                [
                    [
                        ("target", target_image),
                        ("full Voronoi", full_image),
                        ("class overlay: red surface / blue medium", _to_image(renders["class overlay"].image)),
                    ],
                    [
                        ("target", target_image),
                        ("SURFACE original parameters", _to_image(surface.image)),
                        ("SURFACE fixed density occupancy", _to_image(renders["surface occupancy"].image)),
                    ],
                    [
                        ("target", target_image),
                        ("EMPTY original parameters", _to_image(empty.image)),
                        ("EMPTY fixed density occupancy", _to_image(renders["empty occupancy"].image)),
                    ],
                    [
                        ("target", target_image),
                        ("MEDIUM original parameters", _to_image(medium.image)),
                        ("MEDIUM fixed density occupancy", _to_image(renders["medium occupancy"].image)),
                    ],
                ]
            )
            grid.save(output / f"view_{index:02d}_separation.png")
            for name, render in {"target": target, "full": full.image, **{k: v.image for k, v in renders.items()}}.items():
                safe = name.replace(" ", "_")
                _to_image(render).save(output / f"view_{index:02d}_{safe}.png")

    return {
        key: float(np.nanmean(values)) for key, values in totals.items()
    }


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in rows[0]
    }


def _plot_training(run: Path, output: Path) -> None:
    train = _read_csv(run / "metrics" / "train.csv")
    evaluate = _read_csv(run / "metrics" / "eval.csv")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    axes[0, 0].plot(evaluate["iteration"], evaluate["eval/primary_psnr"], marker="o")
    axes[0, 0].set(title="Validation reconstruction", xlabel="iteration", ylabel="PSNR [dB]")
    axes[0, 1].plot(evaluate["iteration"], evaluate["eval/primary_ssim"], marker="o")
    axes[0, 1].set(title="Validation SSIM", xlabel="iteration", ylabel="SSIM")
    axes[0, 2].plot(train["iteration"], train["model/voronoi_count"] / 1000.0)
    axes[0, 2].set(title="Adaptive population", xlabel="iteration", ylabel="cells [k]")
    axes[1, 0].plot(train["iteration"], train["medium/density_p50"], label="p50")
    axes[1, 0].plot(train["iteration"], train["medium/density_p95"], label="p95")
    axes[1, 0].plot(train["iteration"], train["medium/density_p99"], label="p99")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set(title="Activated density", xlabel="iteration", ylabel="density (log)")
    axes[1, 0].legend()
    axes[1, 1].plot(train["iteration"], train["memory/allocated_mib"], label="allocated")
    axes[1, 1].plot(train["iteration"], train["memory/peak_allocated_mib"], label="peak")
    axes[1, 1].set(title="GPU memory", xlabel="iteration", ylabel="MiB")
    axes[1, 1].legend()
    axes[1, 2].scatter(
        train["iteration"], train["voronoi_adaptation/psnr_delta"], s=10
    )
    axes[1, 2].axhline(0.0, color="black", linewidth=1)
    axes[1, 2].set(title="Immediate adaptation shock", xlabel="iteration", ylabel="same-batch PSNR delta")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    fig.suptitle(f"FoamSurf Voronoi warmup: {run.name}")
    fig.savefig(output / "training_dashboard.png", dpi=160)
    plt.close(fig)


def _plot_separation(records: list[dict[str, object]], output: Path) -> None:
    iterations = np.asarray([record["iteration"] for record in records])
    counts = np.asarray([[record["empty_count"], record["surface_count"], record["medium_count"]] for record in records])
    metrics = [record["render_metrics"] for record in records]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes[0, 0].stackplot(iterations, counts[:, 0], counts[:, 1], counts[:, 2], labels=("empty", "surface", "medium"), colors=("#555555", "#d62728", "#1f77b4"))
    axes[0, 0].set(title="Otsu class counts", xlabel="iteration", ylabel="cells")
    axes[0, 0].legend()
    axes[0, 1].plot(iterations, [record["threshold_log10"] for record in records], marker="o")
    axes[0, 1].set(title="Otsu threshold drift", xlabel="iteration", ylabel="log10 local density TV")
    axes[1, 0].plot(iterations, [item["surface_psnr_vs_lego"] for item in metrics], marker="o", label="SURFACE vs lego")
    axes[1, 0].plot(iterations, [item["medium_psnr_vs_cloud"] for item in metrics], marker="o", label="MEDIUM vs clouds")
    axes[1, 0].set(title="Separated layer quality", xlabel="iteration", ylabel="PSNR [dB]")
    axes[1, 0].legend()
    axes[1, 1].plot(iterations, [item["surface_alpha_on_cloud_only"] for item in metrics], marker="o", label="surface → cloud")
    axes[1, 1].plot(iterations, [item["medium_alpha_on_lego_only"] for item in metrics], marker="o", label="medium → lego")
    axes[1, 1].set(title="Cross-layer alpha leakage", xlabel="iteration", ylabel="mean alpha")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    fig.savefig(output / "separation_dashboard.png", dpi=160)
    plt.close(fig)


def _evolution_montage(run: Path, output: Path) -> None:
    iterations = (0, 500, 1000, 2000, 3000, 4000, 5000)
    rows = []
    for iteration in iterations:
        directory = run / "evaluation" / f"iteration_{iteration:06d}"
        target = Image.open(directory / "0000_r_000_panel.png").convert("RGB")
        width = target.width // 6
        target = target.crop((0, 0, width, target.height))
        render = Image.open(directory / "0000_r_000_voronoi_only.png").convert("RGB")
        error = Image.open(directory / "0000_r_000_error.png").convert("RGB")
        trans = Image.open(directory / "0000_r_000_transmittance.png").convert("RGB")
        rows.append([
            (f"target @ {iteration}", target),
            (f"Voronoi @ {iteration}", render),
            (f"absolute error @ {iteration}", error),
            (f"transmittance @ {iteration}", trans),
        ])
    _grid(rows).save(output / "reconstruction_evolution_view00.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument(
        "--iterations", nargs="+", type=int, default=[500, 1000, 2000, 3000, 4000, 5000]
    )
    parser.add_argument("--image-views", type=int, default=4)
    args = parser.parse_args()

    run = Path(args.run).resolve()
    output = run / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    _plot_training(run, output)
    _evolution_montage(run, output)

    records: list[dict[str, object]] = []
    comparison_rows: list[list[tuple[str, Image.Image]]] = []
    for iteration in args.iterations:
        checkpoint = run / "checkpoints" / f"iter_{iteration:08d}.pt"
        checkpoint_output = output / f"iteration_{iteration:06d}"
        checkpoint_output.mkdir(parents=True, exist_ok=True)
        config = ExperimentConfig.load(run / "config.yaml")
        config.output_dir = str(checkpoint_output / "diagnostics")
        config.wandb.mode = "disabled"
        config.validation.image_views = 0
        config.diagnostics.enabled = True
        config.diagnostics.save_per_primitive = False
        trainer = Trainer(config, resume=checkpoint)
        try:
            summary, _, histograms = trainer.evaluator.evaluate(iteration)
            contribution = histograms["medium_contribution"]
            density = trainer.model.medium.densities.reshape(-1)
            local_tv = _local_density_tv(
                density, trainer.topology.adjacency, trainer.topology.offsets
            )
            active = contribution > 0
            score = torch.log10(local_tv + 1.0e-12)
            threshold = _otsu(score[active])
            labels = torch.full_like(score, EMPTY, dtype=torch.int64)
            labels[active & (score >= threshold)] = SURFACE
            labels[active & (score < threshold)] = MEDIUM
            render_metrics = _render_separation(
                trainer, labels, checkpoint_output, iteration, args.image_views
            )
            counts = {name: int((labels == label).sum()) for label, name in CLASS_NAMES.items()}
            record = {
                "iteration": iteration,
                "cell_count": int(labels.numel()),
                "active_count": int(active.sum()),
                "active_fraction": float(active.float().mean()),
                "threshold_log10": threshold,
                "score_p01": float(torch.quantile(score[active], 0.01)),
                "score_p50": float(torch.quantile(score[active], 0.50)),
                "score_p99": float(torch.quantile(score[active], 0.99)),
                "empty_count": counts["empty"],
                "surface_count": counts["surface"],
                "medium_count": counts["medium"],
                "full_eval_psnr": float(summary["eval/primary_psnr"]),
                "render_metrics": render_metrics,
            }
            records.append(record)
            torch.save(
                {
                    "labels": labels.cpu(),
                    "class_names": CLASS_NAMES,
                    "classifier": "local_contrast_otsu",
                    "threshold_log10": threshold,
                    "contribution": contribution.cpu(),
                    "sigma_total_variation": local_tv.cpu(),
                    "source_checkpoint": str(checkpoint),
                },
                checkpoint_output / "cell_labels.pt",
            )
            (checkpoint_output / "classifier.json").write_text(
                json.dumps(record, indent=2) + "\n"
            )
            comparison_rows.append([
                (f"full {iteration}", Image.open(checkpoint_output / "view_00_full.png").convert("RGB")),
                (f"SURFACE {iteration}", Image.open(checkpoint_output / "view_00_surface.png").convert("RGB")),
                (f"MEDIUM {iteration}", Image.open(checkpoint_output / "view_00_medium.png").convert("RGB")),
                (f"EMPTY {iteration}", Image.open(checkpoint_output / "view_00_empty.png").convert("RGB")),
                (f"classes {iteration}", Image.open(checkpoint_output / "view_00_class_overlay.png").convert("RGB")),
            ])
        finally:
            trainer.logger.close()
            del trainer
            torch.cuda.empty_cache()

    (output / "analysis.json").write_text(json.dumps(records, indent=2) + "\n")
    _plot_separation(records, output)
    _grid(comparison_rows).save(output / "checkpoint_separation_comparison_view00.png")
    print(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
