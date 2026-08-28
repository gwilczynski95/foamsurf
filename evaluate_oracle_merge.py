#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from evaluation import Evaluator
from evaluation.metrics import psnr
from foamsurf import NativeHybridRenderer
from foamsurf_data import LayerGroundTruth, load_dataset
from foamsurf_model import VoronoiTopologyController
from training import ExperimentConfig
from training.experiment_logging import ExperimentLogger
from training.oracle_merge import (
    branch_tensor_digest,
    checkpoint_sha256,
    load_oracle_model,
)


def _renderer(config: ExperimentConfig, maximum_rays: int) -> NativeHybridRenderer:
    settings = config.renderer
    return NativeHybridRenderer(
        max_batch_size=maximum_rays,
        near=settings.near,
        far=settings.far,
        transmittance_threshold=settings.validation_transmittance_threshold,
        max_intersections=settings.max_intersections,
        number_of_sides=settings.number_of_sides,
        gaussian_radius_squared=settings.gaussian_radius_squared,
        gaussian_hit_buffer_size=settings.gaussian_hit_buffer_size,
    )


def _source_renderer_config(
    payload: dict, oracle: ExperimentConfig
) -> ExperimentConfig:
    copied = deepcopy(oracle)
    for name, value in payload["config"]["renderer"].items():
        setattr(copied.renderer, name, value)
    return copied


@torch.no_grad()
def _reproduce_branch(
    branch: str,
    payload: dict,
    model,
    topology,
    dataset,
    layers: LayerGroundTruth,
    oracle_config: ExperimentConfig,
) -> dict[str, float | int | bool | None]:
    """Re-evaluate one imported branch with its source renderer contract."""

    source_config = _source_renderer_config(payload, oracle_config)
    renderer = _renderer(source_config, dataset.width * dataset.height)
    if branch == "voronoi":
        topology.synchronize_renderer(renderer, model.medium)
        medium_scene = model.medium.render_scene()

    values: list[float] = []
    for index, camera in enumerate(dataset.test):
        rays_bundle = camera.rays()
        rays = torch.cat(
            (rays_bundle.origins, rays_bundle.directions), dim=-1
        ).contiguous()
        if branch == "gaussian":
            scene = model.gaussians.render_scene(camera.origin)
            renderer.set_gaussian_geometry(scene)
            result = renderer.render_gaussians(
                rays, scene, background=tuple(oracle_config.background)
            )
            target = layers.lego_lego[index].ground_truth(
                tuple(oracle_config.background)
            ).to(result.image.device)
        elif branch == "voronoi":
            result = renderer.render_voronoi(
                rays,
                medium_scene,
                background=tuple(oracle_config.background),
            )
            target = layers.lego_clouds[index].ground_truth(
                tuple(oracle_config.background)
            ).to(result.image.device)
        else:
            raise ValueError(f"unsupported oracle branch: {branch}")
        values.append(float(psnr(result.image.clamp(0.0, 1.0), target).item()))

    measured = float(np.mean(values))
    checkpoint_path_is_best = bool(payload.get("_path_is_best", False))
    expected = float(payload["best_psnr"]) if checkpoint_path_is_best else None
    delta = measured - expected if expected is not None else None
    return {
        "view_count": len(values),
        "mean_psnr": measured,
        "checkpoint_best_psnr": expected,
        "psnr_delta": delta,
        "enforced": checkpoint_path_is_best,
    }


def _build_config(args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig.load(args.config)
    config.experiment_name = "oracle-standalone-merge-frozen-evaluation"
    config.output_dir = str(Path(args.output_dir).resolve())
    config.representation_mode = "hybrid"
    config.dataset.train_target = "both"
    config.gaussian_adaptation.enabled = False
    config.voronoi_adaptation.enabled = False
    config.validation.max_eval_views = args.max_eval_views
    config.validation.image_views = args.image_views
    config.wandb.mode = args.wandb_mode
    config.wandb.name = args.wandb_name
    config.wandb.tags = [
        "oracle",
        "frozen",
        "checkpoint-merge",
        "evaluation-only",
    ]
    config.validate()
    return config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Frozen oracle evaluation from qualified standalone checkpoints."
    )
    parser.add_argument("--gaussian-checkpoint", required=True)
    parser.add_argument("--voronoi-checkpoint", required=True)
    parser.add_argument(
        "--config",
        default="configs/baseline_best_2d_gaussians.yaml",
        help="Oracle renderer/config source; renderer must match Gaussian checkpoint.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-eval-views", type=int, default=-1)
    parser.add_argument("--image-views", type=int, default=4)
    parser.add_argument(
        "--reproduction-tolerance", type=float, default=5.0e-4
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--wandb-name")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("oracle evaluation requires CUDA")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _build_config(args)
    config.save(output_dir / "config.yaml")
    device = torch.device(config.device)
    dataset = load_dataset(
        config.dataset.source_path,
        resolution=config.dataset.resolution,
        device=device,
        llffhold=config.dataset.llffhold,
    )
    layers = LayerGroundTruth.load(
        dataset.test,
        config.dataset.gaussian_ground_truth_path,
        config.dataset.medium_ground_truth_path,
        resolution=config.dataset.resolution,
    )

    gaussian_path = Path(args.gaussian_checkpoint).resolve()
    medium_path = Path(args.voronoi_checkpoint).resolve()
    model, gaussian_payload, medium_payload, compatibility = load_oracle_model(
        gaussian_path, medium_path, config, device
    )
    gaussian_payload["_path_is_best"] = gaussian_path.name == "best.pt"
    medium_payload["_path_is_best"] = medium_path.name == "best.pt"
    topology = VoronoiTopologyController(model.medium)
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("oracle model was not frozen after topology rebuild")

    reproduction = {
        "gaussian": _reproduce_branch(
            "gaussian", gaussian_payload, model, topology, dataset, layers, config
        ),
        "voronoi": _reproduce_branch(
            "voronoi", medium_payload, model, topology, dataset, layers, config
        ),
    }
    for name, result in reproduction.items():
        delta = result["psnr_delta"]
        if result["enforced"] and (
            delta is None or abs(float(delta)) > args.reproduction_tolerance
        ):
            raise RuntimeError(
                f"{name} branch failed source-checkpoint reproduction: "
                f"delta={delta}, tolerance={args.reproduction_tolerance}"
            )

    maximum_rays = max(
        config.optimization.rays_per_batch,
        dataset.width * dataset.height,
    )
    renderer = _renderer(config, maximum_rays)
    renderer.set_gaussian_geometry(
        model.gaussians.render_scene(dataset.test[0].origin)
    )
    topology.synchronize_renderer(renderer, model.medium)
    evaluator = Evaluator(
        renderer,
        model,
        topology,
        dataset,
        config,
        output_dir,
        layer_ground_truth=layers,
    )
    logger = ExperimentLogger(config, output_dir, resume=False)
    started = time.perf_counter()
    try:
        metrics, images, histograms = evaluator.evaluate(iteration=0)
        logger.log_eval(metrics, images, histograms)
    finally:
        logger.close()
    evaluation_seconds = time.perf_counter() - started

    manifest = {
        "experiment_type": "oracle_standalone_checkpoint_merge",
        "oracle": True,
        "evaluation_only": True,
        "optimizer_steps": 0,
        "adaptation_events": 0,
        "gaussian_checkpoint": str(gaussian_path),
        "gaussian_checkpoint_sha256": checkpoint_sha256(gaussian_path),
        "gaussian_branch_tensor_sha256": branch_tensor_digest(
            gaussian_payload, "gaussians"
        ),
        "voronoi_checkpoint": str(medium_path),
        "voronoi_checkpoint_sha256": checkpoint_sha256(medium_path),
        "voronoi_branch_tensor_sha256": branch_tensor_digest(
            medium_payload, "medium"
        ),
        "compatibility": compatibility,
        "source_reproduction": reproduction,
        "renderer": asdict(config.renderer),
        "gaussian_count": model.gaussians.point_count,
        "voronoi_count": model.medium.point_count,
    }
    (output_dir / "oracle_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    summary = {
        "status": "completed_oracle_frozen_evaluation",
        "oracle": True,
        "evaluation_only": True,
        "optimizer_steps": 0,
        "adaptation_events": 0,
        "evaluation_seconds": evaluation_seconds,
        "gaussian_count": model.gaussians.point_count,
        "voronoi_count": model.medium.point_count,
        **metrics,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("FOAMSURF_ORACLE_RESULT=" + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
