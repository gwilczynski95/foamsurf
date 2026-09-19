#!/usr/bin/env python3
"""Privileged layer-GT upper bound for native Voronoi warmup separation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_voronoi_warmup import (
    CLASS_NAMES,
    EMPTY,
    MEDIUM,
    SURFACE,
    _render_separation,
)
from training import ExperimentConfig
from training.trainer import Trainer


def _rays(camera) -> torch.Tensor:
    bundle = camera.rays()
    return torch.cat((bundle.origins, bundle.directions), dim=-1).contiguous()


@torch.no_grad()
def _oracle_labels(trainer: Trainer) -> tuple[torch.Tensor, dict[str, float]]:
    renderer = trainer.renderer
    renderer.transmittance_threshold = (
        trainer.config.renderer.validation_transmittance_threshold
    )
    trainer.topology.synchronize_renderer(renderer, trainer.model.medium)
    scene = trainer.model.medium.render_scene()
    count = trainer.model.medium.point_count
    surface_contribution = torch.zeros(count, device=trainer.device)
    medium_contribution = torch.zeros(count, device=trainer.device)

    for index, camera in enumerate(trainer.dataset.test):
        rays = _rays(camera)
        target = camera.ground_truth(tuple(trainer.config.background))
        lego_alpha = trainer.layer_ground_truth.lego_lego[index].alpha.to(
            trainer.device
        ).squeeze(-1)
        cloud_alpha = trainer.layer_ground_truth.lego_clouds[index].alpha.to(
            trainer.device
        ).squeeze(-1)
        masks = (
            ((lego_alpha > 0.01) & (cloud_alpha <= 0.01), surface_contribution),
            ((cloud_alpha > 0.01) & (lego_alpha <= 0.01), medium_contribution),
        )
        for mask, accumulator in masks:
            if not bool(mask.any()):
                continue
            result = renderer.render(
                rays[mask],
                voronoi=scene,
                background=tuple(trainer.config.background),
                collect_diagnostics=True,
                diagnostic_target=target[mask],
            )
            accumulator += result.diagnostics.medium[:, 0]

    total = surface_contribution + medium_contribution
    active = total > 0
    labels = torch.full((count,), EMPTY, dtype=torch.int64, device=trainer.device)
    labels[active & (surface_contribution >= medium_contribution)] = SURFACE
    labels[active & (medium_contribution > surface_contribution)] = MEDIUM
    stats = {
        "exclusive_ray_surface_contribution": float(surface_contribution.sum()),
        "exclusive_ray_medium_contribution": float(medium_contribution.sum()),
        "active_fraction": float(active.float().mean()),
    }
    return labels, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument(
        "--iterations", nargs="+", type=int, default=[4000, 5000]
    )
    parser.add_argument("--image-views", type=int, default=4)
    args = parser.parse_args()

    run = Path(args.run).resolve()
    records = []
    for iteration in args.iterations:
        output = run / "analysis" / f"iteration_{iteration:06d}" / "oracle"
        output.mkdir(parents=True, exist_ok=True)
        config = ExperimentConfig.load(run / "config.yaml")
        config.output_dir = str(output / "diagnostics")
        config.wandb.mode = "disabled"
        config.validation.image_views = 0
        trainer = Trainer(
            config,
            resume=run / "checkpoints" / f"iter_{iteration:08d}.pt",
        )
        try:
            labels, contribution_stats = _oracle_labels(trainer)
            metrics = _render_separation(
                trainer, labels, output, iteration, args.image_views
            )
            checkpoint_only = torch.load(
                run
                / "analysis"
                / f"iteration_{iteration:06d}"
                / "cell_labels.pt",
                map_location="cpu",
                weights_only=False,
            )["labels"].to(labels.device)
            active = labels != EMPTY
            agreement = float((labels[active] == checkpoint_only[active]).float().mean())
            confusion = {
                f"oracle_{CLASS_NAMES[oracle]}__pred_{CLASS_NAMES[predicted]}": int(
                    ((labels == oracle) & (checkpoint_only == predicted)).sum()
                )
                for oracle in CLASS_NAMES
                for predicted in CLASS_NAMES
            }
            record = {
                "iteration": iteration,
                "counts": {
                    name: int((labels == label).sum())
                    for label, name in CLASS_NAMES.items()
                },
                **contribution_stats,
                "checkpoint_only_agreement_on_oracle_active": agreement,
                "confusion": confusion,
                "render_metrics": metrics,
            }
            records.append(record)
            torch.save(
                {
                    "labels": labels.cpu(),
                    "class_names": CLASS_NAMES,
                    "classifier": "exclusive_layer_contribution_oracle",
                    "source_checkpoint": str(
                        run / "checkpoints" / f"iter_{iteration:08d}.pt"
                    ),
                },
                output / "cell_labels.pt",
            )
            (output / "oracle.json").write_text(
                json.dumps(record, indent=2) + "\n"
            )
        finally:
            trainer.logger.close()
            del trainer
            torch.cuda.empty_cache()

    destination = run / "analysis" / "oracle_analysis.json"
    destination.write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
