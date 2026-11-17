#!/usr/bin/env python3
"""VGG LPIPS of finished runs on every test view, from the final checkpoint, as the baselines in the paper's tables:
torchmetrics LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True) on the float frame in [0, 1].

The frame is the evaluator's main image (evaluation/evaluator.py: hybrid render at the validation transmittance
threshold, the run's background, then the water light and the camera model when the run has them, clamped). Check:
the PSNR of every frame must equal the PSNR the evaluator saved for that view at the final iteration
(joint/evaluation/iteration_<N>/per_view.csv) within PSNR_TOLERANCE, so LPIPS is measured on the same image.
Runs with the evaluation-time water-light extension are refused (it is re-fitted before each evaluation; use
foamsurf/scripts/fps_bench.py --lpips for SeaThru runs).

Writes RUN/joint/lpips_vgg.json (mean and per view) and prints one JSON line per run.
Usage: python scripts/lpips_vgg.py RUN_DIR [RUN_DIR ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PSNR_TOLERANCE = 0.01


def saved_psnr(run: Path, iteration: int) -> dict[str, float]:
    path = run / "joint/evaluation" / f"iteration_{iteration:06d}" / "per_view.csv"
    with open(path) as f:
        return {r["view_name"]: float(r["hybrid_psnr"]) for r in csv.DictReader(f)}


def check_psnr(measured: dict[str, float], saved: dict[str, float], tolerance: float = PSNR_TOLERANCE) -> float:
    """Largest per-view PSNR gap; raises when a view is missing or a gap exceeds the tolerance."""
    if set(measured) != set(saved):
        raise ValueError(f"views differ: rendered {sorted(measured)}, saved {sorted(saved)}")
    gap = max(abs(measured[v] - saved[v]) for v in measured)
    if gap > tolerance:
        raise ValueError(f"frame PSNR differs from the saved evaluation by {gap:.4f} dB")
    return gap


@torch.no_grad()
def run_lpips(run: Path, lpips_net) -> dict:
    from evaluation.evaluator import _rays
    from training.config import ExperimentConfig
    from training.trainer import Trainer

    config = ExperimentConfig.load(run / "joint/config.yaml")
    if config.optimization.water_extrapolation:
        raise ValueError(f"{run}: water-light extension is re-fitted per evaluation; not supported here")
    config.output_dir = str(ROOT / "outputs/lpips_vgg/_trainer" / run.name)
    config.wandb.mode = "disabled"
    trainer = Trainer(config, resume=run / "joint/checkpoints/last.pt")
    model, renderer, evaluator = trainer.model, trainer.renderer, trainer.evaluator
    renderer.transmittance_threshold = config.renderer.validation_transmittance_threshold
    trainer.topology.synchronize_renderer(renderer, model.medium)
    medium = None if config.representation_mode == "gaussian_only" else model.medium.render_scene()
    bg = tuple(config.background)
    per_view = {}
    for camera in trainer.dataset.test:
        scene = model.gaussians.render_scene(camera.origin)
        renderer.set_gaussian_geometry(scene)
        rays = _rays(camera)
        out = renderer.render(rays, gaussians=scene, voronoi=medium, background=bg)
        image = (out.image if evaluator.water_light is None or out.backscatter is None
                 else evaluator.water_light(out.image, out.backscatter, rays))
        image = (image if evaluator.camera_isp is None else evaluator.camera_isp(image, camera)).clamp(0, 1)
        H, W = camera.height, camera.width
        image, gt = image.reshape(H, W, 3), camera.ground_truth(bg).reshape(H, W, 3)
        batch = lambda x: x.permute(2, 0, 1)[None].float()  # noqa: E731
        per_view[camera.name] = dict(psnr=float(-10 * torch.log10(((image - gt) ** 2).mean())),
                                     lpips_vgg=float(lpips_net(batch(gt), batch(image))))
    iteration = config.optimization.iterations
    gap = check_psnr({v: m["psnr"] for v, m in per_view.items()}, saved_psnr(run, iteration))
    return dict(run=str(run), iteration=iteration, views=len(per_view), background=list(bg),
                lpips_vgg=statistics.fmean(m["lpips_vgg"] for m in per_view.values()),
                psnr=statistics.fmean(m["psnr"] for m in per_view.values()), max_psnr_gap_to_evaluation=gap,
                implementation="torchmetrics LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True), "
                               "float frame", per_view=per_view)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    lpips_net = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).cuda().eval()
    for run in args.runs:
        result = run_lpips(run.resolve(), lpips_net)
        (run / "joint/lpips_vgg.json").write_text(json.dumps(result, indent=1))
        print(json.dumps({k: result[k] for k in ("run", "views", "lpips_vgg", "psnr", "max_psnr_gap_to_evaluation")}),
              flush=True)


if __name__ == "__main__":
    main()
