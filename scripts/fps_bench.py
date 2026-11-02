#!/usr/bin/env python3
"""Rendering speed of trained runs on every test view, as a viewer renders a static scene: the OptiX
geometry is uploaded once per run; one frame = view-dependent Gaussian colours + hybrid render + PPISP
camera model (when the run has one), at the evaluation transmittance threshold. Reported per run and
scene: the median frame time over N repetitions per view, FPS, and the PSNR of the frame (so speed options
can be checked for quality). Paths compared:
- eval: what the evaluator times (render only, with the surface pass and input checks);
- view: render(surface=False, check_inputs=False) + colours + PPISP;
The OptiX K-hit buffer is --hit-buffer (default 64; one OptiX module per process, so K=32 is another run).
--threshold overrides the evaluation transmittance threshold (rays end when T falls below it).
Usage: python scripts/fps_bench.py RUN_DIR [RUN_DIR ...] [--hit-buffer 32] [--repeats 20] [--threshold 1e-4] [--json OUT]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.evaluator import _rays  # noqa: E402
from training.camera_isp import pixel_grid  # noqa: E402
from training.config import ExperimentConfig  # noqa: E402
from training.trainer import Trainer  # noqa: E402


def timed(fn, repeats):
    times = []
    for i in range(repeats + 3):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); start.record()
        out = fn()
        end.record(); torch.cuda.synchronize()
        if i >= 3:
            times.append(start.elapsed_time(end))
    return out, statistics.median(times)


def bench(run: Path, hit_buffer, repeats, threshold=None):
    config = ExperimentConfig.load(run / "joint/config.yaml")
    config.output_dir = str(ROOT / "outputs/fps_bench/_trainer" / run.name)
    config.wandb.mode = "disabled"
    config.renderer.gaussian_hit_buffer_size = hit_buffer  # one OptiX module per process
    trainer = Trainer(config, resume=run / "joint/checkpoints/last.pt")
    model, isp = trainer.model, trainer.camera_isp
    bg = tuple(config.background)
    renderers = {"eval": trainer.renderer, "view": trainer.renderer}
    for renderer in set(renderers.values()):
        renderer.transmittance_threshold = (config.renderer.validation_transmittance_threshold
                                            if threshold is None else threshold)
        trainer.topology.synchronize_renderer(renderer, model.medium)
    medium = model.medium.render_scene()
    with torch.no_grad():
        scene_any = model.gaussians.render_scene(trainer.dataset.test[0].origin)
        for renderer in set(renderers.values()):
            renderer.set_gaussian_geometry(scene_any)  # once: the geometry does not depend on the view
    results = {name: {"ms": [], "psnr": []} for name in renderers}
    with torch.no_grad():
        for cam in trainer.dataset.test:
            H, W = cam.height, cam.width
            rays = _rays(cam)
            coords = pixel_grid(H, W, rays.device).reshape(H, W, 2)
            gt = cam.ground_truth(bg).reshape(H, W, 3)
            for name, renderer in renderers.items():
                fast = name != "eval"

                def frame():
                    scene = model.gaussians.render_scene(cam.origin)  # same means tensor: geometry stays valid
                    out = renderer.render(rays, gaussians=scene, voronoi=medium, background=bg,
                                          surface=not fast, check_inputs=not fast)
                    image = out.image.reshape(H, W, 3)
                    return image if isp is None else isp.apply(image, coords, (H, W), None, cam.origin)

                if fast:
                    image, ms = timed(frame, repeats)
                else:  # the evaluator's number: render only
                    scene = model.gaussians.render_scene(cam.origin)
                    _, ms = timed(lambda: renderer.render(rays, gaussians=scene, voronoi=medium, background=bg), repeats)
                    image = frame()
                results[name]["ms"].append(ms)
                results[name]["psnr"].append(float(-10 * torch.log10(((image.clamp(0, 1) - gt) ** 2).mean())))
    summary = {name: {"ms": round(statistics.mean(r["ms"]), 3), "fps": round(1000 / statistics.mean(r["ms"]), 1),
                      "psnr": round(statistics.mean(r["psnr"]), 3)} for name, r in results.items()}
    return {"run": run.name, "hit_buffer": hit_buffer, "threshold": renderer.transmittance_threshold, "gaussians": model.gaussians.point_count, "views": len(trainer.dataset.test),
            "resolution": [trainer.dataset.test[0].width, trainer.dataset.test[0].height], **summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--hit-buffer", type=int, default=64, help="OptiX K-hit buffer (models trained with 64)")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=None, help="evaluation transmittance threshold")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    rows = []
    for run in args.runs:
        row = bench(run.resolve(), args.hit_buffer, args.repeats, args.threshold)
        rows.append(row)
        print(json.dumps(row), flush=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
