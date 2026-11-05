"""Render a trained FoamSurf run: the test or training views of its dataset, or any camera poses.

    python render.py --run RUN_DIR                                    # test views, every mode
    python render.py --run RUN_DIR --views train --modes full,gaussians
    python render.py --run RUN_DIR --poses cameras.json --modes gaussians,voronoi
    python render.py --run RUN_DIR --export-cameras cameras.json      # the dataset cameras as a pose template
    python render.py --run RUN_DIR --scene /data/Curasao-Undistorted  # dataset not at the path it was trained with
    python render.py --run RUN_DIR --modes full --metrics             # test PSNR / SSIM as in the evaluation

Modes (one PNG per view and mode, in OUT/<mode>/<view>.png; --npy also saves float arrays):
  full         the evaluation image: Gaussians + water medium, water light terms, camera model (PPISP)
  raw          the same before the camera model
  gaussians    the Gaussians alone on black: the scene without water ("restoration")
  alpha        accumulated Gaussian alpha
  voronoi      the Voronoi medium alone on black (no Gaussians)
  water        the water light in front of the surfaces in the full render (backscatter)
  depth        surface depth of the Gaussians (normalized per view in the PNG; metric in --npy)

Pose file: a JSON list of cameras in the dataset's (COLMAP) world frame,
  {"name": str, "origin": [x,y,z], "right": [...], "down": [...], "forward": [...],
   "width": int, "height": int, "fx": float, "fy": float, "cx": float, "cy": float}
Novel poses get the camera model's exposure and colour of their nearest training cameras, as the test views.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import torch

MODES = ("full", "raw", "gaussians", "alpha", "voronoi", "water", "depth")


def camera_record(camera) -> dict:
    fx, fy, cx, cy = camera.intrinsics if camera.intrinsics is not None else (
        camera.width / (2 * math.tan(camera.fov_x / 2)), camera.height / (2 * math.tan(camera.fov_y / 2)),
        camera.width / 2, camera.height / 2)
    vector = lambda value: [float(x) for x in value.detach().cpu().reshape(3)]
    return {"name": camera.name, "origin": vector(camera.origin), "right": vector(camera.right),
            "down": vector(camera.down), "forward": vector(camera.forward), "width": camera.width,
            "height": camera.height, "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)}


def camera_from_record(record: dict, device):
    from foamsurf_data.blender import Camera
    width, height = int(record["width"]), int(record["height"])
    tensor = lambda key: torch.tensor(record[key], dtype=torch.float32, device=device)
    fx, fy = float(record["fx"]), float(record["fy"])
    return Camera(name=str(record["name"]), origin=tensor("origin"), right=tensor("right"), down=tensor("down"),
                  forward=tensor("forward"), fov_x=2 * math.atan(width / (2 * fx)), fov_y=2 * math.atan(height / (2 * fy)),
                  foreground=torch.zeros(height, width, 3, device=device), alpha=torch.ones(height, width, 1, device=device),
                  intrinsics=(fx, fy, float(record["cx"]), float(record["cy"])))


def load_run(run: Path, checkpoint: str, scene: Path | None = None):
    """A trainer restored from the run's checkpoint (nothing in the run directory is written). ``scene`` replaces
    the dataset path stored in the run (a run trained on another machine)."""
    from training.config import ExperimentConfig
    from training.trainer import Trainer
    joint = run / "joint" if (run / "joint").is_dir() else run
    config = ExperimentConfig.load(joint / "config.yaml")
    if scene is not None:
        config.dataset.source_path = str(Path(scene).resolve())
    if not Path(config.dataset.source_path).is_dir():
        raise FileNotFoundError(f"scene {config.dataset.source_path} not found: pass --scene with the dataset directory")
    config.output_dir = tempfile.mkdtemp(prefix="foamsurf_render_")
    config.wandb.mode = "disabled"
    trainer = Trainer(config, resume=joint / "checkpoints" / f"{checkpoint}.pt")
    trainer.renderer.transmittance_threshold = config.renderer.validation_transmittance_threshold
    trainer.topology.synchronize_renderer(trainer.renderer, trainer.model.medium)
    if config.optimization.water_extrapolation:
        trainer.fit_water_extrapolation()
    return trainer


@torch.no_grad()
def render_view(trainer, camera, modes) -> dict[str, torch.Tensor]:
    """Images [H, W, C] of the requested modes for one camera."""
    from evaluation.evaluator import _rays
    from training.camera_isp import pixel_grid
    model, renderer = trainer.model, trainer.renderer
    background = tuple(trainer.config.background)
    height, width = camera.height, camera.width
    rays = _rays(camera).reshape(-1, 6)
    gaussians = model.gaussians.render_scene(camera.origin)
    renderer.set_gaussian_geometry(gaussians)
    medium = model.medium.render_scene()
    out = {}
    if {"full", "raw", "water"} & set(modes):
        hybrid = renderer.render(rays, gaussians=gaussians, voronoi=medium, background=background)
        light = trainer.evaluator.water_light or (lambda image, backscatter, rays: image)
        raw = light(hybrid.image, hybrid.backscatter, rays)
        if "raw" in modes:
            out["raw"] = raw
        if "full" in modes:
            isp = trainer.camera_isp
            out["full"] = (raw if isp is None else isp.apply(raw, pixel_grid(height, width, raw.device).reshape(-1, 2),
                                                             (height, width), None, camera.origin)).clamp(0, 1)
        if "water" in modes and hybrid.backscatter is not None:
            out["water"] = light(hybrid.backscatter, hybrid.backscatter, rays)
    if {"gaussians", "alpha", "depth"} & set(modes):
        only = renderer.render_gaussians(rays, gaussians, background=(0.0, 0.0, 0.0))
        if "gaussians" in modes:
            out["gaussians"] = only.image
        if "alpha" in modes:
            out["alpha"] = only.alpha
        if "depth" in modes:
            out["depth"] = only.surface_depth
    if "voronoi" in modes:
        out["voronoi"] = renderer.render_voronoi(rays, medium, background=(0.0, 0.0, 0.0)).image
    return {name: value.reshape(height, width, -1) for name, value in out.items()}


def to_png(name: str, image: torch.Tensor) -> np.ndarray:
    value = image.detach().float().cpu().numpy()
    if name == "depth":
        finite = np.isfinite(value) & (value > 0)
        top = float(np.quantile(value[finite], 0.99)) if finite.any() else 1.0
        value = np.where(finite, value / max(top, 1e-8), 0.0)
    value = np.clip(value, 0.0, 1.0)
    if value.shape[-1] == 1:
        value = value[..., 0]
    return (value * 255.0 + 0.5).astype(np.uint8)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="run directory written by train.py")
    parser.add_argument("--checkpoint", default="last", choices=("last", "best"))
    parser.add_argument("--scene", type=Path, help="dataset directory of the run, when not at the path it was trained with")
    parser.add_argument("--views", default="test", choices=("test", "train", "all"))
    parser.add_argument("--poses", type=Path, help="JSON list of cameras to render instead of the dataset views")
    parser.add_argument("--modes", default=",".join(MODES), help="comma-separated, from: " + ", ".join(MODES))
    parser.add_argument("--out", type=Path, help="output directory (default RUN/renders)")
    parser.add_argument("--npy", action="store_true", help="also save float32 arrays (.npy)")
    parser.add_argument("--metrics", action="store_true",
                        help="dataset views: PSNR / SSIM of the full image against the photos, in OUT/metrics.json")
    parser.add_argument("--export-cameras", type=Path, metavar="JSON", help="write the dataset cameras and stop")
    args = parser.parse_args(argv)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = sorted(set(modes) - set(MODES))
    if unknown:
        parser.error(f"unknown modes: {', '.join(unknown)}")

    trainer = load_run(args.run.resolve(), args.checkpoint, args.scene)
    dataset = trainer.dataset
    if args.export_cameras:
        records = [dict(camera_record(c), split=split) for split, cams in (("train", dataset.train), ("test", dataset.test))
                   for c in cams]
        args.export_cameras.write_text(json.dumps(records, indent=1))
        print(f"wrote {len(records)} cameras to {args.export_cameras}")
        return
    if args.poses:
        cameras = [camera_from_record(r, trainer.device) for r in json.loads(args.poses.read_text())]
    else:
        cameras = {"test": dataset.test, "train": dataset.train, "all": dataset.train + dataset.test}[args.views]
    out = (args.out or args.run / "renders").resolve()
    if args.metrics and (args.poses or "full" not in modes):
        parser.error("--metrics needs dataset views (no --poses) and the full mode")
    scores = {}
    for camera in cameras:
        images = render_view(trainer, camera, modes)
        if args.metrics:
            from evaluation.metrics import psnr, ssim
            target = camera.ground_truth(tuple(trainer.config.background)).reshape(camera.height, camera.width, 3)
            scores[camera.name] = {"psnr": float(psnr(images["full"], target)), "ssim": float(ssim(images["full"], target))}
        for name, image in images.items():
            folder = out / name
            folder.mkdir(parents=True, exist_ok=True)
            Image.fromarray(to_png(name, image)).save(folder / f"{camera.name}.png")
            if args.npy:
                np.save(folder / f"{camera.name}.npy", image.float().cpu().numpy())
        print(camera.name, ", ".join(images), *(f"{k} {v:.3f}" for k, v in scores.get(camera.name, {}).items()), flush=True)
    if args.metrics:
        mean = {k: float(np.mean([v[k] for v in scores.values()])) for k in ("psnr", "ssim")}
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps({"mean": mean, "views": scores}, indent=1))
        print(f"mean PSNR {mean['psnr']:.3f}  SSIM {mean['ssim']:.4f}")
    print(f"renders in {out}")


if __name__ == "__main__":
    main()
