#!/usr/bin/env python3
"""Evaluate surface/medium separation, clear-scene restoration and synthetic layers, or export transmittance."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from evaluation.metrics import ImageMetrics, depth_errors, psnr, view_opacity
from render import load_run, render_view


def summarize(rows):
    if not rows:
        raise ValueError("no test views")
    return {key: statistics.fmean(row[key] for row in rows.values() if row[key] is not None)
            if any(row[key] is not None for row in rows.values()) else None for key in next(iter(rows.values()))}


@torch.no_grad()
def separation(trainer, masks_path):
    with np.load(masks_path) as archive:
        masks = {name: archive[name].astype(bool) for name in archive.files if name != "meta"}
    names = {camera.name for camera in trainer.dataset.test}
    if names != set(masks):
        raise ValueError(f"test views and masks differ: {sorted(names ^ set(masks))}")
    rows = {}
    for camera in trainer.dataset.test:
        alpha = render_view(trainer, camera, ["alpha"])["alpha"].cpu().numpy()
        seabed, water = view_opacity(alpha, masks[camera.name])
        rows[camera.name] = {"op_seabed": seabed, "op_water": water}
    mean = summarize(rows)
    mean["separation"] = None if mean["op_water"] is None else 0.5 * (mean["op_seabed"] + 1 - mean["op_water"])
    return {"mean": mean, "views": rows}


@torch.no_grad()
def restoration(trainer, data, metric):
    from evaluation.evaluator import _rays
    params = json.loads((data / "params.json").read_text())
    fog, scale = params["fog"], params["scale_factor_nerfstudio_per_colmap_unit"]
    beta_d, beta_b, b_inf = [torch.tensor(fog[key], device=trainer.device) for key in ("beta_d", "beta_b", "b_inf")]
    rows = {}
    for camera in trainer.dataset.test:
        height, width = camera.height, camera.width
        clear = torch.as_tensor(np.asarray(Image.open(data / "clear" / f"{camera.name}.png").convert("RGB"),
                                           dtype=np.float32) / 255, device=trainer.device)
        depth = torch.as_tensor(np.load(data / "depth" / f"{camera.name}.npy"), device=trainer.device).float()
        if clear.shape != (height, width, 3) or depth.shape != (height, width):
            raise ValueError(f"restoration references do not match {camera.name} resolution")
        rays = _rays(camera).reshape(-1, 6)
        g = trainer.model.gaussians.render_scene(camera.origin)
        trainer.renderer.set_gaussian_geometry(g)
        result = trainer.renderer.render(rays, gaussians=g, voronoi=trainer.model.medium.render_scene(), background=(0, 0, 0))
        clean = trainer.renderer.render_gaussians(rays, g, background=(0, 0, 0)).image.reshape(height, width, 3)
        direct = clear * torch.exp(-beta_d * depth[..., None])
        backscatter = b_inf * (1 - torch.exp(-beta_b * depth[..., None]))
        rows[camera.name] = {
            **{"restoration_" + key: value for key, value in metric(clean, clear).items()},
            "direct_psnr": float(psnr(result.direct.reshape_as(clear).clamp(0, 1), direct)),
            "backscatter_psnr": float(psnr(result.backscatter.reshape_as(clear).clamp(0, 1), backscatter)),
            **depth_errors(result.surface_depth.reshape_as(depth) * scale, depth),
        }
    return {"mean": summarize(rows), "views": rows, "fog": fog}


@torch.no_grad()
def layers(trainer, object_path=None, cloud_path=None):
    from evaluation.evaluator import _rays
    from evaluation.layer_metrics import decomposition_metrics
    from foamsurf_data.layer_ground_truth import LayerGroundTruth
    truth = trainer.evaluator.layer_ground_truth
    if object_path or cloud_path:
        if not object_path or not cloud_path:
            raise ValueError("--object and --cloud must be supplied together")
        truth = LayerGroundTruth.load(trainer.dataset.test, object_path, cloud_path, trainer.config.dataset.resolution)
    if truth is None:
        raise ValueError("layer ground truth is missing: pass --object and --cloud")
    rows = {}
    bg = tuple(trainer.config.background)
    for index, camera in enumerate(trainer.dataset.test):
        rays = _rays(camera)
        g = trainer.model.gaussians.render_scene(camera.origin)
        trainer.renderer.set_gaussian_geometry(g)
        surface = trainer.renderer.render_gaussians(rays, g, background=bg)
        medium = trainer.renderer.render_voronoi(rays, trainer.model.medium.render_scene(), background=bg)
        obj, cloud = truth.lego_lego[index], truth.lego_clouds[index]
        values, _ = decomposition_metrics(surface.image.clamp(0, 1), surface.alpha, medium.image.clamp(0, 1),
                                          medium.alpha, obj.ground_truth(bg).to(trainer.device),
                                          obj.alpha.to(trainer.device), cloud.ground_truth(bg).to(trainer.device),
                                          cloud.alpha.to(trainer.device), trainer.config.diagnostics.layer_alpha_threshold)
        rows[camera.name] = {"object_psnr": values["lego_lego/psnr"], "cloud_psnr": values["lego_clouds/psnr"],
                             "gauss_on_cloud": values["entanglement/gaussian_alpha_on_cloud_only"]}
    return {"mean": summarize(rows), "views": rows}


@torch.no_grad()
def transmittance(trainer, output, wanted=None):
    from evaluation.evaluator import _rays
    cameras = {camera.name: camera for camera in trainer.dataset.test}
    if wanted and set(wanted) - cameras.keys():
        raise ValueError(f"not test views: {sorted(set(wanted) - cameras.keys())}")
    output.mkdir(parents=True, exist_ok=True)
    rows = {}
    medium = trainer.model.medium.render_scene()
    for name, camera in cameras.items():
        if wanted and name not in wanted:
            continue
        result = trainer.renderer.render_voronoi(_rays(camera), medium, background=tuple(trainer.config.background))
        arrays = {"transmittance": result.transmittance.reshape(camera.height, camera.width).float().cpu().numpy()}
        if result.direct_transmittance is not None:
            arrays["direct_transmittance"] = result.direct_transmittance.reshape(camera.height, camera.width, -1).float().cpu().numpy()
        np.savez_compressed(output / f"{name}_medium_transmittance.npz", **arrays)
        for key, values in arrays.items():
            suffix = "_rgb" if key == "direct_transmittance" else ""
            Image.fromarray((np.clip(values, 0, 1) * 255 + 0.5).astype(np.uint8)).save(
                output / f"{name}_medium_transmittance{suffix}.png")
        rows[name] = {"transmittance": float(arrays["transmittance"].mean())}
    return {"mean": summarize(rows), "views": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("separation", "restoration", "layers", "transmittance"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--run", type=Path, required=True)
        sub.add_argument("--scene", type=Path)
        sub.add_argument("--out", type=Path, required=True, help="output directory; metrics are written to metrics.json")
        if name == "separation":
            sub.add_argument("--masks", type=Path, required=True)
        elif name == "restoration":
            sub.add_argument("--references", type=Path, help="Garden directory with clear/, depth/ and params.json")
            sub.add_argument("--metric-device", default="cpu")
        elif name == "layers":
            sub.add_argument("--object", type=Path)
            sub.add_argument("--cloud", type=Path)
        else:
            sub.add_argument("--views", nargs="+")
    args = parser.parse_args(argv)
    layer_paths = None
    if args.command == "layers" and (args.object or args.cloud):
        if not args.object or not args.cloud:
            parser.error("--object and --cloud must be supplied together")
        layer_paths = (args.object, args.cloud)
    trainer = load_run(args.run.resolve(), scene=args.scene, layer_paths=layer_paths)
    if args.command == "separation":
        result = separation(trainer, args.masks)
    elif args.command == "restoration":
        result = restoration(trainer, args.references or Path(trainer.config.dataset.source_path), ImageMetrics(args.metric_device))
    elif args.command == "layers":
        result = layers(trainer)
    else:
        result = transmittance(trainer, args.out, args.views)
    result.update(run=str(args.run.resolve()), checkpoint="last", evaluation=args.command)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["mean"], indent=2))


if __name__ == "__main__":
    main()
