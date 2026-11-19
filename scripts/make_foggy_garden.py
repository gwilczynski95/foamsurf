#!/usr/bin/env python3
"""Generate the paper's easy/hard Foggy Garden datasets from clear photos and exported camera-z depths.

Depth export uses a trained clear-scene Nerfstudio model (see --export-depth --help).
Image generation needs NumPy, Pillow and pycolmap; Nerfstudio is optional for depth export.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from PIL import Image

from dataclasses import dataclass

import numpy as np

UNITS = ("nerfstudio", "colmap")


@dataclass(frozen=True)
class FogParams:
    name: str
    beta_d: tuple[float, float, float]
    beta_b: tuple[float, float, float]
    b_inf: tuple[float, float, float]


FOG_PRESETS = {
    # Quoted in the paper's implementation details; easy matches Fig. 5 (free fit gives 0.64/0.64/0.495).
    "easy": FogParams("easy", (0.6, 0.6, 0.6), (0.6, 0.6, 0.6), (0.5, 0.5, 0.5)),
    "hard_text": FogParams("hard_text", (0.8, 0.8, 0.8), (0.6, 0.6, 0.6), (0.5, 0.5, 0.5)),
    # Rounded free fit to the paper's Fig. 5 hard image (fit: 1.08 / 0.72 / 0.47; MSE 6x lower than hard_text).
    "hard_fig5": FogParams("hard_fig5", (1.1, 1.1, 1.1), (0.7, 0.7, 0.7), (0.47, 0.47, 0.47)),
}

def ray_distance(z_depth: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Camera-z depth [H, W] -> distance along each pixel's ray (pixel centres at +0.5, COLMAP convention)."""
    height, width = z_depth.shape
    u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    return z_depth * np.sqrt(((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2 + 1.0)


def apply_fog(clear: np.ndarray, depth: np.ndarray, params: FogParams) -> np.ndarray:
    """clear: [H, W, 3] float in [0, 1]; depth: [H, W]. Returns the foggy image in [0, 1]."""
    z = depth[..., None].astype(np.float64)
    direct = clear * np.exp(-np.asarray(params.beta_d) * z)
    backscatter = np.asarray(params.b_inf) * (1.0 - np.exp(-np.asarray(params.beta_b) * z))
    return np.clip(direct + backscatter, 0.0, 1.0)


def depth_in_units(depth_ns: np.ndarray, units: str, scale_factor: float) -> np.ndarray:
    """Convert nerfstudio-scene depth to the requested units.

    Nerfstudio maps COLMAP poses with a rigid transform followed by a uniform scale
    ``scale_factor``, so distances scale by exactly that factor.
    """
    if units == "nerfstudio":
        return depth_ns
    if units == "colmap":
        return depth_ns / scale_factor
    raise ValueError(f"Unknown units {units!r}, expected one of {UNITS}")


def to_float(image_uint8: np.ndarray) -> np.ndarray:
    return image_uint8.astype(np.float64) / 255.0


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def rescale_reconstruction(reconstruction, width: int, height: int, name_map: dict[str, str]):
    """Rescale a pycolmap reconstruction in place to `width` x `height` and rename images.

    Intrinsics scale by width/W and height/H (pycolmap Camera.rescale); 2D keypoints scale the same way.
    """
    ratios = {}
    for camera_id, camera in reconstruction.cameras.items():
        ratios[camera_id] = np.array([width / camera.width, height / camera.height])
        camera.rescale(width, height)
    for image in reconstruction.images.values():
        ratio = ratios[image.camera_id]
        for point in image.points2D:
            point.xy = point.xy * ratio
        image.name = name_map[image.name]
    return reconstruction


def scaled_model(colmap_model: Path, views: list[str], size: tuple[int, int]):
    import pycolmap
    reconstruction = pycolmap.Reconstruction(str(colmap_model))
    rescale_reconstruction(reconstruction, *size, {f"{v}.JPG": f"{v}.png" for v in views})
    return reconstruction


def ray_depths(reconstruction, depth_dir: Path) -> dict[str, np.ndarray]:
    """Per-view ray distance from the rendered z-depth, using the (rescaled) intrinsics of each view."""
    depths = {}
    for image in reconstruction.images.values():
        view = Path(image.name).stem
        fx, fy, cx, cy = reconstruction.cameras[image.camera_id].params
        depths[view] = ray_distance(np.load(depth_dir / f"{view}.npy"), fx, fy, cx, cy)
    return depths


def make_variant(out: Path, preset: str, views: list[str], clear_dir: Path, depths: dict[str, np.ndarray],
                 reconstruction, info_common: dict) -> dict:
    params = FOG_PRESETS[preset]
    for sub in ("images", "clear", "depth", "sparse/0"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    transmission = []
    for view in views:
        clear_u8 = np.asarray(Image.open(clear_dir / f"{view}.JPG").convert("RGB"))
        depth = depths[view]
        assert depth.shape == clear_u8.shape[:2], (view, depth.shape, clear_u8.shape)
        Image.fromarray(to_uint8(apply_fog(to_float(clear_u8), depth, params))).save(out / "images" / f"{view}.png")
        Image.fromarray(clear_u8).save(out / "clear" / f"{view}.png")
        np.save(out / "depth" / f"{view}.npy", depth.astype(np.float32))
        transmission.append(float(np.mean(np.exp(-params.beta_d[0] * depth))))
    reconstruction.write(str(out / "sparse" / "0"))

    names = sorted(f"{v}.png" for v in views)
    test = names[::8]
    train = [name for index, name in enumerate(names) if index % 8]
    (out / "train_list.txt").write_text("\n".join(train) + "\n", encoding="utf8")
    for split in ("val", "test"):
        (out / f"{split}_list.txt").write_text("\n".join(test) + "\n", encoding="utf8")

    info = {"preset": preset, "fog": asdict(params), "num_views": len(views), "num_train": len(train),
            "num_test": len(test), "mean_direct_transmission": float(np.mean(transmission)), **info_common}
    (out / "params.json").write_text(json.dumps(info, indent=2), encoding="utf8")
    print(f"[make] {out.name}: mean direct transmission {info['mean_direct_transmission']:.3f}", flush=True)
    return info




def export_depth() -> None:
    import torch
    import pycolmap
    from nerfstudio.cameras.cameras import Cameras, CameraType
    from nerfstudio.utils.eval_utils import eval_setup
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--colmap-model", type=Path, required=True, help="original full-res sparse model")
    parser.add_argument("--images-dir", type=Path, required=True, help="clear images that will be fogged")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    _, pipeline, checkpoint, step = eval_setup(args.load_config, test_mode="test")
    pipeline.eval()
    model = pipeline.model
    dataset = pipeline.datamanager.eval_dataset  # eval-mode "all": every view
    reconstruction = pycolmap.Reconstruction(str(args.colmap_model))
    camera_of = {image.name: reconstruction.cameras[image.camera_id] for image in reconstruction.images.values()}

    scale_json = args.load_config.parent / "dataparser_transforms.json"
    scale_factor = float(json.loads(scale_json.read_text())["scale"])

    out_depth = args.output_dir / "depth_ns"
    out_depth.mkdir(parents=True, exist_ok=True)
    per_view = {}
    for index, filename in enumerate(dataset.image_filenames):
        name = Path(filename).name
        clear = np.asarray(Image.open(args.images_dir / name).convert("RGB"))
        height, width = clear.shape[:2]
        colmap_camera = pycolmap.Camera(camera_of[name].todict())
        colmap_camera.rescale(width, height)
        assert colmap_camera.model == pycolmap.CameraModelId.PINHOLE, colmap_camera.model
        fx, fy, cx, cy = colmap_camera.params
        camera = Cameras(
            camera_to_worlds=dataset.cameras.camera_to_worlds[index : index + 1],
            fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
            width=width, height=height, camera_type=CameraType.PERSPECTIVE,
        )
        outputs = model.get_outputs_for_camera(camera)
        depth = outputs["depth"][..., 0].float().cpu().numpy()
        rgb = outputs["rgb"].clamp(0, 1).float().cpu().numpy()
        assert depth.shape == (height, width), (depth.shape, (height, width))
        np.save(out_depth / f"{Path(name).stem}.npy", depth.astype(np.float32))
        mse = float(np.mean((rgb - clear / 255.0) ** 2))
        per_view[Path(name).stem] = {
            "psnr_clear": float(-10 * np.log10(max(mse, 1e-12))),
            "depth_p01": float(np.percentile(depth, 1)),
            "depth_p50": float(np.percentile(depth, 50)),
            "depth_p99": float(np.percentile(depth, 99)),
        }
        print(f"[depth] {name}: psnr={per_view[Path(name).stem]['psnr_clear']:.2f} "
              f"depth p1/p50/p99={per_view[Path(name).stem]['depth_p01']:.3f}/"
              f"{per_view[Path(name).stem]['depth_p50']:.3f}/{per_view[Path(name).stem]['depth_p99']:.3f}", flush=True)

    summary = {
        "checkpoint": str(checkpoint),
        "step": step,
        "scale_factor": scale_factor,
        "depth": "splatfacto expected depth (camera z), nerfstudio scene units",
        "mean_psnr_clear": float(np.mean([v["psnr_clear"] for v in per_view.values()])),
        "per_view": per_view,
    }
    (args.output_dir / "depth_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(f"[depth] {len(per_view)} views, scale_factor={scale_factor:.6f}, "
          f"mean splatfacto PSNR on clear views={summary['mean_psnr_clear']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-dir", type=Path, required=True, help="dir with depth_ns/ and depth_summary.json")
    parser.add_argument("--clear-dir", type=Path, required=True)
    parser.add_argument("--colmap-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads((args.depth_dir / "depth_summary.json").read_text())
    views = sorted(summary["per_view"])
    size = Image.open(args.clear_dir / f"{views[0]}.JPG").size
    reconstruction = scaled_model(args.colmap_model, views, size)
    depths = ray_depths(reconstruction, args.depth_dir / "depth_ns")
    info_common = {
        "depth": "distance along camera ray, nerfstudio scene units (from splatfacto expected z-depth)",
        "scale_factor_nerfstudio_per_colmap_unit": summary["scale_factor"], "resolution": list(size),
        "clear_images": str(args.clear_dir), "colmap_model": str(args.colmap_model),
        "splatfacto_checkpoint": summary["checkpoint"],
        "formula": "I = O*exp(-beta_d*z) + b_inf*(1-exp(-beta_b*z))  (WaterSplatting Eq. 6)",
    }
    for preset in ("easy", "hard_fig5"):
        make_variant(args.output_dir / f"garden_{'hard' if preset == 'hard_fig5' else preset}", preset, views, args.clear_dir, depths, reconstruction,
                     info_common)


if __name__ == "__main__":
    import sys
    if "--export-depth" in sys.argv:
        sys.argv.remove("--export-depth")
        export_depth()
    else:
        main()
