#!/usr/bin/env python3
"""Open-water masks of the SeaThru-NeRF test views from the images alone, with Segment Anything.

Usage: python scripts/sam_water_mask.py --checkpoint sam_vit_h_4b8939.pth [--scene SCENE ...]
       [--datasets DIR] [--output-dir configs/water_sam] [--preview-dir DIR]

Writes <output-dir>/<scene>.npz (one boolean water mask per test view, key = view name)
for every scene of PROMPTS (or the --scene ones). A view without water prompts has no
open water in the frame: its mask is all False and SAM is not run.

Needs the `segment_anything` package (pip install --no-deps segment-anything) and the
ViT-H checkpoint. Each view is segmented with the point prompts below (fractions of
width/height): positives on open water, negatives on the seabed. Of SAM's three
proposals the one agreeing with the most prompts is kept (SAM's score breaks ties;
pick="score" keeps the highest-scoring one), the BORDER pixels at the frame edge take the label
just inside them, then holes and specks smaller than MIN_REGION of the image are
removed from both water and seabed. The result does
not depend on any FoamSurf model.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATASETS = "/shared/sets/datasets/SeathruNeRF_dataset"
MIN_REGION = 0.002  # fraction of the image
BORDER = 6  # SAM leaves a few unlabelled-looking rows at the frame edge; copied from inside
# Scene -> test view (every 8th COLMAP image by name) -> (x, y) prompts as fractions of
# the image, inspected view by view.
PROMPTS = {
    "IUI3-RedSea-Undistorted": {
        "MTN_5894": dict(water=[(0.70, 0.10), (0.90, 0.25), (0.50, 0.04)],
                         seabed=[(0.30, 0.60), (0.70, 0.80), (0.85, 0.55)]),
        "MTN_5903": dict(water=[(0.50, 0.10), (0.80, 0.40), (0.95, 0.70), (0.05, 0.05), (0.12, 0.20)],
                         seabed=[(0.20, 0.60), (0.50, 0.80), (0.10, 0.90)]),
        "MTN_5911": dict(water=[(0.50, 0.10), (0.90, 0.15), (0.10, 0.10)],
                         seabed=[(0.50, 0.70), (0.20, 0.90), (0.80, 0.85)]),
        "MTN_5928": dict(water=[(0.50, 0.10), (0.90, 0.30), (0.20, 0.10)],
                         seabed=[(0.50, 0.75), (0.30, 0.90), (0.80, 0.90)]),
    },
    "Curasao-Undistorted": {
        "MTN_1288": dict(water=[(0.60, 0.08), (0.90, 0.20), (0.20, 0.05)],
                         seabed=[(0.30, 0.60), (0.70, 0.80), (0.50, 0.45)]),
        "MTN_1296": dict(water=[(0.85, 0.05), (0.95, 0.15), (0.70, 0.03)],
                         seabed=[(0.30, 0.50), (0.60, 0.70), (0.50, 0.20), (0.20, 0.10)]),
        "MTN_1304": dict(water=[(0.75, 0.05), (0.95, 0.12), (0.60, 0.03)],
                         seabed=[(0.30, 0.60), (0.70, 0.80), (0.10, 0.10)]),
    },
    "JapaneseGradens-RedSea-Undistorted": {
        # Open water only in a thin strip at the top; the bluish band below is reef.
        "MTN_1090": dict(water=[(0.50, 0.02), (0.85, 0.03), (0.15, 0.02)],
                         seabed=[(0.30, 0.12), (0.60, 0.12), (0.90, 0.15), (0.45, 0.07), (0.50, 0.60),
                                 (0.20, 0.80)]),
        "MTN_1098": dict(water=[(0.50, 0.10), (0.20, 0.08), (0.90, 0.20)],
                         seabed=[(0.50, 0.70), (0.20, 0.60), (0.80, 0.80)]),
        # The hazy band under the water strip is reef (as in IUI3).
        "MTN_1106": dict(water=[(0.10, 0.05), (0.30, 0.05), (0.80, 0.03)],
                         seabed=[(0.50, 0.60), (0.30, 0.85), (0.85, 0.70), (0.04, 0.45), (0.10, 0.52)],
                         pick="score"),
    },
    "Panama-Undisorted": {  # the directory name is misspelled in the dataset
        # Open water only in the top-right corner; hazy reef silhouettes elsewhere at the top.
        "MTN_1529": dict(water=[(0.85, 0.03), (0.95, 0.12), (0.75, 0.05)],
                         seabed=[(0.15, 0.12), (0.30, 0.05), (0.35, 0.02), (0.50, 0.12), (0.05, 0.30), (0.50, 0.60),
                                 (0.90, 0.30)]),
        # Hazy seabed up to the top edge, no open water in the frame.
        "MTN_1539": dict(water=[], seabed=[]),
        "MTN_1547": dict(water=[], seabed=[]),
    },
}


def load_masks(path):
    """Per-view boolean water masks (full resolution) from one .npz or every .npz of a
    directory (one per scene; view names are unique across scenes); {} when absent."""
    path = None if path is None else Path(path)
    files = sorted(path.glob("*.npz")) if path is not None and path.is_dir() else (
        [path] if path is not None and path.is_file() else [])
    masks = {}
    for file in files:
        with np.load(file) as data:
            for name in (name for name in data.files if name != "meta"):
                if name in masks:
                    raise ValueError(f"view {name} has masks in two files")
                masks[name] = data[name]
    return masks


def prompt_points(prompts, width, height):
    points = np.array([(x * width, y * height) for x, y in prompts["water"] + prompts["seabed"]], dtype=np.float32)
    labels = np.array([1] * len(prompts["water"]) + [0] * len(prompts["seabed"]))
    return points, labels


def extend_to_border(mask, border=BORDER):
    """Replace the outermost ``border`` rows/columns by the labels just inside them."""
    mask = mask.copy()
    mask[:border] = mask[border]
    mask[-border:] = mask[-border - 1]
    mask[:, :border] = mask[:, border:border + 1]
    mask[:, -border:] = mask[:, -border - 1:-border]
    return mask


def clean(mask, min_region=MIN_REGION):
    """Drop connected water and seabed regions smaller than ``min_region`` of the image."""
    from scipy.ndimage import label
    mask = mask.copy()
    for value in (True, False):
        regions, count = label(mask == value)
        sizes = np.bincount(regions.ravel())
        small = np.flatnonzero(sizes < min_region * mask.size)
        small = small[small > 0]
        mask[np.isin(regions, small)] = not value
    return mask


def segment(predictor, image, prompts):
    """Boolean water mask [H, W] and SAM's score for it (no water prompts: no water, score nan)."""
    if not prompts["water"]:
        return np.zeros(image.shape[:2], dtype=bool), float("nan")
    predictor.set_image(image)
    points, labels = prompt_points(prompts, image.shape[1], image.shape[0])
    masks, scores, _ = predictor.predict(point_coords=points, point_labels=labels, multimask_output=True)
    if prompts.get("pick") == "score":  # a proposal approved by eye that is not the most consistent one
        best = int(np.argmax(scores))
    else:
        best = choose(masks.astype(bool), scores, points, labels)
    return clean(extend_to_border(masks[best])), float(scores[best])


def choose(masks, scores, points, labels):
    """Index of the proposal that agrees with the most prompts (water points inside,
    seabed points outside); SAM's score breaks ties."""
    rows = np.clip(points[:, 1].astype(int), 0, masks.shape[1] - 1)
    columns = np.clip(points[:, 0].astype(int), 0, masks.shape[2] - 1)
    agreement = [int(((mask[rows, columns]) == (labels == 1)).sum()) for mask in masks]
    return max(range(len(masks)), key=lambda index: (agreement[index], scores[index]))


def boundary(mask):
    """Pixels where the mask changes from the pixel above or to the left (no wrap-around)."""
    edge = np.zeros_like(mask)
    edge[1:] |= mask[1:] != mask[:-1]
    edge[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    return edge


def overlay(image, mask, prompts):
    """Image with the water mask tinted and the prompt points drawn."""
    from PIL import ImageDraw
    tinted = image.astype(np.float32)
    tinted[mask] = 0.45 * tinted[mask] + 0.55 * np.array([255, 60, 60])
    tinted[boundary(mask)] = (255, 255, 0)
    picture = Image.fromarray(tinted.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(picture)
    height, width = mask.shape
    for kind, colour in (("water", (0, 255, 0)), ("seabed", (0, 120, 255))):
        for x, y in prompts[kind]:
            cx, cy = x * width, y * height
            draw.ellipse((cx - 9, cy - 9, cx + 9, cy + 9), fill=colour, outline=(0, 0, 0), width=2)
    return picture


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="SAM ViT-H weights")
    parser.add_argument("--scene", action="append", choices=sorted(PROMPTS), help="default: every scene")
    parser.add_argument("--datasets", type=Path, default=Path(DATASETS))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "configs/water_sam")
    parser.add_argument("--preview-dir", type=Path, help="overlay per view: <dir>/<scene>/<view>.png")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from segment_anything import SamPredictor, sam_model_registry
    predictor = SamPredictor(sam_model_registry["vit_h"](checkpoint=str(args.checkpoint)).to(args.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for scene in args.scene or PROMPTS:
        masks, scores = {}, {}
        for view, prompts in PROMPTS[scene].items():
            image = np.asarray(Image.open(args.datasets / scene / "images" / f"{view}.png").convert("RGB"))
            masks[view], scores[view] = segment(predictor, image, prompts)
            print(f"{scene} {view}: water {masks[view].mean():.3f}, SAM score {scores[view]:.3f}")
            if args.preview_dir:
                (args.preview_dir / scene).mkdir(parents=True, exist_ok=True)
                overlay(image, masks[view], prompts).save(args.preview_dir / scene / f"{view}.png")
        meta = dict(method="segment_anything vit_h point prompts", checkpoint=args.checkpoint.name,
                    scene=scene, prompts=PROMPTS[scene], scores=scores)
        output = args.output_dir / f"{scene}.npz"
        np.savez_compressed(output, meta=json.dumps(meta), **masks)
        print(output)


if __name__ == "__main__":
    sys.exit(main())
