#!/usr/bin/env python3
"""Open-water masks of the IUI3 test views from the images alone, with Segment Anything.

Usage: python scripts/sam_water_mask.py --checkpoint sam_vit_h_4b8939.pth [--dataset DATA]
       [--output configs/iui3_water_sam.npz] [--preview preview.png]

Needs the `segment_anything` package (pip install --no-deps segment-anything) and the
ViT-H checkpoint. Each view is segmented with the point prompts below (fractions of
width/height): positives on open water, negatives on the seabed. The highest-scoring
of SAM's three proposals is kept, then holes and specks smaller than
MIN_REGION of the image are removed from both water and seabed. The result does
not depend on any FoamSurf model.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATASET = "/shared/sets/datasets/SeathruNeRF_dataset/IUI3-RedSea-Undistorted"
MIN_REGION = 0.002  # fraction of the image
# (x, y) as fractions of the image; inspected on the four test views.
PROMPTS = {
    "MTN_5894": dict(water=[(0.70, 0.10), (0.90, 0.25), (0.50, 0.04)],
                     seabed=[(0.30, 0.60), (0.70, 0.80), (0.85, 0.55)]),
    "MTN_5903": dict(water=[(0.50, 0.10), (0.80, 0.40), (0.95, 0.70), (0.05, 0.05), (0.12, 0.20)],
                     seabed=[(0.20, 0.60), (0.50, 0.80), (0.10, 0.90)]),
    "MTN_5911": dict(water=[(0.50, 0.10), (0.90, 0.15), (0.10, 0.10)],
                     seabed=[(0.50, 0.70), (0.20, 0.90), (0.80, 0.85)]),
    "MTN_5928": dict(water=[(0.50, 0.10), (0.90, 0.30), (0.20, 0.10)],
                     seabed=[(0.50, 0.75), (0.30, 0.90), (0.80, 0.90)]),
}


def prompt_points(prompts, width, height):
    points = np.array([(x * width, y * height) for x, y in prompts["water"] + prompts["seabed"]], dtype=np.float32)
    labels = np.array([1] * len(prompts["water"]) + [0] * len(prompts["seabed"]))
    return points, labels


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
    """Boolean water mask [H, W] and SAM's score for it."""
    predictor.set_image(image)
    points, labels = prompt_points(prompts, image.shape[1], image.shape[0])
    masks, scores, _ = predictor.predict(point_coords=points, point_labels=labels, multimask_output=True)
    best = int(np.argmax(scores))
    return clean(masks[best].astype(bool)), float(scores[best])


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
    parser.add_argument("--dataset", type=Path, default=Path(DATASET))
    parser.add_argument("--output", type=Path, default=ROOT / "configs/iui3_water_sam.npz")
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from segment_anything import SamPredictor, sam_model_registry
    sam = sam_model_registry["vit_h"](checkpoint=str(args.checkpoint)).to(args.device)
    predictor = SamPredictor(sam)
    masks, scores, previews = {}, {}, []
    for view, prompts in PROMPTS.items():
        image = np.asarray(Image.open(args.dataset / "images" / f"{view}.png").convert("RGB"))
        masks[view], scores[view] = segment(predictor, image, prompts)
        previews.append(overlay(image, masks[view], prompts))
        print(f"{view}: water {masks[view].mean():.3f}, SAM score {scores[view]:.3f}")
    meta = dict(method="segment_anything vit_h point prompts", checkpoint=args.checkpoint.name,
                prompts=PROMPTS, scores=scores)
    np.savez_compressed(args.output, meta=json.dumps(meta), **masks)
    print(args.output)
    if args.preview:
        width, height = previews[0].size
        sheet = Image.new("RGB", (width * 2, height * 2))
        for index, picture in enumerate(previews):
            sheet.paste(picture, ((index % 2) * width, (index // 2) * height))
        sheet.resize((width, height)).save(args.preview)
        print(args.preview)


if __name__ == "__main__":
    sys.exit(main())
