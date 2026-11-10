"""Turn spec/gloss textures into the albedo / metal / rough maps of the tracer's "pbr" material.

    python vis/specgloss_to_pbr.py DIFFUSE SPECULAR GLOSSINESS OUT_DIR PREFIX [--manifest OUT_DIR/manifest.json]

albedo = the diffuse colour, or the specular colour where the diffuse is dark (spec/gloss keeps a metal's or a
coloured lacquer's colour in the specular map); metal = 1 only on unsaturated (grey) parts, so coloured parts
read as lacquered plastic; rough = 1 - glossiness. With --manifest, its first part gets these maps.
Used for the toy submarine (Submarine.rar, YellowSubmarine Spec_gloss textures).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load(path, mode):
    return np.asarray(Image.open(path).convert(mode), dtype=np.float32) / 255.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("diffuse", type=Path)
    parser.add_argument("specular", type=Path)
    parser.add_argument("glossiness", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("prefix")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    diffuse, specular = load(args.diffuse, "RGB") ** 2.2, load(args.specular, "RGB") ** 2.2
    gloss = load(args.glossiness, "L")
    coloured_spec = np.clip((specular.max(-1) - 0.04) / (specular.max(-1) + diffuse.max(-1) + 1e-6) * 1.2, 0, 1)
    albedo = diffuse * (1 - coloured_spec[..., None]) + specular * coloured_spec[..., None]
    srgb = np.clip(albedo, 0, 1) ** (1 / 2.2)
    saturation = (srgb.max(-1) - srgb.min(-1)) / np.maximum(srgb.max(-1), 1e-3)
    metal = np.clip((0.25 - saturation) / 0.1, 0, 1)
    args.out.mkdir(parents=True, exist_ok=True)
    names = {k: f"{args.prefix}_{k}.png" for k in ("albedo", "metal", "rough")}
    Image.fromarray((srgb * 255).round().astype(np.uint8)).save(args.out / names["albedo"])
    Image.fromarray((metal * 255).round().astype(np.uint8)).save(args.out / names["metal"])
    Image.fromarray(((1 - gloss) * 255).round().astype(np.uint8)).save(args.out / names["rough"])
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        manifest["parts"][0]["material"] = {"mode": "pbr", "rgb": [1, 1, 1], "maps": names}
        args.manifest.write_text(json.dumps(manifest, indent=1))
    print("metal fraction", round(float(metal.mean()), 3))


if __name__ == "__main__":
    main()
