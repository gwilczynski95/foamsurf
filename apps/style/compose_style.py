"""One run from the Gaussians of one run and the Voronoi medium of another (e.g. two stylize.py results).

    python compose_style.py --gaussians OUT_FIRE --medium OUT_STARRY --out OUT_BOTH

Both runs must come from the same trained model (camera model, water light and everything else are taken from the
--gaussians run). Runs on the CPU; render the result with render.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from training.checkpoints import load_checkpoint, save_checkpoint


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gaussians", type=Path, required=True)
    parser.add_argument("--medium", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", default="last", choices=("last",))
    args = parser.parse_args(argv)
    surface = load_checkpoint(args.gaussians / "joint" / "checkpoints" / f"{args.checkpoint}.pt")
    water = load_checkpoint(args.medium / "joint" / "checkpoints" / f"{args.checkpoint}.pt")
    surface_style, water_style = surface.get("style") or {}, water.get("style") or {}
    bases = {info.get("base_run") for info in (surface_style.get("gaussians"), water_style.get("voronoi")) if info}
    if len(bases) > 1:
        print(f"warning: stylized from different runs: {sorted(bases)}")
    payload = dict(surface)
    payload["model"] = {k: v for k, v in surface["model"].items() if not k.startswith("medium.")}
    payload["model"].update({k: v for k, v in water["model"].items() if k.startswith("medium.")})
    payload["model_metadata"] = dict(surface["model_metadata"], medium=water["model_metadata"]["medium"])
    payload["medium_optimizer"] = water["medium_optimizer"]
    payload["topology_revision"] = water["topology_revision"]
    payload["voronoi_adaptation"] = None
    payload["style"] = {"gaussians": surface_style.get("gaussians"), "voronoi": water_style.get("voronoi"),
                        "composed_from": [str(args.gaussians.resolve()), str(args.medium.resolve())]}
    joint = args.out / "joint"
    joint.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.gaussians / "joint" / "config.yaml", joint / "config.yaml")
    save_checkpoint(joint / "checkpoints" / f"{args.checkpoint}.pt", payload)
    (args.out / "style_compose.json").write_text(json.dumps(payload["style"], indent=1, default=str))
    print(f"composed run in {args.out}: Gaussians of {args.gaussians}, medium of {args.medium}")


if __name__ == "__main__":
    main()
