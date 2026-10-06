#!/usr/bin/env python3
"""Single public FoamSurf training entry point.

Usage: python train.py --config configs/iui3_5k.yaml --output-dir outputs/my_run
       python train.py --config configs/synthetic.yaml --output-dir outputs/lego
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml

from training.config import ExperimentConfig
from training.end_to_end import EndToEndConfig, EndToEndTrainer, atomic_json, update_config

ROOT = Path(__file__).resolve().parent


def resolve_iui3(config_path, output_dir):
    """Resolve a standalone IUI3 manifest without importing any experiment package."""
    source = yaml.safe_load(Path(config_path).read_text())
    if source.get("warmup_steps") != 0:
        raise ValueError("IUI3 joint training currently requires zero warmup")
    root = Path(output_dir).resolve()
    source["output_dir"] = str(root)
    for stage in ("warmup", "joint"):
        section = source[stage]
        section["output_dir"] = str(root / stage)
        checked = ExperimentConfig()
        update_config(checked, section)
        checked.validate()
    return source


def fit_iui3(manifest, resume=False):
    root = Path(manifest["output_dir"])
    path = root / "pipeline_config.json"
    if root.exists() and any(root.iterdir()):
        if not resume or not path.is_file() or json.loads(path.read_text()) != manifest:
            raise ValueError("Nonempty output requires --resume with the identical configuration")
    else:
        root.mkdir(parents=True, exist_ok=True)
        atomic_json(path, manifest)
        for stage in ("warmup", "joint"):
            (root / (stage + ".yaml")).write_text(yaml.safe_dump(manifest[stage], sort_keys=False))
    for stage in ("warmup", "joint"):
        marker = root / (stage + "_done.json")
        if marker.exists():
            artifact = Path(json.loads(marker.read_text())["artifact"])
            if not artifact.is_file():
                raise FileNotFoundError(artifact)
            continue
        subprocess.run([sys.executable, str(ROOT / "train.py"), "--stage", stage,
                        "--output-dir", str(root)], cwd=ROOT, check=True)
        if not marker.is_file():
            raise RuntimeError(stage + " did not complete")


def run_iui3_stage(output_dir, stage):
    from training.colmap_joint import run_stage
    if stage == "joint":
        import training.colmap_joint_trainer as module
        from training.colmap_joint_trainer import CapacityTrainer
        original = module.ColmapJointTrainer
        try:
            module.ColmapJointTrainer = CapacityTrainer
            return run_stage(output_dir, stage)
        finally:
            module.ColmapJointTrainer = original
    return run_stage(output_dir, stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", choices=("warmup", "partition", "joint"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.stage:
        if not args.output_dir:
            parser.error("--stage requires --output-dir")
        root = args.output_dir.resolve()
        manifest = json.loads((root / "pipeline_config.json").read_text())
        if "capacity_loss" in manifest:
            if args.stage == "partition":
                parser.error("IUI3 has no partition stage")
            run_iui3_stage(root, args.stage)
        else:
            from training.end_to_end import run_stage
            run_stage(root, args.stage)
        return
    if not args.config:
        parser.error("--config is required")
    raw = yaml.safe_load(args.config.read_text())
    output = args.output_dir or Path(raw["output_dir"])
    if "capacity_loss" in raw:
        manifest = resolve_iui3(args.config, output)
        if args.dry_run:
            print(json.dumps(manifest, indent=2))
        else:
            fit_iui3(manifest, resume=args.resume)
    else:
        config = EndToEndConfig.load(args.config, output_dir=output)
        if args.dry_run:
            print(json.dumps(config.manifest(), indent=2))
        else:
            EndToEndTrainer(config).fit(resume=args.resume)


if __name__ == "__main__":
    main()
