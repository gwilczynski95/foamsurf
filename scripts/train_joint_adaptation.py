#!/usr/bin/env python3
"""Fork the same immutable native handoff into seven controlled joint runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from training.checkpoints import load_checkpoint
    from training.config import ExperimentConfig
    from training.joint_adaptation_sweep import build_variant, validate_initialization, install_sweep_controller, VARIANTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/joint_adaptation_sweep.yaml"))
    parser.add_argument("--variant", type=int, required=True, choices=range(len(VARIANTS)))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    settings = yaml.safe_load(Path(args.config).read_text())
    source = Path(settings["source_checkpoint"])
    if not source.is_absolute():
        source = ROOT / source
    source = source.resolve()
    payload = load_checkpoint(source)
    count = validate_initialization(payload)
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to reuse non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    base_path = output / "source_config.yaml"
    base_path.write_text(yaml.safe_dump(payload["config"], sort_keys=False))
    base = ExperimentConfig.load(base_path)
    config = build_variant(base, args.variant, count, settings, output)
    config.save(output / "config.yaml")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    (output / "sweep_manifest.json").write_text(json.dumps(dict(
        variant=args.variant, name=VARIANTS[args.variant][0], source=str(source),
        source_sha256=digest.hexdigest(), initial_medium_count=count,
        baseline="outputs/end_to_end/508222/joint", settings=settings,
        wandb_group=os.environ.get("WANDB_RUN_GROUP")), indent=2))
    del payload
    if args.dry_run:
        print(output / "config.yaml")
        return
    from training.trainer import Trainer
    trainer = Trainer(config, resume=source)
    install_sweep_controller(trainer)
    trainer.fit()


if __name__ == "__main__":
    main()
