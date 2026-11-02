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

from training.config import ExperimentConfig, drop_legacy_keys
from training.end_to_end import EndToEndConfig, EndToEndTrainer, atomic_json, update_config

ROOT = Path(__file__).resolve().parent


def load_iui3_source(config_path):
    """Load the full IUI3 YAML or a small, explicit variant of it."""
    path = Path(config_path)
    raw = yaml.safe_load(path.read_text())
    if "base_config" not in raw:
        return raw, {}
    allowed = {"base_config", "output_dir", "gaussian_rgb_activation",
               "medium_color_activation", "wandb_project", "overrides"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unsupported IUI3 variant fields: {sorted(unknown)}")
    base = yaml.safe_load((path.parent / raw["base_config"]).read_text())
    if "capacity_loss" not in base or "base_config" in base:
        raise ValueError("IUI3 variant must reference a full IUI3 configuration")
    return base, raw


def apply_overrides(manifest, overrides, defaults=None):
    """Apply ``dotted.path=yaml_value`` overrides to known manifest keys only.

    A key absent from the manifest is accepted when ``defaults`` (same shape)
    has it, e.g. a stage config field added after the YAML was written.
    """
    for item in overrides or ():
        path, separator, text = item.partition("=")
        if not separator or not path:
            raise ValueError(f"Override must look like key.path=value: {item!r}")
        keys = path.split(".")
        node, known = manifest, defaults
        for key in keys[:-1]:
            known = known.get(key) if isinstance(known, dict) else None
            if not isinstance(node, dict) or key not in node:
                raise KeyError(f"Unknown override path: {path}")
            node = node[key]
        if not isinstance(node, dict) or (
            keys[-1] not in node and not (isinstance(known, dict) and keys[-1] in known)
        ):
            raise KeyError(f"Unknown override path: {path}")
        node[keys[-1]] = yaml.safe_load(text)
    return manifest


def resolve_iui3(
    config_path,
    output_dir,
    gaussian_rgb_activation=None,
    medium_color_activation=None,
    wandb_project=None,
    overrides=None,
    run_suffix=None,
):
    """Resolve a standalone IUI3 manifest without importing any experiment package."""
    from training.losses import MediumFixConfig

    source, variant = load_iui3_source(config_path)
    # Older configs predate medium_fix; defaults reproduce their behavior.
    source.setdefault("medium_fix", MediumFixConfig().as_dict())
    gaussian_rgb_activation = (gaussian_rgb_activation if gaussian_rgb_activation is not None
                               else variant.get("gaussian_rgb_activation"))
    medium_color_activation = (medium_color_activation if medium_color_activation is not None
                               else variant.get("medium_color_activation"))
    wandb_project = wandb_project if wandb_project is not None else variant.get("wandb_project")
    if source.get("warmup_steps") != 0:
        raise ValueError("IUI3 joint training currently requires zero warmup")
    if gaussian_rgb_activation not in (None, "identity", "sigmoid"):
        raise ValueError("Gaussian RGB activation must be identity or sigmoid")
    if medium_color_activation not in (None, "dc", "sigmoid"):
        raise ValueError("medium color activation must be dc or sigmoid")
    if gaussian_rgb_activation is not None:
        source["joint"]["model"]["gaussian_rgb_activation"] = gaussian_rgb_activation
        source["variant"]["gaussian_rgb_activation"] = gaussian_rgb_activation
    if medium_color_activation is not None:
        source["medium_color_activation"] = medium_color_activation
    if gaussian_rgb_activation is not None or medium_color_activation is not None:
        gaussian = source["joint"]["model"]["gaussian_rgb_activation"]
        medium = source.get("medium_color_activation", "dc")
        suffix = f"-g_{gaussian}-m_{medium}"
        source["joint"]["experiment_name"] += suffix
        source["joint"]["wandb"]["name"] += suffix
        # The zero-step initialization has no metrics worth publishing.
        source["warmup"]["wandb"]["mode"] = "disabled"
    if wandb_project is not None:
        if not wandb_project:
            raise ValueError("W&B project must not be empty")
        source["joint"]["wandb"]["project"] = wandb_project
    from training.warmup_profile import WarmupProfile

    stage_defaults = ExperimentConfig().as_dict()
    # Variant-file overrides first, so command-line --set can still change them.
    overrides = list(variant.get("overrides", [])) + list(overrides or [])
    from training.schedule import pop_schedule, scale_schedule

    schedule_iterations, overrides = pop_schedule(overrides)
    apply_overrides(source, overrides, {"warmup": stage_defaults, "joint": stage_defaults,
                                        "profile": WarmupProfile().as_dict()})
    if schedule_iterations is not None:  # after the overrides: scales the values they set
        scale_schedule(source, schedule_iterations)
    WarmupProfile.from_dict(source["profile"])
    MediumFixConfig(**source["medium_fix"])
    if run_suffix:
        source["joint"]["experiment_name"] += "-" + run_suffix
        source["joint"]["wandb"]["name"] += "-" + run_suffix
    root = Path(output_dir).resolve()
    source["output_dir"] = str(root)
    for stage in ("warmup", "joint"):
        section = source[stage]
        section["output_dir"] = str(root / stage)
        checked = ExperimentConfig()
        update_config(checked, section)
        checked.validate()
    return source


def normalize_saved_manifest(saved):
    """A saved IUI3 manifest as the current code would resolve it.

    Manifests written before medium_fix (or a newer field) used the defaults;
    inactive options removed since then are dropped.
    """
    from training.losses import MediumFixConfig, drop_legacy_medium_fix

    saved = dict(saved)
    saved["medium_fix"] = {**MediumFixConfig().as_dict(),
                           **drop_legacy_medium_fix(saved.get("medium_fix", {}))}
    for stage in ("warmup", "joint"):
        saved[stage] = drop_legacy_keys(saved[stage])
    return saved


def fit_iui3(manifest, resume=False):
    root = Path(manifest["output_dir"])
    path = root / "pipeline_config.json"
    if root.exists() and any(root.iterdir()):
        saved = normalize_saved_manifest(json.loads(path.read_text())) if path.is_file() else None
        if not resume or saved != manifest:
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
        from training.colmap_joint_trainer import MediumFixTrainer
        original = module.ColmapJointTrainer
        try:
            module.ColmapJointTrainer = MediumFixTrainer
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
    parser.add_argument("--gaussian-rgb-activation", choices=("identity", "sigmoid"))
    parser.add_argument("--medium-color-activation", choices=("dc", "sigmoid"))
    parser.add_argument("--wandb-project")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY.PATH=VALUE",
                        help="IUI3 only: override an existing manifest key (repeatable)")
    parser.add_argument("--run-suffix", help="IUI3 only: appended to joint run names")
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
    if "capacity_loss" in raw or "base_config" in raw:
        manifest = resolve_iui3(
            args.config,
            output,
            gaussian_rgb_activation=args.gaussian_rgb_activation,
            medium_color_activation=args.medium_color_activation,
            wandb_project=args.wandb_project,
            overrides=args.overrides,
            run_suffix=args.run_suffix,
        )
        if args.dry_run:
            print(json.dumps(manifest, indent=2))
        else:
            fit_iui3(manifest, resume=args.resume)
    else:
        if any((args.gaussian_rgb_activation, args.medium_color_activation, args.wandb_project,
                args.overrides, args.run_suffix)):
            parser.error("color activation and manifest overrides are supported only for IUI3")
        config = EndToEndConfig.load(args.config, output_dir=output)
        if args.dry_run:
            print(json.dumps(config.manifest(), indent=2))
        else:
            EndToEndTrainer(config).fit(resume=args.resume)


if __name__ == "__main__":
    main()
