#!/usr/bin/env python3
"""Train FoamSurf on one scene.

    python train.py --scene /data/SeathruNeRF_dataset/Curasao-Undistorted --out outputs/curacao
    python train.py --scene /data/foggy_garden/garden_easy --out outputs/garden_easy --config configs/garden_fog.yaml
    python train.py ... --seed 43 --wandb my_project --set joint.optimization.iterations=10000

The scene is a COLMAP dataset (images/ and sparse/0/) or a Blender dataset (transforms_train.json; use
configs/synthetic.yaml, which has no COLMAP water fit). Every 500 iterations and at the end the test views
are rendered and scored; the run directory then holds:
    joint/checkpoints/last.pt (+ best.pt)   the model, its optimizers, camera model and water light terms
    joint/metrics/train.csv, eval.csv       losses and statistics per step, test metrics per evaluation
    joint/evaluation/iteration_*/           test renders (full image, Gaussians alone, medium alone, ...)
    joint/summary.json                      final status, timing and model size
Render a finished run with render.py. --resume continues a run with the same configuration.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml

from training.config import ExperimentConfig, atomic_json, drop_legacy_keys, update_config

ROOT = Path(__file__).resolve().parent


def load_iui3_source(config_path):
    """Load the full YAML (configs/base.yaml) or a variant of it: ``base_config`` plus ``overrides``.

    A variant may itself be based on a variant (configs/garden_fog.yaml on configs/seathru.yaml): the
    overrides of the chain are applied in order, base first."""
    path = Path(config_path)
    raw = yaml.safe_load(path.read_text())
    if "base_config" not in raw:
        return raw, {}
    allowed = {"base_config", "output_dir", "gaussian_rgb_activation",
               "medium_color_activation", "wandb_project", "overrides"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unsupported IUI3 variant fields: {sorted(unknown)}")
    base, base_variant = load_iui3_source(path.parent / raw["base_config"])
    if "capacity_loss" not in base:
        raise ValueError("IUI3 variant must reference a full IUI3 configuration")
    variant = {**base_variant, **{k: v for k, v in raw.items() if k != "overrides"}}
    variant["overrides"] = list(base_variant.get("overrides", [])) + list(raw.get("overrides") or [])
    return base, variant


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


def run_voronoi_first(args, scene):
    """The Voronoi-first pipeline (configs/synthetic_voronoi.yaml, training/end_to_end.py): a medium-only warmup
    whose cells densify on the content, a partition of the cells into surface and the rest, Gaussians from the
    surface cells, then hybrid training. --set takes warmup., joint., both., profile., partition., handoff. keys."""
    from training.end_to_end import EndToEndConfig, EndToEndTrainer
    overrides = [f"both.seed={args.seed}", f"both.wandb.mode={'online' if args.wandb else 'disabled'}"]
    if args.wandb:
        overrides.append(f"both.wandb.project={args.wandb}")
    if scene:
        overrides.append(f"both.dataset.source_path={scene}")
    config = EndToEndConfig.load(args.config, output_dir=args.output_dir, overrides=overrides + args.overrides)
    if args.dry_run:
        print(json.dumps(config.manifest(), indent=2))
    else:
        EndToEndTrainer(config).fit(resume=args.resume)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, help="COLMAP (images/, sparse/0/) or Blender (transforms_*.json) scene")
    parser.add_argument("--out", "--output-dir", dest="output_dir", type=Path, help="run directory")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "seathru.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", metavar="PROJECT", help="log to Weights & Biases (default: off)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY.PATH=VALUE",
                        help="override a configuration value, e.g. joint.optimization.iterations=10000 (repeatable)")
    parser.add_argument("--resume", action="store_true", help="continue an unfinished run")
    parser.add_argument("--dry-run", action="store_true", help="print the resolved configuration and stop")
    parser.add_argument("--stage", choices=("warmup", "partition", "joint"), help=argparse.SUPPRESS)  # internal
    parser.add_argument("--run-suffix", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stage:
        if not args.output_dir:
            parser.error("--stage requires --out")
        root = args.output_dir.resolve()
        if "capacity_loss" in json.loads((root / "pipeline_config.json").read_text()):
            if args.stage == "partition":
                parser.error("this configuration has no partition stage")
            run_iui3_stage(root, args.stage)
        else:
            from training.end_to_end import run_stage
            run_stage(root, args.stage)
        return
    if not args.output_dir:
        parser.error("--out is required")
    scene = None
    if args.scene:
        scene = args.scene.resolve()
        if not ((scene / "sparse").is_dir() or (scene / "transforms_train.json").is_file()):
            parser.error(f"{scene} is not a COLMAP scene (no sparse/) nor a Blender scene (no transforms_train.json)")
    if "partition" in yaml.safe_load(args.config.read_text()):
        run_voronoi_first(args, scene)
        return
    overrides = [f"variant.seed={args.seed}", f"warmup.seed={args.seed}", f"joint.seed={args.seed}"]
    if scene:
        overrides += [f"warmup.dataset.source_path={scene}", f"joint.dataset.source_path={scene}"]
    overrides += [f"joint.wandb.mode={'online' if args.wandb else 'disabled'}", "warmup.wandb.mode=disabled"]
    manifest = resolve_iui3(args.config, args.output_dir, wandb_project=args.wandb,
                            overrides=overrides + args.overrides,
                            run_suffix=args.run_suffix or args.output_dir.name)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
    else:
        fit_iui3(manifest, resume=args.resume)


if __name__ == "__main__":
    main()
