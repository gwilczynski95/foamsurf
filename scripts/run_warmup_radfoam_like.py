#!/usr/bin/env python3
"""Launch a verified initialization experiment, never joint training."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.warmup_sweep import configuration, variants
from warmup_gate import require_certificate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index",type=int,choices=range(50))
    p.add_argument("--suite",choices=["original","organic"],default="original")
    p.add_argument("--config", help="warmup-only YAML wrapper")
    p.add_argument("--output-dir")
    p.add_argument("--certificate")
    p.add_argument("--iterations",type=int)
    p.add_argument("--seed",type=int)
    p.add_argument("--resume")
    p.add_argument("--manifest",action="store_true")
    p.add_argument("--wandb-mode",choices=["online","offline","disabled"],default="offline")
    args=p.parse_args()
    builder, manifest = configuration, variants
    if args.suite == "organic":
        from training.warmup_organic_sweep import configuration as builder, variants as manifest
    if args.manifest:
        print(json.dumps(manifest(),indent=2));return
    wrapper = {}
    if args.config:
        import yaml
        wrapper = yaml.safe_load(Path(args.config).read_text())
        if set(wrapper) - {"variant", "iterations", "profile"}:
            raise ValueError("unknown warmup wrapper fields")
        if args.index is None:
            args.index = int(wrapper.get("variant", 7))
    args.iterations = args.iterations or int(wrapper.get("iterations", 6000 if args.suite == "organic" else 5000))
    if args.index is None or not args.output_dir or not args.certificate:
        p.error("--index, --output-dir and --certificate required")
    certificate=require_certificate(args.certificate)
    config,profile,row=builder(args.index,args.output_dir,args.iterations)
    if wrapper.get("profile"):
        if row["legacy"]:
            raise ValueError("legacy control cannot accept a custom profile")
        from training.warmup_profile import WarmupProfile
        values = profile.as_dict(); values.update(wrapper["profile"])
        custom = WarmupProfile.from_dict(values)
        delta = custom.outer_points - profile.outer_points
        config.voronoi_adaptation.max_points += delta
        config.voronoi_adaptation.min_points += delta
        profile = custom
        row["profile"] = profile.as_dict()
    if args.seed is not None: config.seed=args.seed
    config.wandb.mode=args.wandb_mode
    from training.trainer import Trainer
    from training.warmup_trainer import WarmupTrainer
    trainer=(Trainer(config,resume=args.resume) if row["legacy"] else WarmupTrainer(config,profile,resume=args.resume))
    run=Path(config.output_dir)
    (run/"sweep_variant.json").write_text(json.dumps({**row,"actual_seed":config.seed,
        "iterations":args.iterations,"test_certificate":certificate},indent=2)+"\n")
    if args.resume is None:
        summary,images,histograms=trainer.evaluator.evaluate(0)
        trainer.logger.log_eval(summary,images,histograms)
        trainer.save_voronoi_warmup_milestone(0)
    print(json.dumps(trainer.fit(),indent=2))


if __name__=="__main__":main()
