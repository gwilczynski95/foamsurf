#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training import ExperimentConfig
from training.trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument("--phase", choices=("gaussian", "voronoi"), required=True)
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()
path = ROOT / "configs" / f"stage{1 if args.phase == 'gaussian' else 2}_{args.phase}_only.yaml"
config = ExperimentConfig.load(path)
config.output_dir = args.output_dir
config.optimization.iterations = 4
config.optimization.full_image_batch = False
config.optimization.rays_per_batch = 4096
config.optimization.reconstruction_loss = "mse"
config.validation.eval_interval = 4
config.validation.checkpoint_interval = 4
config.validation.log_interval = 1
config.validation.max_eval_views = 2
config.validation.image_views = 1
config.wandb.mode = "disabled"
if args.phase == "gaussian":
    config.model.gaussian_initial_points = 512
    config.gaussian_adaptation.max_points = 1024
    config.gaussian_adaptation.densify_from = 2
    config.gaussian_adaptation.densify_until = 4
    config.gaussian_adaptation.densification_interval = 2
    config.gaussian_adaptation.opacity_reset_interval = 0
    config.gaussian_adaptation.opacity_reset_at_densify_from = False
else:
    config.model.medium_points = 128
    config.voronoi_adaptation.max_points = 256
    config.voronoi_adaptation.min_points = 32
    config.voronoi_adaptation.densify_from = 2
    config.voronoi_adaptation.densify_until = 4
    config.voronoi_adaptation.interval = 2
config.validate()
trainer = Trainer(config)
inactive = trainer.model.medium if args.phase == "gaussian" else trainer.model.gaussians
before = {name: value.detach().clone() for name, value in inactive.state_dict().items()}
trainer.fit()
for name, value in inactive.state_dict().items():
    if not torch.equal(value, before[name]):
        raise AssertionError(f"inactive parameter changed: {name}")
if any(parameter.grad is not None for parameter in inactive.parameters()):
    raise AssertionError("inactive representation received a gradient")
