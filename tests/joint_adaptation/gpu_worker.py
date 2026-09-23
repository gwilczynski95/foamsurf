"""Tiny integration worker, not collected as a standalone pytest module."""
from pathlib import Path
import sys
import torch
import yaml
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.config import ExperimentConfig
from training.joint_adaptation_sweep import build_variant, validate_initialization, install_sweep_controller
from training.trainer import Trainer

source, output, index = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
payload = torch.load(source, map_location='cpu', weights_only=False)
count = validate_initialization(payload)
output.mkdir()
base_path = output / 'source.yaml'
base_path.write_text(yaml.safe_dump(payload['config']))
base = ExperimentConfig.load(base_path)
settings = dict(interval=2, delay=2, split_fraction=0.1, prune_fraction=0.02,
                minimum_population_ratio=0.5, maximum_population_ratio=1.5,
                binarization_weight=0.001, binarization_start=0, binarization_ramp=2)
c = build_variant(base, index, count, settings, output)
c.wandb.mode = 'disabled'
c.optimization.iterations = 6
c.gaussian_adaptation.enabled = True
c.gaussian_adaptation.densify_from = 1
c.gaussian_adaptation.densify_until = 3
c.gaussian_adaptation.densification_interval = 2
c.gaussian_adaptation.opacity_prune_from = 1
c.gaussian_adaptation.opacity_prune_until = 6
c.gaussian_adaptation.opacity_prune_interval = 2
c.gaussian_adaptation.min_opacity = 0.001
c.gaussian_adaptation.min_opacity_initial = 0.001
c.voronoi_adaptation.densify_until = 6
c.validation.eval_interval = 6
c.validate()
trainer = Trainer(c, resume=source)
install_sweep_controller(trainer)
assert trainer.start_iteration == 1
assert trainer.model.medium.point_count == count
if index in (5,6):
    assert not trainer.voronoi_adaptation.should_collect(2)
    assert trainer.voronoi_adaptation.should_collect(3)
    assert not trainer.voronoi_adaptation.should_adapt(2)
    assert trainer.voronoi_adaptation.should_adapt(4)
trainer.fit()
final = torch.load(output / 'checkpoints/last.pt', map_location='cpu', weights_only=False)
assert final['iteration'] == 6
assert final['medium_optimizer']['state'] and final['gaussian_optimizer']['state']
for value in final['model'].values():
    assert torch.isfinite(value).all()
if index == 0:
    assert final['model_metadata']['medium']['point_count'] == count
if index in (3,4,5,6):
    assert final['voronoi_adaptation']['total_densified'] > 0
assert final['voronoi_adaptation']['total_failed_events'] == 0
