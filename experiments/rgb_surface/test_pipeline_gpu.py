import csv
import gc
import json
from pathlib import Path

import pytest
import torch

from experiments.rgb_surface.train import resolve, fit
from experiments.rgb_surface.trainer import SurfaceTrainer
from training.colmap_joint import install_joint_topology
from training.config import ExperimentConfig


def manifest(tmp_path, full=False):
    # Strongest black-background combination, first paired seed.
    m = resolve(Path(__file__).with_name('sweep.yaml'), 69, tmp_path/'run')
    m['profile']['spatial_count'] = 25000 if full else 256
    for name in ('warmup', 'joint'):
        c = m[name]
        c['dataset']['resolution'] = 1 if full else 16
        c['wandb']['mode'] = 'disabled'
        c['validation'].update(eval_interval=2, checkpoint_interval=2, log_interval=1, max_eval_views=1, image_views=1)
    j = m['joint']
    j['optimization']['iterations'] = 2 if full else 4
    j['gaussian_adaptation'].update(densify_from=1, densify_until=6, densification_interval=2,
        opacity_prune_from=1, opacity_prune_ramp=0, opacity_prune_until=j['optimization']['iterations'])
    j['voronoi_adaptation'].update(densify_from=2, densify_until=6, interval=2, min_points=32)
    m['rgb_surface'].update(start=0, ramp_steps=0, rays=128 if full else 32)
    m['rgb_smoothing']['ramp_steps'] = 2
    return m


def check_output(m):
    output = Path(m['output_dir'])
    warm = torch.load(output/'warmup/checkpoints/last.pt', map_location='cpu', weights_only=False)
    assert warm['iteration'] == 0
    assert not (output/'warmup/metrics/train.csv').exists()
    rows = list(csv.DictReader((output/'joint/metrics/train.csv').open()))
    for key in ('surface_overlap/weighted', 'rgb_smooth/weighted'):
        assert any(float(r[key]) > 0 for r in rows), key
    assert all(float(r['surface_overlap/rays']) == m['rgb_surface']['rays'] for r in rows)
    assert all(float(r['surface_overlap/traversal_truncated_fraction']) == 0 for r in rows)
    final = torch.load(output/'joint/checkpoints/last.pt', map_location='cpu', weights_only=False)
    assert final['voronoi_adaptation']['total_densified'] > 0
    assert all(torch.isfinite(v).all() for v in final['model'].values())
    directory = output/'joint/evaluation'/f'iteration_{m["joint"]["optimization"]["iterations"]:06d}'
    assert list(directory.glob('*_coverage_panel.png'))
    assert list(directory.glob('*_gaussian_alpha.npz'))
    assert list(directory.glob('*_gaussian_only_black.png'))
    assert (directory/'coverage.csv').is_file()
    return output


def test_pipeline_and_resume(tmp_path):
    assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
    m = manifest(tmp_path)
    fit(m)
    fit(m, resume=True)
    out = check_output(m)
    last = out/'joint/checkpoints/last.pt'
    assert json.loads(last.with_suffix('.surface.json').read_text()) == m['rgb_surface']
    c = ExperimentConfig.load(out/'joint.yaml')
    c.output_dir = str(tmp_path/'resumed')
    trainer = SurfaceTrainer(c, resume=last, smoothing=m['rgb_smoothing'], surface=m['rgb_surface'])
    install_joint_topology(trainer)
    assert torch.isfinite(trainer._train_step(5)[0].total)
    assert trainer.surface_metrics['surface_overlap/weighted'] > 0
    trainer.logger.close()
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    last.with_suffix('.surface.json').unlink()
    with pytest.raises(FileNotFoundError):
        fit(m, resume=True)


def test_production(tmp_path):
    assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
    m = manifest(tmp_path, full=True)
    fit(m)
    check_output(m)
