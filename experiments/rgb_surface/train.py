#!/usr/bin/env python3
"""Factorial background x local optical overlap x graph TV, with paired seeds."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
import itertools
import json
import math
import subprocess
import yaml

from experiments.rgb_smooth.train import resolve as smooth_resolve
from experiments.rgb_smooth.loss import SmoothConfig
from experiments.rgb_surface.loss import SurfaceConfig
from training.colmap_joint import run_stage as base_stage
from training.config import ExperimentConfig
from training.end_to_end import atomic_json, update_config


def settings(path):
    value = yaml.safe_load(Path(path).read_text())
    if set(value) != {'base_smoothing', 'project', 'seeds', 'backgrounds', 'overlap_weights', 'tv_multipliers', 'overlap', 'smoothing'}:
        raise ValueError('Invalid surface sweep settings')
    for key in ('seeds', 'backgrounds', 'overlap_weights', 'tv_multipliers'):
        if not value[key] or len(set(value[key])) != len(value[key]):
            raise ValueError(f'Invalid {key}')
    if any(type(s) is not int or s < 0 for s in value['seeds']):
        raise ValueError('Invalid seeds')
    if set(value['backgrounds']) - {'white', 'black'}:
        raise ValueError('Invalid background')
    for key in ('overlap_weights', 'tv_multipliers'):
        if any(not math.isfinite(v) or v < 0 for v in value[key]):
            raise ValueError(f'Invalid {key}')
    SurfaceConfig(**value['overlap'])
    SmoothConfig(**value['smoothing'])
    return value


def choices(value):
    return list(itertools.product(value['backgrounds'], value['overlap_weights'],
                                  value['tv_multipliers'], value['seeds']))


def resolve(path, index, output):
    path = Path(path).resolve()
    s = settings(path)
    variants = choices(s)
    if not 0 <= index < len(variants):
        raise ValueError('Variant out of range')
    bg, ow, tv, seed = variants[index]
    m = smooth_resolve(path.parent/s['base_smoothing'], 0, output)
    m['rgb_surface'] = SurfaceConfig(**dict(s['overlap'], weight=ow)).as_dict()
    sm = dict(s['smoothing'])
    for key in ('attenuation', 'backscatter', 'color'):
        sm[key] *= tv
    m['rgb_smoothing'] = SmoothConfig(**sm).as_dict()
    name = f'{bg}_O{ow:g}_TV{tv:g}_seed{seed}'
    m['variant'].update(name=name, seed=seed, background=bg, overlap_weight=ow, tv_multiplier=tv)
    m['index'] = index
    for stage in ('warmup', 'joint'):
        c = m[stage]
        c['seed'] = seed
        c['background'] = [0., 0., 0.] if bg == 'black' else [1., 1., 1.]
        c['experiment_name'] = f'iui3-surface-{index:02d}-{name}-{stage}'
        c['wandb'].update(entity='mlpolska', project=s['project'], name=c['experiment_name'],
                          tags=['rgb-surface', 'no-warmup', bg, f'O-{ow:g}', f'TV-{tv:g}', f'seed-{seed}', stage])
        config = ExperimentConfig()
        update_config(config, c)
        config.validate()
    return m


def fit(manifest, resume=False):
    output = Path(manifest['output_dir'])
    mp = output/'pipeline_config.json'
    if output.exists() and any(output.iterdir()):
        if not resume or not mp.exists() or json.loads(mp.read_text()) != manifest:
            raise ValueError('Resume requires identical manifest')
    else:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(mp, manifest)
        for stage in ('warmup', 'joint'):
            (output/f'{stage}.yaml').write_text(yaml.safe_dump(manifest[stage], sort_keys=False))
    for stage in ('warmup', 'joint'):
        marker = output/f'{stage}_done.json'
        if marker.exists():
            artifact = Path(json.loads(marker.read_text())['artifact'])
            if not artifact.is_file():
                raise FileNotFoundError(artifact)
            if stage == 'joint':
                for suffix, key in (('.smooth.json', 'rgb_smoothing'), ('.surface.json', 'rgb_surface')):
                    if json.loads(artifact.with_suffix(suffix).read_text()) != manifest[key]:
                        raise ValueError('Checkpoint loss settings mismatch')
            continue
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--stage', stage,
                        '--output-dir', str(output)], cwd=ROOT, check=True)
        if not marker.exists():
            raise RuntimeError('Stage incomplete')


def run_stage(output, stage):
    manifest = json.loads((Path(output)/'pipeline_config.json').read_text())
    if manifest['warmup_steps'] != 0:
        raise ValueError('No warmup in this sweep')
    if stage == 'warmup':
        return base_stage(output, stage)
    import training.colmap_joint_trainer as module
    from experiments.rgb_surface.trainer import SurfaceTrainer
    original = module.ColmapJointTrainer
    try:
        module.ColmapJointTrainer = SurfaceTrainer
        return base_stage(output, stage)
    finally:
        module.ColmapJointTrainer = original


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).with_name('sweep.yaml')))
    parser.add_argument('--variant', type=int, default=0)
    parser.add_argument('--output-dir')
    parser.add_argument('--stage', choices=['warmup', 'joint'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    if args.list:
        for i, (bg, ow, tv, seed) in enumerate(choices(settings(args.config))):
            print(f'{i:02d}  {bg:5}  overlap={ow:g}  TV={tv:g}  seed={seed}')
    elif not args.output_dir:
        parser.error('--output-dir is required')
    elif args.stage:
        run_stage(args.output_dir, args.stage)
    else:
        manifest = resolve(args.config, args.variant, Path(args.output_dir).resolve())
        if args.dry_run:
            print(json.dumps(manifest, indent=2))
        else:
            fit(manifest, args.resume)
