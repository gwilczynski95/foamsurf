#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import argparse
import itertools
import math
import json
import subprocess
import yaml
from training.colmap_joint import resolve as base_resolve, run_stage as base_stage
from training.config import ExperimentConfig
from training.end_to_end import atomic_json, update_config
from experiments.rgb_smooth.loss import SmoothConfig
from experiments.rgb_smooth.train import resolve as smooth_resolve
from experiments.rgb_disentangle.losses import ConflictConfig


def resolve(path,index,output):
    path=Path(path).resolve();s=yaml.safe_load(path.read_text())
    allowed={'base_smoothing','project','seeds','strengths','conflict','smoothing'}
    if set(s)!=allowed:raise ValueError('Invalid conflict sweep settings')
    if not s['seeds'] or len(set(s['seeds']))!=len(s['seeds']):raise ValueError('Invalid seeds')
    if any(type(seed) is not int or seed<0 for seed in s['seeds']):raise ValueError('Invalid seed')
    if not s['strengths'] or len(set(s['strengths']))!=len(s['strengths']):raise ValueError('Invalid strengths')
    if any(not math.isfinite(v) or v<=0 for v in s['strengths']):raise ValueError('Invalid strength')
    choices=[('control',0.,seed) for seed in s['seeds']]
    choices += [(group,strength,seed) for group in ('O','G','S','OG','OS','GS','OGS')
                for strength in s['strengths'] for seed in s['seeds']]
    if not 0<=index<len(choices):raise ValueError('Variant out of range')
    group,strength,seed=choices[index]
    manifest=smooth_resolve(path.parent/s['base_smoothing'],0,output)
    c=ConflictConfig(**s['conflict']).as_dict()
    c['overlap_weight']*=strength if 'O' in group else 0
    c['opacity_weight']*=strength if 'G' in group else 0
    sm=SmoothConfig(**s['smoothing']).as_dict()
    for key in ('attenuation','backscatter','color'):sm[key]*=strength if 'S' in group else 0
    manifest['rgb_conflict']=ConflictConfig(**c).as_dict()
    manifest['rgb_smoothing']=SmoothConfig(**sm).as_dict()
    manifest['index']=index
    name=f'{group}_x{strength:g}_seed{seed}'
    manifest['variant'].update(name=name,seed=seed,group=group,strength=strength)
    for stage in ('warmup','joint'):
        conf=manifest[stage];conf['seed']=seed
        conf['experiment_name']=f'iui3-conflict-{index:02d}-{name}-{stage}'
        conf['wandb'].update(project=s['project'],name=conf['experiment_name'],
                             tags=['rgb-conflict','no-warmup',group,f'seed-{seed}',stage])
        config=ExperimentConfig();update_config(config,conf);config.validate()
    return manifest


def fit(manifest,resume=False):
    output=Path(manifest['output_dir']);mp=output/'pipeline_config.json'
    if output.exists() and any(output.iterdir()):
        if not resume or not mp.exists() or json.loads(mp.read_text())!=manifest:
            raise ValueError('Resume requires identical manifest')
    else:
        output.mkdir(parents=True,exist_ok=True);atomic_json(mp,manifest)
        for stage in ('warmup','joint'):
            (output/f'{stage}.yaml').write_text(yaml.safe_dump(manifest[stage],sort_keys=False))
    for stage in ('warmup','joint'):
        marker=output/f'{stage}_done.json'
        if marker.exists():
            artifact=Path(json.loads(marker.read_text())['artifact'])
            if not artifact.is_file():raise FileNotFoundError(artifact)
            if stage=='joint' and not artifact.with_suffix('.smooth.json').is_file():
                raise FileNotFoundError('Missing smoothing checkpoint sidecar')
            if stage=='joint' and not artifact.with_suffix('.conflict.json').is_file():
                raise FileNotFoundError('Missing conflict checkpoint sidecar')
            if stage=='joint':
                for suffix,key in (('.smooth.json','rgb_smoothing'),('.conflict.json','rgb_conflict')):
                    if json.loads(artifact.with_suffix(suffix).read_text())!=manifest[key]:
                        raise ValueError('Checkpoint loss settings mismatch')
            continue
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--stage',stage,
                        '--output-dir',str(output)],cwd=ROOT,check=True)
        if not marker.exists():raise RuntimeError('Stage incomplete')


def run_stage(output,stage):
    manifest=json.loads((Path(output)/'pipeline_config.json').read_text())
    if manifest['warmup_steps']!=0:raise ValueError('Only zero-warmup runs supported')
    if stage=='warmup':return base_stage(output,stage)
    # Process-local selection. Existing workers never import this module.
    import training.colmap_joint_trainer as module
    from experiments.rgb_disentangle.trainer import ConflictTrainer
    original=module.ColmapJointTrainer
    try:
        module.ColmapJointTrainer=ConflictTrainer
        return base_stage(output,stage)
    finally:module.ColmapJointTrainer=original


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default=str(Path(__file__).with_name('sweep.yaml')))
    parser.add_argument('--variant',type=int,default=0)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--stage',choices=['warmup','joint'])
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.stage:run_stage(args.output_dir,args.stage)
    else:
        manifest=resolve(args.config,args.variant,Path(args.output_dir).resolve())
        if args.dry_run:print(json.dumps(manifest,indent=2))
        else:fit(manifest,args.resume)
