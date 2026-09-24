#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import argparse
import itertools
import json
import subprocess
import yaml
from training.colmap_joint import resolve as base_resolve, run_stage as base_stage
from training.config import ExperimentConfig
from training.end_to_end import atomic_json, update_config
from experiments.rgb_smooth.loss import SmoothConfig


def resolve(path,index,output):
    path=Path(path).resolve();settings=yaml.safe_load(path.read_text())
    allowed={'base_sweep','base_variant','project','seeds','variants','ramp_steps',
             'coefficient_delta','color_delta','epsilon'}
    if set(settings)-allowed:raise ValueError('Unknown smoothing sweep setting')
    choices=list(itertools.product(settings['variants'],settings['seeds']))
    if not 0<=index<len(choices):raise ValueError('Variant out of range')
    variant,seed=choices[index]
    if set(variant)!={'name','attenuation','backscatter','color'}:raise ValueError('Invalid variant')
    smoothing=SmoothConfig(**{k:v for k,v in variant.items() if k!='name'},
        **{k:settings[k] for k in ('ramp_steps','coefficient_delta','color_delta','epsilon')})
    manifest=base_resolve(path.parent/settings['base_sweep'],settings['base_variant'],output)
    manifest['warmup_steps']=0
    manifest['warmup']['optimization']['iterations']=1  # constructor only, NO steps
    manifest['warmup']['voronoi_warmup']['milestone_iterations']=[0]
    manifest['rgb_smoothing']=smoothing.as_dict()
    manifest['index']=index
    manifest['variant'].update(name=f'{variant["name"]}_seed{seed}',warmup_steps=0,seed=seed)
    for name in ('warmup','joint'):
        stage=manifest[name];stage['seed']=seed
        if stage['model']['medium_type']!='rgb_backscatter':raise ValueError('RGB base required')
        stage['experiment_name']=f'iui3-smooth-{index:02d}-{variant["name"]}-seed{seed}-{name}'
        stage['wandb'].update(project=settings['project'],name=stage['experiment_name'],
                              tags=['rgb-smooth','no-warmup',variant['name'],f'seed-{seed}',name])
        config=ExperimentConfig();update_config(config,stage);config.validate()
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
    from experiments.rgb_smooth.trainer import SmoothJointTrainer
    original=module.ColmapJointTrainer
    try:
        module.ColmapJointTrainer=SmoothJointTrainer
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
