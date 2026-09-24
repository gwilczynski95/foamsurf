"""COLMAP -> optional native warmup -> adaptive joint. No partition required.

Separate orchestrator keeps the historical LEGO end-to-end contract unchanged.
"""
from copy import deepcopy
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml
from training.end_to_end import EndToEndConfig, update_config, atomic_json
from training.config import ExperimentConfig
from training.warmup_profile import WarmupProfile

ROOT=Path(__file__).resolve().parents[1]


def variants(settings):
    if 'timing' in settings:
        timing=settings['timing']
        if set(timing)!={'initializations','warmup_lengths'}:
            raise ValueError('Timing requires initializations and warmup_lengths')
        lengths=timing['warmup_lengths']
        if not lengths or any(type(n) is not int or n<0 for n in lengths) or len(set(lengths))!=len(lengths):
            raise ValueError('Warmup lengths must be unique nonnegative integers')
        result=[]
        for init in timing['initializations']:
            if set(init)!={'count','fraction','extent'}:
                raise ValueError('Timing initialization requires count, fraction, extent')
            n,f,e=init['count'],init['fraction'],init['extent']
            for steps in lengths:
                result.append(dict(name=f'spatial_n{n}_f{f:g}_e{e:g}_w{steps}',
                                   mode='spatial',**init,warmup_steps=steps))
        if not result or len({v['name'] for v in result})!=len(result):
            raise ValueError('Timing initializations must be nonempty and unique')
        if settings.get('medium_experiment', False):
            result=[dict(v,medium_type=kind,seed=seed,
                         name=f'{v["name"]}_{kind}_seed{seed}')
                    for v,kind,seed in itertools.product(result,('scalar','rgb_backscatter'),settings['seeds'])]
        return result
    return [dict(name='radfoam_control',mode='pcd')] + [
        dict(name=f'spatial_n{n}_f{f:g}_e{e:g}',mode='spatial',count=n,fraction=f,extent=e)
        for n,f,e in itertools.product(settings['counts'],settings['colmap_fractions'],settings['extents'])]


def resolve(path,index,output):
    path=Path(path).resolve()
    s=yaml.safe_load(path.read_text())
    allowed={'base','warmup_iterations','joint_iterations','counts','colmap_fractions','extents',
             'warmup_overrides','joint_overrides','profile_overrides','timing','medium_experiment','seeds'}
    if set(s)-allowed: raise ValueError('Unknown COLMAP sweep setting')
    choices=variants(s)
    if not 0 <= index < len(choices): raise ValueError('Variant out of range')
    v=choices[index]
    c=EndToEndConfig.load(path.parent/s['base'])
    w,j,p=c.warmup,c.joint,c.profile
    warmup_steps=v.get('warmup_steps',s['warmup_iterations'])
    if not isinstance(warmup_steps,int) or warmup_steps<0: raise ValueError('Invalid warmup length')
    w.optimization.iterations=max(1,warmup_steps)
    j.optimization.iterations=s['joint_iterations']
    w.voronoi_warmup.milestone_iterations=sorted(set([0,warmup_steps]))
    w.voronoi_adaptation.densify_until=max(11000,w.optimization.iterations)
    if v['mode']=='spatial':
        p.site_initialization='spatial';p.outer_points=0
        p.spatial_count=v['count'];p.spatial_colmap_fraction=v['fraction'];p.spatial_extent=v['extent']
        w.model.medium_points=v['count']
    for stage,overrides in [(w,s.get('warmup_overrides',{})),(j,s.get('joint_overrides',{}))]:
        update_config(stage,overrides)
        if 'medium_type' in v:
            stage.model.medium_type=v['medium_type'];stage.seed=v['seed']
    p=WarmupProfile.from_dict(dict(p.as_dict(),**s.get('profile_overrides',{})))
    from foamsurf_data.colmap import _read_points_binary, _read_points_text
    sparse=Path(j.dataset.source_path)/'sparse/0'
    points,_=(_read_points_binary(sparse/'points3D.bin') if (sparse/'points3D.bin').exists()
              else _read_points_text(sparse/'points3D.txt'))
    j.model.gaussian_initial_points=len(points)
    for name,stage in [('warmup',w),('joint',j)]:
        stage.output_dir=str(Path(output)/name)
        family='C' if s.get('medium_experiment') else ('B' if 'timing' in s else 'A')
        stage.experiment_name=f'iui3-{family}{index:02d}-{v["name"]}-{name}'
        stage.wandb.name=stage.experiment_name
        stage.wandb.project=j.wandb.project
        stage.wandb.mode=j.wandb.mode
        stage.wandb.tags=[f'colmap-stage-{family}',v['name'],name]
        stage.validate()
    if w.model.medium_bbox_min is not None or j.model.medium_bbox_min is not None:
        raise ValueError('COLMAP joint forbids manual bounds')
    if w.model.medium_type != j.model.medium_type:
        raise ValueError('Warmup and joint must use the same medium model')
    if w.dataset != j.dataset: raise ValueError('Stages must use identical datasets')
    if w.dataset.train_target!='both' or j.radfoam_prior.enabled:
        raise ValueError('Only full RGB, no external RadFoam prior')
    return dict(variant=v,index=index,warmup_steps=warmup_steps,warmup=w.as_dict(),
                joint=j.as_dict(),profile=p.as_dict(),output_dir=str(Path(output).resolve()))


def fit(path,index,output,resume=False):
    output=Path(output).resolve()
    manifest=resolve(path,index,output)
    mp=output/'pipeline_config.json'
    if output.exists() and any(output.iterdir()):
        if not resume or not mp.exists() or json.loads(mp.read_text())!=manifest:
            raise ValueError('Nonempty output; resume requires identical configuration')
    else:
        output.mkdir(parents=True,exist_ok=True)
        atomic_json(mp,manifest)
        for name in ('warmup','joint'):
            (output/f'{name}.yaml').write_text(yaml.safe_dump(manifest[name],sort_keys=False))
    for stage in ('warmup','joint'):
        marker=output/f'{stage}_done.json'
        if marker.exists():
            if not Path(json.loads(marker.read_text())['artifact']).is_file():
                raise FileNotFoundError('Missing completed stage checkpoint')
            continue
        subprocess.run([sys.executable,str(ROOT/'scripts/train_colmap_joint.py'),
                        '--stage',stage,'--output-dir',str(output)],cwd=ROOT,check=True)
        if not marker.exists(): raise RuntimeError(f'{stage} incomplete')


def transfer_medium(payload,device):
    """Exact physical transfer, independent of the warmup's canonical color export."""
    import torch
    from foamsurf_model.medium import VoronoiMediumModel
    state={n:payload['model']['medium.'+n] for n in VoronoiMediumModel.PARAMETER_NAMES}
    if payload.get('warmup',{}).get('profile',{}).get('site_initialization') not in ('pcd','spatial'):
        raise ValueError('Expected DC COLMAP warmup')
    state['raw_color']=payload['warmup']['raw_color']
    metadata=dict(payload['model_metadata']['medium'],color_activation='dc',guard_count=0)
    medium=VoronoiMediumModel.from_checkpoint(state,metadata,payload['model']['medium.bbox_min'],
                                            payload['model']['medium.bbox_max'],device)
    # Outer origin is not a permanent protection from adaptation in joint.
    medium.guard_mask.zero_()
    return medium


def install_joint_topology(trainer):
    from training.warmup_trainer import WarmupTopology
    trainer.topology.__class__=WarmupTopology
    trainer.topology.organic=True
    # Existing controller clamps child proposals to metadata bounds. Expand those
    # to current sites before proposing children; do not clip displaced parents.
    controller=trainer.voronoi_adaptation
    original=controller.propose_plan
    def plan(*args,**kwargs):
        import torch
        m=trainer.model.medium
        m.bbox_min=torch.minimum(m.bbox_min,m.sites.detach().amin(0))
        m.bbox_max=torch.maximum(m.bbox_max,m.sites.detach().amax(0))
        return original(*args,**kwargs)
    controller.propose_plan=plan


def run_stage(output,stage):
    import torch
    import numpy as np
    import random
    from training.checkpoints import load_checkpoint
    output=Path(output).resolve()
    manifest=json.loads((output/'pipeline_config.json').read_text())
    config=ExperimentConfig.load(output/f'{stage}.yaml')
    # Avoid reusing one W&B id for both stages.
    if os.environ.get('WANDB_RUN_ID'):
        os.environ['WANDB_RUN_ID']+='-'+stage
    last=output/stage/'checkpoints/last.pt'
    if stage=='warmup':
        from training.warmup_trainer import WarmupTrainer
        trainer=WarmupTrainer(config,WarmupProfile.from_dict(manifest['profile']),resume=last if last.exists() else None)
        if not last.exists():
            trainer._checkpoint(output/'warmup/checkpoints/initialization.pt',0)
            save_initialization_preview(trainer,output)
        if manifest['warmup_steps']==0:
            trainer._checkpoint(last,0);trainer.logger.close()
        elif trainer.start_iteration<=config.optimization.iterations:
            result=trainer.fit()
            if result['status']!='completed': raise RuntimeError('Warmup incomplete')
        else: trainer.logger.close()
    elif stage=='joint':
        from training.colmap_joint_trainer import ColmapJointTrainer as Trainer
        from foamsurf_data import load_dataset
        from foamsurf_model.gaussians import SurfaceGaussianModel
        from foamsurf_model.scene import HybridSceneModel
        if last.exists(): trainer=Trainer(config,resume=last)
        else:
            random.seed(config.seed);np.random.seed(config.seed);torch.manual_seed(config.seed);torch.cuda.manual_seed_all(config.seed)
            payload=load_checkpoint(output/'warmup/checkpoints/last.pt')
            medium=transfer_medium(payload,config.device)
            dataset=load_dataset(config.dataset.source_path,resolution=config.dataset.resolution,
                                 device=config.device,llffhold=config.dataset.llffhold)
            points,colors=dataset.initial_point_cloud()
            gaussians=SurfaceGaussianModel(points,colors,config.model,config.device)
            config.model.gaussian_initial_points=len(points)
            config.model.medium_points=medium.point_count
            if medium.point_count>config.voronoi_adaptation.max_points:
                raise ValueError('Warmup exceeds joint population budget; refusing implicit truncation')
            # Metadata includes all retained sites; no inherited small scene box.
            medium.bbox_min=torch.minimum(medium.bbox_min,medium.sites.detach().amin(0))
            medium.bbox_max=torch.maximum(medium.bbox_max,medium.sites.detach().amax(0))
            config.save(output/'joint.yaml')
            trainer=Trainer(config,initial_model=HybridSceneModel(gaussians,medium))
            trainer._checkpoint(output/'joint/checkpoints/initialization.pt',0)
            trainer._checkpoint(last,0)
            trainer._evaluate_and_checkpoint(0)
            atomic_json(output/'handoff_report.json',dict(source_count=medium.point_count,
                retained_count=medium.point_count,gaussian_count=len(points),removed_count=0,
                color_activation='dc',fresh_optimizers=True,joint_iteration=0))
        install_joint_topology(trainer)
        if trainer.start_iteration<=config.optimization.iterations:
            result=trainer.fit()
            if result['status']!='completed': raise RuntimeError('Joint incomplete')
        else: trainer.logger.close()
    else: raise ValueError(stage)
    atomic_json(output/f'{stage}_done.json',dict(artifact=str(last)))


def save_initialization_preview(trainer,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    m=trainer.model.medium
    sites=m.sites.detach().cpu().numpy()
    points,_=trainer.dataset.initial_point_cloud()
    cameras=np.stack([c.origin.cpu().numpy() for c in trainer.dataset.train])
    ids=np.random.default_rng(0).choice(len(sites),min(10000,len(sites)),replace=False)
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,(a,b) in zip(axes,[(0,1),(0,2),(1,2)]):
        ax.scatter(sites[ids,a],sites[ids,b],s=1,alpha=.15,label='Voronoi')
        ax.scatter(points[:,a],points[:,b],s=1,alpha=.3,label='COLMAP')
        ax.scatter(cameras[:,a],cameras[:,b],s=15,label='Train cameras')
        ax.set_xlabel('XYZ'[a]);ax.set_ylabel('XYZ'[b]);ax.legend()
    fig.tight_layout();fig.savefig(output/'initialization.png',dpi=130);plt.close(fig)
    atomic_json(output/'initialization_geometry.json',getattr(m,'initialization_report',dict(mode='pcd')))
