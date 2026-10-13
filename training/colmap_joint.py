"""IUI3 bootstrap, exact medium transfer and joint-stage execution."""
import json
import os
from pathlib import Path

from training.config import ExperimentConfig
from training.end_to_end import atomic_json
from training.warmup_profile import WarmupProfile

def transfer_medium(payload,device,color_activation='dc',color_mode='per_cell'):
    """Exact physical transfer, independent of the warmup's canonical color export.

    color_mode='global' starts the shared color at the mean transferred color.
    """
    import torch
    from foamsurf_model.medium import VoronoiMediumModel
    state={n:payload['model']['medium.'+n] for n in VoronoiMediumModel.PARAMETER_NAMES}
    if payload.get('warmup',{}).get('profile',{}).get('site_initialization') not in ('pcd','spatial'):
        raise ValueError('Expected DC COLMAP warmup')
    if color_activation == 'dc':
        state['raw_color']=payload['warmup']['raw_color']
    elif color_activation == 'sigmoid':
        color=(0.5+0.28209479177387814*payload['warmup']['raw_color']).clamp_min(0)
        if bool((color >= 1).any()):
            raise ValueError('Sigmoid cannot transfer warmup colors >= 1 exactly')
        state['raw_color']=torch.logit(color.clamp_min(1e-6))
    else:
        raise ValueError('Unknown medium color activation')
    metadata=dict(payload['model_metadata']['medium'],color_activation=color_activation,guard_count=0)
    medium=VoronoiMediumModel.from_checkpoint(state,metadata,payload['model']['medium.bbox_min'],
                                            payload['model']['medium.bbox_max'],device)
    # Outer origin is not a permanent protection from adaptation in joint.
    medium.guard_mask.zero_()
    if color_mode=='global':
        medium.enable_global_color(medium.colors.detach().mean(0))
    elif color_mode!='per_cell':
        raise ValueError('Unknown medium color mode')
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
            medium=transfer_medium(payload,config.device,
                manifest.get('medium_color_activation','dc'),
                manifest.get('medium_fix',{}).get('color_mode','per_cell'))
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
                color_activation=medium.color_activation,color_mode=medium.color_mode,
                fresh_optimizers=True,joint_iteration=0))
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
