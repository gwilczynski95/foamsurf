#!/usr/bin/env python3
"""Checkpoint-only and privileged analyses, in a separate post-training process.

Outer sites are label -1 and never sampled as surface/medium/empty occupancy.
Two Otsu stages use TRAIN RGB contribution and inner-inner density edges only.
Ground-truth layers enter exclusively after these predictions have been fixed.
"""
import argparse
import json
from pathlib import Path
import sys
import gc
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training import ExperimentConfig
from training.trainer import Trainer
from training.warmup_quantiles import render_quantiles
from scripts.analyze_voronoi_warmup import (
    _otsu,_render_separation,_rays,_plot_training,_read_csv,_to_image,_grid,
    EMPTY,SURFACE,MEDIUM,CLASS_NAMES)


def quantiles(values):
    if not values.numel():return {}
    qs=torch.tensor([.01,.05,.5,.95,.99],device=values.device)
    return {str(float(q)):float(v) for q,v in zip(qs,torch.quantile(values.float(),qs))}


def classify(sigma,contribution,adjacency,offsets,outer):
    inner=~outer
    labels=torch.full_like(contribution,-1,dtype=torch.long)
    labels[inner]=EMPTY
    degree=offsets[1:].long()-offsets[:-1].long()
    owners=torch.repeat_interleave(torch.arange(len(sigma),device=sigma.device),degree)
    neighbors=adjacency.long()
    keep=inner[owners]&inner[neighbors]
    owners,neighbors=owners[keep],neighbors[keep]
    sums=torch.zeros_like(sigma).index_add_(0,owners,(sigma[owners]-sigma[neighbors]).abs())
    counts=torch.zeros_like(sigma).index_add_(0,owners,torch.ones_like(owners,dtype=sigma.dtype))
    tv=sums/counts.clamp_min(1)
    cscore=torch.log10(contribution.clamp_min(1e-12))
    cthreshold=_otsu(cscore[inner])
    active=inner&(contribution>0)&(cscore>=cthreshold)
    score=torch.log10(tv.clamp_min(1e-12))
    threshold=_otsu(score[active]) if bool(active.any()) else None
    if threshold is not None:
        labels[active&(score>=threshold)]=SURFACE
        labels[active&(score<threshold)]=MEDIUM
    return labels,tv,{'contribution_log10_threshold':cthreshold,'tv_log10_threshold':threshold}


@torch.no_grad()
def oracle_contribution(trainer):
    result=torch.zeros((trainer.model.medium.point_count,2),device=trainer.device)
    for i,camera in enumerate(trainer.dataset.test):
        lego=trainer.layer_ground_truth.lego_lego[i].alpha.to(trainer.device).squeeze(-1)
        cloud=trainer.layer_ground_truth.lego_clouds[i].alpha.to(trainer.device).squeeze(-1)
        rays=_rays(camera)
        for j,mask in enumerate(((lego>.01)&(cloud<=.01),(cloud>.01)&(lego<=.01))):
            if bool(mask.any()):
                render=trainer.renderer.render(rays[mask],voronoi=trainer.model.medium.render_scene(),
                    background=tuple(trainer.config.background),collect_diagnostics=True)
                result[:,j]+=render.diagnostics.medium[:,0]
    return result


@torch.no_grad()
def quantile_diagnostics(trainer,output,oracle,views):
    widths={'all':[],'lego_only':[],'cloud_only':[]}
    valid_counts={key:[0,0] for key in widths}
    owner_count=torch.zeros(trainer.model.medium.point_count,device=trainer.device)
    alphas=[];validity=[]
    # Local generator: analysis is reproducible and independent of training RNG.
    gen=torch.Generator(device=trainer.device).manual_seed(12345)
    for i,camera in enumerate(trainer.dataset.test):
        rays=_rays(camera).reshape(-1,6)
        selected=torch.arange(0,len(rays),64,device=trainer.device)
        sample=rays[selected]
        q=torch.rand(len(sample),2,device=trainer.device,generator=gen).clamp_min(1e-7).sort(descending=True).values
        depth,owners=render_quantiles(trainer.renderer,trainer.model.medium.render_scene(),sample,q)
        valid=(depth>0).all(-1)
        width=(depth[:,1]-depth[:,0]).abs()
        idx=owners[owners>=0].long()
        owner_count.index_add_(0,idx,torch.ones_like(idx,dtype=owner_count.dtype))
        lego=trainer.layer_ground_truth.lego_lego[i].alpha.to(trainer.device).reshape(-1)[selected]
        cloud=trainer.layer_ground_truth.lego_clouds[i].alpha.to(trainer.device).reshape(-1)[selected]
        masks={'all':torch.ones_like(valid),'lego_only':(lego>.01)&(cloud<=.01),'cloud_only':(cloud>.01)&(lego<=.01)}
        for name,mask in masks.items():
            valid_counts[name][0]+=int((valid&mask).sum());valid_counts[name][1]+=int(mask.sum())
            widths[name].append(width[valid&mask].cpu())
        alpha=trainer.renderer.render_voronoi(sample,trainer.model.medium.render_scene()).alpha.flatten()
        alphas.append(alpha.cpu());validity.append(valid.float().cpu())
        if i<views:
            qfixed=torch.tensor([.9,.5],device=trainer.device).expand(len(rays),2).contiguous()
            ds,_=render_quantiles(trainer.renderer,trainer.model.medium.render_scene(),rays,qfixed)
            ok=(ds>0).all(-1).reshape(camera.height,camera.width)
            dw=(ds[:,1]-ds[:,0]).abs().reshape(camera.height,camera.width)
            fig,axes=plt.subplots(1,2,figsize=(10,4))
            axes[0].imshow(ok.cpu(),vmin=0,vmax=1);axes[0].set_title('valid q=(0.9,0.5)')
            axes[1].imshow(torch.where(ok,dw,torch.nan).cpu(),cmap='magma');axes[1].set_title('quantile width (scene units)')
            fig.tight_layout();fig.savefig(output/f'view_{i:02d}_quantiles.png');plt.close(fig)
    stats={name:{'valid':valid_counts[name][0],'sampled':valid_counts[name][1],
                 'width_quantiles':quantiles(torch.cat(values))} for name,values in widths.items()}
    a=torch.cat(alphas);v=torch.cat(validity)
    stats['alpha_validity_correlation']=float(torch.corrcoef(torch.stack((a,v)))[0,1]) if a.std()>0 and v.std()>0 else None
    stats['owners_by_oracle']={name:float(owner_count[oracle==label].sum()) for label,name in CLASS_NAMES.items()}
    return stats,owner_count



def plot_objective(run, destination):
    train=_read_csv(run/'metrics'/'train.csv')
    required={'warmup/rgb_loss','warmup/weighted_opacity_loss',
              'warmup/weighted_quantile_loss','warmup/target_alpha_mean',
              'warmup/rendered_alpha_mean'}
    if not required.issubset(train):
        return
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    iteration=train['iteration']
    for key,label in (('warmup/rgb_loss','RGB'),
                      ('warmup/weighted_opacity_loss','weighted opacity'),
                      ('warmup/weighted_quantile_loss','weighted quantile')):
        axes[0].plot(iteration,np.maximum(train[key],1e-12),label=label)
    axes[0].set_yscale('log')
    axes[0].set(title='Warmup objective terms',xlabel='iteration',ylabel='loss (log)')
    axes[0].legend();axes[0].grid(alpha=.25)
    axes[1].plot(iteration,train['warmup/target_alpha_mean'],label='target')
    axes[1].plot(iteration,train['warmup/rendered_alpha_mean'],label='rendered')
    axes[1].set(title='Opacity supervision',xlabel='iteration',ylabel='mean alpha',ylim=(-.02,1.02))
    axes[1].legend();axes[1].grid(alpha=.25)
    fig.savefig(destination/'objective_dashboard.png',dpi=160)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True)
    p.add_argument('--iterations',nargs='+',type=int)
    p.add_argument('--image-views',type=int,default=4)
    args=p.parse_args()
    run=Path(args.run).resolve();destination=run/'warmup_analysis';destination.mkdir(exist_ok=True)
    checkpoints=sorted((run/'checkpoints').glob('iter_*.pt'))
    records=[]
    for checkpoint in checkpoints:
        iteration=int(checkpoint.stem.split('_')[-1])
        if args.iterations is not None and iteration not in args.iterations:continue
        output=destination/f'iteration_{iteration:06d}';output.mkdir(exist_ok=True)
        if (output/'metrics.json').exists():
            records.append(json.loads((output/'metrics.json').read_text()));continue
        config=ExperimentConfig.load(run/'config.yaml')
        # Analysis does not train; a fresh diagnostics directory is allowed.
        config.output_dir=str(output/'diagnostics');config.wandb.mode='disabled'
        config.validation.image_views=0
        trainer=Trainer(config,resume=checkpoint)
        try:
            trainer.renderer.transmittance_threshold=config.renderer.validation_transmittance_threshold
            with torch.no_grad():
                evaluation,_,_ = trainer.evaluator.evaluate(iteration)
                medium=trainer.model.medium;outer=medium.guard_mask
                contribution=torch.zeros(medium.point_count,device=trainer.device)
                cell_stats=torch.zeros((medium.point_count,5),device=trainer.device)
                intersections=[];truncated=[]
                for camera in trainer.dataset.train:
                    rendered=trainer.renderer.render(_rays(camera),voronoi=medium.render_scene(),
                        background=tuple(config.background),collect_diagnostics=True,
                        diagnostic_target=camera.ground_truth(tuple(config.background)))
                    stats=rendered.diagnostics.medium
                    cell_stats[:,:4]+=stats[:,:4]
                    cell_stats[:,4]=torch.maximum(cell_stats[:,4],stats[:,4])
                    contribution+=stats[:,0]
                    intersections.append(rendered.num_intersections.flatten()[::64].float().cpu())
                    truncated.append(float(rendered.diagnostics.traversal_truncated.mean()))
                sigma=medium.densities.flatten()
                labels,tv,thresholds=classify(sigma,contribution,trainer.topology.adjacency,trainer.topology.offsets,outer)
                metrics=_render_separation(trainer,labels,output,iteration,args.image_views)
                layer=oracle_contribution(trainer)
                total=layer.sum(-1)
                oracle=torch.full_like(labels,EMPTY)
                oracle[(total>0)&(layer[:,0]>=layer[:,1])]=SURFACE
                oracle[(total>0)&(layer[:,0]<layer[:,1])]=MEDIUM
                oracle[outer]=-1
                odir=output/'oracle';odir.mkdir(exist_ok=True)
                ometrics=_render_separation(trainer,oracle,odir,iteration,args.image_views)
                confusion={};weighted={}
                for a,aname in CLASS_NAMES.items():
                    for b,bname in CLASS_NAMES.items():
                        mask=(oracle==a)&(labels==b)
                        confusion[f'{aname}->{bname}']=int(mask.sum())
                        weighted[f'{aname}->{bname}']=float(total[mask].sum())
                qstats,owners=quantile_diagnostics(trainer,output,oracle,args.image_views)
                raw=medium.raw_density.flatten()
                # Inference checkpoints are beta1. Recover the actual warmup raw
                # values analytically so histogram ordering matches current sites.
                payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
                profile=payload.get('warmup',{}).get('profile')
                if profile:
                    from training.warmup_profile import inverse_density
                    raw=inverse_density(sigma,profile['density_beta'],profile['activation_scale'])
                del payload
                distributions={name:{'raw':quantiles(raw[mask]),'sigma':quantiles(sigma[mask]),
                    'local_tv':quantiles(tv[mask]),'contribution':float(contribution[mask].sum()),
                    'count':int(mask.sum())} for name,mask in [('inner',~outer),('outer',outer)]}
                record={'iteration':iteration,'metrics':metrics,'full_evaluation':evaluation,'oracle_metrics':ometrics,**thresholds,
                    'intersections':quantiles(torch.cat(intersections)), 'truncated_ray_fraction':float(np.mean(truncated)),
                    'counts':{name:int((labels==label).sum()) for label,name in CLASS_NAMES.items()},
                    'confusion':confusion,'contribution_weighted_confusion':weighted,
                    'oracle_cloud_contribution_mislabeled_surface':float(layer[(oracle==MEDIUM)&(labels==SURFACE),1].sum()),
                    'total_oracle_cloud_contribution':float(layer[(oracle==MEDIUM),1].sum()),
                    'distributions':distributions,'quantiles':qstats}
                torch.save({'labels':labels.cpu(),'oracle_labels':oracle.cpu(),'outer_mask':outer.cpu(),
                    'sites':medium.sites.detach().cpu(),'sigma':sigma.cpu(),'raw_density':raw.cpu(),
                    'contribution':contribution.cpu(),'local_tv':tv.cpu(),'quantile_owner_count':owners.cpu(),
                    'oracle_layer_contribution':layer.cpu(),'class_names':{**CLASS_NAMES,-1:'outer'},
                    'per_ray_statistics':cell_stats.cpu(),
                    'statistics_columns':['contribution','visibility','path_length','weighted_path_length','max_rgb_error'],
                    'contribution_per_visible_ray':(contribution/cell_stats[:,1].clamp_min(1)).cpu(),
                    'checkpoint_index':trainer.topology.last_permutation.cpu(),
                    'source_checkpoint':str(checkpoint)},output/'cell_features.pt')
                # Labels consumed by future initialization must follow ORIGINAL
                # checkpoint order, not an evaluation-time topology permutation.
                source_labels=torch.empty_like(labels)
                source_outer=torch.empty_like(outer)
                source_labels[trainer.topology.last_permutation]=labels
                source_outer[trainer.topology.last_permutation]=outer
                torch.save({'labels':source_labels.cpu(),'outer_mask':source_outer.cpu(),
                    'class_names':{**CLASS_NAMES,-1:'outer'}, 'classifier':'two_stage_inner_otsu',
                    'source_checkpoint':str(checkpoint),'order':'source_checkpoint'},output/'cell_labels.pt')
                fig,axes=plt.subplots(1,4,figsize=(20,4))
                for mask,name in [(~outer,'inner'),(outer,'outer')]:
                    if not bool(mask.any()):continue
                    for ax,values,title in zip(axes,(sigma,tv,contribution,raw),('log10 sigma','log10 local TV','log10 contribution','raw density')):
                        selected=values[mask] if title=='raw density' else torch.log10(values[mask].clamp_min(1e-12))
                        selected=selected[torch.isfinite(selected)]
                        if selected.numel():ax.hist(selected.cpu().numpy(),bins=100,alpha=.5,label=name)
                        ax.set_title(title);ax.legend()
                fig.tight_layout();fig.savefig(output/'distributions.png');plt.close(fig)
                (output/'metrics.json').write_text(json.dumps(record,indent=2)+'\n')
                records.append(record)
        finally:
            trainer.logger.close();del trainer;gc.collect();torch.cuda.empty_cache()
    (destination/'analysis.json').write_text(json.dumps(records,indent=2)+'\n')
    if (run/'metrics/train.csv').exists():
        _plot_training(run,destination)
        plot_objective(run,destination)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,key,title in zip(axes,('full_psnr','surface_alpha_on_cloud_only','medium_psnr_vs_cloud'),
        ('Full PSNR','Surface leakage into clouds','Medium PSNR')):
        ax.plot([r['iteration'] for r in records],[r['metrics'][key] for r in records],label='checkpoint-only')
        ax.plot([r['iteration'] for r in records],[r['oracle_metrics'][key] for r in records],label='oracle')
        ax.set_title(title);ax.legend();ax.set_xlabel('iteration')
    fig.tight_layout();fig.savefig(destination/'separation_evolution.png');plt.close(fig)


if __name__=='__main__':main()
