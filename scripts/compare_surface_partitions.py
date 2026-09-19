#!/usr/bin/env python3
"""Four checkpoint-only classifier families; CPU plots and optional GPU renders.

Uses features already measured by analyze_warmup_radfoam_like.py. No training.
GT, when present, is consulted only AFTER all predictions have been produced.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.surface_partition import classify, variants, checkpoint_order


def plot_partition(data, result, output):
    sigma = np.log10(np.maximum(data['sigma'], 1e-12))
    tv = np.log10(np.maximum(data['local_tv'], 1e-12))
    active = (~data['outer_mask']) & (data['contribution'] > 0)
    fig, axes = plt.subplots(1,3,figsize=(17,4.5), constrained_layout=True)
    for ax, values, title in zip(axes[:2], [sigma,tv], ['log10 sigma','log10 local TV']):
        edges = np.histogram_bin_edges(values[active], bins=128)
        for mask, label, color in [(active & ~result.surface,'NON-SURFACE','royalblue'),
                                   (result.surface,'SURFACE','tomato')]:
            ax.hist(values[mask], bins=edges, color=color, alpha=.65, label=label)
        ax.set(xlabel=title, ylabel='Number of active inner cells'); ax.legend()
    threshold = result.info['threshold_log10_tv']
    if threshold is not None:
        axes[1].axvline(threshold,color='black',ls='--',label=f'cut={threshold:.3f}')
        axes[1].legend()
    centers = (result.edges[1:]+result.edges[:-1])/2
    axes[2].plot(centers,result.histogram,label='Fit histogram',alpha=.5)
    axes[2].plot(centers,result.smooth,label='Smoothed histogram')
    if threshold is not None: axes[2].axvline(threshold,color='black',ls='--')
    axes[2].set(xlabel='log10 local TV',ylabel=f'Weight (contribution power {result.info["power"]})')
    axes[2].legend()
    fig.suptitle(f'{result.info["name"]}: {result.info["status"]} | {result.info["reason"]}')
    fig.savefig(output/'histograms.png',dpi=140); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,6),constrained_layout=True)
    ids=np.flatnonzero(active)
    ids=np.random.default_rng(0).choice(ids,min(len(ids),20000),replace=False)
    ax.scatter(sigma[ids],tv[ids],s=1,c=np.where(result.surface[ids],'tomato','royalblue'),alpha=.3)
    if result.model is not None:
        xs=np.linspace(sigma[active].min(),sigma[active].max(),200)
        ys=np.linspace(tv[active].min(),tv[active].max(),200)
        xx,yy=np.meshgrid(xs,ys)
        z=np.column_stack((xx.ravel(),yy.ravel()))
        p=result.model.predict_proba((z-result.center)/result.scale)[:,result.component].reshape(xx.shape)
        if p.min()<.5<p.max(): ax.contour(xx,yy,p,levels=[.5],colors='black',linewidths=2)
        ax.set_title('GMM: black contour = P(SURFACE)=0.5; no scalar cutoff')
    elif threshold is not None:
        ax.axhline(threshold,color='black',ls='--'); ax.set_title('Black line = TV cutoff')
    ax.set(xlabel='log10 sigma',ylabel='log10 local TV')
    fig.savefig(output/'decision_boundary.png',dpi=140);plt.close(fig)


def render_partitions(run, checkpoint, feature_data, results, destination, views, batch_size):
    from training import ExperimentConfig
    from training.trainer import Trainer
    from scripts.analyze_voronoi_warmup import _rays,_to_image,_grid,_masked_scene,_occupancy_scene,_label_scene
    config=ExperimentConfig.load(run/'config.yaml')
    config.output_dir=str(destination/'render_context')
    config.wandb.mode='disabled'
    trainer=Trainer(config,resume=checkpoint)
    trainer.renderer.max_batch_size=batch_size
    try:
        trainer.renderer.transmittance_threshold=config.renderer.validation_transmittance_threshold
        permutation=trainer.topology.last_permutation.cpu().numpy()
        saved_sites=checkpoint_order(feature_data['sites'],feature_data['checkpoint_index'])
        actual=trainer.model.medium.sites.detach().cpu().numpy()
        if not np.allclose(actual,saved_sites[permutation],atol=1e-5,rtol=1e-5):
            raise ValueError('Feature/checkpoint geometry mismatch: refusing mislabeled rendering')
        with torch.no_grad():
            for result in results:
                directory=destination/result.info['name']; directory.mkdir(exist_ok=True)
                source=checkpoint_order(result.surface,feature_data['checkpoint_index'])
                mask=torch.as_tensor(source[permutation],device=trainer.device)
                # Every non-surface cell, INCLUDING outer sites, is retained.
                labels=torch.where(mask,1,2)
                for i,camera in enumerate(trainer.dataset.test[:views]):
                    rays=_rays(camera)
                    images={'target':_to_image(camera.ground_truth(tuple(config.background)))}
                    # Sequential rendering, CPU images only: no full evaluator,
                    # no Gaussian-only render and no retained GPU render dict.
                    for name in ['full','surface','non_surface','overlay','surface_occupancy','non_surface_occupancy']:
                        if name=='full': scene=trainer.model.medium.render_scene()
                        elif name=='overlay': scene=_label_scene(trainer.model,labels)
                        elif name.endswith('_occupancy'):
                            scene=_occupancy_scene(trainer.model,labels,1 if name=='surface_occupancy' else 2)
                        else: scene=_masked_scene(trainer.model,labels,1 if name=='surface' else 2)
                        flat=rays.reshape(-1,6)
                        pixels=[]
                        background=(0.,0.,0.) if name.endswith('_occupancy') or name=='overlay' else tuple(config.background)
                        for start in range(0,len(flat),batch_size):
                            rendered=trainer.renderer.render_voronoi(flat[start:start+batch_size],scene,background=background)
                            pixels.append(rendered.image.detach().cpu().reshape(-1,3))
                            del rendered
                        images[name]=_to_image(torch.cat(pixels).reshape(camera.height,camera.width,3))
                        del scene,pixels
                    grid=_grid([
                        [('Target',images['target']),('Full Voronoi',images['full']),('RED surface / BLUE non-surface',images['overlay'])],
                        [('SURFACE',images['surface']),('NON-SURFACE (includes outer)',images['non_surface']),('SURFACE fixed sigma=0.5',images['surface_occupancy'])],
                    ])
                    grid.save(directory/f'view_{i:02d}_partition.png')
                    for name,image in images.items(): image.save(directory/f'view_{i:02d}_{name}.png')
                print('RENDERED',result.info['name'],flush=True)
    finally:
        trainer.logger.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--iteration',type=int,default=5000)
    parser.add_argument('--render',action='store_true')
    parser.add_argument('--views',type=int,default=4)
    parser.add_argument('--batch-size',type=int,default=8192)
    parser.add_argument('--methods',nargs='+',help='Optional exact variant names')
    args=parser.parse_args()
    if args.views<1 or args.batch_size<1: parser.error('views and batch size must be positive')
    torch.set_num_threads(4)
    run=args.run.resolve()
    source=run/f'warmup_analysis/iteration_{args.iteration:06d}/cell_features.pt'
    saved=torch.load(source,map_location='cpu',weights_only=False)
    data={k:v.numpy() for k,v in saved.items() if isinstance(v,torch.Tensor)}
    checkpoint=run/f'checkpoints/iter_{args.iteration:08d}.pt'
    if Path(saved['source_checkpoint']).resolve()!=checkpoint.resolve():
        raise ValueError('Feature source does not match requested checkpoint')
    destination=run/f'surface_partitions/iteration_{args.iteration:06d}'
    destination.mkdir(parents=True,exist_ok=True)
    settings=variants()
    if args.methods:
        unknown=set(args.methods)-{s['name'] for s in settings}
        if unknown: raise ValueError(f'Unknown variants: {unknown}')
        settings=[s for s in settings if s['name'] in args.methods]
    results=[]
    for setting in settings:
        result=classify(data['sigma'],data['local_tv'],data['contribution'],data['outer_mask'],**setting)
        directory=destination/setting['name'];directory.mkdir(exist_ok=True)
        plot_partition(data,result,directory)
        ordered=checkpoint_order(result.surface,data['checkpoint_index'])
        torch.save(dict(surface_mask=torch.from_numpy(ordered),order='source_checkpoint',
                        source_checkpoint=str(checkpoint),classifier=result.info),directory/'surface_mask.pt')
        results.append(result)
    # Evaluation-only oracle attribution. Never passed to classify().
    for result in results:
        info=result.info
        info['surface_count']=int(result.surface.sum())
        info['non_surface_count']=int((~result.surface).sum())
        if 'oracle_layer_contribution' in data:
            layer=data['oracle_layer_contribution']
            info['oracle_evaluation']={
                'lego_contribution_retained':float(layer[result.surface,0].sum()/max(layer[:,0].sum(),1e-30)),
                'cloud_contribution_to_surface':float(layer[result.surface,1].sum()/max(layer[:,1].sum(),1e-30))}
        (destination/info['name']/'result.json').write_text(json.dumps(info,indent=2)+'\n')
        print(json.dumps(info),flush=True)
    (destination/'summary.json').write_text(json.dumps([r.info for r in results],indent=2)+'\n')
    if args.render:
        render_partitions(run,checkpoint,data,results,destination,args.views,args.batch_size)


if __name__=='__main__': main()
