"""Real IUI3 end-to-end smoke, isolated by the test gate from K16 suites."""
from pathlib import Path
import subprocess
import sys
import yaml
import torch
import pytest
from training.colmap_joint import fit, transfer_medium, install_joint_topology

ROOT=Path(__file__).resolve().parents[2]

@pytest.fixture(autouse=True)
def release_gpu():
    yield
    import gc
    gc.collect()
    torch.cuda.empty_cache()

def config_file(tmp_path,steps=6,full=False):
    s=yaml.safe_load((ROOT/'configs/colmap_joint_stage_a.yaml').read_text())
    s['base']=str(ROOT/'configs/end_to_end_iui3.yaml')
    s['counts']=[25000 if full else 256];s['colmap_fractions']=[0.];s['extents']=[1.]
    s['warmup_iterations']=steps;s['joint_iterations']=steps
    common=dict(dataset=dict(resolution=1 if full else 16),wandb=dict(mode='disabled'),
                validation=dict(eval_interval=steps or 1,checkpoint_interval=2,log_interval=1,max_eval_views=1,image_views=1))
    s['warmup_overrides']=dict(common,optimization=dict(rays_per_batch=1000000 if full else 2048),
                              voronoi_adaptation=dict(densify_from=2,densify_until=10))
    j=s['joint_overrides']
    j.update(common)
    j['gaussian_adaptation'].update(densify_from=1,densify_until=max(steps,2),densification_interval=2,
        opacity_prune_from=1,opacity_prune_ramp=0,opacity_prune_until=max(steps,1),min_opacity=.1)
    j['disentanglement'].update(opacity_binarization_start=0,opacity_binarization_ramp=0,
                              graph_tv_weight=1e-5,graph_tv_start=0,graph_tv_ramp=0)
    j['voronoi_adaptation'].update(densify_from=2,densify_until=max(steps,2),interval=2,
                                  min_points=32,max_points=50000,prune_patience=1)
    s['profile_overrides']=dict(resolutions=[1],resolution_steps=[0],density_warmup=0,
        population_target=40000 if full else 1024,population_horizon=20,population_min_interval=1)
    path=tmp_path/'sweep.yaml';path.write_text(yaml.safe_dump(s))
    return path

@pytest.mark.parametrize('timing',[False,True])
@pytest.mark.parametrize('zero',[False,True])
def test_pipeline(tmp_path,zero,timing):
    assert torch.cuda.is_available()
    path=config_file(tmp_path)
    if zero:
        s=yaml.safe_load(path.read_text());s['warmup_iterations']=0;path.write_text(yaml.safe_dump(s))
    index=1
    if timing:
        s=yaml.safe_load(path.read_text())
        s['timing']=dict(initializations=[dict(count=256,fraction=0.,extent=1.)],
                         warmup_lengths=[0 if zero else 6])
        path.write_text(yaml.safe_dump(s));index=0
    out=tmp_path/'run'
    fit(path,index,out)
    fit(path,index,out,resume=True)
    warm=torch.load(out/'warmup/checkpoints/last.pt',map_location='cpu',weights_only=False)
    initial=torch.load(out/'joint/checkpoints/initialization.pt',map_location='cpu',weights_only=False)
    final=torch.load(out/'joint/checkpoints/last.pt',map_location='cpu',weights_only=False)
    assert warm['iteration']==(0 if zero else 6)
    assert final['iteration']==6 and initial['iteration']==0
    assert len(initial['model']['gaussians.means'])==21907
    assert not initial['medium_optimizer']['state'] and not initial['gaussian_optimizer']['state']
    a=transfer_medium(warm,'cpu')
    x=torch.argsort(a.sites[:,0]);y=torch.argsort(initial['model']['medium.sites'][:,0])
    torch.testing.assert_close(a.sites[x],initial['model']['medium.sites'][y])
    torch.testing.assert_close(a.raw_color[x],initial['model']['medium.raw_color'][y])
    torch.testing.assert_close(a.raw_density[x],initial['model']['medium.raw_density'][y])
    for t in final['model'].values(): assert torch.isfinite(t).all()
    assert final['voronoi_adaptation']['total_densified']>0
    # Actual Trainer resume must retain DC and continue optimizing.
    from training.config import ExperimentConfig
    from training.colmap_joint_trainer import ColmapJointTrainer as Trainer
    c=ExperimentConfig.load(out/'joint.yaml');c.output_dir=str(tmp_path/'resumed')
    t=Trainer(c,resume=out/'joint/checkpoints/last.pt');install_joint_topology(t)
    assert t.model.medium.color_activation=='dc'
    torch.testing.assert_close(t.birth_steps.cpu(),final['colmap_joint']['birth_steps'])
    perm=t.topology.last_permutation.cpu()
    for name,value in final['colmap_joint']['medium_accumulators'].items():
        torch.testing.assert_close(getattr(t.voronoi_adaptation,name).cpu(),value[perm])
    loss,*_=t._train_step(7)
    assert torch.isfinite(loss.total)
    t.logger.close()

def test_exact_render_transfer_and_warmup_resume(tmp_path):
    from training.colmap_joint import resolve
    from training.config import ExperimentConfig
    from training.end_to_end import update_config
    from training.warmup_profile import WarmupProfile
    from training.warmup_trainer import WarmupTrainer
    from scripts.analyze_voronoi_warmup import _rays
    path=config_file(tmp_path)
    manifest=resolve(path,1,tmp_path/'render')
    config=ExperimentConfig()
    update_config(config,manifest['warmup'])
    trainer=WarmupTrainer(config,WarmupProfile.from_dict(manifest['profile']))
    with torch.no_grad():
        trainer.model.medium.raw_color.fill_(4.)
        rays=_rays(trainer.dataset.test[0]).reshape(-1,6)[:128]
        before=trainer.renderer.render_voronoi(rays,trainer.model.medium.render_scene(),background=(1.,1.,1.)).image.clone()
        checkpoint=tmp_path/'exact.pt'
        trainer._checkpoint(checkpoint,1)
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        medium=transfer_medium(payload,trainer.device)
        trainer.topology.synchronize_renderer(trainer.renderer,medium)
        after=trainer.renderer.render_voronoi(rays,medium.render_scene(),background=(1.,1.,1.)).image
        assert medium.colors.min()>1
        torch.testing.assert_close(before,after,rtol=2e-5,atol=2e-5)
    trainer.logger.close()
    config.output_dir=str(tmp_path/'warmup_resumed')
    resumed=WarmupTrainer(config,WarmupProfile.from_dict(manifest['profile']),resume=checkpoint)
    with torch.no_grad():
        restored=resumed.renderer.render_voronoi(rays,resumed.model.medium.render_scene(),background=(1.,1.,1.)).image
        torch.testing.assert_close(restored,before,rtol=2e-5,atol=2e-5)
    resumed.logger.close()
