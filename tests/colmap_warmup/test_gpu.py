"""Real IUI3 population: warmup, growth, resume, then native handoff/joint."""
from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys
import torch
import yaml
from training.end_to_end import EndToEndConfig

ROOT=Path(__file__).resolve().parents[2]


def smoke_config(tmp_path):
    c=EndToEndConfig.load(ROOT/'configs/end_to_end_iui3.yaml')
    for config in [c.warmup,c.joint]:
        config.dataset.resolution=16
        config.optimization.iterations=6
        config.optimization.rays_per_batch=2048
        config.validation.eval_interval=6
        config.validation.log_interval=1
        config.validation.checkpoint_interval=2
        config.validation.max_eval_views=1
        config.validation.image_views=1
        config.wandb.mode='disabled'
        config.gaussian_adaptation.enabled=False
        config.gaussian_adaptation.densify_from=1
        config.gaussian_adaptation.densify_until=6
        config.gaussian_adaptation.opacity_prune_from=1
        config.gaussian_adaptation.opacity_prune_until=6
    c.warmup.voronoi_warmup.milestone_iterations=[0,2,6]
    c.warmup.voronoi_adaptation.densify_from=4
    c.warmup.voronoi_adaptation.densify_until=6
    c.profile.population_target=35000
    c.profile.population_horizon=20
    c.profile.population_min_interval=100
    c.profile.error_map_chunk=4096
    c.profile.resolutions=[2,1]
    c.profile.resolution_steps=[0,2]
    c.profile.density_warmup=0
    c.warmup.save(tmp_path/'warmup.yaml'); c.joint.save(tmp_path/'joint.yaml')
    wrapper=dict(output_dir=str(tmp_path/'run'),warmup=dict(config='warmup.yaml',profile=c.profile.as_dict()),
                 joint=dict(config='joint.yaml'),partition=dict(method='otsu_w1',batch_size=2048,image_views=1),
                 handoff=dict(gaussian_count=64,pca_neighbors=3,reset_density=1e-8))
    (tmp_path/'pipeline.yaml').write_text(yaml.safe_dump(wrapper))
    return c,tmp_path/'pipeline.yaml'


def test_real_colmap_pipeline(tmp_path):
    assert torch.cuda.is_available()
    _,path=smoke_config(tmp_path)
    command=[sys.executable,str(ROOT/'scripts/train_end_to_end.py'),'--config',str(path)]
    subprocess.run(command,cwd=ROOT,check=True)
    root=tmp_path/'run'
    warm=torch.load(root/'warmup/checkpoints/last.pt',map_location='cpu',weights_only=False)
    final=torch.load(root/'joint/checkpoints/last.pt',map_location='cpu',weights_only=False)
    assert warm['iteration']==final['iteration']==6
    assert warm['voronoi_adaptation']['total_densified']>0
    assert warm['config']['model']['medium_points']==19716
    assert 'raw_color' in warm['warmup']
    for tensor in final['model'].values(): assert torch.isfinite(tensor).all()
    report=json.loads((root/'handoff_report.json').read_text())
    assert report['removed_surface_count']+report['retained_voronoi_count']==report['source_count']
    assert (root/'transition/view_00_initialization.png').is_file()
    subprocess.run(command+['--resume'],cwd=ROOT,check=True)


def test_pcd_checkpoint_restores_physical_field(tmp_path):
    from training.warmup_trainer import WarmupTrainer
    c,_=smoke_config(tmp_path)
    c.warmup.output_dir=str(tmp_path/'first')
    trainer=WarmupTrainer(c.warmup,c.profile)
    assert trainer.model.medium.point_count==24716
    assert int(trainer.model.medium.outer_mask.sum())==5000
    assert trainer.model.medium.raw_color.abs().max()==0
    trainer._train_step(1)
    trainer._train_step(2)
    checkpoint=tmp_path/'first/checkpoints/resume.pt'
    trainer._checkpoint(checkpoint,2)
    def field(t):
        m=t.model.medium;order=torch.argsort(m.sites[:,0])
        return torch.cat([m.sites,m.densities,m.colors],-1)[order].detach().cpu()
    expected=field(trainer)
    resumed_config=deepcopy(c.warmup);resumed_config.output_dir=str(tmp_path/'resumed')
    resumed=WarmupTrainer(resumed_config,c.profile,resume=checkpoint)
    torch.testing.assert_close(field(resumed),expected)
    for i in [3,4,5]:
        loss,*_=resumed._train_step(i)
        assert torch.isfinite(loss.total)


def test_full_resolution_million_ray_growth(tmp_path):
    subprocess.run([sys.executable,str(ROOT/'tests/colmap_warmup/production_worker.py'),
                    str(tmp_path/'production')],cwd=ROOT,check=True)
