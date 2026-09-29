from pathlib import Path
import csv,json,gc
import pytest
import torch
from experiments.rgb_disentangle.train import resolve,fit
from experiments.rgb_disentangle.trainer import ConflictTrainer
from training.colmap_joint import install_joint_topology
from training.config import ExperimentConfig


def manifest(tmp_path,full=False):
    # OGS, standard strength, seed 42: control3 + 6groups*9 + offset3 = index60.
    m=resolve(Path(__file__).with_name('sweep.yaml'),60,tmp_path/'run')
    assert m['variant']['group']=='OGS'
    m['profile']['spatial_count']=25000 if full else 256
    for name in ('warmup','joint'):
        c=m[name];c['dataset']['resolution']=1 if full else 16;c['wandb']['mode']='disabled'
        c['validation'].update(eval_interval=2,checkpoint_interval=2,log_interval=1,max_eval_views=1,image_views=1)
    j=m['joint'];j['optimization']['iterations']=2 if full else 4
    j['gaussian_adaptation'].update(densify_from=1,densify_until=6,densification_interval=2,
        opacity_prune_from=1,opacity_prune_ramp=0,opacity_prune_until=j['optimization']['iterations'])
    j['voronoi_adaptation'].update(densify_from=2,densify_until=6,interval=2,min_points=32)
    # Full-resolution test also exercises the production regularizer ray budget.
    m['rgb_conflict'].update(overlap_start=0,opacity_start=0,overlap_ramp=0,opacity_ramp=0,
        rays=128 if full else 32,samples=1024 if full else 128)
    m['rgb_smoothing']['ramp_steps']=2
    return m


def test_pipeline_resume_and_isolation(tmp_path):
    assert torch.cuda.is_available(),'Required RTX4090 test'
    m=manifest(tmp_path);fit(m);fit(m,resume=True)
    out=Path(m['output_dir']);last=out/'joint/checkpoints/last.pt'
    warm=torch.load(out/'warmup/checkpoints/last.pt',map_location='cpu',weights_only=False)
    assert warm['iteration']==0 and not (out/'warmup/metrics/train.csv').exists()
    rows=list(csv.DictReader((out/'joint/metrics/train.csv').open()))
    for key in ('conflict/overlap_weighted','conflict/opacity_weighted','rgb_smooth/weighted'):
        assert any(float(r[key])>0 for r in rows),key
    assert all(float(r['conflict/rays'])==32 for r in rows)
    final=torch.load(last,map_location='cpu',weights_only=False)
    assert final['voronoi_adaptation']['total_densified']>0
    assert all(torch.isfinite(v).all() for v in final['model'].values())
    assert json.loads(last.with_suffix('.conflict.json').read_text())==m['rgb_conflict']
    config=ExperimentConfig.load(out/'joint.yaml');config.output_dir=str(tmp_path/'resumed')
    trainer=ConflictTrainer(config,resume=last,smoothing=m['rgb_smoothing'],conflict=m['rgb_conflict'])
    install_joint_topology(trainer)
    assert torch.isfinite(trainer._train_step(5)[0].total)
    assert trainer.conflict_metrics['conflict/overlap_weighted']>0
    trainer.logger.close();del trainer;gc.collect();torch.cuda.empty_cache()
    # Missing sidecar must reject resume even when a completed marker exists.
    last.with_suffix('.conflict.json').unlink()
    with pytest.raises(FileNotFoundError):fit(m,resume=True)


def test_full_resolution(tmp_path):
    assert torch.cuda.is_available(),'Required RTX4090 test'
    m=manifest(tmp_path,full=True);fit(m)
    out=Path(m['output_dir']);rows=list(csv.DictReader((out/'joint/metrics/train.csv').open()))
    assert all(float(r['conflict/rays'])==128 for r in rows)
    for key in ('conflict/overlap_weighted','conflict/opacity_weighted','rgb_smooth/weighted'):
        assert any(float(r[key])>0 for r in rows),key
    for row in csv.DictReader((out/'joint/metrics/eval.csv').open()):
        assert float(row['eval/proxy_responsibility/truncated_ray_fraction'])==0
