from pathlib import Path
import csv,json
import pytest
import torch
from experiments.rgb_smooth.train import resolve,fit
from experiments.rgb_smooth.trainer import SmoothJointTrainer
from training.colmap_joint import install_joint_topology
from training.config import ExperimentConfig


def manifest(tmp_path,variant=12,full=False):
    m=resolve(Path(__file__).with_name('sweep.yaml'),variant,tmp_path/'run')
    m['profile']['spatial_count']=25000 if full else 256
    for name in ('warmup','joint'):
        c=m[name];c['dataset']['resolution']=1 if full else 16;c['wandb']['mode']='disabled'
        c['validation'].update(eval_interval=2,checkpoint_interval=2,log_interval=1,max_eval_views=1,image_views=1)
    j=m['joint'];j['optimization']['iterations']=2 if full else 4
    j['gaussian_adaptation'].update(densify_from=1,densify_until=6,densification_interval=2,
                                    opacity_prune_from=1,opacity_prune_ramp=0,opacity_prune_until=j['optimization']['iterations'])
    j['voronoi_adaptation'].update(densify_from=2,densify_until=6,interval=2,min_points=32)
    m['rgb_smoothing']['ramp_steps']=2
    return m

def test_pipeline_resume_and_integration(tmp_path):
    m=manifest(tmp_path);fit(m);fit(m,resume=True)
    out=Path(m['output_dir'])
    warm=torch.load(out/'warmup/checkpoints/last.pt',map_location='cpu',weights_only=False)
    assert warm['iteration']==0
    assert not (out/'warmup/metrics/train.csv').exists()
    rows=list(csv.DictReader((out/'joint/metrics/train.csv').open()))
    assert any(float(r['rgb_smooth/weighted'])>0 for r in rows)
    assert int(float(rows[-1]['rgb_smooth/edge_count']))>0
    final=out/'joint/checkpoints/last.pt'
    saved=torch.load(final,map_location='cpu',weights_only=False)
    assert saved['voronoi_adaptation']['total_densified']>0
    assert json.loads(final.with_suffix('.smooth.json').read_text())==m['rgb_smoothing']
    for value in saved['model'].values():assert torch.isfinite(value).all()
    config=ExperimentConfig.load(out/'joint.yaml');config.output_dir=str(tmp_path/'resumed')
    trainer=SmoothJointTrainer(config,resume=final,smoothing=m['rgb_smoothing']);install_joint_topology(trainer)
    assert torch.isfinite(trainer._train_step(5)[0].total)
    assert trainer.smooth_metrics['rgb_smooth/weighted']>0
    trainer.logger.close()

def test_production(tmp_path):
    m=manifest(tmp_path,full=True);fit(m)
    out=Path(m['output_dir'])
    rows=list(csv.DictReader((out/'joint/metrics/train.csv').open()))
    assert float(rows[-1]['rgb_smooth/weighted'])>0
    for row in csv.DictReader((out/'joint/metrics/eval.csv').open()):
        assert float(row['eval/proxy_responsibility/truncated_ray_fraction'])==0
