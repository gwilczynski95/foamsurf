import importlib.util
from pathlib import Path
import csv
import torch
import pytest
import yaml
from training.colmap_joint import fit, transfer_medium, install_joint_topology

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('colmap_smoke_helpers',ROOT/'tests/colmap_joint/test_gpu.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)


@pytest.mark.parametrize('zero',[False,True])
def test_rgb_pipeline_and_resume(tmp_path,zero):
    path=helpers.config_file(tmp_path,steps=6)
    s=yaml.safe_load(path.read_text())
    if zero:s['warmup_iterations']=0
    for stage in ['warmup_overrides','joint_overrides']:
        s[stage].setdefault('model',{})['medium_type']='rgb_backscatter'
        s[stage].setdefault('disentanglement',{})['graph_tv_weight']=0
    path.write_text(yaml.safe_dump(s))
    out=tmp_path/'run';fit(path,1,out);fit(path,1,out,resume=True)
    warm=torch.load(out/'warmup/checkpoints/last.pt',map_location='cpu',weights_only=False)
    initial=torch.load(out/'joint/checkpoints/initialization.pt',map_location='cpu',weights_only=False)
    final=torch.load(out/'joint/checkpoints/last.pt',map_location='cpu',weights_only=False)
    assert warm['iteration']==(0 if zero else 6)
    assert initial['iteration']==0 and final['iteration']==6
    assert initial['model']['medium.raw_density'].shape[1]==6
    assert not initial['medium_optimizer']['state']
    medium=transfer_medium(warm,'cpu')
    x=medium.sites[:,0].argsort();y=initial['model']['medium.sites'][:,0].argsort()
    torch.testing.assert_close(medium.raw_density[x],initial['model']['medium.raw_density'][y])
    torch.testing.assert_close(medium.raw_color[x],initial['model']['medium.raw_color'][y])
    for v in final['model'].values():assert torch.isfinite(v).all()
    assert final['voronoi_adaptation']['total_densified']>0
    assert final['model_metadata']['medium']['medium_type']=='rgb_backscatter'
    from training.config import ExperimentConfig
    from training.colmap_joint_trainer import ColmapJointTrainer
    cfg=ExperimentConfig.load(out/'joint.yaml');cfg.output_dir=str(tmp_path/'resume')
    trainer=ColmapJointTrainer(cfg,resume=out/'joint/checkpoints/last.pt');install_joint_topology(trainer)
    assert trainer.model.medium.is_rgb
    assert trainer.model.medium.raw_density.shape[1]==6
    assert torch.isfinite(trainer._train_step(7)[0].total)
    trainer.logger.close()
    rows=list(csv.DictReader(open(out/'joint/metrics/eval.csv')))
    assert 'eval/proxy_responsibility/medium_fraction_mean' in rows[-1]
    assert 'eval/responsibility/medium_fraction_mean' not in rows[-1]
    assert list((out/'joint/evaluation').rglob('*_backscatter.png'))


def test_rgb_production(tmp_path):
    path=helpers.config_file(tmp_path,steps=2,full=True)
    s=yaml.safe_load(path.read_text())
    for stage in ['warmup_overrides','joint_overrides']:
        s[stage].setdefault('model',{})['medium_type']='rgb_backscatter'
        s[stage].setdefault('disentanglement',{})['graph_tv_weight']=0
    s['joint_overrides']['voronoi_adaptation']['max_points']=500000
    path.write_text(yaml.safe_dump(s))
    out=tmp_path/'production';fit(path,1,out)
    for stage in ['warmup','joint']:
        rows=list(csv.DictReader(open(out/stage/'metrics/eval.csv')))
        assert rows
        for r in rows:assert float(r['eval/proxy_responsibility/truncated_ray_fraction'])==0
