from pathlib import Path
import numpy as np
import pytest
import torch
from training.colmap_initialization import sample_spatial
from training.colmap_joint import resolve, transfer_medium
from training.warmup_profile import WarmupProfile
from foamsurf_model.medium import VoronoiMediumModel

ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('fraction',[0.,.1,.25])
@pytest.mark.parametrize('extent',[1.,2.])
def test_spatial(fraction,extent):
    points=np.random.default_rng(3).normal(size=(100,3)).astype('float32')
    cameras=np.array([[0,0,-4],[1,0,-3]],dtype='float32')
    np.random.seed(8);a,report=sample_spatial(points,cameras,1000,fraction,extent)
    np.random.seed(8);b,_=sample_spatial(points,cameras,1000,fraction,extent)
    np.testing.assert_array_equal(a,b)
    assert a.shape==(1000,3) and np.isfinite(a).all()
    assert report['colmap_count']==int(1000*fraction)
    assert (a[:,2]<cameras[:,2].min()).any()
    assert (a[:,2]>points[:,2].max()).any()

def test_all_configs(tmp_path):
    configs=[resolve(ROOT/'configs/colmap_joint_stage_a.yaml',i,tmp_path/str(i)) for i in range(19)]
    assert len({c['variant']['name'] for c in configs})==19
    for c in configs:
        j=c['joint'];p=c['profile']
        assert j['model']['gaussian_initial_points']==21907
        assert j['voronoi_adaptation']['enabled']
        assert j['gaussian_adaptation']['opacity_prune_until']==30000
        assert j['disentanglement']['opacity_binarization_weight']==.001
        WarmupProfile.from_dict(p)
    with pytest.raises(ValueError): resolve(ROOT/'configs/colmap_joint_stage_a.yaml',19,tmp_path)

def test_transfer_dc_and_legacy():
    n=40
    state={'medium.sites':torch.randn(n,3),'medium.raw_density':torch.randn(n,1),
           'medium.raw_color':torch.zeros(n,3),'medium.bbox_min':torch.full((3,),-10.),
           'medium.bbox_max':torch.full((3,),10.)}
    dc=torch.full((n,3),4.) # physical RGB >1 must NOT be clipped
    p=dict(model=state,model_metadata={'medium':dict(point_count=n,guard_count=5)},
           warmup=dict(profile=dict(site_initialization='spatial'),raw_color=dc))
    m=transfer_medium(p,'cpu')
    assert m.colors.min()>1 and not m.guard_mask.any()
    torch.testing.assert_close(m.densities,torch.nn.functional.softplus(state['medium.raw_density']))
    copy=VoronoiMediumModel.from_checkpoint(m.named_primitive_parameters(),m.checkpoint_metadata(),m.bbox_min,m.bbox_max,'cpu')
    torch.testing.assert_close(copy.colors,m.colors)
    old=VoronoiMediumModel.from_checkpoint(m.named_primitive_parameters(),dict(point_count=n),m.bbox_min,m.bbox_max,'cpu')
    torch.testing.assert_close(old.colors,torch.sigmoid(dc))

@pytest.mark.parametrize('kwargs',[dict(spatial_count=0),dict(spatial_colmap_fraction=-1),dict(spatial_extent=float('nan'))])
def test_invalid(kwargs):
    with pytest.raises(ValueError):
        WarmupProfile.from_dict(dict(site_initialization='spatial',outer_points=0,**kwargs))

def test_timing_configs(tmp_path):
    lengths=[0,100,250,500,1000,2000,4000]
    for group,reference in enumerate([2,6]):
        old=resolve(ROOT/'configs/colmap_joint_stage_a.yaml',reference,tmp_path/'same')
        for offset,steps in enumerate(lengths):
            c=resolve(ROOT/'configs/colmap_joint_stage_b.yaml',group*7+offset,tmp_path/'same')
            assert c['warmup_steps']==steps
            assert c['warmup']['optimization']['iterations']==max(steps,1)
            assert c['warmup']['voronoi_warmup']['milestone_iterations']==sorted({0,steps})
            assert c['profile']==old['profile']
            assert c['joint']['optimization']['iterations']==30000
            # Only display names/tags differ; joint numerical settings stay identical.
            for key in c['joint']:
                if key not in ('experiment_name','wandb'):
                    assert c['joint'][key]==old['joint'][key],key
            assert c['joint']['wandb']['project']=='foamsurf_iui3'
            if steps==1000:
                for key in c['warmup']:
                    if key not in ('experiment_name','wandb'):
                        assert c['warmup'][key]==old['warmup'][key],key
    with pytest.raises(ValueError):
        resolve(ROOT/'configs/colmap_joint_stage_b.yaml',14,tmp_path)

@pytest.mark.parametrize('lengths', [[-1],[True],[1.5],[0,0],[]])
def test_invalid_timing_lengths(lengths):
    from training.colmap_joint import variants
    with pytest.raises(ValueError):
        variants(dict(timing=dict(initializations=[dict(count=25000,fraction=0.,extent=2.)],
                                 warmup_lengths=lengths)))
