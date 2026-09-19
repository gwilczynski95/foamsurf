from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
import torch

from training.end_to_end import EndToEndConfig, EndToEndTrainer, atomic_json
from training.native_handoff import handoff_plan, local_tv, initialize_native_hybrid

ROOT=Path(__file__).resolve().parents[2]


def test_baseline_contract():
    c=EndToEndConfig.load(ROOT/'configs/end_to_end.yaml')
    assert c.warmup.optimization.iterations==5000
    assert c.joint.optimization.iterations==30000
    assert c.profile.quantile_weight==0
    assert c.profile.population_policy=='radfoam'
    assert c.profile.ray_sampling=='mixed'
    assert c.profile.outer_positions_trainable
    assert c.warmup.model.medium_points+c.profile.outer_points==131072
    assert c.partition['method']=='weighted_valley_w1_s2'
    assert c.joint.gaussian_adaptation.densify_until==15000
    assert not c.joint.voronoi_adaptation.enabled
    assert not c.joint.radfoam_prior.enabled


def test_invalid_cross_stage_settings():
    for mutation in [lambda c:setattr(c.joint.dataset,'source_path','different'),
                     lambda c:setattr(c.joint.voronoi_adaptation,'enabled',True),
                     lambda c:c.partition.update(method='gmm_auto'),
                     lambda c:c.handoff.update(reset_density=0),
                     lambda c:setattr(c.joint,'seed',42)]:
        c=EndToEndConfig.load(ROOT/'configs/end_to_end.yaml')
        mutation(c)
        with pytest.raises(ValueError): c.validate()


def test_tv_excludes_outer_edges():
    sigma=torch.tensor([1.,4.,9.])
    # triangle adjacency; outer third vertex must have no influence
    tv=local_tv(sigma,torch.tensor([1,2,0,2,0,1]),torch.tensor([0,2,4,6]),torch.tensor([False,False,True]))
    torch.testing.assert_close(tv,torch.tensor([3.,3.,0.]))


def fixture_plan():
    # Chain 0--1--2--3--4--5. Remove 0,1; only retained cell 2 changes.
    adj=torch.tensor([1,0,2,1,3,2,4,3,5,4])
    offsets=torch.tensor([0,1,3,5,7,9,10])
    permutation=np.array([3,0,5,1,4,2])
    surface=np.array([True,True,False,False,False,False])
    return handoff_plan(surface,adj,offsets,permutation,100,7),permutation


def test_boundary_and_sampling_permutation():
    plan,index=fixture_plan()
    assert plan['surface_mask'].nonzero().flatten().tolist()==[0,3]
    assert plan['affected_mask'].nonzero().flatten().tolist()==[5]
    assert set(plan['gaussian_source_indices'].tolist())=={0,3}
    assert len(plan['gaussian_source_indices'])==2  # cap to available surface
    again,_=fixture_plan()
    torch.testing.assert_close(plan['gaussian_source_indices'],again['gaussian_source_indices'])
    with pytest.raises(ValueError):
        handoff_plan(np.zeros(6,bool),torch.tensor([],dtype=torch.long),torch.zeros(7,dtype=torch.long),np.arange(6),10,0)


def test_native_density_transfer_and_trainability(tmp_path):
    c=EndToEndConfig.load(ROOT/'configs/end_to_end.yaml').joint
    c.device='cpu';c.model.gaussian_initial_scale_neighbors=1
    plan,_=fixture_plan()
    sites=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.],[0.,0.,1.],[1.,1.,1.]])
    raw=torch.tensor([[-20.],[-5.],[-1.],[0.],[5.],[10.]])
    color=torch.arange(18,dtype=torch.float32).reshape(6,3)/10
    sites=torch.cat((sites,torch.rand((30,3),generator=torch.Generator().manual_seed(5))+2))
    raw=torch.cat((raw,torch.full((30,1),-2.)))
    color=torch.cat((color,torch.zeros(30,3)))
    plan['surface_mask']=torch.cat((plan['surface_mask'],torch.zeros(30,dtype=torch.bool)))
    plan['affected_mask']=torch.cat((plan['affected_mask'],torch.zeros(30,dtype=torch.bool)))
    source=tmp_path/'warmup/checkpoints/last.pt';source.parent.mkdir(parents=True)
    state={'medium.sites':sites,'medium.raw_density':raw,'medium.raw_color':color,
           'medium.bbox_min':torch.tensor([-2.]*3),'medium.bbox_max':torch.tensor([2.]*3)}
    torch.save(dict(format_version=1,baseline='B0',config={'representation_mode':'voronoi_only'},model=state),source)
    plan.update(source_checkpoint=str(source),source_sites=sites,classifier={'status':'candidate'})
    (tmp_path/'partition').mkdir();torch.save(plan,tmp_path/'partition/handoff.pt')
    model,report=initialize_native_hybrid(tmp_path,c,dict(gaussian_count=100,pca_neighbors=3,reset_density=1e-8))
    keep=~plan['surface_mask'];affected=plan['affected_mask'][keep]
    assert model.medium.point_count==34 and model.gaussians.point_count==2
    assert not model.medium.guard_mask.any()
    for p in model.medium.parameters(): assert p.requires_grad
    torch.testing.assert_close(model.medium.raw_density[~affected],raw[keep][~affected],rtol=0,atol=0)
    torch.testing.assert_close(model.medium.densities[affected],torch.full((1,1),1e-8))
    torch.testing.assert_close(model.medium.raw_color,color[keep],rtol=0,atol=0)
    torch.testing.assert_close(model.gaussians.means,sites[plan['gaussian_source_indices']])
    assert report['retained_voronoi_count']==34 and report['reset_neighbor_count']==1
    assert c.model.medium_points==34 and c.voronoi_adaptation.min_points==34


def test_orchestration_resume_and_failure(tmp_path,monkeypatch):
    c=EndToEndConfig.load(ROOT/'configs/end_to_end.yaml',tmp_path/'run')
    calls=[]
    def worker(command,**kwargs):
        stage=command[command.index('--stage')+1];calls.append(stage)
        root=Path(c.output_dir)
        if stage=='partition' and calls.count('partition')==1:
            raise RuntimeError('simulated interrupted stage')
        artifact=root/f'{stage}.artifact';artifact.touch()
        atomic_json(root/f'{stage}_done.json',dict(artifact=str(artifact)))
        if stage=='joint':
            (root/'joint').mkdir();atomic_json(root/'joint/summary.json',dict(status='completed'))
    monkeypatch.setattr('training.end_to_end.subprocess.run',worker)
    with pytest.raises(RuntimeError):EndToEndTrainer(c).fit()
    assert calls==['warmup','partition']
    EndToEndTrainer(c).fit(resume=True)
    assert calls==['warmup','partition','partition','joint']
    EndToEndTrainer(c).fit(resume=True)
    assert len(calls)==4
    with pytest.raises(FileExistsError):EndToEndTrainer(c).fit()
    changed=deepcopy(c);changed.partition['method']='otsu_w1'
    with pytest.raises(ValueError):EndToEndTrainer(changed).fit(resume=True)
