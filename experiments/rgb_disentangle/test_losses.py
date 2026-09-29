from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
import torch
from experiments.rgb_disentangle.losses import ConflictConfig,opacity_loss,overlap_profile,concentration,ramp
from experiments.rgb_disentangle.renderer import gaussian_tau,nearest_ids,compute_regularizers
from experiments.rgb_disentangle.train import resolve
from experiments.ray_heatmaps.analyze import metrics


def test_opacity_threshold_empty_and_gradient():
    a=torch.tensor([0.,.1,.2,.8],dtype=torch.double,requires_grad=True)
    loss,mask=opacity_loss(a)
    assert loss.item()==pytest.approx(.5)
    assert mask.item()==.5
    loss.backward();torch.testing.assert_close(a.grad,torch.tensor([0.,0.,-.5,-.5],dtype=a.dtype))
    a=torch.zeros(5,requires_grad=True);loss,_=opacity_loss(a);loss.backward()
    assert loss==0 and torch.equal(a.grad,torch.zeros_like(a))


def test_numpy_heatmap_equivalence_and_isolation():
    rng=np.random.default_rng(3)
    b=rng.uniform(.01,.5,(3,40,3));c=rng.uniform(.1,.9,(3,40,3))
    g=np.zeros((3,40));g[:,20]=3
    expected,_=metrics(b*.3,b,c,g,.1,.3)
    bt=torch.tensor(b,requires_grad=True);ct=torch.tensor(c,requires_grad=True);gt=torch.tensor(g,requires_grad=True)
    loss,p=overlap_profile(bt,ct,gt,.1,band_fraction=.3/4)
    np.testing.assert_allclose(p['score'].detach(),expected['suspect_overlap'],rtol=1e-10,atol=1e-12)
    loss.backward()
    assert bt.grad.abs().sum()>0 and ct.grad.abs().sum()>0
    assert gt.grad is None


def test_gradcheck_medium():
    torch.manual_seed(2)
    b=(torch.rand(2,12,3,dtype=torch.double)*.3+.1).requires_grad_()
    c=(torch.rand(2,12,3,dtype=torch.double)*.5+.2).requires_grad_()
    g=torch.zeros(2,12,dtype=torch.double);g[:,5]=2
    assert torch.autograd.gradcheck(lambda x,y:overlap_profile(x,y,g,.1,.15)[0],(b,c),eps=1e-6,atol=1e-5)


@pytest.mark.parametrize('case',['empty','dense','no_gaussians','zero_colors'])
def test_extreme_profiles_finite(case):
    b=torch.full((2,32,3),0. if case=='empty' else (1e5 if case=='dense' else .2),requires_grad=True)
    c=torch.full_like(b,0. if case=='zero_colors' else .5,requires_grad=True)
    g=torch.zeros(2,32);g[:,0]=0 if case=='no_gaussians' else 15
    loss,p=overlap_profile(b,c,g,.1)
    assert torch.isfinite(loss) and all(torch.isfinite(x).all() for x in p.values())
    loss.backward();assert torch.isfinite(b.grad).all() and torch.isfinite(c.grad).all()
    if case in ('empty','no_gaussians','zero_colors'):assert loss==0


def test_subdivision_and_ray_duplication():
    x=torch.tensor([[.1,.4,.2,.3]])
    torch.testing.assert_close(concentration(x),concentration(x.repeat_interleave(2,dim=1)/2))
    b=torch.full((1,10,3),.2);c=torch.ones_like(b);g=torch.zeros(1,10);g[:,3]=2
    one=overlap_profile(b,c,g,.1,.2)[0]
    many=overlap_profile(b.repeat(5,1,1),c.repeat(5,1,1),g.repeat(5,1),.1,.2)[0]
    torch.testing.assert_close(one,many)


def plane_scene(dtype=torch.double,device='cpu'):
    def t(v):return torch.tensor(v,dtype=dtype,device=device)
    return SimpleNamespace(means=t([[0,0,2]]),quaternions=t([[1,0,0,0]]),
        log_scales=t([[0,0]]),raw_kappa=t([[-30]]),opacity_logits=t([[np.log(4)]]))


def test_native_hit_profile_formula_and_missing():
    scene=plane_scene();rays=torch.tensor([[0,0,0,0,0,1],[0,0,0,0,0,1]],dtype=torch.double)
    depth=torch.tensor([[2.,float('nan')]],dtype=torch.double);ids=torch.tensor([[0,-1]])
    tau=gaussian_tau(scene,rays,depth,ids,40,4.)
    assert tau[0].sum().item()==pytest.approx(-np.log(.2))
    assert tau[1].sum()==0 and tau[0].argmax()==20
    assert not tau.requires_grad


def test_nearest_refresh_after_moving_sites():
    sites=torch.tensor([[0.,0,1],[0,0,3]])
    rays=torch.tensor([[0.,0,0,0,0,1]])
    assert nearest_ids(sites,rays,4,4.,1).tolist()==[[0,0,1,1]]
    sites[:]=sites.flip(0)
    assert nearest_ids(sites,rays,4,4.,1).tolist()==[[1,1,0,0]]


@pytest.mark.parametrize('kwargs',[{'overlap_weight':-1},{'opacity_weight':float('nan')},
    {'samples':1},{'rays':0},{'overlap_start':-1},{'opacity_ramp':1.5},
    {'band_fraction':0},{'opacity_threshold':1},{'range_multiplier':float('inf')}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):ConflictConfig(**kwargs)


def test_ramps_and_off_path():
    assert [ramp(t,10,5) for t in (9,10,12,15,20)]==[0,0,.4,1,1]
    assert ramp(10,10,0)==1
    def forbidden(*a,**kw):raise AssertionError('Off path rendered')
    med=SimpleNamespace(raw_density=torch.zeros(1))
    loss,logs=compute_regularizers(forbidden,None,None,med,ConflictConfig(),1,1,10,42)
    assert loss==0 and logs['rays']==0


def test_all_manifests():
    path=Path(__file__).with_name('sweep.yaml')
    groups=set();names=set();controls=0
    for i in range(66):
        m=resolve(path,i,Path('/tmp')/f'conflict_manifest_{i}')
        assert m['warmup_steps']==0
        assert m['joint']['model']['medium_type']=='rgb_backscatter'
        assert m['joint']['wandb']['project']=='foamsurf_iui3_rgb_conflict'
        g=m['variant']['group'];groups.add(g);names.add(m['variant']['name'])
        assert (m['rgb_conflict']['overlap_weight']>0)==('O' in g)
        assert (m['rgb_conflict']['opacity_weight']>0)==('G' in g)
        assert (m['rgb_smoothing']['color']>0)==('S' in g)
        controls+=g=='control'
    assert groups=={'control','O','G','S','OG','OS','GS','OGS'} and len(names)==66 and controls==3
    with pytest.raises(ValueError):resolve(path,66,'/tmp/not_written')
