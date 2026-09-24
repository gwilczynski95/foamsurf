import pytest
import torch
from experiments.rgb_smooth.loss import SmoothConfig, smooth_loss
from experiments.rgb_smooth.train import resolve
from pathlib import Path

CFG=Path(__file__).with_name('sweep.yaml')

def tensors():
    coefficients=torch.tensor([[.1,.2,.3,.4,.5,.6],[.2,.4,.6,.8,1.,1.2]],dtype=torch.float64,requires_grad=True)
    colors=torch.tensor([[.2,.3,.4],[.3,.2,.8]],dtype=torch.float64,requires_grad=True)
    return coefficients,colors,torch.tensor([0]),torch.tensor([1])

def test_formula_and_gradients():
    a,c,i,j=tensors();s=SmoothConfig(attenuation=.01,backscatter=.02,color=.1,ramp_steps=0)
    actual,parts=smooth_loss(a,c,i,j,s,1)
    expected=.01*torch.nn.functional.huber_loss(torch.log(a[:1,:3]+s.epsilon),torch.log(a[1:,:3]+s.epsilon),delta=.1)
    expected+=.02*torch.nn.functional.huber_loss(torch.log(a[:1,3:]+s.epsilon),torch.log(a[1:,3:]+s.epsilon),delta=.1)
    expected+=.1*torch.nn.functional.huber_loss(c[:1],c[1:],delta=.05)
    torch.testing.assert_close(actual,expected)
    assert torch.autograd.gradcheck(lambda x,y:smooth_loss(x,y,i,j,s,1)[0],(a,c))
    grad=torch.autograd.grad(actual,(a,c))
    assert all(torch.isfinite(g).all() for g in grad)

def test_constant_empty_zero_and_ramp():
    a,c,i,j=tensors();s=SmoothConfig(attenuation=1,backscatter=1,color=1,ramp_steps=100)
    full=smooth_loss(a,c,i,j,s,100)[0]
    torch.testing.assert_close(smooth_loss(a,c,i,j,s,25)[0],full*.25)
    assert smooth_loss(a,c,i,j,s,0)[0]==0
    assert smooth_loss(a,c,i[:0],j[:0],s,100)[0]==0
    assert smooth_loss(a,c,i,j,SmoothConfig(),100)[0]==0
    assert smooth_loss(a[:1].expand(2,-1),c[:1].expand(2,-1),i,j,s,100)[0]==0

def test_color_only_and_edge_normalization():
    a,c,i,j=tensors();s=SmoothConfig(color=1,ramp_steps=0)
    loss,_=smooth_loss(a,c,i,j,s,1)
    ga,gc=torch.autograd.grad(loss,(a,c))
    assert (ga==0).all() and gc.abs().sum()>0
    torch.testing.assert_close(loss,smooth_loss(a,c,i.repeat(4),j.repeat(4),s,1)[0])
    torch.testing.assert_close(loss,smooth_loss(a,c,j,i,s,1)[0])

@pytest.mark.parametrize('bad',[dict(color=-1),dict(attenuation=float('nan')),dict(epsilon=0),dict(ramp_steps=-1),dict(color_delta=0)])
def test_invalid(bad):
    with pytest.raises(ValueError):SmoothConfig(**bad)

def test_all_configs(tmp_path):
    manifests=[resolve(CFG,i,tmp_path/str(i)) for i in range(15)]
    for i,m in enumerate(manifests):
        assert m['warmup_steps']==0
        assert m['warmup']['voronoi_warmup']['milestone_iterations']==[0]
        assert m['joint']['seed']==42+i%3
        assert m['joint']['model']['medium_type']=='rgb_backscatter'
        assert m['joint']['wandb']['project']=='foamsurf_iui3_rgb_smooth'
        assert m['joint']['optimization']['iterations']==30000
        assert m['profile']['spatial_count']==25000
        assert m['joint']['disentanglement']['graph_tv_weight']==0
        SmoothConfig(**m['rgb_smoothing'])
    with pytest.raises(ValueError):resolve(CFG,15,tmp_path)
