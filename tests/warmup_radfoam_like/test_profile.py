import json
import math
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
from training.warmup_profile import WarmupProfile, inverse_density, opacity_loss, quantile_loss
from training.warmup_sweep import variants, configuration


@pytest.mark.parametrize("beta", [1.,5.,10.,20.])
def test_density_roundtrip_and_derivative(beta):
    sigma=torch.tensor([1e-12,1e-8,.0006715,.005,.0693147,1.,100.],dtype=torch.float64)
    for scale in (1.,2.5):
        raw=inverse_density(sigma,beta,scale).requires_grad_()
        activated=scale*F.softplus(raw,beta=beta)
        torch.testing.assert_close(activated,sigma,rtol=1e-8,atol=1e-12)
        activated.sum().backward()
        torch.testing.assert_close(raw.grad,scale*torch.sigmoid(beta*raw))


def test_known_density_values():
    assert F.softplus(torch.tensor(0.,dtype=torch.float64),beta=10).item()==pytest.approx(math.log(2)/10)
    assert F.softplus(torch.tensor(-.5,dtype=torch.float64),beta=10).item()==pytest.approx(.0006715348489)


def test_opacity_loss_matches_radfoam_and_detaches_target():
    rendered=torch.tensor([[.2],[.8]],requires_grad=True)
    target=torch.tensor([[0.],[1.]],requires_grad=True)
    loss=opacity_loss(rendered,target)
    assert loss.item()==pytest.approx(.04)
    loss.backward()
    torch.testing.assert_close(rendered.grad,torch.tensor([[.2],[-.2]]))
    assert target.grad is None
    with pytest.raises(ValueError,match="equal shape"):
        opacity_loss(rendered,target.flatten())


def test_loss_reduction_preserves_transparent_ray_downweighting():
    depths=torch.tensor([[1.,3.],[-1.,2.],[-1.,-1.]],requires_grad=True)
    loss,valid,width=quantile_loss(depths)
    assert loss.item()==pytest.approx(2/3)
    loss.backward()
    torch.testing.assert_close(depths.grad,torch.tensor([[-1/3,1/3],[0.,0.],[0.,0.]]))


@pytest.mark.parametrize("step",[0,1,150,500,1999,2000,5000,10000,18000,20000,20001])
def test_cosine_against_independent_formula(step):
    p=WarmupProfile()
    for key,init,final,warm,horizon in (("medium_sites",2e-4,5e-6,0,18000),
        ("medium_density",.1,.01,2000,20000),("medium_color",.005,.0005,0,20000)):
        expected=(0. if step>horizon else init*step/warm if warm and step<warm else
                  final+(init-final)*math.cos(math.pi*(step-warm)/(horizon-warm)/2)**2)
        assert p.rates(step)[key]==pytest.approx(expected,abs=1e-15)


@pytest.mark.parametrize("changes",[
    {"resolutions":[4,1],"resolution_steps":[0]},
    {"resolution_steps":[1,150,500]}, {"resolution_steps":[0,500,150]},
    {"resolution_steps":[0,150,150]}, {"resolutions":[4,0,1]},
    {"density_beta":0}, {"density_warmup":20000}, {"opacity_weight":-1},
    {"opacity_weight":float("nan")}, {"fixed_quantiles":[.1,.9]},
])
def test_invalid_profile(changes):
    values=WarmupProfile().as_dict();values.update(changes)
    with pytest.raises(ValueError):WarmupProfile.from_dict(values)


def test_fifty_unique_variants_and_fixed_population(tmp_path):
    rows=variants()
    assert [r['id'] for r in rows]==list(range(50))
    signatures={json.dumps([r['profile'],r['config_overrides']],sort_keys=True) for r in rows}
    assert len(signatures)==50
    for i in range(50):
        config,profile,row=configuration(i,tmp_path/str(i))
        assert config.representation_mode=='voronoi_only'
        assert not config.gaussian_adaptation.enabled
        assert config.voronoi_adaptation.max_points>=config.model.medium_points+profile.outer_points
        assert profile.opacity_weight==(0. if i < 3 else 1.)
    config,profile,_=configuration(47,tmp_path/'fixed')
    assert config.voronoi_adaptation.freeze_at_capacity
    assert config.voronoi_adaptation.max_points==config.model.medium_points+profile.outer_points


def test_quantile_and_resolution_boundaries():
    p=WarmupProfile()
    assert [p.resolution(i) for i in (0,149,150,499,500)]==[4,4,2,2,1]
    assert p.quantile_strength(0)==0
    assert p.quantile_strength(5000)==pytest.approx(5e-5)
    assert p.quantile_strength(10000)==pytest.approx(1e-4)


def test_joint_modules_do_not_import_warmup():
    root=Path(__file__).resolve().parents[2]
    for path in (root/'training/trainer.py',root/'training/optimizer.py',
                 root/'foamsurf_model/medium.py',root/'renderer/foamsurf/cuda.py'):
        source=path.read_text()
        assert 'import WarmupTrainer' not in source
        assert 'from .warmup_' not in source
    joint_trainer=(root/'training/trainer.py').read_text()
    assert 'batch.alpha' not in joint_trainer
    assert 'opacity_loss' not in joint_trainer
    assert 'return F.softplus(self.raw_density)' in (root/'foamsurf_model/medium.py').read_text()

