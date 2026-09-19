import numpy as np
import pytest
from evaluation.surface_partition import classify, variants, checkpoint_order, otsu_threshold


def fixture():
    rng=np.random.default_rng(17)
    x=np.r_[rng.normal(-1,.12,1500),rng.normal(1.6,.12,1500)]
    sigma=10**np.r_[rng.normal(-.1,.12,1500),rng.normal(2,.12,1500)]
    return sigma,10**x,np.ones(3000),np.zeros(3000,dtype=bool)


@pytest.mark.parametrize('setting',variants(),ids=lambda s:s['name'])
def test_two_modes_and_scale_equivariance(setting):
    data=fixture()
    a=classify(*data,**setting)
    assert a.info['status']=='candidate'
    assert a.surface[:1500].mean()<.02
    assert a.surface[1500:].mean()>.98
    b=classify(data[0]*7,data[1]*7,data[2]*12,data[3],**setting)
    np.testing.assert_array_equal(a.surface,b.surface)


@pytest.mark.parametrize('method',['valley','otsu','gmm'])
def test_degenerate_and_excluded(method):
    for contribution in [np.zeros(20),np.ones(20)]:
        result=classify(np.ones(20),np.ones(20),contribution,np.zeros(20,bool),method=method)
        assert result.info['status']=='uncertain'
        assert not result.surface.any()
    data=list(fixture())
    data[3][-10:]=True;data[2][-20:-10]=0
    result=classify(*data,method=method,power=1)
    assert not result.surface[-20:].any()


def test_weighted_histogram_invariant_to_splitting_mass():
    data=fixture()
    a=classify(*data,method='otsu',power=1)
    b=classify(np.repeat(data[0],2),np.repeat(data[1],2),np.repeat(data[2]/2,2),
               np.repeat(data[3],2),method='otsu',power=1)
    assert a.info['threshold_log10_tv']==pytest.approx(b.info['threshold_log10_tv'])
    np.testing.assert_array_equal(np.repeat(a.surface,2),b.surface)


def test_nonfinite_negative_and_shape_rejected():
    data=list(fixture())
    for value in [np.nan,np.inf,-1]:
        broken=data[0].copy();broken[0]=value
        with pytest.raises(ValueError):classify(broken,*data[1:],method='otsu')
    with pytest.raises(ValueError):classify(data[0][:-1],*data[1:],method='otsu')


def test_checkpoint_order_roundtrip():
    index=np.array([2,0,3,1]);mask=np.array([True,False,False,True])
    np.testing.assert_array_equal(checkpoint_order(mask,index)[index],mask)
    for bad in [[0,0,2,3],[0,1,2,4],[0,1,2],[0.,1.,2.,3.]]:
        with pytest.raises(ValueError):checkpoint_order(mask,bad)


def test_valley_unimodal_abstains():
    x=np.linspace(-2,2,1000)
    weights=np.exp(-x*x*4)
    r=classify(10**x,10**x,weights,np.zeros(1000,bool),method='valley',power=1)
    assert r.info['status']=='uncertain'


def test_otsu_known_plateau_and_empty():
    assert otsu_threshold([0,0],[0,1,2]) is None
    assert otsu_threshold([1,0,0,1],[0,1,2,3,4])==pytest.approx(2)


def test_variant_names_unique():
    names=[s['name'] for s in variants()]
    assert len(names)==len(set(names))==3
