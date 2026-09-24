import numpy as np
from experiments.ray_heatmaps.analyze import concentration,metrics,native

def test_concentration():
    assert np.allclose(concentration(np.ones((2,100)),.1),0)
    x=np.zeros((1,100));x[0,20]=1
    assert np.allclose(concentration(x,.1),.99)
    assert np.allclose(concentration(x.repeat(2,axis=1)/2,.05),.99)
    assert np.allclose(concentration(x*0,.1),0)

def test_homogeneous_and_zero():
    a=np.ones((1,100,3))*.2;b=a*2;c=np.ones_like(a)*.7
    maps,p=metrics(a,b,c,np.zeros((1,100)),.1,.2)
    assert np.allclose(p['backscatter'],.7*(1-np.exp(-4)))
    assert np.allclose(maps['intrinsic_source_concentration'],0,atol=1e-12)
    assert np.allclose(maps['suspect_overlap'],0)
    for v in metrics(a*0,b*0,c,np.zeros((1,100)),.1,.2)[0].values():assert np.all(np.isfinite(v))

def test_gaussian_plane(tmp_path):
    f=native(tmp_path)
    # unit planar Gaussian z=2, opacity .8, kappa=1
    g=np.array([[0,0,2,1,0,0,0,1,0,0,0,1,.8,1.]])
    o=np.zeros((2,3));r=np.array([[0,0,1.],[1,0,0.]])
    tau=np.zeros((2,40));f(2,1,40,4.,o,r,g,tau)
    assert np.allclose(tau[0].sum(),-np.log(.2))
    assert tau[1].sum()==0
    assert np.argmax(tau[0])==20

def test_overlap_localization():
    n=100;a=np.zeros((1,n,3));b=a.copy();c=np.ones_like(a)
    b[:,50,:]=10;g=np.zeros((1,n));g[:,50]=5
    together,_=metrics(a,b,c,g,.1,.3)
    g=np.roll(g,30,axis=1)
    apart,_=metrics(a,b,c,g,.1,.3)
    assert together['suspect_overlap'][0]>.1
    assert apart['suspect_overlap'][0]==0
