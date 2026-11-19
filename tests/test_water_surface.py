import torch
from apps.draining.water_surface import dielectric


def test_normal_incidence_fresnel_and_direction():
    d=torch.tensor([[0.,0.,-1.]])
    n=torch.tensor([[0.,0.,1.]])
    reflected,refracted,f=dielectric(d,n,torch.tensor([False]))
    torch.testing.assert_close(reflected,-d)
    torch.testing.assert_close(refracted,d)
    torch.testing.assert_close(f,torch.tensor([((1-1.333)/(1+1.333))**2]))


def test_total_internal_reflection():
    d=torch.tensor([[.9,0.,(1-.9**2)**.5]])
    n=torch.tensor([[0.,0.,1.]])
    reflected,_,f=dielectric(d,n,torch.tensor([True]))
    assert reflected[0,2]<0
    torch.testing.assert_close(f,torch.ones(1))


def test_snell_law_oblique_air_to_water():
    d=torch.tensor([[.6,0.,-.8]])
    n=torch.tensor([[0.,0.,1.]])
    _,refracted,f=dielectric(d,n,torch.tensor([False]))
    torch.testing.assert_close(refracted[:,0],torch.tensor([.6/1.333]))
    torch.testing.assert_close(refracted.norm(dim=-1),torch.ones(1))
    assert 0<f.item()<1


def test_heightfield_intersection_flat_and_parallel():
    from types import SimpleNamespace
    from apps.draining.water_surface import WaterSurface
    water=WaterSurface.__new__(WaterSurface)
    water.up=torch.tensor([0.,0.,1.])
    water.scale=1.
    water.r=SimpleNamespace(far=10.)
    water.waves=lambda p: (torch.zeros(len(p)),water.up.expand(len(p),3))
    rays=torch.tensor([[0.,0.,2.,0.,0.,-1.],[0.,0.,-2.,0.,0.,1.],[0.,0.,2.,1.,0.,0.]])
    distance,_,_,valid,inside=water.intersection(rays,0.)
    torch.testing.assert_close(distance[:2],torch.tensor([2.,2.]))
    assert valid.tolist()==[True,True,False]
    assert inside.tolist()==[False,True,False]


def test_final_wave_multiplier_doubles_only_the_ending():
    from apps.draining.water_surface import final_wave_multiplier
    assert final_wave_multiplier(0,12,2)==1
    assert final_wave_multiplier(8.4,12,2)==1
    assert final_wave_multiplier(12,12,2)==2
    assert final_wave_multiplier(24,12,2)==2
    values=[final_wave_multiplier(i/24,12,2) for i in range(575)]
    assert all(a<=b for a,b in zip(values,values[1:]))
    assert max(b-a for a,b in zip(values,values[1:]))<.025


def test_height_and_slopes_use_same_final_multiplier():
    from types import SimpleNamespace
    from apps.draining.water_surface import WaterSurface
    water=WaterSurface.__new__(WaterSurface)
    water.u=torch.tensor([1.,0.,0.]);water.v=torch.tensor([0.,1.,0.]);water.up=torch.tensor([0.,0.,1.]);water.time=0.
    water.ocean=SimpleNamespace(update=lambda time:None,sample=lambda x,y:(torch.ones_like(x)*.1,torch.ones_like(x)*.2,torch.ones_like(x)*.3))
    points=torch.zeros(2,3)
    water.wave_multiplier=1.
    h1,n1=water.waves(points)
    water.wave_multiplier=2.
    h2,n2=water.waves(points)
    torch.testing.assert_close(h2,2*h1)
    torch.testing.assert_close(-n2[:,:2]/n2[:,2:],2*(-n1[:,:2]/n1[:,2:]))
