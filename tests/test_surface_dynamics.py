import math
import torch
from apps.draining.surface_dynamics import ShallowWater


def make_solver(wall=False):
    terrain=torch.full((16,96),-1.)
    if wall:
        terrain[:,62:]=1.
    solver=ShallowWater(terrain,torch.ones_like(terrain,dtype=torch.bool),torch.tensor([0.,0.]),torch.tensor([1.,.16]),1.)
    solver.boundaries(0.)
    return solver


def test_pulse_reflects_from_rock_without_penetration():
    results=[]
    for wall in [False,True]:
        s=make_solver(wall)
        x=torch.linspace(0,1,96)
        s.h[:]=.01*torch.exp(-((x-.25)/.035).square())
        c=math.sqrt(s.gravity*.3)
        s.u[:,1:-1]=c/.3*(s.h[:,:-1]+s.h[:,1:])/2
        dt=.002
        for _ in range(400):
            s.step(dt)
        assert torch.isfinite(s.h).all()
        weights=s.h.square().mean(0)
        results.append(float((weights*x).sum()/weights.sum()))
        if wall:
            assert s.h[:,61:].abs().max()==0
            assert s.u[:,61:].abs().max()==0
            assert s.u[:,:60].sum()<0  # The wave now travels back from the rock.
    assert results[1]<results[0]-.2


def test_still_water_is_still_and_closed_domain_conserves_volume():
    s=make_solver(True)
    for _ in range(10):
        s.step(.002)
    assert s.h.abs().max()==0
    s.h[:,20:25]=.01
    mass=s.h.sum().item()
    for _ in range(100):
        s.step(.002)
    assert abs(s.h.sum().item()-mass)<1e-5


def test_large_boundary_forcing_is_limited_to_linear_regime():
    class Incoming:
        def update(self,time):
            self.time=time
        def sample(self,x,y):
            h=2*torch.sin(20*x-10*self.time)
            return h,torch.zeros_like(h),torch.zeros_like(h)
    s=make_solver(True)
    incoming=Incoming()
    # Start at rest, then introduce an excessively large incoming wave.
    incoming.sample=lambda x,y:(torch.zeros_like(x),torch.zeros_like(x),torch.zeros_like(x))
    s.update(0.,0.,incoming)
    del incoming.sample
    for i in range(1,101):
        s.update(0.,i*.01,incoming)
    assert torch.isfinite(s.h).all()
    assert s.h.abs().max()<.2
