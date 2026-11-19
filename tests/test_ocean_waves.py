import torch
from apps.draining.ocean_waves import OceanWaves


def test_ocean_is_periodic_deterministic_and_continuous():
    ocean=OceanWaves(1.,'cpu',resolution=64)
    x=torch.tensor([.2,.7,1.3])
    y=torch.tensor([.5,.1,2.4])
    first=ocean.sample(x,y)[0]
    torch.testing.assert_close(ocean.sample(x+ocean.length,y)[0],first,atol=1e-7,rtol=1e-4)
    assert first.abs().max()<=ocean.amplitude
    other=OceanWaves(1.,'cpu',resolution=64)
    torch.testing.assert_close(other.sample(x,y)[0],first)
    ocean.update(.001)
    assert (ocean.sample(x,y)[0]-first).abs().max()<1e-4
    ocean.update(1.)
    assert not torch.allclose(ocean.sample(x,y)[0],first)
    assert torch.isfinite(ocean.grid).all()


def test_strength_updates_same_time_without_resampling():
    ocean=OceanWaves(1.,'cpu',resolution=64)
    original=ocean.grid.clone()
    ocean.strength=3.
    ocean.update(0.)
    torch.testing.assert_close(ocean.grid,original*3)


def test_long_swell_height_bound_and_late_growth():
    ocean=OceanWaves(1.,'cpu',resolution=64)
    ocean.strength=7.
    ocean.update(2.)
    x=torch.linspace(-4,4,400)
    h,dx,dy=ocean.sample(x,x*.3)
    assert torch.isfinite(h).all() and torch.isfinite(dx).all() and torch.isfinite(dy).all()
    assert h.abs().max()<=ocean.amplitude
    assert h.std()>.02
