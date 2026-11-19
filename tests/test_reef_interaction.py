from apps.draining.reef_interaction import late_wave_strength


def test_late_waves_grow_smoothly_without_changing_early_animation():
    assert late_wave_strength(0,12)==1
    assert late_wave_strength(6,12)==1
    assert late_wave_strength(12,12)==3
    values=[late_wave_strength(i/24,12) for i in range(289)]
    assert all(a<=b for a,b in zip(values,values[1:]))
    assert max(b-a for a,b in zip(values,values[1:]))<.03


def test_contact_foam_persists_then_decays_and_emits_finite_spray():
    from types import SimpleNamespace
    import torch
    from apps.draining.reef_interaction import ReefInteraction
    y,x=torch.meshgrid(torch.linspace(-.1,.1,48),torch.linspace(-.1,.1,48),indexing='ij')
    origins=torch.stack((x.flatten(),y.flatten(),1+.7*x.flatten()),-1)
    directions=torch.tensor([0.,0.,-1.]).expand_as(origins)
    water=SimpleNamespace(up=torch.tensor([0.,0.,1.]),u=torch.tensor([1.,0.,0.]),v=torch.tensor([0.,1.,0.]),
                          scale=1.,depth=torch.ones(x.numel()),rays=torch.cat((origins,directions),-1))
    water.waves=lambda p: (torch.zeros(len(p)),torch.tensor([.6,0.,.8]).expand(len(p),3))
    contact=ReefInteraction(water,None)
    contact.update(0.,0.)
    contact.update(0.,.1)
    assert contact.foam.max()>0
    assert len(contact.particles)>0 and torch.isfinite(contact.particles).all()
    previous=contact.foam.max().item()
    contact.update(10.,.2)
    assert 0<contact.foam.max()<previous
    contact.foam.zero_()
    contact.rock_face.zero_()
    contact.update(0.,.3)
    assert contact.foam.max()==0  # No contact foam on a flat seabed.
