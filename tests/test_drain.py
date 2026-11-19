import torch
from examples.draining.drain import water_intervals


def test_water_intervals_enter_exit_parallel_and_outside():
    rays = torch.tensor([[0.,0.,2.,0.,0.,-1.], [0.,0.,-2.,0.,0.,1.],
                         [0.,0.,2.,1.,0.,0.], [0.,0.,-2.,1.,0.,0.],
                         [0.,0.,2.,0.,0.,1.], [0.,0.,-2.,0.,0.,-1.]])
    expected = torch.tensor([[2.,10.],[0.,2.],[0.,0.],[0.,10.],[0.,0.],[0.,10.]])
    torch.testing.assert_close(water_intervals(rays, torch.tensor([0.,0.,1.]), 0., 0., 10.), expected)


def test_final_level_rise_changes_only_last_five_seconds():
    from examples.draining.drain import final_level_rise
    values=[final_level_rise(i,574,24,.4,5) for i in range(574)]
    assert all(v==0 for v in values[:455])
    assert values[455]>0
    assert abs(values[-1]-.4)<1e-12
    assert all(a<=b for a,b in zip(values,values[1:]))
    assert max(b-a for a,b in zip(values,values[1:]))<.006
    assert final_level_rise(573,574,24,0,5)==0


def test_post_rise_tail_keeps_original_timing_and_holds_level():
    from examples.draining.drain import final_level_rise
    original=[final_level_rise(i,574,24,.4,5) for i in range(574)]
    extended=[final_level_rise(i,574,24,.4,5) for i in range(814)]
    assert extended[:574]==original
    assert all(abs(v-.4)<1e-12 for v in extended[574:])
