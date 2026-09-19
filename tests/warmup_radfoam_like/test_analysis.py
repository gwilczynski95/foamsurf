import torch
import pytest
from scripts.analyze_warmup_radfoam_like import classify


def test_outer_exclusion_and_no_gt_input():
    # Complete four-site graph. An extreme scaffold density cannot change
    # inner-inner TV, and no GT labels enter this function.
    adjacency=torch.tensor([1,2,3,0,2,3,0,1,3,0,1,2])
    offsets=torch.tensor([0,3,6,9,12])
    outer=torch.tensor([False,False,False,True])
    sigma=torch.tensor([.01,.1,10.,100000.])
    contrib=torch.tensor([1e-9,1.,10.,100.])
    labels,tv,thresholds=classify(sigma,contrib,adjacency,offsets,outer)
    assert labels[3]==-1
    assert tv[0].item()==pytest.approx((.09+9.99)/2)
    sigma[3]=0
    other,other_tv,_=classify(sigma,contrib,adjacency,offsets,outer)
    torch.testing.assert_close(tv,other_tv)
    torch.testing.assert_close(labels,other)
