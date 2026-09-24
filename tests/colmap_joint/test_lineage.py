import torch
from foamsurf_model.gaussian_adaptation import GaussianTopologyPlan
from training.colmap_joint_trainer import map_birth_steps

def test_birth_mapping_clone_split_prune():
    plan=GaussianTopologyPlan(torch.tensor([0,2]),torch.tensor([2]),torch.tensor([1]),4)
    birth=map_birth_steps(torch.tensor([0,0,3,0]),plan,9)
    assert birth.tolist()==[0,3,9,9,9]
