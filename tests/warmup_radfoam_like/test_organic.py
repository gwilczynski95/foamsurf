from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from foamsurf_data.rays import generate_camera_rays
from training.warmup_sampling import MixedRaySampler
from training.warmup_population_math import (
    population_masks, sample_parents, offspring_sites, growth_interval)
from training.warmup_profile import WarmupProfile
from training.warmup_organic_sweep import variants, configuration


class Dataset:
    device = torch.device("cpu")
    def __init__(self):
        self.train = []
        for i in range(3):
            self.train.append(SimpleNamespace(
                width=3, height=2, origin=torch.tensor([float(i), 0., 0.]),
                right=torch.tensor([1.,0.,0.]), down=torch.tensor([0.,1.,0.]),
                forward=torch.tensor([0.,0.,1.]), fov_x=.8, fov_y=.6,
                foreground=torch.arange(18).float().reshape(2,3,3)/100+i/10,
                alpha=torch.arange(6).float().reshape(2,3,1)/10+i/10))
    def stacked_training_data(self):
        return {
            **{key:torch.stack([getattr(c, name) for c in self.train])
               for key,name in (("origins","origin"),("rights","right"),
                                ("downs","down"),("forwards","forward"))},
            **{key:torch.tensor([getattr(c,key) for c in self.train])
               for key in ("fov_x","fov_y")},
            "foreground":torch.cat([c.foreground.reshape(-1,3) for c in self.train]),
            "alpha":torch.cat([c.alpha.reshape(-1,1) for c in self.train])}


def test_mixed_indices_keep_ray_rgb_alpha_aligned():
    dataset = Dataset()
    sampler = MixedRaySampler(dataset, 300, (.2,.4,.6))
    ids = torch.tensor([17,0,7,5,6,12,7])
    batch = sampler.batch_from_indices(ids)
    for j,index in enumerate(ids.tolist()):
        camera,pixel = divmod(index,6)
        c = dataset.train[camera]
        rays = generate_camera_rays(c.origin,c.right,c.down,c.forward,3,2,.8,.6)
        torch.testing.assert_close(batch.rays.directions[j],rays.directions.reshape(-1,3)[pixel])
        torch.testing.assert_close(batch.rays.origins[j],c.origin)
        alpha = c.alpha.reshape(-1,1)[pixel]
        torch.testing.assert_close(batch.alpha[j],alpha)
        torch.testing.assert_close(batch.target[j],
            c.foreground.reshape(-1,3)[pixel]+torch.tensor([.2,.4,.6])*(1-alpha))
    state = torch.get_rng_state()
    first = sampler.next()
    torch.set_rng_state(state)
    second = sampler.next()
    assert first.rays.camera_indices.unique().numel() == 3
    torch.testing.assert_close(first.target,second.target,rtol=0,atol=0)
    torch.testing.assert_close(first.rays.directions,second.rays.directions,rtol=0,atol=0)


def test_pruning_protects_neighbors_and_uses_strict_thresholds():
    # Chain 0--1--2--3 plus isolated 4. Active 0 protects 1; large 3 survives.
    adj = torch.tensor([1,0,2,1,3,2])
    off = torch.tensor([0,1,3,5,6,6])
    contribution = torch.tensor([.02,0.,0.,0.,.01])
    radii = torch.tensor([.02,.02,.02,.1,.02])
    expected = torch.tensor([False,False,True,False,True])
    torch.testing.assert_close(population_masks(contribution,radii,adj,off,.01,.1),expected)


def test_sampling_is_weighted_without_replacement_and_handles_empty():
    torch.manual_seed(22)
    error = torch.tensor([0.,1.,8.,-3.])
    radii = torch.tensor([100.,1.,1.,1.])
    assert set(sample_parents(error,radii,99).tolist()) == {1,2}
    hits = sum(int(sample_parents(error,radii,1)[0]) == 2 for _ in range(1000))
    assert 830 < hits < 940
    assert sample_parents(torch.zeros(4),radii,5).numel() == 0
    with pytest.raises(FloatingPointError):
        sample_parents(torch.tensor([float("nan")]),torch.ones(1),1)


def test_children_are_quarter_neighbor_with_bounded_jitter():
    points = torch.tensor([[0.,0.,0.],[4.,0.,0.]])
    parents = torch.tensor([0,1])
    children = offspring_sites(points,torch.tensor([1,0]),parents)
    expected = torch.tensor([[1.,0.,0.],[3.,0.,0.]])
    torch.testing.assert_close((children-expected).norm(dim=1),torch.full((2,),.1))


def test_linear_growth_interval_and_validation():
    assert growth_interval(83000,131072,2097152,1.15,2000,11000) == 100
    assert growth_interval(371004,131072,2097152,1.15,2000,11000) == 254
    with pytest.raises(ValueError):
        growth_interval(4,4,4,1.15,2,5)


@pytest.mark.parametrize("changes",[
    {"ray_sampling":"mixed","loss":"l1_dssim"},
    {"population_policy":"radfoam"},
    {"population_factor":1.}, {"population_target":0},
    {"error_map_chunk":0}, {"prune_radius":float("nan")},
    {"site_initialization":"invalid"}, {"topology_schedule":"invalid"},
])
def test_invalid_organic_profile(changes):
    with pytest.raises(ValueError):
        replace(WarmupProfile(),**changes).validate()


def test_organic_manifest_is_isolated_and_paired(tmp_path):
    rows = variants()
    assert len(rows) == 16
    for i in range(16):
        config,profile,row = configuration(i,tmp_path/str(i))
        assert config.representation_mode == "voronoi_only"
        assert not config.gaussian_adaptation.enabled
        assert config.model.appearance == "rgb"
        assert config.optimization.iterations == 6000
        if i >= 2:
            assert config.optimization.rays_per_batch == 1000000
            assert config.voronoi_adaptation.densify_from == 2000
            assert profile.population_target == 2097152
    for i in range(2,16,2):
        a,b=rows[i],rows[i+1]
        assert (a["seed"],b["seed"]) == (0,42)
        assert a["profile"] == b["profile"] and a["inner"] == b["inner"]
    config,profile,_ = configuration(2,tmp_path/"full")
    assert config.model.medium_points + profile.outer_points == 131072
    assert profile.population_policy == "radfoam"
