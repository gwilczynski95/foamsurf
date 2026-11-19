"""Initial triangulation of trained sites (loading a checkpoint): an ambiguous from-scratch triangulation is reached
incrementally from an offset start without changing the sites; only if that fails are the non-guard sites jittered
(guards never move); the returned permutation is the composition of the build and the rebuild orders."""
from types import SimpleNamespace

import pytest
import torch

import foamsurf_model.topology as topo


class Fake:
    """Constructor fails ``fail_build`` times, rebuild fails ``fail_rebuild`` times; permutations are fixed."""
    builds, rebuilds = [], []
    fail_build = fail_rebuild = 0
    perm_build = perm_rebuild = None

    def __init__(self, sites):
        Fake.builds.append(sites.clone())
        if len(Fake.builds) <= Fake.fail_build:
            raise topo.foamsurf_topology.TriangulationFailedError("ambiguous triangulation")
        self._perm = Fake.perm_build if Fake.perm_build is not None else torch.arange(len(sites))

    def permutation(self):
        return self._perm

    def rebuild(self, sites, incremental):
        Fake.rebuilds.append((sites.clone(), incremental))
        if len(Fake.rebuilds) <= Fake.fail_rebuild:
            raise topo.foamsurf_topology.TriangulationFailedError("ambiguous triangulation")
        if Fake.perm_rebuild is None:
            return False
        self._perm = Fake.perm_rebuild
        return True


def medium(n=6, guards=2):
    sites = torch.nn.Parameter(torch.rand(n, 3, generator=torch.Generator().manual_seed(0)))
    guard = torch.zeros(n, dtype=torch.bool)
    guard[-guards:] = True
    return SimpleNamespace(sites=sites, guard_mask=guard, bbox_min=torch.full((3,), -1.0), bbox_max=torch.full((3,), 2.0))


@pytest.fixture
def fake(monkeypatch):
    def install(fail_build=0, fail_rebuild=0, perm_build=None, perm_rebuild=None):
        Fake.builds, Fake.rebuilds = [], []
        Fake.fail_build, Fake.fail_rebuild = fail_build, fail_rebuild
        Fake.perm_build, Fake.perm_rebuild = perm_build, perm_rebuild
        monkeypatch.setattr(topo, "foamsurf_topology", SimpleNamespace(
            Triangulation=Fake, TriangulationFailedError=RuntimeError))
    return install


def test_unambiguous_sites_are_untouched(fake):
    fake()
    m = medium()
    before = m.sites.detach().clone()
    _, perm = topo.VoronoiTopologyController._initial_triangulation(m)
    assert torch.equal(m.sites.detach(), before) and len(Fake.builds) == 1 and torch.equal(perm, torch.arange(6))


def test_ambiguous_sites_are_reached_incrementally_and_kept_exact(fake):
    p1 = torch.tensor([5, 4, 3, 2, 1, 0])
    p2 = torch.tensor([1, 0, 2, 3, 5, 4])
    fake(fail_build=1, perm_build=p1, perm_rebuild=p2)
    m = medium()
    before = m.sites.detach().clone()
    _, perm = topo.VoronoiTopologyController._initial_triangulation(m)
    assert torch.equal(m.sites.detach(), before)                       # the sites are not changed
    assert not torch.equal(Fake.builds[1], before)                     # the start was offset
    rebuilt, incremental = Fake.rebuilds[0]
    assert incremental and torch.equal(rebuilt, before[p1])            # rebuilt to the exact sites, in build order
    assert torch.equal(perm, p1[p2])                                   # composed permutation


def test_training_style_perturbation_keeps_the_exact_sites(fake):
    fake(fail_build=1, fail_rebuild=3)  # the three offset starts build but cannot be rebuilt to the exact sites
    m = medium()
    before = m.sites.detach().clone()
    topo.VoronoiTopologyController._initial_triangulation(m)
    assert torch.equal(m.sites.detach(), before)                    # rendering keeps the exact sites
    assert len(Fake.builds) == 5                                     # exact, 3 offsets, 1 perturbed: accepted
    assert 0 < float((Fake.builds[-1] - before).abs().max()) < 1e-4  # a training-sized perturbation


def test_jitter_fallback_when_nothing_else_builds(fake):
    fake(fail_build=14)  # exact + 3 offsets + 10 perturbed builds fail; the first jitter build succeeds
    m = medium()
    before = m.sites.detach().clone()
    topo.VoronoiTopologyController._initial_triangulation(m)
    after = m.sites.detach()
    assert torch.equal(after[m.guard_mask], before[m.guard_mask])
    moved = (after - before).norm(dim=-1)
    assert bool((moved[~m.guard_mask] > 0).all()) and float(moved.max()) < 1e-5


def test_gives_up_after_the_limit(fake):
    fake(fail_build=100)
    m = medium()
    before = m.sites.detach().clone()
    with pytest.raises(topo.foamsurf_topology.TriangulationFailedError, match="after 3 jittered attempts"):
        topo.VoronoiTopologyController._initial_triangulation(m, max_attempts=3)
    assert torch.equal(m.sites.detach(), before)


def test_reflected_jitter_moves_sites_off_the_box_faces():
    sites = torch.tensor([[2.0, 0.5, -1.0], [0.0, 0.0, 0.0]])  # the first lies on two faces (x = max, z = min)
    movable = torch.tensor([True, True])
    lo, hi = torch.full((3,), -1.0), torch.full((3,), 2.0)
    clamped = topo.jittered_sites(sites, movable, 3, 1e-3, lo, hi)
    reflected = topo.jittered_sites(sites, movable, 3, 1e-3, lo, hi, reflect=True)
    assert bool(((reflected >= lo) & (reflected <= hi)).all())
    assert bool((reflected[0, 0] < 2.0) and (reflected[0, 2] > -1.0))
    assert (reflected - sites).abs().max() <= 4e-3 + 1e-9
    assert torch.equal(reflected[1], clamped[1])


def test_training_imports_do_not_initialize_cuda():
    import os
    import subprocess
    import sys

    subprocess.run([sys.executable, "-c", """
import sys
import torch
import training.end_to_end
import training.native_handoff
import training.trainer
assert 'foamsurf_topology' not in sys.modules
assert not torch.cuda.is_initialized()
"""], env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), check=True)
