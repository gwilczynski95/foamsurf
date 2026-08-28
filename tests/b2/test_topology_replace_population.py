from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import sys
import types

import pytest
import torch
from torch import nn

topology_stub = types.ModuleType("foamsurf_topology")
topology_stub.TriangulationFailedError = RuntimeError
sys.modules.setdefault("foamsurf_topology", topology_stub)

import foamsurf_model.topology as topology_module
from foamsurf_model.medium import VoronoiMediumModel
from foamsurf_model.topology import VoronoiTopologyController


class AmbiguousTriangulation(RuntimeError):
    pass


class IdentityTriangulation:
    def __init__(self, sites: torch.Tensor):
        self.sites = sites.detach().clone()

    def permutation(self) -> torch.Tensor:
        return torch.arange(self.sites.shape[0], device=self.sites.device)

    def point_adjacency(self) -> torch.Tensor:
        return torch.empty(0, dtype=torch.int64, device=self.sites.device)

    def point_adjacency_offsets(self) -> torch.Tensor:
        return torch.zeros(
            self.sites.shape[0] + 1,
            dtype=torch.int64,
            device=self.sites.device,
        )


def _medium_and_optimizer():
    medium = VoronoiMediumModel.__new__(VoronoiMediumModel)
    nn.Module.__init__(medium)
    medium.device = torch.device("cpu")
    medium.register_buffer("bbox_min", torch.zeros(3))
    medium.register_buffer("bbox_max", torch.ones(3))
    medium.register_buffer(
        "guard_mask", torch.tensor([False, False, False, True])
    )
    medium.sites = nn.Parameter(
        torch.tensor(
            [[0.1, 0.1, 0.1], [0.3, 0.3, 0.3],
             [0.6, 0.6, 0.6], [0.9, 0.9, 0.9]]
        )
    )
    medium.raw_density = nn.Parameter(torch.arange(4.0).reshape(4, 1))
    medium.raw_color = nn.Parameter(torch.arange(12.0).reshape(4, 3))
    optimizer = torch.optim.Adam(
        [
            {"params": [medium.sites], "name": "medium_sites"},
            {"params": [medium.raw_density], "name": "medium_density"},
            {"params": [medium.raw_color], "name": "medium_color"},
        ],
        lr=0.01,
    )
    sum(parameter.square().sum() for parameter in medium.parameters()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    controller = VoronoiTopologyController.__new__(VoronoiTopologyController)
    controller.revision = 7
    controller.last_replacement_retries = 0
    return medium, optimizer, controller


def test_replace_population_retries_before_committing(monkeypatch) -> None:
    medium, optimizer, controller = _medium_and_optimizer()
    calls = []

    def triangulate(sites):
        calls.append(sites.detach().clone())
        if len(calls) <= 2:
            raise AmbiguousTriangulation("ambiguous triangulation")
        return IdentityTriangulation(sites)

    monkeypatch.setattr(
        topology_module,
        "foamsurf_topology",
        SimpleNamespace(
            Triangulation=triangulate,
            TriangulationFailedError=AmbiguousTriangulation,
        ),
    )
    child = medium.sites.detach()[[1]].clone()
    controller.replace_population(
        medium,
        optimizer,
        keep_indices=torch.tensor([0, 2, 3]),
        parent_indices=torch.tensor([1]),
        child_sites=child,
        max_attempts=4,
    )

    assert len(calls) == 3
    assert controller.last_replacement_retries == 2
    assert controller.revision == 8
    assert medium.point_count == 4
    assert not torch.equal(medium.sites.detach()[-1], child[0])
    assert medium.guard_mask.tolist() == [False, False, True, False]
    for group in optimizer.param_groups:
        state = optimizer.state[group["params"][0]]
        torch.testing.assert_close(state["exp_avg"][-1], torch.zeros_like(state["exp_avg"][-1]))
        torch.testing.assert_close(state["exp_avg_sq"][-1], torch.zeros_like(state["exp_avg_sq"][-1]))


def test_replace_population_rolls_back_after_all_retries(monkeypatch) -> None:
    medium, optimizer, controller = _medium_and_optimizer()
    parameters = {
        name: getattr(medium, name) for name in medium.PARAMETER_NAMES
    }
    values = {name: value.detach().clone() for name, value in parameters.items()}
    guard_mask = medium.guard_mask.clone()
    optimizer_state = deepcopy(optimizer.state_dict())

    def fail(_sites):
        raise AmbiguousTriangulation("ambiguous triangulation")

    monkeypatch.setattr(
        topology_module,
        "foamsurf_topology",
        SimpleNamespace(
            Triangulation=fail,
            TriangulationFailedError=AmbiguousTriangulation,
        ),
    )
    with pytest.raises(AmbiguousTriangulation):
        controller.replace_population(
            medium,
            optimizer,
            keep_indices=torch.tensor([0, 2, 3]),
            parent_indices=torch.tensor([1]),
            child_sites=medium.sites.detach()[[1]].clone(),
            max_attempts=3,
        )

    assert controller.last_replacement_retries == 3
    assert controller.revision == 7
    torch.testing.assert_close(medium.guard_mask, guard_mask)
    for name, parameter in parameters.items():
        assert getattr(medium, name) is parameter
        torch.testing.assert_close(parameter, values[name])
    assert optimizer.state_dict().keys() == optimizer_state.keys()
    for current_group, old_group in zip(
        optimizer.state_dict()["param_groups"], optimizer_state["param_groups"]
    ):
        assert current_group == old_group
    for key, old_state in optimizer_state["state"].items():
        for name, old_value in old_state.items():
            current = optimizer.state_dict()["state"][key][name]
            if torch.is_tensor(old_value):
                torch.testing.assert_close(current, old_value)
            else:
                assert current == old_value
