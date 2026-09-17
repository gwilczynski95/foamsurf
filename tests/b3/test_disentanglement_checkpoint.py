from __future__ import annotations

import sys
import types

import torch

topology_stub = types.ModuleType("foamsurf_topology")
topology_stub.TriangulationFailedError = RuntimeError
sys.modules.setdefault("foamsurf_topology", topology_stub)

from training import ExperimentConfig
from training.checkpoints import checkpoint_payload


class _StateOwner:
    def __init__(self, state):
        self._state = state

    def state_dict(self):
        return self._state


class _Model(_StateOwner):
    def checkpoint_metadata(self):
        return {"gaussians": {}, "medium": {}}


def test_checkpoint_contains_exact_disentanglement_controller_state(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    state = {
        "density_stat_ema": 0.25,
        "density_cap_value": 0.5,
        "density_cap_frozen": True,
        "last_cap_update_iteration": 5_000,
    }
    payload = checkpoint_payload(
        _Model({"value": torch.tensor(1.0)}),
        _StateOwner({"gaussian": 1}),
        _StateOwner({"medium": 2}),
        ExperimentConfig.load("configs/baseline_clean_joint.yaml"),
        iteration=5_000,
        training_seconds=12.5,
        best_psnr=31.0,
        topology_revision=4,
        disentanglement_state=state,
    )
    assert payload["disentanglement"] == state


def test_checkpoint_keeps_backward_compatible_none_state(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    payload = checkpoint_payload(
        _Model({}),
        _StateOwner({}),
        _StateOwner({}),
        ExperimentConfig.load("configs/baseline_clean_joint.yaml"),
        iteration=1,
        training_seconds=0.0,
        best_psnr=0.0,
        topology_revision=0,
    )
    assert payload["format_version"] == 1
    assert payload["disentanglement"] is None
