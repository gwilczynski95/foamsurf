from __future__ import annotations

from pathlib import Path

import torch

from training.determinism import tensor_difference, tensor_digest, tensors_digest


def test_tensor_digests_are_stable_and_name_sensitive() -> None:
    value = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert tensor_digest(value) == tensor_digest(value.clone())
    assert tensor_digest(value) != tensor_digest(value + 1.0)
    assert tensors_digest({"a": value}) != tensors_digest({"b": value})


def test_tensor_difference_reports_exact_and_changed_values() -> None:
    reference = torch.tensor([1.0, 2.0, 3.0])
    exact = tensor_difference(reference, reference.clone())
    changed = tensor_difference(reference, torch.tensor([1.0, 2.5, 3.0]))
    assert exact["different_elements"] == 0
    assert exact["max_abs"] == 0.0
    assert changed["different_elements"] == 1
    assert changed["max_abs"] == 0.5


def test_audit_sbatch_contracts() -> None:
    root = Path(__file__).resolve().parents[2]
    replay = (root / "scripts" / "determinism_audit_replay.sbatch").read_text()
    training = (
        root / "scripts" / "determinism_audit_training.sbatch"
    ).read_text()
    assert "#SBATCH --array=0-2%1" in replay
    assert "#SBATCH --array=0-5%1" in training
    assert "foamsurf_determinism_audit" in replay
    assert "foamsurf_determinism_audit" in training
    assert "CUBLAS_WORKSPACE_CONFIG" in replay
    assert "CUBLAS_WORKSPACE_CONFIG" in training
