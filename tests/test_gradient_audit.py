"""scripts/gradient_audit.py: the directional finite-difference check and the renderer audit."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import gradient_audit as audit  # noqa: E402


def test_directional_check_agrees_on_a_correct_gradient():
    target = torch.randn(50, 3)

    def loss(p):
        return ((p["x"].double().sin() * target).sum() + (p["y"].double() ** 3).sum())

    report = audit.directional_check(loss, dict(x=torch.randn(50, 3), y=torch.randn(7)))
    assert all(row["best"] < 1e-3 for row in report.values())


class _WrongBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x * 2.0

    @staticmethod
    def backward(ctx, grad):
        return grad * 1.5  # true derivative is 2


def test_directional_check_catches_a_wrong_backward():
    report = audit.directional_check(lambda p: _WrongBackward.apply(p["x"]).double().sum(), dict(x=torch.randn(20)))
    assert report["x"]["best"] > 0.2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_renderer_gradient_matches_its_own_finite_differences():
    """2000 Gaussians, 200 cells, 4096 rays, threshold 1e-4, far 1000, some rays fill the 64-hit buffer."""
    report, saturation = audit.synthetic_audit()
    audit.print_report(report)
    print(saturation)
    assert saturation["full_fraction"] > 0.01  # the audit really exercises the truncated buffer
    for name, row in report.items():
        assert row["best"] < 0.02, (name, row)
