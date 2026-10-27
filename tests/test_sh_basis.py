"""foamsurf_model/sh.py: the basis-product evaluate_sh equals the term-by-term 3DGS formula."""
import pytest
import torch

from foamsurf_model.sh import C0, C1, C2, C3, evaluate_sh, sh_basis


def reference(degree, c, d):
    """The previous term-by-term implementation (3DGS convention)."""
    x, y, z = d[..., 0:1], d[..., 1:2], d[..., 2:3]
    r = C0 * c[..., 0]
    if degree > 0:
        r = r - C1 * y * c[..., 1] + C1 * z * c[..., 2] - C1 * x * c[..., 3]
    if degree > 1:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        r = (r + C2[0] * xy * c[..., 4] + C2[1] * yz * c[..., 5] + C2[2] * (2 * zz - xx - yy) * c[..., 6]
             + C2[3] * xz * c[..., 7] + C2[4] * (xx - yy) * c[..., 8])
    if degree > 2:
        r = (r + C3[0] * y * (3 * xx - yy) * c[..., 9] + C3[1] * xy * z * c[..., 10]
             + C3[2] * y * (4 * zz - xx - yy) * c[..., 11] + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * c[..., 12]
             + C3[4] * x * (4 * zz - xx - yy) * c[..., 13] + C3[5] * z * (xx - yy) * c[..., 14]
             + C3[6] * x * (xx - 3 * yy) * c[..., 15])
    return r


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_values_and_gradients_match_the_term_by_term_formula(degree):
    torch.manual_seed(degree)
    coefficients = torch.randn(50, 3, 16, dtype=torch.float64, requires_grad=True)
    directions = torch.nn.functional.normalize(torch.randn(50, 3, dtype=torch.float64), dim=-1).requires_grad_(True)
    new = evaluate_sh(degree, coefficients, directions)
    old = reference(degree, coefficients, directions)
    torch.testing.assert_close(new, old)
    weights = torch.randn_like(new)
    grads_new = torch.autograd.grad((new * weights).sum(), (coefficients, directions), allow_unused=True)
    grads_old = torch.autograd.grad((old * weights).sum(), (coefficients, directions), allow_unused=True)
    for a, b, like in zip(grads_new, grads_old, (coefficients, directions)):  # degree 0: no direction gradient
        torch.testing.assert_close(torch.zeros_like(like) if a is None else a, torch.zeros_like(like) if b is None else b)


def test_basis_size_and_dc():
    d = torch.tensor([[0.0, 0.0, 1.0]])
    assert [sh_basis(k, d).shape[-1] for k in range(4)] == [1, 4, 9, 16]
    assert sh_basis(0, d).item() == pytest.approx(C0)
