from __future__ import annotations

import torch


C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)


def rgb_to_sh_dc(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / C0


def evaluate_sh(
    degree: int, coefficients: torch.Tensor, directions: torch.Tensor
) -> torch.Tensor:
    if not 0 <= degree <= 3:
        raise ValueError("SH degree must be between 0 and 3")
    required = (degree + 1) ** 2
    if coefficients.shape[-1] < required:
        raise ValueError("not enough SH coefficients")
    if coefficients.shape[:-2] != directions.shape[:-1]:
        raise ValueError("coefficient and direction batch shapes do not match")

    # Basis [..., 1, required] times the coefficients in one product: indexing the coefficients one by
    # one makes every backward allocate and zero a full copy of them per coefficient.
    return (coefficients[..., :required] * sh_basis(degree, directions)[..., None, :]).sum(-1)


def sh_basis(degree: int, directions: torch.Tensor) -> torch.Tensor:
    """Real SH basis values [..., (degree + 1)^2] (3DGS sign convention) of unit directions [..., 3]."""
    x, y, z = directions.unbind(-1)
    values = [torch.full_like(x, C0)]
    if degree > 0:
        values += [-C1 * y, C1 * z, -C1 * x]
    if degree > 1:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        values += [C2[0] * xy, C2[1] * yz, C2[2] * (2.0 * zz - xx - yy), C2[3] * xz, C2[4] * (xx - yy)]
    if degree > 2:
        values += [
            C3[0] * y * (3.0 * xx - yy),
            C3[1] * xy * z,
            C3[2] * y * (4.0 * zz - xx - yy),
            C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
            C3[4] * x * (4.0 * zz - xx - yy),
            C3[5] * z * (xx - yy),
            C3[6] * x * (xx - 3.0 * yy),
        ]
    return torch.stack(values, -1)
