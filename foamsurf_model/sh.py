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

    x = directions[..., 0:1]
    y = directions[..., 1:2]
    z = directions[..., 2:3]
    result = C0 * coefficients[..., 0]

    if degree > 0:
        result = (
            result
            - C1 * y * coefficients[..., 1]
            + C1 * z * coefficients[..., 2]
            - C1 * x * coefficients[..., 3]
        )
    if degree > 1:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        result = (
            result
            + C2[0] * xy * coefficients[..., 4]
            + C2[1] * yz * coefficients[..., 5]
            + C2[2] * (2.0 * zz - xx - yy) * coefficients[..., 6]
            + C2[3] * xz * coefficients[..., 7]
            + C2[4] * (xx - yy) * coefficients[..., 8]
        )
    if degree > 2:
        result = (
            result
            + C3[0] * y * (3.0 * xx - yy) * coefficients[..., 9]
            + C3[1] * xy * z * coefficients[..., 10]
            + C3[2] * y * (4.0 * zz - xx - yy) * coefficients[..., 11]
            + C3[3]
            * z
            * (2.0 * zz - 3.0 * xx - 3.0 * yy)
            * coefficients[..., 12]
            + C3[4] * x * (4.0 * zz - xx - yy) * coefficients[..., 13]
            + C3[5] * z * (xx - yy) * coefficients[..., 14]
            + C3[6] * x * (xx - 3.0 * yy) * coefficients[..., 15]
        )
    return result
