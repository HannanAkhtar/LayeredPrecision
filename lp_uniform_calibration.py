"""Weight-only, four-code uniform residual quantizers for Experiment 4.

U0 uses the analytical midpoint scale; U searches a fixed 29-point scale
grid independently in each output channel. Exact midpoint ties use the lower
code. Metadata contains only one FP32 scale per channel.
"""

from __future__ import annotations

import torch

ALPHAS = tuple(0.35 + 0.0125 * i for i in range(29))
LEVELS = (-1.5, -0.5, 0.5, 1.5)


def _encode(matrix: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # The three midpoints are -scale, 0, and scale. Strict comparison sends
    # exact ties to the lower index, including zero.
    return ((matrix > -scale[:, None]).to(torch.uint8) +
            (matrix > 0).to(torch.uint8) +
            (matrix > scale[:, None]).to(torch.uint8))


@torch.no_grad()
def quantize_uniform(residual: torch.Tensor, family: str) -> tuple[dict, torch.Tensor]:
    if family not in ("U0", "U"):
        raise ValueError(family)
    if residual.ndim < 2 or not residual.is_floating_point() or not bool(torch.isfinite(residual).all()):
        raise ValueError("Expected finite floating-point channel-axis-0 residual")
    matrix = residual.reshape(residual.shape[0], -1)
    amplitude = matrix.abs().amax(dim=1)
    scale = amplitude * 0.5
    alpha_index = torch.full((matrix.shape[0],), 12, dtype=torch.uint8, device=matrix.device)
    if family == "U":
        # Accumulate objectives in FP64; the decoded reconstruction remains
        # FP32 so it matches the actual stored decoder metadata.
        best = torch.full_like(amplitude, float("inf"), dtype=torch.float64)
        for index, alpha in enumerate(ALPHAS):
            candidate = amplitude * alpha
            codes = _encode(matrix, candidate)
            decoded = (codes.to(matrix.dtype) * candidate[:, None] - 1.5 * candidate[:, None])
            objective = (matrix.double() - decoded.double()).square().sum(dim=1)
            take = objective < best
            best = torch.where(take, objective, best)
            scale = torch.where(take, candidate, scale)
            alpha_index = torch.where(take, torch.tensor(index, device=matrix.device, dtype=torch.uint8), alpha_index)
    codes = _encode(matrix, scale)
    decoded = codes.to(matrix.dtype) * scale[:, None] - 1.5 * scale[:, None]
    shape = (matrix.shape[0],) + (1,) * (residual.ndim - 1)
    params = {"family": family, "quantized": codes.reshape_as(residual),
              "scale": scale.reshape(shape), "shape": tuple(residual.shape),
              "bits": 2, "axis": 0, "alpha_index": alpha_index}
    return params, decoded.reshape_as(residual)


def dequantize_uniform(params: dict) -> torch.Tensor:
    codes = params["quantized"]
    scale = params["scale"]
    return codes.to(scale.dtype) * scale - 1.5 * scale
