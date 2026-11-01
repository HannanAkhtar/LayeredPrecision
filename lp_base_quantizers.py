"""Weight-only 5-bit LP base quantizers for Experiment 5.

The validated 2-bit U and N implementations are deliberately left untouched.
Codes are unsigned 0..31 and all decoder metadata is FP32 except affine
zero-points. Strict threshold comparisons assign an exact tie to the lower code.
"""

from __future__ import annotations

import math
import torch

from layered_quantization import IntegerFormat
from quantizer_variants import dequantize_variant

FAMILIES = ("A", "P", "U", "N", "L")
GAMMAS = tuple(0.7 + 0.025 * i for i in range(25))
LOG_RATIO = math.sqrt(2.0)


def _matrix(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim < 2 or not weight.is_floating_point() or not bool(torch.isfinite(weight).all()):
        raise ValueError("Expected finite floating-point channel-axis-0 tensor")
    return weight.reshape(weight.shape[0], -1)


def _view(weight: torch.Tensor) -> tuple[int, ...]:
    return (weight.shape[0],) + (1,) * (weight.ndim - 1)


def _assign(matrix: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    thresholds = ((centers[:, :-1] + centers[:, 1:]) * .5).contiguous()
    return torch.searchsorted(thresholds, matrix.contiguous(), right=False).to(torch.uint8)


def _symmetric_levels(scale: torch.Tensor) -> torch.Tensor:
    factors = torch.arange(32, device=scale.device, dtype=scale.dtype) - 15.5
    return scale[:, None] * factors[None, :]


def _log_levels(scale: torch.Tensor) -> torch.Tensor:
    powers = torch.tensor([LOG_RATIO ** i for i in range(16)], device=scale.device, dtype=scale.dtype)
    factors = torch.cat((-powers.flip(0), powers))
    return scale[:, None] * factors[None, :]


def _decode_centers(params: dict, centers: torch.Tensor) -> torch.Tensor:
    codes = params["quantized"]
    return centers.gather(1, codes.reshape(codes.shape[0], -1).long()).reshape_as(codes)


@torch.no_grad()
def quantize_base(weight: torch.Tensor, family: str, *, max_iterations: int = 30,
                  tolerance: float = 1e-6) -> tuple[dict, torch.Tensor]:
    if family not in FAMILIES and family != "U0":
        raise ValueError(family)
    matrix = _matrix(weight)
    channels, width = matrix.shape
    if family == "A":
        params, decoded = IntegerFormat("channel", 0).quantize(weight, 5)
        params["family"] = family
        return params, decoded
    if family == "P":
        # Same paired-tail percentile convention as the validated 2-bit P.
        q = torch.tensor([.005, .995], dtype=weight.dtype, device=weight.device)
        low, high = torch.quantile(matrix, q, dim=1)
        low = torch.minimum(low, torch.zeros_like(low))
        high = torch.maximum(high, torch.zeros_like(high))
        scale = (high - low) / 31
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        zp = (-low / safe).round().clamp(0, 31).to(torch.uint8)
        codes = torch.where(scale[:, None] > 0,
                            (matrix / safe[:, None] + zp[:, None]).round().clamp(0, 31),
                            torch.zeros_like(matrix)).to(torch.uint8)
        decoded = torch.where(scale[:, None] > 0,
                              scale[:, None] * (codes.to(weight.dtype)-zp[:, None]),
                              low[:, None])
        params = {"family": family, "format": "int", "bits": 5, "axis": 0,
                  "quantized": codes.reshape_as(weight), "scale": scale.reshape(_view(weight)),
                  "zero_point": zp.reshape(_view(weight)), "t_min": low.reshape(_view(weight)),
                  "clip_percentile": 99.5}
        return params, decoded.reshape_as(weight)
    if family in ("U", "U0", "L"):
        amplitude = matrix.abs().amax(1)
        base_scale = amplitude / (16 if family != "L" else LOG_RATIO ** 15)
        best_scale = base_scale.clone()
        best_error = torch.full((channels,), float("inf"), device=matrix.device, dtype=torch.float64)
        selected = torch.full((channels,), 12, device=matrix.device, dtype=torch.uint8)
        candidates = (1.0,) if family == "U0" else GAMMAS
        for index, gamma in enumerate(candidates):
            scale = base_scale * gamma
            centers = _log_levels(scale) if family == "L" else _symmetric_levels(scale)
            codes = _assign(matrix, centers)
            decoded = centers.gather(1, codes.long())
            error = (matrix.double()-decoded.double()).square().sum(1)
            better = error < best_error
            best_error = torch.where(better, error, best_error)
            best_scale = torch.where(better, scale, best_scale)
            selected = torch.where(better, torch.tensor(index, dtype=torch.uint8, device=matrix.device), selected)
        centers = _log_levels(best_scale) if family == "L" else _symmetric_levels(best_scale)
        codes = _assign(matrix, centers)
        decoded = centers.gather(1, codes.long())
        params = {"family": family, "bits": 5, "axis": 0,
                  "quantized": codes.reshape_as(weight), "scale": best_scale.reshape(_view(weight)),
                  "gamma_index": selected, "zero_exact_for_nonzero_scale": False}
        if family == "L":
            params["global_log_ratio"] = LOG_RATIO
        return params, decoded.reshape_as(weight)
    if max_iterations < 1 or tolerance < 0:
        raise ValueError("Invalid Lloyd settings")
    ordered = matrix.sort(1).values
    positions = (torch.arange(32, device=matrix.device, dtype=torch.float64) + .5) * (width-1) / 32
    left, right = positions.floor().long(), positions.ceil().long()
    fraction = (positions-left).to(matrix.dtype)
    centers = ordered[:, left]*(1-fraction) + ordered[:, right]*fraction
    iterations = torch.zeros(channels, dtype=torch.int16, device=matrix.device)
    for iteration in range(max_iterations):
        codes = _assign(matrix, centers).long()
        totals = torch.zeros_like(centers).scatter_add_(1, codes, matrix)
        counts = torch.zeros_like(centers).scatter_add_(1, codes, torch.ones_like(matrix))
        updated = torch.where(counts > 0, totals/counts.clamp_min(1), centers).sort(1).values
        change = (updated-centers).abs().amax(1)
        scale = matrix.abs().amax(1).clamp_min(torch.finfo(matrix.dtype).tiny)
        converged = change <= tolerance*scale
        iterations += (~(iterations > 0)).to(torch.int16) if iteration == 0 else (~converged).to(torch.int16)
        centers = updated
        if bool(converged.all()):
            break
    codes = _assign(matrix, centers)
    decoded = centers.gather(1, codes.long())
    duplicate = (centers[:, 1:] == centers[:, :-1]).any(1)
    params = {"family": "N", "bits": 5, "axis": 0, "quantized": codes.reshape_as(weight),
              "centers": centers, "iterations": iteration+1,
              "duplicate_centroid_channels": int(duplicate.sum()),
              "converged": iteration+1 < max_iterations}
    return params, decoded.reshape_as(weight)


def dequantize_base(params: dict) -> torch.Tensor:
    family = params["family"]
    if family in ("A", "P"):
        return IntegerFormat.dequantize(params) if family == "A" else dequantize_variant(params)
    if family in ("U", "U0", "L"):
        scale = params["scale"].reshape(-1)
        centers = _log_levels(scale) if family == "L" else _symmetric_levels(scale)
    elif family == "N":
        centers = params["centers"]
    else:
        raise ValueError(family)
    return _decode_centers(params, centers)
