"""Prespecified two-bit quantizers for the first LP residual.

The signed symmetric convention uses codes {-1, 0, 1} and levels
{-scale, 0, +scale}. Code -2 is unused: exact zero and four symmetric,
equally spaced levels cannot coexist at two bits. Affine methods use unsigned
codes {0, 1, 2, 3} and levels scale * (code - integer_zero_point).
"""

from __future__ import annotations

import math
import torch

from layered_quantization import IntegerFormat


VARIANTS = (
    "symmetric_minmax", "affine_minmax", "symmetric_percentile",
    "affine_percentile", "affine_mse",
)


def _channel_view(tensor: torch.Tensor, axis: int) -> tuple[torch.Tensor, int | None]:
    if tensor.ndim <= 1:
        return tensor.reshape(1, -1), None
    if not -tensor.ndim <= axis < tensor.ndim:
        raise ValueError(f"channel_axis {axis} is invalid for shape {tuple(tensor.shape)}")
    axis = axis % tensor.ndim
    return tensor.movedim(axis, 0).reshape(tensor.shape[axis], -1), axis


def _metadata(tensor: torch.Tensor, bits: int, axis: int | None,
              scale: torch.Tensor, zero_point: torch.Tensor,
              codes: torch.Tensor, **extra) -> dict:
    return {
        "format": "int", "bits": bits, "scale": scale,
        "zero_point": zero_point, "quantized": codes,
        "shape": tensor.shape, "granularity": "channel",
        "selected_granularity": "channel" if axis is not None else "tensor",
        "axis": axis, "channel_axis": 0 if axis is None else axis,
        "use_zero_point": True, **extra,
    }


def _reshape(values: torch.Tensor, tensor: torch.Tensor, axis: int | None) -> torch.Tensor:
    if axis is None:
        return values.reshape(())
    shape = [1] * tensor.ndim
    shape[axis] = tensor.shape[axis]
    return values.reshape(shape)


def _affine(tensor: torch.Tensor, bits: int, low: torch.Tensor,
            high: torch.Tensor, axis: int | None, **extra) -> tuple[dict, torch.Tensor]:
    qmax = 2 ** bits - 1
    low = torch.minimum(low, torch.zeros_like(low))
    high = torch.maximum(high, torch.zeros_like(high))
    scale = (high - low) / qmax
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    zp = (-low / safe).round().clamp(0, qmax).to(torch.uint8)
    s = _reshape(safe, tensor, axis)
    z = _reshape(zp, tensor, axis)
    active = _reshape(scale > 0, tensor, axis)
    codes = torch.where(active, (tensor / s + z).round().clamp(0, qmax),
                        torch.zeros_like(tensor)).to(torch.uint8)
    reconstructed = torch.where(active, _reshape(scale, tensor, axis) *
                                (codes.to(tensor.dtype) - z),
                                _reshape(low, tensor, axis))
    return _metadata(tensor, bits, axis, _reshape(scale, tensor, axis), z,
                     codes, t_min=_reshape(low, tensor, axis),
                     clip_low=_reshape(low, tensor, axis),
                     clip_high=_reshape(high, tensor, axis), **extra), reconstructed


def _symmetric(tensor: torch.Tensor, bits: int, threshold: torch.Tensor,
               axis: int | None, **extra) -> tuple[dict, torch.Tensor]:
    if bits != 2:
        raise ValueError("The prespecified signed symmetric convention requires 2 bits")
    scale = _reshape(threshold, tensor, axis)
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = torch.where(scale > 0, (tensor / safe).round().clamp(-1, 1),
                        torch.zeros_like(tensor)).to(torch.int8)
    reconstructed = scale * codes.to(tensor.dtype)
    return _metadata(tensor, bits, axis, scale,
                     torch.zeros_like(scale, dtype=torch.int8), codes,
                     signed=True, effective_levels=3,
                     clip_abs=scale, **extra), reconstructed


@torch.no_grad()
def quantize_variant(tensor: torch.Tensor, variant: str, bits: int = 2,
                     channel_axis: int = 0, clip_percentile: float = 99.5,
                     mse_scale_min: float = 0.5, mse_scale_max: float = 1.5,
                     mse_steps: int = 17) -> tuple[dict, torch.Tensor]:
    """Return stored codes/metadata and a dequantized residual reconstruction.

    MSE search tests every integer zero point and a fixed scale grid per channel.
    The original affine min/max result initializes the winner, so optimization
    cannot increase calibration MSE. No task labels enter calibration.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {VARIANTS}")
    if bits != 2:
        raise ValueError("First-residual variants require exactly 2 bits")
    if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
        raise ValueError("Expected a finite floating-point tensor")
    if not 50 <= clip_percentile <= 100:
        raise ValueError("clip_percentile must be in [50, 100] for paired affine tails")
    if variant == "affine_minmax":
        return IntegerFormat("channel", channel_axis).quantize(tensor, bits)
    matrix, axis = _channel_view(tensor, channel_axis)
    low = matrix.amin(dim=1)
    high = matrix.amax(dim=1)
    if variant == "symmetric_minmax":
        return _symmetric(tensor, bits, matrix.abs().amax(dim=1), axis)
    if variant == "symmetric_percentile":
        threshold = torch.quantile(matrix.abs(), clip_percentile / 100, dim=1)
        return _symmetric(tensor, bits, threshold, axis,
                          clip_percentile=clip_percentile)
    if variant == "affine_percentile":
        tail = (100 - clip_percentile) / 100
        thresholds = torch.quantile(matrix, torch.tensor([tail, 1 - tail],
                                    device=tensor.device, dtype=tensor.dtype), dim=1)
        return _affine(tensor, bits, thresholds[0], thresholds[1], axis,
                       clip_percentile=clip_percentile)

    if (mse_steps < 2 or not math.isfinite(mse_scale_min) or
            not math.isfinite(mse_scale_max) or mse_scale_min <= 0 or
            mse_scale_max < mse_scale_min):
        raise ValueError("Invalid MSE scale grid")
    baseline, reconstructed = IntegerFormat("channel", channel_axis).quantize(tensor, bits)
    baseline["calibration"] = "affine_mse_grid"
    baseline["mse_scale_grid"] = [mse_scale_min, mse_scale_max, mse_steps]
    best_scale = baseline["scale"].clone()
    best_zp = baseline["zero_point"].clone()
    best_codes = baseline["quantized"].clone()
    best_t_min = baseline["t_min"].clone()
    best_reconstructed = reconstructed.clone()
    best_error = ((matrix - _channel_view(reconstructed, channel_axis)[0]).double()
                  .square().mean(dim=1))
    base_scale = baseline["scale"].reshape(-1)
    for factor in torch.linspace(mse_scale_min, mse_scale_max, mse_steps,
                                 device=tensor.device, dtype=tensor.dtype):
        candidate_scale = base_scale * factor
        scale = _reshape(candidate_scale, tensor, axis)
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        for zero_point in range(4):
            codes = torch.where(scale > 0,
                                (tensor / safe + zero_point).round().clamp(0, 3),
                                torch.zeros_like(tensor)).to(torch.uint8)
            candidate = scale * (codes.to(tensor.dtype) - zero_point)
            error = ((matrix - _channel_view(candidate, channel_axis)[0]).double()
                     .square().mean(dim=1))
            better = error < best_error
            if not better.any():
                continue
            mask = _reshape(better, tensor, axis)
            best_error = torch.where(better, error, best_error)
            best_scale = torch.where(mask, scale, best_scale)
            best_zp = torch.where(mask, torch.full_like(best_zp, zero_point), best_zp)
            best_codes = torch.where(mask, codes, best_codes)
            best_t_min = torch.where(mask, torch.zeros_like(best_t_min), best_t_min)
            best_reconstructed = torch.where(mask, candidate, best_reconstructed)
    baseline.update(scale=best_scale, zero_point=best_zp,
                    quantized=best_codes, t_min=best_t_min)
    return baseline, best_reconstructed


def dequantize_variant(params: dict) -> torch.Tensor:
    """Reconstruct exactly from codes and the stored per-channel metadata."""
    codes = params["quantized"].to(params["scale"].dtype)
    scale = params["scale"]
    zero_point = params["zero_point"].to(scale.dtype)
    reconstructed = scale * (codes - zero_point)
    if "t_min" in params:
        reconstructed = torch.where(scale > 0, reconstructed, params["t_min"])
    return reconstructed
