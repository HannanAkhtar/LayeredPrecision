"""FP32 affine LP quantization with channel or row-local group granularity."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch

from layered_quantization import IntegerFormat


def _check(weight: torch.Tensor, bits: int, granularity: str, group_size: int):
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("Eligible weight must be a floating point matrix")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("Nonfinite weight values are not supported")
    if bits < 1 or bits > 16:
        raise ValueError("Affine bit width must be in [1, 16]")
    if granularity not in ("channel", "group") or group_size < 1:
        raise ValueError("Invalid granularity or group size")


def affine_quantize(weight: torch.Tensor, bits: int, granularity: str = "channel",
                    group_size: int = 128) -> tuple[dict, torch.Tensor]:
    """Reuse IntegerFormat's affine fit and rounding on FP32 row-local regions."""
    _check(weight, bits, granularity, group_size)
    x = weight.to(torch.float32)
    rows, cols = x.shape
    if granularity == "channel":
        params, decoded = IntegerFormat("channel", 0).quantize(x, bits)
        return {"bits": bits, "granularity": granularity, "group_size": cols,
                "shape": [rows, cols], "codes": params["quantized"],
                "scale": params["scale"].reshape(rows),
                "zero_point": params["zero_point"].reshape(rows),
                "t_min": params["t_min"].reshape(rows)}, decoded
    full, tail = divmod(cols, group_size)
    codes = torch.empty((rows, cols), dtype=torch.uint8 if bits <= 8 else torch.uint16)
    decoded = torch.empty_like(x)
    scales, zeros, minima = [], [], []
    if full:
        regions = x[:, :full * group_size].reshape(rows * full, group_size)
        p, d = IntegerFormat("channel", 0).quantize(regions, bits)
        codes[:, :full * group_size] = p["quantized"].reshape(rows, full * group_size)
        decoded[:, :full * group_size] = d.reshape(rows, full * group_size)
        scales.append(p["scale"].reshape(rows, full))
        zeros.append(p["zero_point"].reshape(rows, full))
        minima.append(p["t_min"].reshape(rows, full))
    if tail:
        p, d = IntegerFormat("channel", 0).quantize(x[:, full * group_size:], bits)
        codes[:, full * group_size:] = p["quantized"]
        decoded[:, full * group_size:] = d
        scales.append(p["scale"].reshape(rows, 1))
        zeros.append(p["zero_point"].reshape(rows, 1))
        minima.append(p["t_min"].reshape(rows, 1))
    return {"bits": bits, "granularity": granularity, "group_size": group_size,
            "shape": [rows, cols], "codes": codes,
            "scale": torch.cat(scales, 1), "zero_point": torch.cat(zeros, 1),
            "t_min": torch.cat(minima, 1)}, decoded


def affine_decode(params: dict) -> torch.Tensor:
    rows, cols = params["shape"]
    bits, granularity, group_size = (params[k] for k in ("bits", "granularity", "group_size"))
    if granularity not in ("channel", "group"):
        raise ValueError("Unknown granularity")
    groups = 1 if granularity == "channel" else math.ceil(cols / group_size)
    if tuple(params["scale"].shape) not in ((rows,), (rows, groups)):
        raise ValueError("Invalid scale shape")
    if tuple(params["codes"].shape) != (rows, cols):
        raise ValueError("Invalid code shape")
    checked_codes = params["codes"].to(torch.int32) if bits > 8 else params["codes"]
    if int(checked_codes.min()) < 0 or int(checked_codes.max()) > (1 << bits) - 1:
        raise ValueError("Code out of range")
    result = torch.empty((rows, cols), dtype=torch.float32)
    for g in range(groups):
        start = g * group_size
        stop = min(start + group_size, cols)
        if granularity == "channel":
            start, stop = 0, cols
        scale = params["scale"].reshape(rows, groups)[:, g:g+1].float()
        zero = params["zero_point"].reshape(rows, groups)[:, g:g+1].float()
        minimum = params["t_min"].reshape(rows, groups)[:, g:g+1].float()
        result[:, start:stop] = torch.where(scale > 0,
            scale * (params["codes"][:, start:stop].float() - zero), minimum)
    return result


def encode_layers(weight: torch.Tensor, bits: Iterable[int], granularity: str,
                  group_size: int = 128):
    original = weight.to(torch.float32)
    prefix = torch.zeros_like(original)
    for layer, width in enumerate(bits):
        residual = original - prefix if layer else original
        params, contribution = affine_quantize(residual, width, granularity, group_size)
        prefix = prefix + contribution if layer else contribution
        error = original - prefix
        residual_error = residual - contribution
        bf16_error = original - prefix.to(torch.bfloat16).float()
        signal = float(original.square().mean())
        mse = float(error.square().mean())
        yield params, {"layer": layer, "bits": width, "nominal_bits": None,
                       "parameter_mse_fp32": mse,
                       "parameter_mse_after_bf16_cast": float(bf16_error.square().mean()),
                       "sqnr_db_fp32": 10 * math.log10(signal / mse) if mse and signal else (None if not mse else 0.0),
                       "residual_energy_fp32": float(residual.square().mean()),
                       "residual_mse_fp32": float(residual_error.square().mean())}, prefix
