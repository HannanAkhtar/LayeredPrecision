"""FP32 affine LP quantization with channel or row-local group granularity."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from .rtn_reproduction import MODES, quantize_base

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
                "zero_point": params["zero_point"].reshape(rows)}, decoded
    full, tail = divmod(cols, group_size)
    codes = torch.empty((rows, cols), dtype=torch.uint8 if bits <= 8 else torch.uint16)
    decoded = torch.empty_like(x)
    scales, zeros = [], []
    if full:
        regions = x[:, :full * group_size].reshape(rows * full, group_size)
        p, d = IntegerFormat("channel", 0).quantize(regions, bits)
        codes[:, :full * group_size] = p["quantized"].reshape(rows, full * group_size)
        decoded[:, :full * group_size] = d.reshape(rows, full * group_size)
        scales.append(p["scale"].reshape(rows, full))
        zeros.append(p["zero_point"].reshape(rows, full))
    if tail:
        p, d = IntegerFormat("channel", 0).quantize(x[:, full * group_size:], bits)
        codes[:, full * group_size:] = p["quantized"]
        decoded[:, full * group_size:] = d
        scales.append(p["scale"].reshape(rows, 1))
        zeros.append(p["zero_point"].reshape(rows, 1))
    return {"bits": bits, "granularity": granularity, "group_size": group_size,
            "shape": [rows, cols], "codes": codes,
            "scale": torch.cat(scales, 1), "zero_point": torch.cat(zeros, 1)}, decoded


def _symmetric_regions(regions: torch.Tensor, bits: int):
    """Restricted-range signed symmetric RTN; exact zero is a representable level.

    q in [-Q,Q], Q=2^(bits-1)-1. Code q+2^(bits-1) is unsigned for
    compatibility with the existing bitstream. Code zero is unused.
    """
    offset = 1 << (bits - 1)
    qmax = offset - 1
    scale = regions.abs().amax(dim=1, keepdim=True) / qmax
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    signed = torch.round(regions / safe).clamp(-qmax, qmax)
    codes = (signed + offset).to(torch.uint8 if bits <= 8 else torch.uint16)
    return codes, signed * scale, scale


def symmetric_rtn_quantize(weight: torch.Tensor, bits: int, granularity: str = "group",
                           group_size: int = 128) -> tuple[dict, torch.Tensor]:
    """A calibration-free *symmetric* RTN control, not a verified RRQ reproduction.

    Unlike affine asymmetric quantization, the group zero point is a fixed
    implicit code offset and therefore is not serialized per group.
    At two bits this is a ternary grid {-s,0,+s} (one unused code).
    """
    _check(weight, bits, granularity, group_size)
    if bits < 2:
        raise ValueError("symmetric_rtn requires at least 2 bits")
    x = weight.to(torch.float32)
    rows, cols = x.shape
    codes = torch.empty((rows, cols), dtype=torch.uint8 if bits <= 8 else torch.uint16)
    decoded = torch.empty_like(x)
    if granularity == "channel":
        c, d, scale = _symmetric_regions(x, bits)
        codes.copy_(c)
        decoded.copy_(d)
        scales = scale.reshape(rows)
        group_size = cols
    else:
        full, tail = divmod(cols, group_size)
        groups = []
        if full:
            region = x[:, :full * group_size].reshape(rows * full, group_size)
            c, d, scale = _symmetric_regions(region, bits)
            codes[:, :full * group_size] = c.reshape(rows, full * group_size)
            decoded[:, :full * group_size] = d.reshape(rows, full * group_size)
            groups.append(scale.reshape(rows, full))
        if tail:
            c, d, scale = _symmetric_regions(x[:, full * group_size:], bits)
            codes[:, full * group_size:] = c
            decoded[:, full * group_size:] = d
            groups.append(scale.reshape(rows, 1))
        scales = torch.cat(groups, dim=1)
    return {"family": "symmetric_rtn", "bits": bits, "granularity": granularity,
            "group_size": group_size, "shape": [rows, cols], "codes": codes,
            "scale": scales}, decoded


AUTOROUND_RTN_SOURCE = "intel/auto-round@6c824f741be358983baba87aff83f9c5aa0a01e7:auto_round/data_type/int.py:quant_tensor_rtn_sym"


def autoround_rtn_sym_quantize(weight: torch.Tensor, bits: int, granularity: str = "group",
                               group_size: int = 128) -> tuple[dict, torch.Tensor]:
    """Plain AutoRound signed RTN, including its signed FP16 scale and zero padding.

    The stored code is signed_integer + 2**(bits-1); the offset is implicit.
    No group zero point is stored. The returned reconstruction has input dtype,
    matching the reference quantize/dequantize function.
    """
    _check(weight, bits, granularity, group_size)
    if bits < 2:
        raise ValueError("autoround_rtn_sym requires at least 2 bits")
    rows, cols = weight.shape
    size = cols if granularity == "channel" else group_size
    pad_len = (-cols) % size
    padded = F.pad(weight, (0, pad_len)) if pad_len else weight
    regions = padded.reshape(-1, size)
    maxq = 1 << (bits - 1)
    wmin = torch.clamp(regions.min(-1)[0], max=0)
    wmax = torch.clamp(regions.max(-1)[0], min=0)
    negative_abs = -wmin
    max_v = (2 * (wmax < negative_abs).int() - 1) * torch.maximum(wmax, negative_abs)
    scale = (max_v / maxq).to(torch.float16)
    scale = torch.where(scale < 0, torch.clamp(scale, max=-1e-5),
                        torch.clamp(scale, min=1e-5))
    signed = regions.div(scale.unsqueeze(-1)).round_().clamp_(-maxq, maxq - 1)
    # AutoRound multiplies the rounded values by the converted scale before
    # casting back to the input dtype. Keep that operation order.
    reconstructed = signed.clone().mul_(scale.unsqueeze(-1)).to(weight.dtype)
    codes = (signed.to(torch.int32) + maxq).to(torch.uint8 if bits <= 8 else torch.uint16)
    codes = codes.reshape(rows, -1)[:, :cols].contiguous()
    reconstructed = reconstructed.reshape(rows, -1)[:, :cols].contiguous()
    scale = scale.reshape(rows, -1)
    if granularity == "channel":
        scale = scale.reshape(rows)
    return {"family": "autoround_rtn_sym", "bits": bits, "granularity": granularity,
            "group_size": size, "shape": [rows, cols], "codes": codes,
            "scale": scale}, reconstructed


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
        family = params.get("family", "affine_asymmetric")
        if family in ("symmetric_rtn", "autoround_rtn_sym", *MODES):
            zero = float(1 << (bits - 1))
        elif family == "affine_asymmetric":
            zero = params["zero_point"].reshape(rows, groups)[:, g:g+1].float()
        else:
            raise ValueError(f"Unknown quantizer family {family}")
        contribution = scale * (params["codes"][:, start:stop].float() - zero)
        if "t_min" in params:  # Read artifacts written before zero-scale metadata was elided.
            minimum = params["t_min"].reshape(rows, groups)[:, g:g+1].float()
            contribution = torch.where(scale > 0, contribution, minimum)
        result[:, start:stop] = contribution
    return result


def encode_layers(weight: torch.Tensor, bits: Iterable[int], granularity: str,
                  group_size: int = 128, quantizer_family: str = "affine_asymmetric",
                  compute_device: str = "cpu"):
    original = weight.to(torch.float32)
    prefix = torch.zeros_like(original)
    for layer, width in enumerate(bits):
        residual = original - prefix if layer else original
        if quantizer_family == "symmetric_rtn":
            params, contribution = symmetric_rtn_quantize(residual, width, granularity, group_size)
        elif quantizer_family == "autoround_rtn_sym":
            params, contribution = autoround_rtn_sym_quantize(residual, width, granularity, group_size)
        elif quantizer_family in MODES:
            if layer or width != 2 or granularity != "group":
                raise ValueError("Phase 1 modes require a single 2-bit G128 base")
            params, contribution = quantize_base(weight, quantizer_family,
                                                  group_size=group_size,
                                                  compute_device=compute_device)
        elif quantizer_family == "affine_asymmetric":
            params, contribution = affine_quantize(residual, width, granularity, group_size)
        else:
            raise ValueError(f"Unknown quantizer family {quantizer_family}")
        prefix = prefix + contribution if layer else contribution
        error = original - prefix.float()
        residual_error = residual.float() - contribution.float()
        bf16_error = original - prefix.to(torch.bfloat16).float()
        signal = float(original.square().mean())
        mse = float(error.square().mean())
        metrics = {"layer": layer, "bits": width, "nominal_bits": None,
                       "parameter_mse_fp32": mse,
                       "parameter_mse_after_bf16_cast": float(bf16_error.square().mean()),
                       "sqnr_db_fp32": 10 * math.log10(signal / mse) if mse and signal else (None if not mse else 0.0),
                       "residual_energy_fp32": float(residual.square().mean()),
                       "residual_mse_fp32": float(residual_error.square().mean())}
        if quantizer_family in MODES:
            independent = affine_decode(params)
            before = independent - contribution.float()
            after = independent.to(contribution.dtype).to(torch.bfloat16).float() - contribution.to(torch.bfloat16).float()
            metrics.update(qdq_decode_max_abs_before_cast=float(before.abs().max()),
                           qdq_decode_rmse_before_cast=float(torch.sqrt(before.square().mean())),
                           qdq_decode_max_abs_after_bf16_cast=float(after.abs().max()),
                           qdq_decode_rmse_after_bf16_cast=float(torch.sqrt(after.square().mean())))
        yield params, metrics, prefix
