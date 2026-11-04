"""Pinned AutoRound 6c824f7 plain and optimized symmetric RTN experiments.

The two algorithms mirror quant_tensor_rtn_sym and quant_tensor_opt_rtn_sym
from auto_round/data_type/int.py. The quantizer is deliberately limited to
one 2-bit G128 base; no existing LP quantizer uses this module.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


AUTOROUND_COMMIT = "6c824f741be358983baba87aff83f9c5aa0a01e7"
MODES = {
    "autoround_plain_fp16": ("plain", torch.float16),
    "autoround_plain_bf16": ("plain", torch.bfloat16),
    "autoround_opt_fp16": ("optimized_search_scales", torch.float16),
}


def _reciprocal(tensor: torch.Tensor) -> torch.Tensor:
    """Pinned auto_round.utils.common.get_reciprocal."""
    eps = 1e-5 if tensor.dtype == torch.float16 else 1e-30
    safe = tensor.abs() >= eps
    safe_tensor = torch.where(safe, tensor, torch.ones_like(tensor))
    return torch.where(safe, 1.0 / safe_tensor, torch.zeros_like(tensor))


def _search_scales(data: torch.Tensor, bits: int) -> torch.Tensor:
    """Pinned int.py search_scales, with qw=1 and the 2-bit search grid."""
    nmax = int(2.0 ** (bits - 1))
    imax = torch.abs(data).argmax(dim=-1, keepdim=True)
    group_max = torch.take_along_dim(data, imax, dim=-1)
    iscales = -nmax * _reciprocal(group_max)
    scales = _reciprocal(iscales)
    levels = torch.empty_like(data)
    torch.round(iscales * data, out=levels)
    levels.clamp_(-nmax, nmax - 1)
    best_loss = ((scales * levels - data).to(torch.float32)) ** 2
    best_loss = torch.sum(best_loss, dim=-1)
    search_min, step = 90, 0.01
    for index in range(-search_min, search_min + 1):
        if index == 0:
            continue
        candidate_inverse = -(nmax - step * index) * _reciprocal(group_max)
        candidate_levels = torch.empty_like(data)
        torch.round(candidate_inverse * data, out=candidate_levels)
        candidate_levels.clamp_(-nmax, nmax - 1)
        candidate_scales = _reciprocal(candidate_inverse)
        loss = ((candidate_scales * candidate_levels - data).to(torch.float32)) ** 2
        loss = torch.sum(loss, dim=-1)
        replace = loss < best_loss
        if replace.any():
            scales[replace] = candidate_scales[replace]
            best_loss[replace] = loss[replace]
    return scales


def quantize_base(weight: torch.Tensor, family: str, *, group_size: int = 128,
                  compute_device: str = "cpu", chunk_groups: int = 32768) -> tuple[dict, torch.Tensor]:
    if family not in MODES:
        raise ValueError(f"Unknown RTN reproduction mode: {family}")
    if weight.ndim != 2 or not weight.is_floating_point() or not bool(torch.isfinite(weight).all()):
        raise ValueError("Expected finite floating point weight matrix")
    if group_size != 128 or chunk_groups < 1:
        raise ValueError("Phase 1 requires G128 and a positive chunk size")
    algorithm, dtype = MODES[family]
    rows, cols = weight.shape
    pad_len = (-cols) % group_size
    input_weight = weight.to(dtype)
    padded = F.pad(input_weight, (0, pad_len)) if pad_len else input_weight
    regions = padded.reshape(-1, group_size)
    code_parts, scale_parts, qdq_parts = [], [], []
    for start in range(0, regions.shape[0], chunk_groups):
        group = regions[start:start + chunk_groups].to(compute_device)
        maxq = 2
        if algorithm == "plain":
            wmin = torch.clamp(group.min(-1)[0], max=0)
            wmax = torch.clamp(group.max(-1)[0], min=0)
            wmin_abs = -(wmin * 1.0)
            wmax_abs = wmax * 1.0
            max_v = (2 * (wmax_abs < wmin_abs).int() - 1) * torch.max(wmax_abs, wmin_abs)
            scale = (max_v / maxq).to(torch.float16)
        else:
            scale = _search_scales(group, 2).squeeze(-1)
        scale = torch.where(scale < 0, torch.clamp(scale, max=-1e-5),
                            torch.clamp(scale, min=1e-5))
        signed = group.div(scale.unsqueeze(-1)).round_().clamp_(-maxq, maxq - 1)
        qdq = signed.clone().mul_(scale.unsqueeze(-1)).to(dtype)
        code_parts.append((signed.to(torch.int16) + maxq).to(torch.uint8).cpu())
        scale_parts.append(scale.cpu())
        qdq_parts.append(qdq.cpu())
    codes = torch.cat(code_parts).reshape(rows, -1)[:, :cols].contiguous()
    scales = torch.cat(scale_parts).reshape(rows, -1).contiguous()
    reconstructed = torch.cat(qdq_parts).reshape(rows, -1)[:, :cols].contiguous()
    return {"family": family, "bits": 2, "granularity": "group", "group_size": 128,
            "shape": [rows, cols], "codes": codes, "scale": scales,
            "qdq_dtype": str(dtype).removeprefix("torch."), "algorithm": algorithm,
            "compute_device": compute_device}, reconstructed
