"""Channel, stage, and pooled statistics for recursively decoded LP residuals."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch

from analyze_mse_layer_sensitivity import STAGES, stage_of


ABS_QUANTILES = (0.50, 0.90, 0.99, 0.995, 0.999)
RAW_LOG_EDGES = np.linspace(-12, 0, 241)
NORMALIZED_EDGES = np.linspace(-8, 8, 241)


def _quantiles(sorted_values: torch.Tensor, probabilities=ABS_QUANTILES):
    count = sorted_values.shape[1]
    values = []
    for probability in probabilities:
        position = (count - 1)*probability
        left, right = math.floor(position), math.ceil(position)
        fraction = position-left
        values.append(sorted_values[:, left]*(1-fraction) + sorted_values[:, right]*fraction)
    return values


def _ratio(num: torch.Tensor, den: torch.Tensor):
    return torch.where(den > 0, num/den.clamp_min(torch.finfo(num.dtype).tiny),
                       torch.full_like(num, float("nan")))


def _optional(value):
    number = float(value)
    return number if math.isfinite(number) else None


@torch.no_grad()
def characterize_tensor(name: str, enhancement: int, residual: torch.Tensor,
                        affine_params: dict, clipped_params: dict):
    """Return one row per output channel plus mergeable raw/normalized histograms."""
    matrix = residual.reshape(residual.shape[0], -1)
    values = matrix.double()
    mean = values.mean(dim=1)
    centered = values - mean[:, None]
    variance = centered.square().mean(dim=1)
    std = variance.sqrt()
    rms = values.square().mean(dim=1).sqrt()
    mean_abs = values.abs().mean(dim=1)
    sorted_abs = values.abs().sort(dim=1).values
    p50, p90, p99, p995, p999 = _quantiles(sorted_abs)
    max_abs = sorted_abs[:, -1]
    third = centered.pow(3).mean(dim=1)
    fourth = centered.pow(4).mean(dim=1)
    skew = _ratio(third, std.pow(3))
    kurtosis = _ratio(fourth, std.pow(4)) - 3
    positive = (values > 0).double().mean(dim=1)
    negative = (values < 0).double().mean(dim=1)
    zero = (values == 0).double().mean(dim=1)
    low = values.amin(dim=1)
    high = values.amax(dim=1)
    positive_range = high.clamp_min(0)
    negative_range = (-low).clamp_min(0)
    affine_scale = affine_params["scale"].reshape(-1).double()
    clipped_scale = clipped_params["scale"].reshape(-1).double()
    zero_point = affine_params["zero_point"].reshape(-1)
    codes = affine_params["quantized"].reshape(matrix.shape)
    counts = torch.stack([(codes == code).sum(dim=1) for code in range(4)], dim=1)
    proportions = counts.double()/matrix.shape[1]
    entropy = -(torch.where(proportions > 0, proportions*proportions.clamp_min(1e-300).log2(), 0)).sum(dim=1)
    zero_code_fraction = proportions.gather(1, zero_point.long()[:, None]).flatten()
    stats = {"mean": mean, "std": std, "rms": rms, "mean_abs": mean_abs,
             "median_abs": p50, "max_abs": max_abs, "skewness": skew,
             "excess_kurtosis": kurtosis, "positive_fraction": positive,
             "negative_fraction": negative, "zero_fraction": zero,
             "normalized_mean": _ratio(mean, std), "abs_p50": p50,
             "abs_p90": p90, "abs_p99": p99, "abs_p995": p995,
             "abs_p999": p999, "tail_p99_p50": _ratio(p99, p50),
             "tail_max_p99": _ratio(max_abs, p99),
             "affine_scale": affine_scale, "affine_zero_point": zero_point.double(),
             "clipped_affine_scale": clipped_scale,
             "clipped_to_affine_scale": _ratio(clipped_scale, affine_scale),
             "absolute_range": high-low,
             "positive_negative_range_ratio": _ratio(positive_range, negative_range),
             "code_entropy": entropy, "extreme_code_fraction": proportions[:, 0]+proportions[:, 3],
             "zero_code_fraction": zero_code_fraction}
    cpu = {key: value.cpu().numpy() for key, value in stats.items()}
    count_cpu = counts.cpu().numpy()
    rows = []
    for channel in range(matrix.shape[0]):
        row = {"enhancement": enhancement, "name": name, "stage": stage_of(name),
               "channel": channel, "weights": matrix.shape[1],
               "degenerate_std": bool(std[channel] == 0)}
        row.update({key: _optional(values[channel]) for key, values in cpu.items()})
        row.update({f"code{code}_count": int(count_cpu[channel, code]) for code in range(4)})
        rows.append(row)
    raw_values = torch.log10(values.abs().clamp_min(1e-12)).flatten().float()
    raw_counts = torch.histc(raw_values, bins=len(RAW_LOG_EDGES)-1,
                             min=float(RAW_LOG_EDGES[0]), max=float(RAW_LOG_EDGES[-1])).cpu().numpy()
    normalized = (values/std.clamp_min(torch.finfo(torch.float64).tiny)[:, None])
    valid = (std > 0)[:, None].expand_as(values)
    normal_counts = torch.histc(normalized[valid].float(), bins=len(NORMALIZED_EDGES)-1,
                                min=float(NORMALIZED_EDGES[0]),
                                max=float(NORMALIZED_EDGES[-1])).cpu().numpy()
    return rows, raw_counts, normal_counts


def summarize_channels(rows: list[dict], enhancement: int, stage: str) -> dict:
    selected = [row for row in rows if row["enhancement"] == enhancement and
                (stage == "all" or row["stage"] == stage)]
    if not selected:
        raise ValueError(f"No channel statistics for E{enhancement}/{stage}")
    weights = np.array([row["weights"] for row in selected], dtype=np.float64)
    total = int(weights.sum())
    summary = {"enhancement": enhancement, "stage": stage,
               "channels": len(selected), "weights": total,
               "degenerate_channels": sum(row["degenerate_std"] for row in selected)}
    numeric_keys = [key for key in selected[0] if key not in
                    {"enhancement", "name", "stage", "channel", "weights", "degenerate_std"}
                    and not key.startswith("code") or key in ("code_entropy",)]
    # Each weighted mean is explicitly a weight-weighted mean of channel values.
    # Code occupancy also has a pooled (marginal) statistic below.
    for key in numeric_keys:
        values = np.array([np.nan if row[key] is None else row[key] for row in selected], dtype=np.float64)
        valid = np.isfinite(values)
        if valid.any():
            summary[f"{key}_weighted_mean"] = float(np.average(values[valid], weights=weights[valid]))
            summary[f"{key}_channel_median"] = float(np.median(values[valid]))
            summary[f"{key}_channel_q25"] = float(np.quantile(values[valid], .25))
            summary[f"{key}_channel_q75"] = float(np.quantile(values[valid], .75))
            summary[f"{key}_channel_p90"] = float(np.quantile(values[valid], .90))
        else:
            summary[f"{key}_weighted_mean"] = None
            summary[f"{key}_channel_median"] = None
            summary[f"{key}_channel_q25"] = None
            summary[f"{key}_channel_q75"] = None
            summary[f"{key}_channel_p90"] = None
    for code in range(4):
        count = sum(row[f"code{code}_count"] for row in selected)
        summary[f"code{code}_count"] = count
        summary[f"code{code}_fraction"] = count/total
    fractions = [summary[f"code{k}_fraction"] for k in range(4)]
    summary["marginal_code_entropy"] = -sum(p*math.log2(p) for p in fractions if p)
    summary["parameter_weighted_conditional_code_entropy"] = sum(
        row["weights"]*row["code_entropy"] for row in selected)/total
    summary["pooled_residual_rms"] = math.sqrt(sum(
        row["weights"]*row["rms"]**2 for row in selected)/total)
    scales = np.array([row["affine_scale"] for row in selected])
    summary["affine_scale_channel_cv"] = float(scales.std()/scales.mean()) if scales.mean() else None
    return summary


def summarize_all(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    globals_ = [summarize_channels(rows, level, "all") for level in (1, 2, 3)]
    stages = [summarize_channels(rows, level, stage)
              for level in (1, 2, 3) for stage in STAGES]
    return globals_, stages
