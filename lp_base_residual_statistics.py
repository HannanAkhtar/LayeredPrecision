"""Per-channel and stage-level FP32 weight / alternative-base residual statistics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from analyze_mse_layer_sensitivity import stage_of, write_json
from evaluate_lp_quantizer_adaptivity import load_torch, save_csv
from evaluate_imagenet import build_model
from analyze_mse_layer_sensitivity import eligible_groups

RAW_EDGES = np.linspace(-0.5, 0.5, 401)
STANDARD_EDGES = np.linspace(-8.0, 8.0, 401)
FAMILIES = ("A", "P", "U", "N", "L")


def histogram(values: torch.Tensor, edges: np.ndarray) -> tuple[np.ndarray, int, int]:
    low, high = float(edges[0]), float(edges[-1])
    below = int((values < low).sum())
    above = int((values > high).sum())
    counts = torch.histc(values.float().flatten(), bins=len(edges)-1, min=low, max=high)
    return counts.long().numpy(), below, above


def tensor_statistics(name: str, values: torch.Tensor) -> tuple[list[dict], dict]:
    matrix = values.reshape(values.shape[0], -1).double()
    channels, width = matrix.shape
    mean = matrix.mean(1)
    centered = matrix-mean[:, None]
    std = centered.square().mean(1).sqrt()
    rms = matrix.square().mean(1).sqrt()
    sorted_abs = matrix.abs().sort(1).values
    positions = [(width-1)*p for p in (.5, .9, .99, .999)]
    quantiles = []
    for position in positions:
        left, right = math.floor(position), math.ceil(position)
        quantiles.append(sorted_abs[:, left]*(right-position)+sorted_abs[:, right]*(position-left)
                         if left != right else sorted_abs[:, left])
    p50, p90, p99, p999 = quantiles
    scale = std.clamp_min(torch.finfo(torch.float64).tiny)
    skew = centered.pow(3).mean(1)/scale.pow(3)
    kurt = centered.pow(4).mean(1)/scale.pow(4)-3
    stats = {"mean": mean, "std": std, "rms": rms, "skewness": skew,
             "excess_kurtosis": kurt, "positive_fraction": (matrix > 0).double().mean(1),
             "negative_fraction": (matrix < 0).double().mean(1),
             "zero_fraction": (matrix == 0).double().mean(1),
             "abs_p50": p50, "abs_p90": p90, "abs_p99": p99, "abs_p999": p999,
             "tail_p99_p50": p99/p50.clamp_min(torch.finfo(torch.float64).tiny),
             "tail_p999_p99": p999/p99.clamp_min(torch.finfo(torch.float64).tiny),
             "dynamic_range": matrix.amax(1)-matrix.amin(1),
             "bias_to_rms": mean/rms.clamp_min(torch.finfo(torch.float64).tiny)}
    cpu = {key: value.numpy() for key, value in stats.items()}
    rows = []
    for channel in range(channels):
        row = {"name": name, "stage": stage_of(name), "channel": channel, "weights": width,
               "degenerate_std": bool(std[channel] == 0)}
        row.update({key: float(value[channel]) if np.isfinite(value[channel]) else None
                    for key, value in cpu.items()})
        rows.append(row)
    standardized = centered/scale[:, None]
    valid = (std > 0)[:, None].expand_as(matrix)
    raw, raw_below, raw_above = histogram(matrix, RAW_EDGES)
    normal, norm_below, norm_above = histogram(standardized[valid], STANDARD_EDGES)
    return rows, {"raw": raw, "raw_below": raw_below, "raw_above": raw_above,
                  "standardized": normal, "standard_below": norm_below,
                  "standard_above": norm_above,
                  "degenerate_values": int((~valid).sum())}


def summary(rows: list[dict], distribution: str, stage: str) -> dict:
    selected = [row for row in rows if stage == "all" or row["stage"] == stage]
    weights = np.array([row["weights"] for row in selected], dtype=np.float64)
    result = {"distribution": distribution, "stage": stage, "channels": len(selected),
              "weights": int(weights.sum()), "degenerate_channels": sum(row["degenerate_std"] for row in selected)}
    for key in ("mean", "std", "rms", "skewness", "excess_kurtosis", "positive_fraction",
                "negative_fraction", "zero_fraction", "abs_p50", "abs_p90", "abs_p99",
                "abs_p999", "tail_p99_p50", "tail_p999_p99", "dynamic_range", "bias_to_rms"):
        values = np.array([np.nan if row[key] is None else row[key] for row in selected])
        valid = np.isfinite(values)
        result[key + "_weighted_mean"] = float(np.average(values[valid], weights=weights[valid])) if valid.any() else None
        result[key + "_channel_median"] = float(np.median(values[valid])) if valid.any() else None
    result["pooled_rms"] = math.sqrt(sum(row["weights"]*row["rms"]**2 for row in selected)/weights.sum())
    scales = np.array([row["std"] for row in selected])
    result["channel_std_cv"] = float(scales.std()/scales.mean()) if scales.mean() else None
    return result


def add_hist(a: dict, b: dict) -> dict:
    if a is None:
        return {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in b.items()}
    for key, value in b.items():
        a[key] += value
    return a


def divergence(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    a = first / first.sum()
    b = second / second.sum()
    m = .5*(a+b)
    js = .5*(np.sum(a[a > 0]*np.log2(a[a > 0]/m[a > 0])) +
              np.sum(b[b > 0]*np.log2(b[b > 0]/m[b > 0])))
    tv = .5*np.abs(a-b).sum()
    return float(js), float(tv)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/lp_base_enhancement_study"))
    args = parser.parse_args()
    out = args.outdir.resolve()
    model = build_model("resnet50")
    eligible = {name for group in eligible_groups(model).values() for name in group}
    original = {name: parameter.detach().cpu() for name, parameter in model.named_parameters() if name in eligible}
    if len(original) != 54:
        raise AssertionError("Eligible tensor count changed")
    distributions = ["FP32"] + [f"R1_{family}" for family in FAMILIES]
    weights = {"FP32": original}
    for family in FAMILIES:
        base = load_torch(out / "decoded_prefixes" / f"B_{family}.pt")
        weights[f"R1_{family}"] = {name: original[name]-base[name] for name in original}
    histograms = {}
    all_channels = []
    global_rows, residual_rows = [], []
    for distribution in distributions:
        rows = []
        by_stage = {}
        for i, (name, tensor) in enumerate(weights[distribution].items(), 1):
            part, hist = tensor_statistics(name, tensor)
            rows.extend(part)
            by_stage[stage_of(name)] = add_hist(by_stage.get(stage_of(name)), hist)
            if i % 18 == 0:
                print(f"Statistics {distribution}: {i}/54", flush=True)
        all_channels.extend({"distribution": distribution, **row} for row in rows)
        by_stage["all"] = None
        for stage in tuple(key for key in by_stage if key != "all"):
            by_stage["all"] = add_hist(by_stage["all"], by_stage[stage])
        histograms[distribution] = by_stage
        target = global_rows if distribution == "FP32" else residual_rows
        for stage in ("all", "conv1", "layer1", "layer2", "layer3", "layer4", "fc"):
            target.append(summary(rows, distribution, stage))
    for row in residual_rows:
        first = histograms["FP32"][row["stage"]]
        second = histograms[row["distribution"]][row["stage"]]
        for label in ("raw", "standardized"):
            a = np.r_[first[label], first["raw_below" if label=="raw" else "standard_below"],
                      first["raw_above" if label=="raw" else "standard_above"]]
            b = np.r_[second[label], second["raw_below" if label=="raw" else "standard_below"],
                      second["raw_above" if label=="raw" else "standard_above"]]
            js, tv = divergence(a, b)
            row[label+"_js_bits"] = js
            row[label+"_tv"] = tv
        row["raw_overflow_fraction"] = (second["raw_below"]+second["raw_above"])/row["weights"]
        row["standard_overflow_fraction"] = (second["standard_below"]+second["standard_above"])/row["weights"]
    save_csv(out / "weight_distribution_statistics.csv", global_rows)
    save_csv(out / "base_residual_statistics.csv", residual_rows)
    save_csv(out / "distribution_channel_statistics.csv", all_channels)
    payload = {"raw_edges": RAW_EDGES, "standard_edges": STANDARD_EDGES}
    for distribution, stages in histograms.items():
        for stage, hist in stages.items():
            for name, value in hist.items():
                payload[f"{distribution}__{stage}__{name}"] = value
    np.savez_compressed(out / "distribution_histograms.npz", **payload)


if __name__ == "__main__":
    main()
