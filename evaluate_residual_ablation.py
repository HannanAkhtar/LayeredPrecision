#!/usr/bin/env python3
"""ResNet-50 first-residual LP quantizer ablation on labeled ImageNet validation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader

from evaluate_imagenet import (EXCLUDE_PATTERNS, MODEL_REGISTRY, build_dataset,
                               build_model, evaluate, restore_model)
from layered_quantization import IntegerFormat, LayeredQuantizer, parameter_is_quantized
from quantizer_variants import VARIANTS, dequantize_variant, quantize_variant


REFERENCES = {"fp32": 80.74, "base_5": 76.246,
              "affine_minmax": 79.996, "monolithic_7": 80.586}
CSV_FIELDS = ("method", "status", "evaluation_scope", "images", "top1", "top5",
              "cross_entropy", "residual_mse", "parameter_mse", "sqnr_db",
              "nominal_bits_per_quantized_weight", "stored_codes", "effective_levels",
              "calibration_seconds", "actual_storage_status", "reference_top1",
              "reference_delta_top1")


def inspect_dataset(dataset, full: bool) -> None:
    """Reject a label layout whose class indices cannot be verified."""
    classes = set(dataset.class_to_idx.values())
    if not dataset.classes or not all(name.isdigit() for name in dataset.classes):
        raise ValueError("ImageNet class folders must be numeric 0..999 labels; "
                         "an unlabeled test split cannot yield Top-1 accuracy")
    if full and (len(dataset) != 50_000 or classes != set(range(1000))):
        raise ValueError("Full evaluation needs 50,000 labeled images and numeric "
                         "class folders 0..999; use --max-batches for a smoke test")
    if full and set(Counter(dataset.targets).values()) != {50}:
        raise ValueError("Full ImageNet validation needs 50 labeled images per class")
    if not dataset.samples:
        raise ValueError("No labeled images found")


def prepare_base(original_state: dict[str, torch.Tensor], model,
                 device: torch.device) -> dict[str, tuple[torch.Tensor, torch.Tensor, dict]]:
    """Compute each 5-bit base and first residual once from original FP32."""
    integer = IntegerFormat("channel", 0)
    prepared = {}
    for name, parameter in model.named_parameters():
        if not parameter_is_quantized(name, parameter, EXCLUDE_PATTERNS, False):
            continue
        weight = original_state[name].to(device)
        params, base = integer.quantize(weight, 5)
        prepared[name] = (base.cpu(), (weight - base).cpu(), {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in params.items()
        })
    return prepared


def verify_affine_prefix(original_state, prepared, device) -> None:
    """Check codes, scales, zero points and decoded prefix against original LP."""
    names = list(prepared)
    selected = names[:1] + names[len(names)//2:len(names)//2+1] + names[-1:]
    for name in dict.fromkeys(selected):
        weight = original_state[name].to(device)
        base, residual, _ = prepared[name]
        old = LayeredQuantizer("channel", 0).quantize_layered(weight, [5, 2])
        old_base = old["layers_info"][0]["quant_params"]
        old_residual = old["layers_info"][1]["quant_params"]
        new_base, new_base_value = IntegerFormat("channel", 0).quantize(weight, 5)
        new_residual, new_value = quantize_variant(residual.to(device), "affine_minmax")
        for key in ("quantized", "scale", "zero_point"):
            if not torch.equal(new_base[key], old_base[key]):
                raise AssertionError(f"{name}: 5-bit {key} differs from original LP")
            if not torch.equal(new_residual[key], old_residual[key]):
                raise AssertionError(f"{name}: 2-bit {key} differs from original LP")
        if not torch.equal(new_base_value, base.to(device)) or not torch.equal(
                residual.to(device), weight - new_base_value):
            raise AssertionError(f"{name}: shared base or residual changed")
        if not torch.equal(base.to(device) + new_value,
                           LayeredQuantizer.reconstruct_from_layers(old["layers_info"])):
            raise AssertionError(f"{name}: LP prefix reconstruction differs")


def _stats_update(stats: dict, params: dict) -> None:
    scale = params["scale"].detach().float().reshape(-1)
    zp = params["zero_point"].detach().float().reshape(-1)
    stats["scale_min"] = min(stats["scale_min"], scale.min().item())
    stats["scale_max"] = max(stats["scale_max"], scale.max().item())
    stats["scale_sum"] += scale.double().sum().item()
    stats["zero_point_min"] = min(stats["zero_point_min"], zp.min().item())
    stats["zero_point_max"] = max(stats["zero_point_max"], zp.max().item())
    stats["zero_point_sum"] += zp.double().sum().item()
    stats["channel_count"] += scale.numel()
    if "clip_abs" in params:
        stats["clip_threshold_min"] = min(stats["clip_threshold_min"], scale.min().item())
        stats["clip_threshold_max"] = max(stats["clip_threshold_max"], scale.max().item())
    elif "clip_low" in params:
        stats["clip_threshold_min"] = min(
            stats["clip_threshold_min"], params["clip_low"].min().item())
        stats["clip_threshold_max"] = max(
            stats["clip_threshold_max"], params["clip_high"].max().item())


def _finish_stats(stats: dict) -> dict:
    count = stats["channel_count"]
    result = {"scale_min": stats["scale_min"], "scale_max": stats["scale_max"],
              "scale_mean": stats["scale_sum"] / count,
              "zero_point_min": stats["zero_point_min"],
              "zero_point_max": stats["zero_point_max"],
              "zero_point_mean": stats["zero_point_sum"] / count,
              "channel_count": count}
    if math.isfinite(stats["clip_threshold_min"]):
        result["clip_threshold_min"] = stats["clip_threshold_min"]
        result["clip_threshold_max"] = stats["clip_threshold_max"]
    return result


def reconstruct(model, original_state, prepared, method: str,
                device: torch.device, clip_percentile: float,
                mse_scale_min: float, mse_scale_max: float, mse_steps: int,
                outdir: Path) -> dict:
    restore_model(model, original_state, device)
    stats = dict(scale_min=float("inf"), scale_max=-float("inf"), scale_sum=0.0,
                 zero_point_min=float("inf"), zero_point_max=-float("inf"),
                 zero_point_sum=0.0, channel_count=0,
                 clip_threshold_min=float("inf"), clip_threshold_max=-float("inf"))
    residual_sse = parameter_sse = signal_ss = 0.0
    count = 0
    quantizer_artifact = {}
    calibration_seconds = 0.0
    for name, parameter in model.named_parameters():
        if name not in prepared:
            continue
        base, residual, _ = prepared[name]
        base = base.to(device)
        residual = residual.to(device)
        if method == "base_5":
            reconstructed = base
        elif method == "monolithic_7":
            _, reconstructed = IntegerFormat("channel", 0).quantize(
                original_state[name].to(device), 7)
        else:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            params, contribution = quantize_variant(
                residual, method, clip_percentile=clip_percentile,
                mse_scale_min=mse_scale_min, mse_scale_max=mse_scale_max,
                mse_steps=mse_steps)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            calibration_seconds += time.perf_counter() - start
            if not torch.equal(dequantize_variant(params), contribution):
                raise AssertionError(f"{name}: stored codes do not reconstruct")
            quantizer_artifact[name] = {
                key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                for key, value in params.items()
            }
            reconstructed = base + contribution
            residual_sse += (residual.double() - contribution.double()).square().sum().item()
            _stats_update(stats, params)
        original = original_state[name].to(device)
        parameter_sse += (original.double() - reconstructed.double()).square().sum().item()
        signal_ss += original.double().square().sum().item()
        count += original.numel()
        parameter.copy_(reconstructed)
    quantizer_stats = _finish_stats(stats) if stats["channel_count"] else None
    artifact_path = None
    if quantizer_artifact:
        artifact_path = outdir / f"{method}_codes_and_metadata.pt"
        torch.save(quantizer_artifact, artifact_path)
    return {"residual_mse": residual_sse / count if quantizer_stats else None,
            "parameter_mse": parameter_sse / count,
            "sqnr_db": 10 * math.log10(signal_ss / parameter_sse)
                       if parameter_sse > 0 and signal_ss > 0 else None,
            "stored_codes": count if quantizer_stats else None,
            "effective_levels": 3 if method.startswith("symmetric") else
                                (4 if quantizer_stats else None),
            "nominal_bits_per_quantized_weight":
                5 if method == "base_5" else 7,
            "calibration_seconds": calibration_seconds if quantizer_stats else None,
            "quantizer_stats": quantizer_stats,
            "quantizer_artifact": artifact_path.name if artifact_path else None}


def save_results(outdir: Path, metadata: dict, rows: list[dict]) -> None:
    payload = {"schema_version": 1, "metadata": metadata, "results": rows}
    (outdir / "residual_ablation.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    with (outdir / "residual_ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})


def plot_results(outdir: Path, rows: list[dict]) -> None:
    valid = [row for row in rows if row["status"] == "success"]
    if not valid:
        return
    for field, label in (("top1", "Top-1 accuracy (%)"),
                         ("cross_entropy", "Cross-entropy loss"),
                         ("residual_mse", "Residual MSE"),
                         ("sqnr_db", "Parameter SQNR (dB)")):
        selected = [row for row in valid if row.get(field) is not None]
        if not selected:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        values = [row[field] for row in selected]
        bars = ax.bar([row["method"] for row in selected], values)
        if field in {"top1", "cross_entropy", "sqnr_db"}:
            spread = max(values) - min(values)
            margin = max(spread * 0.08, 0.05 if field == "top1" else 0.01)
            ax.set_ylim(min(values) - margin, max(values) + margin * 2)
            ax.bar_label(bars, labels=[f"{value:.3f}" for value in values],
                         padding=3, fontsize=8)
        ax.set_ylabel(label + (" (zoomed axis)" if field in
                               {"top1", "cross_entropy", "sqnr_db"} else ""))
        ax.tick_params(axis="x", labelrotation=35)
        fig.tight_layout()
        fig.savefig(outdir / f"{field}.png", dpi=150)
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["resnet50"], default="resnet50")
    parser.add_argument("--data-dir", default="test")
    parser.add_argument("--outdir", default="outputs/residual_ablation")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--variants", nargs="+", default=["all"])
    parser.add_argument("--clip-percentile", type=float, default=99.5)
    parser.add_argument("--mse-scale-min", type=float, default=0.5)
    parser.add_argument("--mse-scale-max", type=float, default=1.5)
    parser.add_argument("--mse-steps", type=int, default=17)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or (args.max_batches is not None
            and args.max_batches <= 0):
        raise ValueError("batch size and max batches must be positive; workers nonnegative")
    variants = list(VARIANTS) if args.variants == ["all"] else args.variants
    if len(set(variants)) != len(variants) or any(v not in VARIANTS for v in variants):
        raise ValueError(f"--variants must be unique names from {VARIANTS} or all")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    if not any(path.is_dir() for path in data_dir.iterdir()):
        raise ValueError("No class folders found. The official unlabeled ImageNet "
                         "test split cannot produce Top-1 or Top-5 accuracy")
    weights = MODEL_REGISTRY[args.model][1]
    transform = weights.transforms()
    dataset = build_dataset(data_dir, transform)
    inspect_dataset(dataset, args.max_batches is None)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    metadata = {"model": args.model, "weights": str(weights),
                "lp_schedule": [5, 2, 2, 2], "quantized_prefix": [5, 2],
                "granularity": "channel", "axis": 0,
                "exclude_patterns": list(EXCLUDE_PATTERNS), "include_1d": False,
                "inference_dtype": "float32", "transform": str(transform),
                "data_dir": str(data_dir.resolve()), "dataset_size": len(dataset),
                "class_to_idx": dataset.class_to_idx,
                "batch_size": args.batch_size, "max_batches": args.max_batches,
                "device": str(device), "torch_version": str(torch.__version__),
                "torchvision_version": str(torchvision.__version__),
                "clip_percentile": args.clip_percentile,
                "mse_scale_grid": [args.mse_scale_min, args.mse_scale_max, args.mse_steps],
                "evaluation_scope": "full" if args.max_batches is None else "smoke",
                "base_artifact": "base_5_codes_and_metadata.pt",
                "storage_note": "Actual coded storage pending: signed codes and clipped/optimized "
                                "metadata need a compatible decoder; tensor containers are not counted."}
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    result_path = outdir / "residual_ablation.json"
    if args.resume and result_path.exists():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if previous.get("metadata") != metadata:
            raise ValueError("Saved run metadata differs; choose a new outdir")
        rows = previous["results"]
    else:
        rows = []
    model = build_model(args.model).to(device)
    original_state = {name: tensor.detach().cpu().clone()
                      for name, tensor in model.state_dict().items()}
    prepared = prepare_base(original_state, model, device)
    torch.save({name: values[2] for name, values in prepared.items()},
               outdir / "base_5_codes_and_metadata.pt")
    verify_affine_prefix(original_state, prepared, device)
    order = ["fp32", "base_5", "affine_minmax", "monolithic_7"] + [
        variant for variant in variants if variant != "affine_minmax"]
    for method in order:
        if method not in REFERENCES and args.max_batches is None:
            mismatches = [old for old in rows if old["method"] in REFERENCES and
                          old["reference_delta_top1"] is not None and
                          abs(old["reference_delta_top1"]) > 0.1]
            if mismatches:
                raise RuntimeError("Baseline reproduction differs by over 0.1 pp; "
                                   "investigate reference rows before ablation")
        if any(row["method"] == method and row["status"] == "success" and
               (not row.get("quantizer_artifact") or
                (outdir / row["quantizer_artifact"]).is_file()) for row in rows):
            continue
        print(f"\nEvaluating {method} ({metadata['evaluation_scope']})")
        if method == "fp32":
            restore_model(model, original_state, device)
            reconstruction = {}
        else:
            reconstruction = reconstruct(model, original_state, prepared, method, device,
                                         args.clip_percentile, args.mse_scale_min,
                                         args.mse_scale_max, args.mse_steps, outdir)
        metrics = evaluate(model, loader, device, args.max_batches)
        reference = REFERENCES.get(method) if args.max_batches is None else None
        row = {"method": method, "status": "success",
               "evaluation_scope": metadata["evaluation_scope"],
               "images": min(len(dataset), args.max_batches * args.batch_size)
                         if args.max_batches is not None else len(dataset),
               **metrics, **reconstruction,
               "settings": {"clip_percentile": args.clip_percentile}
                           if "percentile" in method else
                           {"mse_scale_grid": metadata["mse_scale_grid"]}
                           if method == "affine_mse" else {},
               "actual_storage_status": "pending", "reference_top1": reference,
               "reference_delta_top1": metrics["top1"] - reference
                                       if reference is not None else None}
        rows = [old for old in rows if old["method"] != method] + [row]
        save_results(outdir, metadata, rows)
        if reference is not None and abs(row["reference_delta_top1"]) > 0.1:
            print(f"Reference discrepancy for {method}: {row['reference_delta_top1']:+.3f} pp. "
                  "Check dataset mapping, checkpoint and transforms before interpretation.")
        if method == "monolithic_7" and args.max_batches is None:
            mismatches = [old for old in rows if old["method"] in REFERENCES and
                          old["reference_delta_top1"] is not None and
                          abs(old["reference_delta_top1"]) > 0.1]
            if mismatches:
                plot_results(outdir, rows)
                raise RuntimeError("Baseline reproduction differs by over 0.1 pp; "
                                   "investigate reference rows before ablation")
    plot_results(outdir, rows)
    print(f"Saved {result_path} and {outdir / 'residual_ablation.csv'}")


if __name__ == "__main__":
    main()
