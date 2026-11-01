#!/usr/bin/env python3
"""Reproduce the paper's full-resolution ImageNet PTQ experiment."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path
from collections.abc import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.models as tvm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from bitplane import build_bitplane_database, apply_bitplane_reconstruction
from layered_quantization import LayeredQuantizer, apply_partial_reconstruction


MODEL_REGISTRY = {
    "efficientnet_b2": (tvm.efficientnet_b2, tvm.EfficientNet_B2_Weights.IMAGENET1K_V1),
    "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V2),
    "vit_b_16": (tvm.vit_b_16, tvm.ViT_B_16_Weights.IMAGENET1K_V1),
    "convnext_base": (tvm.convnext_base, tvm.ConvNeXt_Base_Weights.IMAGENET1K_V1),
    "vgg16": (tvm.vgg16, tvm.VGG16_Weights.IMAGENET1K_V1),
    "swin_v2_b": (tvm.swin_v2_b, tvm.Swin_V2_B_Weights.IMAGENET1K_V1),
}

PAPER_CONFIGS = {
    "efficientnet_b2": {"monolithic": [5, 7, 9, 11], "layered": [5, 2, 2, 2]},
    "resnet50": {"monolithic": [5, 7, 9, 11], "layered": [5, 2, 2, 2]},
    "vit_b_16": {"monolithic": [5, 7, 9, 11], "layered": [5, 2, 2, 2]},
    "convnext_base": {"monolithic": [4, 6, 8, 10], "layered": [4, 2, 2, 2]},
    "vgg16": {"monolithic": [4, 6, 8, 10], "layered": [4, 2, 2, 2]},
    "swin_v2_b": {"monolithic": [4, 6, 8, 10], "layered": [4, 2, 2, 2]},
}

RESULT_TYPES = ("Monolithic", "Layered Precision", "Bit-Plane Truncation")
METHOD_TO_RESULT = {
    "monolithic": "Monolithic",
    "lp": "Layered Precision",
    "bitplane": "Bit-Plane Truncation",
}
EXCLUDE_PATTERNS = ("norm", "bn")
CSV_FIELDS = [
    "model", "config", "type", "total_bits", "bits", "formats",
    "parent_config", "status", "coverage", "int_granularity",
    "int_channel_axis", "int_axis_overhead_threshold", "int_use_zero_point",
    "int_selection_summary", "source_bits", "retained_bits", "dropped_bits",
    "decode_mode", "top1", "top5", "cross_entropy", "error",
]


class NumericImageFolder(ImageFolder):
    """Map integer-named class directories directly to their integer labels."""

    def find_classes(self, directory: str):
        classes = sorted(
            (entry.name for entry in os.scandir(directory) if entry.is_dir()),
            key=int,
        )
        return classes, {name: int(name) for name in classes}


def build_model(model_name: str) -> nn.Module:
    factory, weights = MODEL_REGISTRY[model_name]
    return factory(weights=weights).eval()


def official_metrics(model_name: str) -> dict:
    weights = MODEL_REGISTRY[model_name][1]
    metrics = weights.meta.get("_metrics", {}).get("ImageNet-1K", {})
    return {
        "official_top1": metrics.get("acc@1"),
        "official_top5": metrics.get("acc@5"),
    }


def build_dataset(data_dir: Path, transform):
    class_dirs = [entry.name for entry in os.scandir(data_dir) if entry.is_dir()]
    dataset_cls = (
        NumericImageFolder
        if class_dirs and all(name.isdigit() for name in class_dirs)
        else ImageFolder
    )
    dataset = dataset_cls(str(data_dir), transform=transform)
    labels = set(dataset.class_to_idx.values())
    if labels and (min(labels) < 0 or max(labels) >= 1000):
        raise ValueError(
            "ImageNet labels must be integer class indices in [0, 999]"
        )
    print(f"   Label mapping: {dataset_cls.__name__}")
    return dataset


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> dict:
    model.eval()
    correct1 = correct5 = total = 0
    loss_sum = 0.0
    limit = len(loader) if max_batches is None else min(len(loader), max_batches)
    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += nn.functional.cross_entropy(
            logits, labels, reduction="sum"
        ).item()
        correct1 += (logits.argmax(dim=1) == labels).sum().item()
        top5 = logits.topk(5, dim=1).indices
        correct5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.numel()
        print(
            f"   [{batch_idx + 1}/{limit}] "
            f"top1={100 * correct1 / total:.2f}%  "
            f"top5={100 * correct5 / total:.2f}%  ce={loss_sum / total:.4f}",
            end="\r",
            flush=True,
        )
    print()
    if total == 0:
        raise RuntimeError("The evaluation loader produced no images")
    return {
        "top1": round(100 * correct1 / total, 4),
        "top5": round(100 * correct5 / total, 4),
        "cross_entropy": round(loss_sum / total, 6),
    }


def empty_results() -> dict[str, list[dict]]:
    return {result_type: [] for result_type in RESULT_TYPES}


def result_key(row: dict) -> tuple:
    return tuple(row.get("bits", [])), tuple(row.get("formats", []))


def upsert(results: dict[str, list[dict]], result_type: str, row: dict) -> None:
    key = result_key(row)
    results[result_type] = [
        old for old in results[result_type] if result_key(old) != key
    ] + [row]


def completed_keys(rows: Iterable[dict]) -> set:
    return {
        result_key(row) for row in rows if row.get("status") == "success"
    }


def selection_summary(parameter_count: int, layers: int = 1) -> dict:
    total = parameter_count * layers
    return {
        "counts": {"channel:axis=0": total},
        "total_int_quantizers": total,
    }


def successful_metrics(metrics: dict, coverage: float, parameter_count: int, layers: int) -> dict:
    return {
        **metrics,
        "coverage": coverage,
        "status": "success",
        "int_granularity": ["channel"],
        "int_channel_axis": 0,
        "int_axis_overhead_threshold": 0.02,
        "int_use_zero_point": True,
        "int_selection_summary": selection_summary(parameter_count, layers),
    }


def restore_model(model: nn.Module, state: dict[str, torch.Tensor], device: torch.device) -> None:
    model.load_state_dict(
        {name: tensor.to(device) for name, tensor in state.items()},
        strict=True,
    )


def run_metadata(
    model_name: str,
    data_dir: Path,
    transform,
    dataset_size: int,
    methods: Sequence[str],
    max_batches: int | None,
) -> dict:
    config = PAPER_CONFIGS[model_name]
    return {
        "schema_version": 1,
        "release": "paper_release",
        "model": model_name,
        "weights": str(MODEL_REGISTRY[model_name][1]),
        "data_dir": str(data_dir.resolve()),
        "dataset_size": dataset_size,
        "transform": str(transform),
        "methods": list(methods),
        "monolithic_bits": list(config["monolithic"]),
        "layered_bits": list(config["layered"]),
        "bitplane_source_bits": sum(config["layered"]),
        "bitplane_target_bits": list(config["monolithic"]),
        "exclude_patterns": list(EXCLUDE_PATTERNS),
        "include_1d_parameters": False,
        "int_granularity": "channel",
        "int_channel_axis": 0,
        "int_use_zero_point": True,
        "fold_bn_before_quant": False,
        "bn_calibration_batches": 0,
        "max_batches": max_batches,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
    }


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, path)


def csv_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def row_for_csv(model_name: str, result_type: str, row: dict) -> dict:
    output = {field: "" for field in CSV_FIELDS}
    output.update({
        "model": model_name,
        "type": result_type,
        **row,
        "bits": ",".join(map(str, row.get("bits", []))),
        "formats": ",".join(row.get("formats", [])),
    })
    return {field: csv_value(output.get(field, "")) for field in CSV_FIELDS}


def baseline_csv_row(model_name: str, baseline: dict) -> dict:
    row = {field: "" for field in CSV_FIELDS}
    row.update({
        "model": model_name,
        "config": "fp32",
        "type": "Baseline",
        "total_bits": 32,
        "formats": "fp32",
        "status": "success",
        "top1": baseline.get("top1", ""),
        "top5": baseline.get("top5", ""),
        "cross_entropy": baseline.get("cross_entropy", ""),
    })
    return row


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def save_model(
    outdir: Path,
    model_name: str,
    baseline: dict,
    results: dict[str, list[dict]],
    metadata: dict,
    plots: bool = False,
) -> None:
    model_dir = outdir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(model_dir / "quantization_results.json", {
        "base_model": baseline,
        "structured_results": results,
        "run_metadata": metadata,
    })
    rows = [baseline_csv_row(model_name, baseline)]
    for result_type in RESULT_TYPES:
        rows.extend(
            row_for_csv(model_name, result_type, row)
            for row in results[result_type]
        )
    write_csv(model_dir / "quantization_results.csv", rows)
    if plots:
        plot_model(model_dir, model_name, baseline, results)


def load_resume(outdir: Path, model_name: str, metadata: dict):
    path = outdir / model_name / "quantization_results.json"
    if not path.exists():
        return {}, empty_results()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("run_metadata") != metadata:
        raise RuntimeError(
            f"Cannot resume {model_name}: metadata in {path} does not match "
            "this run. Use a new output directory for a different experiment."
        )
    results = empty_results()
    for result_type in RESULT_TYPES:
        results[result_type] = list(
            payload.get("structured_results", {}).get(result_type, [])
        )
    return payload.get("base_model", {}), results


def plot_model(
    model_dir: Path,
    model_name: str,
    baseline: dict,
    results: dict[str, list[dict]],
) -> None:
    labels = {
        "top1": "Top-1 Accuracy (%)",
        "top5": "Top-5 Accuracy (%)",
        "cross_entropy": "Cross-Entropy Loss",
    }
    markers = {
        "Monolithic": "o",
        "Layered Precision": "s",
        "Bit-Plane Truncation": "^",
    }
    colors = {
        "Monolithic": "#2878B5",
        "Layered Precision": "#F28E2B",
        "Bit-Plane Truncation": "#3A923A",
    }
    for metric, ylabel in labels.items():
        fig, axis = plt.subplots(figsize=(7, 4))
        for result_type in RESULT_TYPES:
            points = sorted(
                (int(row["total_bits"]), float(row[metric]))
                for row in results[result_type]
                if row.get("status") == "success" and row.get(metric) is not None
            )
            if points:
                xs, ys = zip(*points)
                axis.plot(
                    xs,
                    ys,
                    marker=markers[result_type],
                    color=colors[result_type],
                    linewidth=1.6,
                    label=result_type,
                )
        if metric in baseline:
            axis.axhline(
                baseline[metric], color="gray", linestyle=":", label="FP32"
            )
        axis.set_xlabel("Total bits")
        axis.set_ylabel(ylabel)
        axis.set_title(f"{model_name}: {ylabel}")
        axis.grid(alpha=0.3)
        axis.legend()
        fig.tight_layout()
        fig.savefig(model_dir / f"plot_{metric}.png", dpi=300)
        plt.close(fig)


def save_combined(outdir: Path, all_results: dict[str, tuple[dict, dict]]) -> None:
    rows: list[dict] = []
    for model_name, (baseline, results) in all_results.items():
        rows.append(baseline_csv_row(model_name, baseline))
        for result_type in RESULT_TYPES:
            rows.extend(
                row_for_csv(model_name, result_type, row)
                for row in results[result_type]
            )
    write_csv(outdir / "all_models_results.csv", rows)


def evaluate_one_model(
    model_name: str,
    loader: DataLoader,
    device: torch.device,
    outdir: Path,
    metadata: dict,
    methods: Sequence[str],
    max_batches: int | None,
    resume: bool,
) -> tuple[dict, dict]:
    print(f"\n{'=' * 64}\nModel: {model_name}\n{'=' * 64}")
    model = build_model(model_name).to(device)
    original_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    baseline, results = (
        load_resume(outdir, model_name, metadata)
        if resume else ({}, empty_results())
    )

    def checkpoint():
        save_model(outdir, model_name, baseline, results, metadata, plots=False)

    if all(key in baseline for key in ("top1", "top5", "cross_entropy")):
        print("\nFP32 baseline ... skipped (checkpoint)")
    else:
        print("\nFP32 baseline ...")
        baseline = evaluate(model, loader, device, max_batches)
        baseline.update({
            "weights": str(MODEL_REGISTRY[model_name][1]),
            **{
                key: value
                for key, value in official_metrics(model_name).items()
                if value is not None
            },
        })
        checkpoint()
    print(
        f"   Top-1: {baseline['top1']:.2f}%  Top-5: {baseline['top5']:.2f}%  "
        f"CE: {baseline['cross_entropy']:.4f}"
    )

    config = PAPER_CONFIGS[model_name]
    if "monolithic" in methods:
        done = completed_keys(results["Monolithic"])
        print("\n--- Monolithic ---")
        for bits in config["monolithic"]:
            key = ((bits,), ("int",))
            if key in done:
                print(f"Monolithic {bits} ... skipped (checkpoint)")
                continue
            print(f"Monolithic {bits} ...")
            restore_model(model, original_state, device)
            try:
                quantizer = LayeredQuantizer("channel", 0)
                _, info = quantizer.apply_layered_quantization_to_model(
                    model,
                    [bits],
                    ["int"],
                    EXCLUDE_PATTERNS,
                    include_1d_parameters=False,
                    inplace=True,
                )
                parameter_count = len(info["per_parameter_info"])
                metrics = evaluate(model, loader, device, max_batches)
                row = {
                    "config": str(bits),
                    "bits": [bits],
                    "formats": ["int"],
                    "total_bits": bits,
                    **successful_metrics(
                        metrics, float(info["coverage"]), parameter_count, 1
                    ),
                }
            except Exception as exc:
                row = {
                    "config": str(bits),
                    "bits": [bits],
                    "formats": ["int"],
                    "total_bits": bits,
                    "status": "failed",
                    "error": str(exc),
                }
                print(f"   Failed: {exc}")
            upsert(results, "Monolithic", row)
            checkpoint()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if "lp" in methods:
        bits_path = config["layered"]
        done = completed_keys(results["Layered Precision"])
        expected = [
            (tuple(bits_path[:depth]), tuple(["int"] * depth))
            for depth in range(1, len(bits_path) + 1)
        ]
        print("\n--- Layered Precision ---")
        if all(key in done for key in expected):
            print("LP path ... skipped (checkpoint)")
        else:
            restore_model(model, original_state, device)
            try:
                quantizer = LayeredQuantizer("channel", 0)
                _, info = quantizer.apply_layered_quantization_to_model(
                    model,
                    bits_path,
                    ["int"] * len(bits_path),
                    EXCLUDE_PATTERNS,
                    include_1d_parameters=False,
                    inplace=True,
                )
                database = info["per_parameter_info"]
                parameter_count = len(database)
                for depth in range(1, len(bits_path) + 1):
                    prefix = bits_path[:depth]
                    key = (tuple(prefix), tuple(["int"] * depth))
                    if key in done:
                        print(f"LP {'-'.join(map(str, prefix))} ... skipped")
                        continue
                    label = "-".join(map(str, prefix))
                    print(f"LP {label} ...")
                    apply_partial_reconstruction(model, database, depth)
                    metrics = evaluate(model, loader, device, max_batches)
                    row = {
                        "config": label,
                        "bits": prefix,
                        "formats": ["int"] * depth,
                        "total_bits": sum(prefix),
                        "parent_config": "-".join(map(str, bits_path)),
                        **successful_metrics(
                            metrics,
                            float(info["coverage"]),
                            parameter_count,
                            depth,
                        ),
                    }
                    upsert(results, "Layered Precision", row)
                    checkpoint()
            except Exception as exc:
                label = "-".join(map(str, bits_path))
                upsert(results, "Layered Precision", {
                    "config": label,
                    "bits": bits_path,
                    "formats": ["int"] * len(bits_path),
                    "total_bits": sum(bits_path),
                    "status": "failed",
                    "error": str(exc),
                })
                checkpoint()
                print(f"   Failed: {exc}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if "bitplane" in methods:
        source_bits = sum(config["layered"])
        targets = config["monolithic"]
        done = completed_keys(results["Bit-Plane Truncation"])
        print("\n--- Bit-Plane Truncation ---")
        needed = [((source_bits, target), ("bitplane_shift",)) for target in targets]
        if all(key in done for key in needed):
            print("Bit-plane path ... skipped (checkpoint)")
        else:
            restore_model(model, original_state, device)
            database = build_bitplane_database(
                model,
                source_bits,
                "channel",
                0,
                EXCLUDE_PATTERNS,
                include_1d_parameters=False,
            )
            parameter_count = len(database["per_parameter_info"])
            for target_bits in targets:
                key = ((source_bits, target_bits), ("bitplane_shift",))
                if key in done:
                    print(f"Bit-plane {source_bits}->{target_bits} ... skipped")
                    continue
                label = f"{source_bits}->{target_bits}"
                print(f"Bit-plane {label} ...")
                try:
                    info = apply_bitplane_reconstruction(
                        model, database, target_bits
                    )
                    metrics = evaluate(model, loader, device, max_batches)
                    row = {
                        "config": label,
                        "bits": [source_bits, target_bits],
                        "formats": ["bitplane_shift"],
                        "total_bits": target_bits,
                        "source_bits": source_bits,
                        "retained_bits": target_bits,
                        "dropped_bits": source_bits - target_bits,
                        "decode_mode": "shift",
                        **successful_metrics(
                            metrics,
                            float(info["coverage"]),
                            parameter_count,
                            1,
                        ),
                    }
                except Exception as exc:
                    row = {
                        "config": label,
                        "bits": [source_bits, target_bits],
                        "formats": ["bitplane_shift"],
                        "total_bits": target_bits,
                        "source_bits": source_bits,
                        "retained_bits": target_bits,
                        "dropped_bits": source_bits - target_bits,
                        "decode_mode": "shift",
                        "status": "failed",
                        "error": str(exc),
                    }
                    print(f"   Failed: {exc}")
                upsert(results, "Bit-Plane Truncation", row)
                checkpoint()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    save_model(outdir, model_name, baseline, results, metadata, plots=True)
    del model, original_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return baseline, results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-resolution ImageNet evaluation for the paper presets."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="ImageNet validation directory in ImageFolder layout",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help=f"Subset of {list(MODEL_REGISTRY)} or 'all'",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHOD_TO_RESULT),
        default=list(METHOD_TO_RESULT),
        help="Methods to evaluate; default runs all reported comparisons",
    )
    parser.add_argument("--outdir", default="outputs/imagenet")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional smoke-test limit; omit for paper results",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only when saved run metadata matches exactly",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"ImageNet directory not found: {data_dir}")
    model_names = list(MODEL_REGISTRY) if args.models == ["all"] else args.models
    unknown = [name for name in model_names if name not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown models {unknown}; available: {list(MODEL_REGISTRY)}")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data: {data_dir.resolve()}")
    print(f"Models: {model_names}")
    print(f"Methods: {args.methods}")
    print("Quantizer: channel-wise affine INT, axis=0, explicit zero point")

    all_results: dict[str, tuple[dict, dict]] = {}
    for model_name in model_names:
        weights = MODEL_REGISTRY[model_name][1]
        transform = weights.transforms()
        print(f"\nBuilding {model_name} dataset with {weights} transforms")
        print(f"   {transform}")
        dataset = build_dataset(data_dir, transform)
        if args.max_batches is None and (
            len(dataset) != 50_000 or len(dataset.classes) != 1_000
        ):
            raise RuntimeError(
                "Paper evaluation requires the complete ImageNet validation "
                f"set (50,000 images, 1,000 classes); found "
                f"{len(dataset):,} images and {len(dataset.classes):,} classes. "
                "Use --max-batches only for an explicitly non-paper smoke test."
            )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        metadata = run_metadata(
            model_name,
            data_dir,
            transform,
            len(dataset),
            args.methods,
            args.max_batches,
        )
        baseline, results = evaluate_one_model(
            model_name,
            loader,
            device,
            outdir,
            metadata,
            args.methods,
            args.max_batches,
            args.resume,
        )
        all_results[model_name] = (baseline, results)
        save_combined(outdir, all_results)
        del loader, dataset
        gc.collect()

    save_combined(outdir, all_results)
    print(f"\nSaved results under {outdir}")


if __name__ == "__main__":
    main()
