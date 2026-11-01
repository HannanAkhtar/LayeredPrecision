#!/usr/bin/env python3
"""Evaluate monolithic and Layered Precision PTQ over NGP checkpoints."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from layered_quantization import LayeredQuantizer, apply_partial_reconstruction
from ngp_data import load_audio, load_image, resolve_source_path
from ngp_metrics import log_spectral_distance, psnr, si_snr, snr, ssim
from ngp_models import build_ngp_model


EXCLUDE_PATTERNS = ("bias", "norm")


def parse_bits_csv(value: str) -> list[int]:
    try:
        bits = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected comma-separated integer bit widths, got {value!r}"
        ) from exc
    if not bits or any(bit <= 0 for bit in bits):
        raise argparse.ArgumentTypeError(
            "Bit widths must be a non-empty comma-separated list of positive integers"
        )
    return bits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate monolithic and Layered Precision PTQ on NGP datasets."
    )
    parser.add_argument("--checkpoints-root", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument(
        "--data-root",
        default=None,
        help="Relocated DIV2K or LibriSpeech root when checkpoint paths moved",
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--monolithic-bits",
        type=parse_bits_csv,
        required=True,
        metavar="BITS",
        help="Comma-separated monolithic widths, for example 6,7,8,9,10,12,14",
    )
    parser.add_argument(
        "--layered-bits",
        type=parse_bits_csv,
        required=True,
        metavar="BITS",
        help="Comma-separated LP base and residual widths, for example 8,2,2,2",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("monolithic", "lp"),
        default=("monolithic", "lp"),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def find_checkpoints(root: Path, manifest: Path | None) -> list[Path]:
    if manifest is None:
        return sorted(root.rglob("best_model.pth"))
    with manifest.open("r", encoding="utf-8") as handle:
        rows = json.load(handle).get("samples", [])
    checkpoints = []
    for row in rows:
        saved = Path(row.get("checkpoint", ""))
        candidates = [saved, root / row.get("sample_id", "") / "best_model.pth"]
        resolved = next((path for path in candidates if path.is_file()), None)
        if resolved is not None and resolved not in checkpoints:
            checkpoints.append(resolved)
    return checkpoints


def modality_metrics(modality: str) -> list[str]:
    return ["mse", "psnr_db", "ssim"] if modality == "image" else [
        "mse", "snr_db", "si_snr_db", "lsd"
    ]


def reconstruct_sample(
    checkpoint: dict,
    data_root: Path | None,
) -> dict:
    sample = checkpoint["sample"]
    source = resolve_source_path(
        sample,
        saved_root=checkpoint.get("args", {}).get("root"),
        data_root=data_root,
    )
    if checkpoint["modality"] == "audio":
        loaded = load_audio(
            source,
            sample_rate=int(
                sample.get("requested_sample_rate", sample.get("sample_rate", 16000))
            ),
            start_sec=float(sample.get("start_sec", 0.0)),
            duration_sec=float(sample.get("duration_sec", 5.0)),
            allow_short=bool(sample.get("allow_short", False)),
        )
        if loaded is None:
            raise RuntimeError(f"Audio clip is shorter than requested: {source}")
    else:
        loaded = load_image(
            source,
            max_image_side=int(sample.get("max_image_side", 512)),
        )
    return {**sample, **loaded, "resolved_source_path": str(source.resolve())}


def rebuild_model(checkpoint: dict, sample: dict, device: torch.device):
    model = build_ngp_model(
        checkpoint["modality"],
        checkpoint["ngp_config"],
        out_channels=int(sample.get("channels", 1)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def restore_model(
    model: torch.nn.Module,
    original_state: dict[str, torch.Tensor],
    device: torch.device,
) -> None:
    model.load_state_dict(
        {name: tensor.to(device) for name, tensor in original_state.items()},
        strict=True,
    )


@torch.no_grad()
def evaluate_model(model, sample: dict, device: torch.device) -> dict:
    grid = sample["grid"].to(device)
    target = sample["target"].to(device)
    prediction = model(grid)
    if "height" in sample:
        prediction = prediction.clamp(0, 1)
        target = target.clamp(0, 1)
        return {
            "mse": F.mse_loss(prediction, target).item(),
            "psnr_db": psnr(prediction, target),
            "ssim": ssim(
                prediction,
                target,
                int(sample["height"]),
                int(sample["width"]),
            ),
        }
    return {
        "mse": F.mse_loss(prediction, target).item(),
        "snr_db": snr(prediction, target),
        "si_snr_db": si_snr(prediction, target),
        "lsd": log_spectral_distance(prediction, target),
    }


def make_row(
    sample: dict,
    result_type: str,
    config: str,
    total_bits: int,
    metrics: dict,
    status: str = "success",
    error: str = "",
) -> dict:
    return {
        "sample_id": sample["sample_id"],
        "dataset": sample.get("dataset", ""),
        "source_path": sample.get("resolved_source_path", sample.get("source_path", "")),
        "type": result_type,
        "config": config,
        "total_bits": total_bits,
        "status": status,
        **metrics,
        "error": error,
    }


def print_metrics(label: str, metrics: dict, modality: str) -> None:
    if modality == "image":
        print(
            f"  {label}: PSNR={metrics['psnr_db']:.2f} "
            f"SSIM={metrics['ssim']:.4f} MSE={metrics['mse']:.6g}"
        )
    else:
        print(
            f"  {label}: SNR={metrics['snr_db']:.2f} "
            f"SI-SNR={metrics['si_snr_db']:.2f} "
            f"LSD={metrics['lsd']:.4f} MSE={metrics['mse']:.6g}"
        )


def row_key(row: dict) -> tuple[str, str, str]:
    return row.get("sample_id", ""), row.get("type", ""), row.get("config", "")


def upsert_rows(rows: list[dict], additions: Iterable[dict]) -> list[dict]:
    by_key = {row_key(row): row for row in rows}
    for row in additions:
        by_key[row_key(row)] = row
    return list(by_key.values())


def required_keys(
    methods: Sequence[str],
    monolithic_bits: Sequence[int],
    layered_bits: Sequence[int],
) -> set:
    needed = {("Baseline", "fp32")}
    if "monolithic" in methods:
        needed.update(("Monolithic", str(bits)) for bits in monolithic_bits)
    if "lp" in methods:
        needed.update(
            ("Layered Precision", "-".join(map(str, layered_bits[:depth])))
            for depth in range(1, len(layered_bits) + 1)
        )
    return needed


def complete_samples(
    rows: list[dict],
    methods: Sequence[str],
    monolithic_bits: Sequence[int],
    layered_bits: Sequence[int],
) -> set:
    needed = required_keys(methods, monolithic_bits, layered_bits)
    found = defaultdict(set)
    for row in rows:
        if row.get("status") == "success":
            found[row.get("sample_id")].add((row.get("type"), row.get("config")))
    return {sample_id for sample_id, keys in found.items() if needed.issubset(keys)}


def run_checkpoint(
    path: Path,
    device: torch.device,
    data_root: Path | None,
    methods: Sequence[str],
    monolithic_bits: Sequence[int],
    layered_bits: Sequence[int],
) -> tuple[list[dict], str]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    modality = checkpoint.get("modality")
    if modality not in {"image", "audio"}:
        raise ValueError(f"Unsupported checkpoint modality {modality!r}: {path}")
    sample = reconstruct_sample(checkpoint, data_root)
    print(f"\n[{sample['sample_id']}] {modality} checkpoint={path}")
    model = rebuild_model(checkpoint, sample, device)
    original_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }

    rows: list[dict] = []
    metrics = evaluate_model(model, sample, device)
    rows.append(make_row(sample, "Baseline", "fp32", 32, metrics))
    print_metrics("baseline", metrics, modality)
    if "monolithic" in methods:
        for bits in monolithic_bits:
            restore_model(model, original_state, device)
            try:
                quantizer = LayeredQuantizer("tensor", 0)
                quantizer.apply_layered_quantization_to_model(
                    model,
                    [bits],
                    ["int"],
                    EXCLUDE_PATTERNS,
                    include_1d_parameters=True,
                    inplace=True,
                )
                metrics = evaluate_model(model, sample, device)
                rows.append(make_row(
                    sample, "Monolithic", str(bits), bits, metrics
                ))
                print_metrics(f"mono {bits}", metrics, modality)
            except Exception as exc:
                rows.append(make_row(
                    sample,
                    "Monolithic",
                    str(bits),
                    bits,
                    {},
                    status="failed",
                    error=str(exc),
                ))
                print(f"  mono {bits} failed: {exc}")

    if "lp" in methods:
        restore_model(model, original_state, device)
        path_bits = list(layered_bits)
        try:
            quantizer = LayeredQuantizer("tensor", 0)
            _, info = quantizer.apply_layered_quantization_to_model(
                model,
                path_bits,
                ["int"] * len(path_bits),
                EXCLUDE_PATTERNS,
                include_1d_parameters=True,
                inplace=True,
            )
            database = info["per_parameter_info"]
            for depth in range(1, len(path_bits) + 1):
                prefix = path_bits[:depth]
                label = "-".join(map(str, prefix))
                apply_partial_reconstruction(model, database, depth)
                metrics = evaluate_model(model, sample, device)
                rows.append(make_row(
                    sample,
                    "Layered Precision",
                    label,
                    sum(prefix),
                    metrics,
                ))
                print_metrics(f"lp {label}", metrics, modality)
        except Exception as exc:
            label = "-".join(map(str, path_bits))
            rows.append(make_row(
                sample,
                "Layered Precision",
                label,
                sum(path_bits),
                {},
                status="failed",
                error=str(exc),
            ))
            print(f"  LP failed: {exc}")

    del model, original_state
    gc.collect()
    torch.cuda.empty_cache()
    return rows, modality


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, path)


def write_rows(path: Path, rows: list[dict], metrics: Sequence[str]) -> None:
    fields = [
        "sample_id", "dataset", "source_path", "type", "config",
        "total_bits", "status", *metrics, "error",
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (
            item.get("sample_id", ""),
            int(item.get("total_bits", 0)),
            item.get("type", ""),
            item.get("config", ""),
        )):
            writer.writerow({field: row.get(field, "") for field in fields})
    os.replace(tmp, path)


def finite_values(rows: Iterable[dict], metric: str) -> list[float]:
    values = []
    for row in rows:
        try:
            value = float(row.get(metric, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def summarize(rows: list[dict], metrics: Sequence[str], path: Path) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        if row.get("status") == "success" and row.get("type") != "Baseline":
            groups[(row["type"], row["config"], int(row["total_bits"]))].append(row)
    output = []
    for (result_type, config, total_bits), group in sorted(
        groups.items(), key=lambda item: (item[0][2], item[0][0], item[0][1])
    ):
        summary = {
            "type": result_type,
            "config": config,
            "total_bits": total_bits,
            "count": len(group),
            "snr_negative_pct": "",
        }
        for metric in metrics:
            values = finite_values(group, metric)
            if not values:
                continue
            average = mean(values)
            summary.update({
                f"{metric}_mean": average,
                f"{metric}_median": median(values),
                f"{metric}_std": math.sqrt(
                    sum((value - average) ** 2 for value in values) / len(values)
                ) if len(values) > 1 else 0.0,
                f"{metric}_min": min(values),
                f"{metric}_max": max(values),
            })
            if metric == "snr_db":
                summary["snr_negative_pct"] = (
                    100 * sum(value < 0 for value in values) / len(values)
                )
        output.append(summary)

    fields = [
        "type", "config", "total_bits", "count",
        *[
            f"{metric}_{stat}"
            for metric in metrics
            for stat in ("mean", "median", "std", "min", "max")
        ],
        "snr_negative_pct",
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in output:
            writer.writerow({field: row.get(field, "") for field in fields})
    os.replace(tmp, path)
    return output


def plot_summary(rows: list[dict], metrics: Sequence[str], outdir: Path) -> None:
    labels = {
        "mse": "MSE",
        "psnr_db": "PSNR (dB)",
        "ssim": "SSIM",
        "snr_db": "SNR (dB)",
        "si_snr_db": "SI-SNR (dB)",
        "lsd": "LSD",
    }
    for metric in metrics:
        fig, axis = plt.subplots(figsize=(7, 4))
        for result_type, marker, color in (
            ("Monolithic", "o", "#2878B5"),
            ("Layered Precision", "s", "#F28E2B"),
        ):
            selected = sorted(
                (
                    int(row["total_bits"]),
                    float(row[f"{metric}_mean"]),
                )
                for row in rows
                if row["type"] == result_type
                and row.get(f"{metric}_mean") not in (None, "")
            )
            if selected:
                xs, ys = zip(*selected)
                axis.plot(
                    xs,
                    ys,
                    marker=marker,
                    color=color,
                    linewidth=1.8,
                    label=result_type,
                )
        axis.set_xlabel("Total bits")
        axis.set_ylabel(labels[metric])
        axis.set_title(f"Dataset mean {labels[metric]}")
        axis.grid(alpha=0.3)
        axis.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"plot_{metric}.png", dpi=300)
        fig.savefig(outdir / f"plot_{metric}.pdf", bbox_inches="tight")
        plt.close(fig)


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metadata_for_run(
    args: argparse.Namespace,
    checkpoints: Sequence[Path],
    modality: str,
) -> dict:
    return {
        "schema_version": 1,
        "release": "paper_release",
        "checkpoints_root": str(Path(args.checkpoints_root).resolve()),
        "checkpoint_sample_ids": [path.parent.name for path in checkpoints],
        "manifest": str(Path(args.manifest).resolve()) if args.manifest else None,
        "data_root": str(Path(args.data_root).resolve()) if args.data_root else None,
        "modality": modality,
        "methods": list(args.methods),
        "monolithic_bits": list(args.monolithic_bits),
        "layered_bits": list(args.layered_bits),
        "int_granularity": "tensor",
        "int_use_zero_point": True,
        "exclude_patterns": list(EXCLUDE_PATTERNS),
        "include_1d_parameters": True,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("NGP evaluation requires CUDA and tiny-cuda-nn")
    root = Path(args.checkpoints_root)
    checkpoints = find_checkpoints(
        root,
        Path(args.manifest) if args.manifest else None,
    )
    if not checkpoints:
        raise RuntimeError(f"No NGP checkpoints found under {root}")

    first = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    modality = first.get("modality")
    if modality not in {"image", "audio"}:
        raise ValueError(f"Unsupported first-checkpoint modality: {modality!r}")
    metadata = metadata_for_run(args, checkpoints, modality)
    metrics = modality_metrics(modality)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows_path = outdir / "quantization_results_per_sample.csv"
    summary_path = outdir / "quantization_results_summary.csv"
    metadata_path = outdir / "run_metadata.json"

    rows: list[dict] = []
    if args.resume:
        if not metadata_path.exists():
            raise RuntimeError("--resume requested but run_metadata.json is missing")
        with metadata_path.open("r", encoding="utf-8") as handle:
            saved_metadata = json.load(handle)
        if saved_metadata != metadata:
            raise RuntimeError(
                "Cannot resume: saved metadata differs. Use a new output directory."
            )
        rows = load_csv(rows_path)
    done = complete_samples(
        rows,
        args.methods,
        args.monolithic_bits,
        args.layered_bits,
    )

    print(f"Device: cuda")
    print(f"Modality: {modality}")
    print(f"Checkpoints: {len(checkpoints)}")
    print(f"Output: {outdir}")
    print(f"Monolithic bits: {args.monolithic_bits}")
    print(f"Layered bits: {args.layered_bits}")
    print("Quantizer: tensor-wise affine INT with explicit zero point")
    atomic_json(metadata_path, metadata)
    failures = []

    for checkpoint_path in checkpoints:
        sample_id = checkpoint_path.parent.name
        if sample_id in done:
            print(f"Skipping {sample_id} (complete checkpoint rows)")
            continue
        try:
            additions, checkpoint_modality = run_checkpoint(
                checkpoint_path,
                torch.device("cuda"),
                Path(args.data_root) if args.data_root else None,
                args.methods,
                args.monolithic_bits,
                args.layered_bits,
            )
            if checkpoint_modality != modality:
                raise ValueError("Mixed image/audio checkpoints are not supported")
            rows = upsert_rows(rows, additions)
            write_rows(rows_path, rows, metrics)
            summarize(rows, metrics, summary_path)
        except Exception as exc:
            failures.append({
                "checkpoint": str(checkpoint_path),
                "error": str(exc),
            })
            atomic_json(outdir / "failures.json", {"failures": failures})
            print(f"Failed {checkpoint_path}: {exc}")

    if not rows:
        raise RuntimeError("No evaluation rows were written")
    write_rows(rows_path, rows, metrics)
    summary_rows = summarize(rows, metrics, summary_path)
    plot_summary(summary_rows, metrics, outdir)
    atomic_json(outdir / "failures.json", {"failures": failures})
    print(f"\nSaved {rows_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
