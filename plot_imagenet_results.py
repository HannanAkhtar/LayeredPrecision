#!/usr/bin/env python3
"""Create publication plots from a clean ImageNet result CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate_imagenet import MODEL_REGISTRY


METHODS = (
    ("Monolithic", "Monolithic", "o", "#2878B5"),
    ("Layered Precision", "Layered Precision", "s", "#F28E2B"),
    ("Bit-Plane Truncation", "Bit-plane coding", "^", "#3A923A"),
)
METRICS = {
    "top1": "Top-1 accuracy (%)",
    "top5": "Top-5 accuracy (%)",
    "cross_entropy": "Cross-entropy loss",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the paper ImageNet curves from all_models_results.csv."
    )
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="One or more output directories or all_models_results.csv files",
    )
    parser.add_argument("--outdir", required=True)
    return parser.parse_args()


def as_float(value: str, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field} value {value!r}") from exc


def load_results(paths: Sequence[Path]) -> list[dict]:
    rows = []
    seen: dict[tuple[str, str, str], dict] = {}
    for path in paths:
        csv_path = path / "all_models_results.csv" if path.is_dir() else path
        if not csv_path.is_file():
            raise FileNotFoundError(f"Results CSV not found: {csv_path}")
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            source_rows = list(csv.DictReader(handle))
        for source in source_rows:
            if source.get("status") != "success":
                continue
            key = (
                source.get("model", ""),
                source.get("type", ""),
                source.get("config", ""),
            )
            normalized = dict(source)
            normalized["total_bits"] = int(
                as_float(source.get("total_bits", ""), "total_bits")
            )
            for metric in METRICS:
                value = source.get(metric, "")
                normalized[metric] = (
                    None if value in (None, "") else as_float(value, metric)
                )
            previous = seen.get(key)
            if previous is not None and previous != normalized:
                raise ValueError(
                    "Conflicting duplicate result for "
                    f"model={key[0]} type={key[1]} config={key[2]}"
                )
            if previous is None:
                seen[key] = normalized
                rows.append(normalized)
    if not rows:
        raise RuntimeError("No successful result rows found")
    return rows


def model_order(rows: Sequence[dict]) -> list[str]:
    present = {row["model"] for row in rows}
    ordered = [name for name in MODEL_REGISTRY if name in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def baseline(rows: Sequence[dict], model: str, metric: str) -> float | None:
    values = [
        row[metric]
        for row in rows
        if row["model"] == model and row["type"] == "Baseline" and row[metric] is not None
    ]
    if len(values) > 1 and len(set(values)) > 1:
        raise ValueError(f"Conflicting FP32 baselines for {model}")
    return values[0] if values else None


def method_points(
    rows: Sequence[dict], model: str, result_type: str, metric: str
) -> list[tuple[int, float]]:
    points = [
        (row["total_bits"], row[metric])
        for row in rows
        if row["model"] == model
        and row["type"] == result_type
        and row[metric] is not None
    ]
    return sorted(points)


def plot_metric(
    rows: Sequence[dict],
    models: Sequence[str],
    metric: str,
    outdir: Path,
) -> None:
    columns = 2 if len(models) > 1 else 1
    plot_rows = math.ceil(len(models) / columns)
    figure, axes = plt.subplots(
        plot_rows,
        columns,
        figsize=(6.4 * columns, 3.8 * plot_rows),
        squeeze=False,
    )
    for index, model in enumerate(models):
        axis = axes[index // columns][index % columns]
        fp32 = baseline(rows, model, metric)
        if fp32 is not None:
            axis.axhline(
                fp32,
                color="#555555",
                linestyle=":",
                linewidth=1.4,
                label="FP32",
            )
        for result_type, label, marker, color in METHODS:
            points = method_points(rows, model, result_type, metric)
            if not points:
                continue
            xs, ys = zip(*points)
            axis.plot(
                xs,
                ys,
                marker=marker,
                color=color,
                linewidth=1.8,
                markersize=5,
                label=label,
            )
        axis.set_title(model)
        axis.set_xlabel("Total bits")
        axis.set_ylabel(METRICS[metric])
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    for index in range(len(models), plot_rows * columns):
        axes[index // columns][index % columns].axis("off")
    figure.tight_layout()
    stem = f"imagenet_{metric}_vs_bits"
    figure.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(outdir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_top1_gap(rows: Sequence[dict], models: Sequence[str], outdir: Path) -> None:
    columns = 2 if len(models) > 1 else 1
    plot_rows = math.ceil(len(models) / columns)
    figure, axes = plt.subplots(
        plot_rows,
        columns,
        figsize=(6.4 * columns, 3.8 * plot_rows),
        squeeze=False,
    )
    for index, model in enumerate(models):
        axis = axes[index // columns][index % columns]
        fp32 = baseline(rows, model, "top1")
        if fp32 is None:
            axis.axis("off")
            continue
        axis.axhline(0.0, color="#555555", linestyle=":", linewidth=1.4)
        for result_type, label, marker, color in METHODS:
            points = method_points(rows, model, result_type, "top1")
            if not points:
                continue
            xs, ys = zip(*points)
            axis.plot(
                xs,
                [fp32 - value for value in ys],
                marker=marker,
                color=color,
                linewidth=1.8,
                markersize=5,
                label=label,
            )
        axis.set_title(model)
        axis.set_xlabel("Total bits")
        axis.set_ylabel("FP32 minus quantized top-1 (pp)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    for index in range(len(models), plot_rows * columns):
        axes[index // columns][index % columns].axis("off")
    figure.tight_layout()
    figure.savefig(outdir / "imagenet_top1_gap.pdf", bbox_inches="tight")
    figure.savefig(outdir / "imagenet_top1_gap.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    rows = load_results([Path(path) for path in args.results])
    models = model_order(rows)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        plot_metric(rows, models, metric, outdir)
    plot_top1_gap(rows, models, outdir)
    print(f"Saved ImageNet plots to {outdir}")


if __name__ == "__main__":
    main()
