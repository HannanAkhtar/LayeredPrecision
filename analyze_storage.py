#!/usr/bin/env python3
"""Measure storage for the exact ImageNet and NGP paper presets.

The historical columns reproduce the original symbol-stream accounting. The
complete columns additionally count quantizer side information and all model
state that remains in FP32, making the reported representation decodable.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch
import torchvision

from evaluate_imagenet import (
    EXCLUDE_PATTERNS as IMAGENET_EXCLUDES,
    MODEL_REGISTRY,
    PAPER_CONFIGS,
    build_model,
)
from evaluate_ngp_dataset import (
    EXCLUDE_PATTERNS as NGP_EXCLUDES,
    find_checkpoints,
    parse_bits_csv,
)
from layered_quantization import LayeredQuantizer
from ngp_models import build_ngp_model
from storage_utils import fixed_state_for_model, measure_layered_storage, saving_percent


MEASURE_FIELDS = [
    "entropy_bits",
    "entropy_bpw",
    "huffman_bits",
    "huffman_bpw",
    "payload_bytes",
    "codebook_bytes",
    "header_bytes",
    "storage_bytes",
    "metadata_bytes",
    "fixed_state_bytes",
    "complete_storage_bytes",
    "global_huffman_bits",
    "global_huffman_bpw",
    "global_payload_bytes",
    "global_codebook_bytes",
    "global_header_bytes",
    "global_storage_bytes",
    "global_metadata_bytes",
    "global_fixed_state_bytes",
    "global_complete_storage_bytes",
    "raw_bits",
    "raw_bpw",
    "raw_payload_bytes",
    "raw_codebook_bytes",
    "raw_header_bytes",
    "raw_storage_bytes",
    "raw_metadata_bytes",
    "raw_fixed_state_bytes",
    "raw_complete_storage_bytes",
    "num_symbols",
]

ROW_FIELDS = [
    "subject",
    "modality",
    "model",
    "checkpoint",
    "type",
    "config",
    "total_bits",
    "status",
    "error",
    *MEASURE_FIELDS,
    "storage_mib",
    "complete_storage_mib",
    "global_storage_mib",
    "global_complete_storage_mib",
    "raw_storage_mib",
    "raw_complete_storage_mib",
]

SIMULCAST_FIELDS = [
    "subject",
    "modality",
    "model",
    "operating_set",
    "simulcast_storage_bytes",
    "lp_storage_bytes",
    "storage_reduction_pct",
    "simulcast_complete_storage_bytes",
    "lp_complete_storage_bytes",
    "complete_storage_reduction_pct",
    "simulcast_global_storage_bytes",
    "lp_global_storage_bytes",
    "global_storage_reduction_pct",
    "simulcast_global_complete_storage_bytes",
    "lp_global_complete_storage_bytes",
    "global_complete_storage_reduction_pct",
    "simulcast_raw_storage_bytes",
    "lp_raw_storage_bytes",
    "raw_storage_reduction_pct",
    "simulcast_raw_complete_storage_bytes",
    "lp_raw_complete_storage_bytes",
    "raw_complete_storage_reduction_pct",
]


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--write-artifacts",
        action="store_true",
        help="Keep encoded streams and side information, not only their sizes",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip complete subjects after validating run metadata",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Storage analysis for the paper experiment presets."
    )
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    imagenet = subparsers.add_parser("imagenet")
    add_common_arguments(imagenet)
    imagenet.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help=f"Subset of {list(MODEL_REGISTRY)} or 'all'",
    )

    ngp = subparsers.add_parser("ngp")
    add_common_arguments(ngp)
    ngp.add_argument("--checkpoints-root", required=True)
    ngp.add_argument("--manifest", default=None)
    ngp.add_argument(
        "--monolithic-bits",
        type=parse_bits_csv,
        required=True,
        metavar="BITS",
        help="Comma-separated monolithic widths",
    )
    ngp.add_argument(
        "--layered-bits",
        type=parse_bits_csv,
        required=True,
        metavar="BITS",
        help="Comma-separated LP base and residual widths",
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    os.replace(temporary, path)


def load_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def prepare_run(outdir: Path, metadata: dict, resume: bool) -> list[dict]:
    outdir.mkdir(parents=True, exist_ok=True)
    metadata_path = outdir / "run_metadata.json"
    rows_path = outdir / "storage_results.csv"
    rows: list[dict] = []
    if resume and metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous != metadata:
            raise RuntimeError(
                "Cannot resume: saved metadata differs. Use a new output directory."
            )
        rows = load_csv(rows_path)
    elif resume and rows_path.exists():
        raise RuntimeError(
            "Cannot resume storage_results.csv without run_metadata.json"
        )
    atomic_json(metadata_path, metadata)
    return rows


def safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def artifact_prefix(
    outdir: Path,
    subject: str,
    result_type: str,
    config: str,
    enabled: bool,
) -> Path | None:
    if not enabled:
        return None
    directory = outdir / "artifacts" / safe_component(subject)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{safe_component(result_type.lower())}_{safe_component(config)}"


def quantize_model(
    model: torch.nn.Module,
    bits: Sequence[int],
    granularity: str,
    channel_axis: int,
    exclude_patterns: Sequence[str],
    include_1d_parameters: bool,
) -> tuple[dict, dict[str, torch.Tensor]]:
    quantizer = LayeredQuantizer(granularity, channel_axis)
    _, summary = quantizer.apply_layered_quantization_to_model(
        model,
        list(bits),
        ["int"] * len(bits),
        exclude_patterns,
        include_1d_parameters=include_1d_parameters,
        inplace=True,
    )
    information = summary["per_parameter_info"]
    return information, fixed_state_for_model(model, information)


def measured_row(
    *,
    subject: str,
    modality: str,
    model_name: str,
    checkpoint: str,
    result_type: str,
    config: str,
    total_bits: int,
    information: dict,
    fixed_state: dict[str, torch.Tensor],
    depth: int,
    write_path: Path | None,
) -> dict:
    measurement = measure_layered_storage(
        information,
        layers_to_use=depth,
        fixed_state=fixed_state,
        write_path=write_path,
    )
    row = {
        "subject": subject,
        "modality": modality,
        "model": model_name,
        "checkpoint": checkpoint,
        "type": result_type,
        "config": config,
        "total_bits": total_bits,
        "status": "success",
        "error": "",
        **measurement,
    }
    for prefix in ("", "global_", "raw_"):
        row[f"{prefix}storage_mib"] = (
            float(row[f"{prefix}storage_bytes"]) / (1024.0**2)
        )
        row[f"{prefix}complete_storage_mib"] = (
            float(row[f"{prefix}complete_storage_bytes"]) / (1024.0**2)
        )
    return row


def replace_subject(rows: list[dict], subject: str, additions: list[dict]) -> list[dict]:
    retained = [row for row in rows if row.get("subject") != subject]
    return retained + additions


def expected_keys(monolithic: Sequence[int], layered: Sequence[int]) -> set:
    expected = {("Monolithic", str(bits)) for bits in monolithic}
    expected.update(
        ("Layered Precision", "-".join(map(str, layered[:depth])))
        for depth in range(1, len(layered) + 1)
    )
    return expected


def complete_subjects(
    rows: Iterable[dict],
    configs_by_subject: dict[str, tuple[Sequence[int], Sequence[int]]],
) -> set:
    found = defaultdict(set)
    for row in rows:
        if row.get("status") == "success":
            found[row.get("subject", "")].add(
                (row.get("type", ""), row.get("config", ""))
            )
    return {
        subject
        for subject, (monolithic, layered) in configs_by_subject.items()
        if expected_keys(monolithic, layered).issubset(found[subject])
    }


def measure_model_presets(
    *,
    subject: str,
    modality: str,
    model_name: str,
    checkpoint: str,
    model_factory,
    monolithic: Sequence[int],
    layered: Sequence[int],
    granularity: str,
    channel_axis: int,
    exclude_patterns: Sequence[str],
    include_1d_parameters: bool,
    outdir: Path,
    write_artifacts: bool,
) -> list[dict]:
    rows: list[dict] = []
    for bits in monolithic:
        print(f"  Monolithic {bits} bits", flush=True)
        model = model_factory()
        information, fixed_state = quantize_model(
            model,
            [bits],
            granularity,
            channel_axis,
            exclude_patterns,
            include_1d_parameters,
        )
        rows.append(measured_row(
            subject=subject,
            modality=modality,
            model_name=model_name,
            checkpoint=checkpoint,
            result_type="Monolithic",
            config=str(bits),
            total_bits=bits,
            information=information,
            fixed_state=fixed_state,
            depth=1,
            write_path=artifact_prefix(
                outdir, subject, "monolithic", str(bits), write_artifacts
            ),
        ))
        del model, information, fixed_state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    label = "-".join(map(str, layered))
    print(f"  Layered Precision {label}", flush=True)
    model = model_factory()
    information, fixed_state = quantize_model(
        model,
        layered,
        granularity,
        channel_axis,
        exclude_patterns,
        include_1d_parameters,
    )
    for depth in range(1, len(layered) + 1):
        prefix = list(layered[:depth])
        config = "-".join(map(str, prefix))
        rows.append(measured_row(
            subject=subject,
            modality=modality,
            model_name=model_name,
            checkpoint=checkpoint,
            result_type="Layered Precision",
            config=config,
            total_bits=sum(prefix),
            information=information,
            fixed_state=fixed_state,
            depth=depth,
            write_path=artifact_prefix(
                outdir, subject, "layered", config, write_artifacts
            ),
        ))
    del model, information, fixed_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def to_float(row: dict, field: str) -> float:
    value = row.get(field, 0.0)
    return float(value) if value not in (None, "") else 0.0


def simulcast_row(subject_rows: Sequence[dict]) -> dict | None:
    successful = [row for row in subject_rows if row.get("status") == "success"]
    monolithic = {
        int(float(row["total_bits"])): row
        for row in successful
        if row.get("type") == "Monolithic"
    }
    layered = [row for row in successful if row.get("type") == "Layered Precision"]
    if not monolithic or not layered:
        return None
    operating_points = sorted({int(float(row["total_bits"])) for row in layered})
    if any(bits not in monolithic for bits in operating_points):
        return None
    deepest = max(layered, key=lambda row: int(float(row["total_bits"])))
    first = successful[0]
    output = {
        "subject": first.get("subject", ""),
        "modality": first.get("modality", ""),
        "model": first.get("model", ""),
        "operating_set": "/".join(map(str, operating_points)),
    }
    for prefix in ("", "global_", "raw_"):
        for complete in (False, True):
            label = f"{prefix}{'complete_' if complete else ''}storage_bytes"
            name = f"{prefix}{'complete_' if complete else ''}storage"
            simulcast = sum(to_float(monolithic[bits], label) for bits in operating_points)
            lp_value = to_float(deepest, label)
            output[f"simulcast_{name}_bytes"] = simulcast
            output[f"lp_{name}_bytes"] = lp_value
            output[f"{name}_reduction_pct"] = saving_percent(lp_value, simulcast)
    return output


def build_simulcast_rows(rows: Sequence[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("subject", "")].append(row)
    output = []
    for subject in sorted(grouped):
        comparison = simulcast_row(grouped[subject])
        if comparison is not None:
            output.append(comparison)
    return output


def aggregate_ngp_rows(rows: Sequence[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if row.get("status") == "success":
            grouped[(row.get("modality"), row.get("type"), row.get("config"))].append(row)
    summaries = []
    additive = [field for field in MEASURE_FIELDS if not field.endswith("_bpw")]
    for (modality, result_type, config), group in sorted(grouped.items()):
        symbols = sum(to_float(row, "num_symbols") for row in group)
        summary = {
            "subject": "dataset_total",
            "modality": modality,
            "model": "ngp",
            "checkpoint": "",
            "type": result_type,
            "config": config,
            "total_bits": group[0].get("total_bits", ""),
            "status": "success",
            "error": "",
        }
        for field in additive:
            summary[field] = sum(to_float(row, field) for row in group)
        summary["num_symbols"] = symbols
        denominator = symbols if symbols else 1.0
        summary["entropy_bpw"] = summary["entropy_bits"] / denominator
        summary["huffman_bpw"] = summary["huffman_bits"] / denominator
        summary["global_huffman_bpw"] = summary["global_huffman_bits"] / denominator
        summary["raw_bpw"] = summary["raw_bits"] / denominator
        for prefix in ("", "global_", "raw_"):
            summary[f"{prefix}storage_mib"] = (
                summary[f"{prefix}storage_bytes"] / (1024.0**2)
            )
            summary[f"{prefix}complete_storage_mib"] = (
                summary[f"{prefix}complete_storage_bytes"] / (1024.0**2)
            )
        summaries.append(summary)
    return summaries


def save_outputs(outdir: Path, rows: list[dict], ngp: bool) -> None:
    ordered = sorted(
        rows,
        key=lambda row: (
            row.get("subject", ""),
            int(float(row.get("total_bits", 0) or 0)),
            row.get("type", ""),
            row.get("config", ""),
        ),
    )
    write_csv(outdir / "storage_results.csv", ordered, ROW_FIELDS)
    comparisons = build_simulcast_rows(ordered)
    write_csv(outdir / "simulcast_vs_lp.csv", comparisons, SIMULCAST_FIELDS)
    if ngp:
        summary = aggregate_ngp_rows(ordered)
        write_csv(outdir / "storage_results_summary.csv", summary, ROW_FIELDS)
        aggregate_comparison = simulcast_row(summary)
        write_csv(
            outdir / "simulcast_vs_lp_summary.csv",
            [aggregate_comparison] if aggregate_comparison else [],
            SIMULCAST_FIELDS,
        )


def run_imagenet(args: argparse.Namespace) -> None:
    models = list(MODEL_REGISTRY) if args.models == ["all"] else args.models
    unknown = [name for name in models if name not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown models {unknown}; available: {list(MODEL_REGISTRY)}")
    metadata = {
        "schema_version": 1,
        "release": "paper_release",
        "experiment": "imagenet",
        "models": models,
        "weights": {name: str(MODEL_REGISTRY[name][1]) for name in models},
        "configs": {name: PAPER_CONFIGS[name] for name in models},
        "quantizer": {
            "format": "affine_int",
            "granularity": "channel",
            "channel_axis": 0,
            "use_zero_point": True,
            "exclude_patterns": list(IMAGENET_EXCLUDES),
            "include_1d_parameters": False,
        },
        "write_artifacts": bool(args.write_artifacts),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
    }
    outdir = Path(args.outdir)
    rows = prepare_run(outdir, metadata, args.resume)
    configs = {
        name: (PAPER_CONFIGS[name]["monolithic"], PAPER_CONFIGS[name]["layered"])
        for name in models
    }
    done = complete_subjects(rows, configs) if args.resume else set()
    for model_name in models:
        if model_name in done:
            print(f"Skipping {model_name} (complete)")
            continue
        print(f"\n[{model_name}]")
        additions = measure_model_presets(
            subject=model_name,
            modality="imagenet",
            model_name=model_name,
            checkpoint=str(MODEL_REGISTRY[model_name][1]),
            model_factory=lambda name=model_name: build_model(name),
            monolithic=PAPER_CONFIGS[model_name]["monolithic"],
            layered=PAPER_CONFIGS[model_name]["layered"],
            granularity="channel",
            channel_axis=0,
            exclude_patterns=IMAGENET_EXCLUDES,
            include_1d_parameters=False,
            outdir=outdir,
            write_artifacts=args.write_artifacts,
        )
        rows = replace_subject(rows, model_name, additions)
        save_outputs(outdir, rows, ngp=False)
    save_outputs(outdir, rows, ngp=False)


def ngp_factory(checkpoint: dict, device: torch.device):
    modality = checkpoint["modality"]
    sample = checkpoint.get("sample", {})
    default_channels = 1 if modality == "audio" else 3
    model = build_ngp_model(
        modality,
        checkpoint["ngp_config"],
        out_channels=int(sample.get("channels", default_channels)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.eval()


def run_ngp(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("NGP storage analysis requires CUDA and tiny-cuda-nn")
    root = Path(args.checkpoints_root)
    manifest = Path(args.manifest) if args.manifest else None
    checkpoints = find_checkpoints(root, manifest)
    if not checkpoints:
        raise RuntimeError(f"No NGP checkpoints found under {root}")
    checkpoint_ids = [path.parent.name for path in checkpoints]
    metadata = {
        "schema_version": 1,
        "release": "paper_release",
        "experiment": "ngp",
        "checkpoints_root": str(root.resolve()),
        "manifest": str(manifest.resolve()) if manifest else None,
        "checkpoint_sample_ids": checkpoint_ids,
        "monolithic_bits": list(args.monolithic_bits),
        "layered_bits": list(args.layered_bits),
        "quantizer": {
            "format": "affine_int",
            "granularity": "tensor",
            "channel_axis": 0,
            "use_zero_point": True,
            "exclude_patterns": list(NGP_EXCLUDES),
            "include_1d_parameters": True,
        },
        "write_artifacts": bool(args.write_artifacts),
        "torch_version": torch.__version__,
    }
    outdir = Path(args.outdir)
    rows = prepare_run(outdir, metadata, args.resume)

    modalities = {}
    for path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        modality = checkpoint.get("modality")
        if modality not in {"image", "audio"}:
            raise ValueError(f"Unsupported modality {modality!r}: {path}")
        modalities[path.parent.name] = modality
    configs = {
        subject: (args.monolithic_bits, args.layered_bits)
        for subject in modalities
    }
    distinct_modalities = set(modalities.values())
    if len(distinct_modalities) != 1:
        raise ValueError(
            "Mixed image/audio checkpoint roots are not supported; "
            "analyze each dataset separately."
        )
    done = complete_subjects(rows, configs) if args.resume else set()
    failures = []
    device = torch.device("cuda")
    for path in checkpoints:
        subject = path.parent.name
        if subject in done:
            print(f"Skipping {subject} (complete)")
            continue
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        modality = checkpoint["modality"]
        print(f"\n[{subject}] {modality}")
        try:
            additions = measure_model_presets(
                subject=subject,
                modality=modality,
                model_name="ngp",
                checkpoint=str(path.resolve()),
                model_factory=lambda saved=checkpoint: ngp_factory(saved, device),
                monolithic=args.monolithic_bits,
                layered=args.layered_bits,
                granularity="tensor",
                channel_axis=0,
                exclude_patterns=NGP_EXCLUDES,
                include_1d_parameters=True,
                outdir=outdir,
                write_artifacts=args.write_artifacts,
            )
            rows = replace_subject(rows, subject, additions)
            save_outputs(outdir, rows, ngp=True)
        except Exception as exc:
            failures.append({"checkpoint": str(path), "error": str(exc)})
            atomic_json(outdir / "failures.json", {"failures": failures})
            print(f"Failed {path}: {exc}")
    save_outputs(outdir, rows, ngp=True)
    atomic_json(outdir / "failures.json", {"failures": failures})
    if failures:
        print(f"Completed with {len(failures)} failed checkpoints")


def main() -> None:
    args = parse_args()
    if args.experiment == "imagenet":
        run_imagenet(args)
    else:
        run_ngp(args)


if __name__ == "__main__":
    main()
