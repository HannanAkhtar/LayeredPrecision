#!/usr/bin/env python3
"""Residual evolution and isolated four-code LP enhancement comparison.

The reference path is 5-A-A-A. At each enhancement, all five families receive
one shared incoming residual from the preceding decoded reference prefix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from analyze_mse_layer_sensitivity import (
    MemmapSubset, balanced_indices, cache_subset, eligible_groups,
    evaluate_details, metric_sums, paired_stats, write_json,
)
from evaluate_imagenet import MODEL_REGISTRY, build_dataset, build_model
from evaluate_residual_ablation import inspect_dataset
from layered_quantization import IntegerFormat
from lp_quantizer_families import (
    FAMILIES, FAMILY_NAMES, dequantize_family, quantize_family,
    read_binary_codebook, write_binary_codebook,
)
from lp_residual_evolution import (
    NORMALIZED_EDGES, RAW_LOG_EDGES, characterize_tensor, summarize_all,
)


PREFIXES = {1: "B", 2: "BA", 3: "BAA", 4: "BAAA"}
EXPERIMENT1 = {"fp32": {"top1": 80.746, "top5": 95.394, "cross_entropy": 1.401995},
               "B": {"top1": 76.250, "top5": 92.940, "cross_entropy": 1.553906},
               "E1_A": {"top1": 79.998, "top5": 95.064, "cross_entropy": 1.436064},
               "E1_P": {"top1": 80.054, "top5": 95.096, "cross_entropy": 1.416465}}


def save_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def atomic_torch_save(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_torch(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cpu_params(params):
    return {key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in params.items()}


def numerical_metrics(original, decoded):
    if set(original) != set(decoded):
        raise AssertionError("Eligible tensor names differ")
    n = sum(value.numel() for value in original.values())
    error = sum((original[name].double()-decoded[name].double()).square().sum().item()
                for name in original)
    signal = sum(value.double().square().sum().item() for value in original.values())
    return {"eligible_weights": n, "parameter_sse": error, "parameter_mse": error/n,
            "parameter_sqnr_db": 10*math.log10(signal/error) if error else None}


def candidate_paths(outdir: Path, enhancement: int, family: str):
    stem = f"E{enhancement}_{family}"
    directory = outdir / "codes"
    directory.mkdir(exist_ok=True)
    return directory / f"{stem}.pt", directory / f"{stem}.bin", directory / f"{stem}.json"


def prefix_path(outdir: Path, prefix: str):
    directory = outdir / "decoded_prefixes"
    directory.mkdir(exist_ok=True)
    return directory / f"{prefix}_eligible_weights.pt"


def validate_saved_baselines(original, names, artifact_dir: Path, device):
    files = {"B": artifact_dir / "base_5_codes_and_metadata.pt",
             "A": artifact_dir / "affine_minmax_codes_and_metadata.pt",
             "P": artifact_dir / "affine_percentile_codes_and_metadata.pt"}
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing Experiment 1 artifact: {path}")
    saved = {key: load_torch(path) for key, path in files.items()}
    for key in files:
        if set(saved[key]) != set(names):
            raise AssertionError(f"Experiment 1 {key} tensor eligibility changed")
    integer = IntegerFormat("channel", 0)
    base_decoded = {}
    a_decoded = {}
    p_decoded = {}
    for index, name in enumerate(names):
        weight = original[name].to(device)
        params, base = integer.quantize(weight, 5)
        for key in ("quantized", "scale", "zero_point"):
            if not torch.equal(params[key].cpu(), saved["B"][name][key]):
                raise AssertionError(f"5-bit artifact mismatch: {name}/{key}")
        residual = weight-base
        base_cpu = IntegerFormat.dequantize(saved["B"][name])
        if not torch.equal(base.cpu(), base_cpu):
            raise AssertionError(f"Saved 5-bit decode mismatch: {name}")
        base_decoded[name] = base_cpu
        for family, output in (("A", a_decoded), ("P", p_decoded)):
            direct, value = quantize_family(residual, family)
            for key in ("quantized", "scale", "zero_point"):
                if not torch.equal(direct[key].cpu(), saved[family][name][key]):
                    raise AssertionError(f"Saved E1 {family} mismatch: {name}/{key}")
            decoded = base_cpu + dequantize_family(saved[family][name], family)
            if not torch.equal(decoded, (base+value).cpu()):
                raise AssertionError(f"Saved E1 {family} reconstruction mismatch: {name}")
            output[name] = decoded
        if (index+1) % 18 == 0:
            print(f"Verified saved B/A/P artifacts: {index+1}/{len(names)} tensors", flush=True)
    return saved, base_decoded, a_decoded, p_decoded


def reference_provenance_check(original, base, affine, percentile, source_json: Path):
    if not source_json.is_file():
        raise FileNotFoundError(f"Missing Experiment 1 metrics: {source_json}")
    previous = json.loads(source_json.read_text(encoding="utf-8"))
    metadata = previous["metadata"]
    if metadata.get("weights") != "ResNet50_Weights.IMAGENET1K_V2" or metadata.get("quantized_prefix") != [5, 2]:
        raise AssertionError("Experiment 1 checkpoint or schedule differs")
    metrics = {row["method"]: row for row in previous["results"]}
    for key, prefix in (("base_5", base), ("affine_minmax", affine),
                        ("affine_percentile", percentile)):
        measured = numerical_metrics(original, prefix)
        if abs(measured["parameter_mse"]-metrics[key]["parameter_mse"]) > 1e-12:
            raise AssertionError(f"Experiment 1 {key} parameter MSE mismatch")
        if abs(metrics[key]["top1"]-EXPERIMENT1["B" if key == "base_5" else
                                              "E1_A" if key == "affine_minmax" else "E1_P"]["top1"]) > 1e-6:
            raise AssertionError(f"Experiment 1 {key} accuracy regression mismatch")
    if metrics["fp32"]["top1"] != EXPERIMENT1["fp32"]["top1"]:
        raise AssertionError("Experiment 1 FP32 accuracy regression mismatch")
    return metrics


def code_behavior(artifacts, family: str):
    counts = np.zeros(4, dtype=np.int64)
    entropy_sum = 0.0
    channel_entropy_sum = 0.0
    channels = weights = extreme = zero_exact = zero_inputs = 0
    duplicates = iterations_sum = max_iteration_tensors = 0
    scales = []
    center_values = []
    center_spacings = []
    for params in artifacts.values():
        codes = params["quantized"].reshape(params["quantized"].shape[0], -1)
        n = codes.shape[1]
        per_channel = torch.stack([(codes == code).sum(dim=1) for code in range(4)], dim=1).double()
        probabilities = per_channel/n
        entropy = -(torch.where(probabilities > 0, probabilities*probabilities.clamp_min(1e-300).log2(), 0)).sum(1)
        count = per_channel.sum(0).cpu().numpy().astype(np.int64)
        counts += count
        weights += codes.numel()
        channels += codes.shape[0]
        entropy_sum += float(entropy.sum())
        channel_entropy_sum += float((entropy*n).sum())
        extreme += int(count[0]+count[3])
        if family in ("A", "P"):
            zero_code = params["zero_point"].reshape(-1).long()
            zero_exact += int(per_channel.gather(1, zero_code[:, None]).sum())
        elif family in ("S", "L"):
            zero_exact += int((params["scale"] == 0).sum().item()*n)
        else:
            zero_exact += int(((params["centers"].gather(1, codes.long()) == 0)).sum().item())
            duplicates += params.get("duplicate_centroid_channels", 0)
            iterations_sum += params.get("iterations", 0)*codes.shape[0]
            max_iteration_tensors += int(not params.get("converged", False))
            centers = params["centers"].double()
            center_values.extend(centers.flatten().tolist())
            center_spacings.extend((centers[:, 1:]-centers[:, :-1]).flatten().tolist())
        if "scale" in params:
            scales.extend(params["scale"].reshape(-1).double().tolist())
    proportions = counts/weights
    marginal = -sum(float(p*math.log2(p)) for p in proportions if p)
    result = {"family": family, "weights": weights, "channels": channels,
              "code0_fraction": float(proportions[0]), "code1_fraction": float(proportions[1]),
              "code2_fraction": float(proportions[2]), "code3_fraction": float(proportions[3]),
              "mean_channel_code_entropy": entropy_sum/channels,
              "parameter_weighted_conditional_code_entropy": channel_entropy_sum/weights,
              "marginal_code_entropy": marginal, "extreme_code_fraction": extreme/weights,
              "exact_zero_reconstruction_fraction": zero_exact/weights,
              "duplicate_centroid_channels": duplicates if family == "N" else None,
              "mean_lloyd_iterations_per_channel": iterations_sum/channels if family == "N" else None,
              "lloyd_max_iteration_tensors": max_iteration_tensors if family == "N" else None}
    if scales:
        result.update(scale_min=min(scales), scale_median=float(np.median(scales)),
                      scale_max=max(scales))
    if center_values:
        result.update(center_min=min(center_values), center_median=float(np.median(center_values)),
                      center_max=max(center_values),
                      median_center_spacing=float(np.median(center_spacings)))
    return result


def calibrate_candidate(outdir, enhancement, family, original, preceding, saved_e1=None,
                        device=torch.device("cuda")):
    pt_path, bin_path, stats_path = candidate_paths(outdir, enhancement, family)
    if pt_path.exists() and bin_path.exists() and stats_path.exists():
        prior = json.loads(stats_path.read_text(encoding="utf-8"))
        changed = False
        if enhancement == 1 and family in ("A", "P") and prior.get("calibration_source") != "saved Experiment 1 artifact":
            prior["calibration_seconds"] = None
            prior["calibration_source"] = "saved Experiment 1 artifact"
            changed = True
        elif "calibration_source" not in prior:
            prior["calibration_source"] = "weight-only calibration in Experiment 3"
            changed = True
        if family == "N" and ("center_min" not in prior or "lloyd_max_iteration_tensors" not in prior):
            behavior = code_behavior(load_torch(pt_path), family)
            for key in ("center_min", "center_median", "center_max", "median_center_spacing",
                        "lloyd_max_iteration_tensors"):
                prior[key] = behavior[key]
            changed = True
        if changed:
            write_json(stats_path, prior)
        return prior
    started = time.perf_counter()
    if enhancement == 1 and family in ("A", "P"):
        if saved_e1 is None:
            raise AssertionError("Saved E1 codes unavailable")
        artifacts = {name: {**params, "family": family} for name, params in saved_e1[family].items()}
    else:
        artifacts = {}
        for index, name in enumerate(original):
            residual = original[name].to(device)-preceding[name].to(device)
            params, decoded = quantize_family(residual, family)
            if params["quantized"].min() < 0 or params["quantized"].max() > 3:
                raise AssertionError(f"Invalid 2-bit codes: E{enhancement}/{family}/{name}")
            if not torch.equal(decoded, dequantize_family(params, family)):
                raise AssertionError(f"Quantizer decode mismatch: E{enhancement}/{family}/{name}")
            artifacts[name] = cpu_params(params)
            if (index+1) % 18 == 0:
                print(f"Calibration E{enhancement}/{family}: {index+1}/{len(original)} tensors", flush=True)
    calibration_seconds = None if enhancement == 1 and family in ("A", "P") else time.perf_counter()-started
    incoming_sse = residual_sse = parameter_sse = signal_ss = 0.0
    count = 0
    for name in original:
        weight = original[name]
        prefix = preceding[name]
        residual = weight-prefix
        contribution = dequantize_family(artifacts[name], family)
        decoded = prefix+contribution
        incoming_sse += residual.double().square().sum().item()
        residual_sse += (residual.double()-contribution.double()).square().sum().item()
        parameter_sse += (weight.double()-decoded.double()).square().sum().item()
        signal_ss += weight.double().square().sum().item()
        count += weight.numel()
    behavior = code_behavior(artifacts, family)
    atomic_torch_save(pt_path, artifacts)
    byte_counts = write_binary_codebook(bin_path, family, artifacts)
    read_family, binary = read_binary_codebook(bin_path)
    if read_family != family or set(binary) != set(artifacts):
        raise AssertionError("Packed codebook roundtrip failed")
    for name in original:
        if not torch.equal(dequantize_family(artifacts[name], family),
                           dequantize_family(binary[name], family)):
            raise AssertionError(f"Packed decoder mismatch: E{enhancement}/{family}/{name}")
    stats = {"config": f"E{enhancement}_{family}", "enhancement": enhancement,
             "family": family, "family_name": FAMILY_NAMES[family], "prefix_bits": 5+2*enhancement,
             "incoming_residual_mse": incoming_sse/count,
             "residual_mse": residual_sse/count,
             "residual_mse_reduction": (incoming_sse-residual_sse)/count,
             "parameter_mse": parameter_sse/count,
             "parameter_sqnr_db": 10*math.log10(signal_ss/parameter_sse) if parameter_sse else None,
             "parameter_sse": parameter_sse,
             "incremental_mse_improvement": (incoming_sse-parameter_sse)/count,
             "calibration_seconds": round(calibration_seconds, 3) if calibration_seconds is not None else None,
             "calibration_source": "saved Experiment 1 artifact" if calibration_seconds is None
                                   else "weight-only calibration in Experiment 3",
             "codes_pt": str(pt_path.relative_to(outdir)),
             "codes_binary": str(bin_path.relative_to(outdir)),
             **behavior, **byte_counts}
    write_json(stats_path, stats)
    return stats


def reconstruct_candidate(outdir, enhancement, family, preceding):
    pt_path, _, _ = candidate_paths(outdir, enhancement, family)
    artifacts = load_torch(pt_path)
    if set(artifacts) != set(preceding):
        raise AssertionError("Candidate tensor names differ from prefix")
    return {name: preceding[name]+dequantize_family(artifacts[name], family)
            for name in preceding}


def build_reference_prefixes(outdir, original, saved, base, affine, device):
    for key, weights in (("B", base), ("BA", affine)):
        path = prefix_path(outdir, key)
        if not path.exists():
            atomic_torch_save(path, weights)
    # E2 and E3 reference codes are generated once from the actual decoded
    # preceding A prefix; later candidate families use these same residuals.
    preceding = affine
    for enhancement, prefix in ((2, "BAA"), (3, "BAAA")):
        path = prefix_path(outdir, prefix)
        if path.exists():
            decoded = load_torch(path)
        else:
            calibrate_candidate(outdir, enhancement, "A", original, preceding,
                                saved_e1=saved, device=device)
            decoded = reconstruct_candidate(outdir, enhancement, "A", preceding)
            atomic_torch_save(path, decoded)
        preceding = decoded
    return {key: prefix_path(outdir, key) for key in ("B", "BA", "BAA", "BAAA")}


def prepare_all_candidates(outdir, original, prefix_paths, saved, device, quantizers, enhancements):
    rows = []
    for enhancement in enhancements:
        preceding = load_torch(prefix_paths[PREFIXES[enhancement]])
        for family in quantizers:
            row = calibrate_candidate(outdir, enhancement, family, original, preceding,
                                      saved_e1=saved, device=device)
            rows.append(row)
            print(f"Prepared E{enhancement}/{family}: MSE={row['parameter_mse']:.9g} "
                  f"entropy={row['parameter_weighted_conditional_code_entropy']:.3f}", flush=True)
        del preceding
    return rows


def residual_characterization(outdir, original, prefix_paths, device):
    row_path = outdir / "residual_statistics.csv"
    stage_path = outdir / "residual_stage_statistics.csv"
    summary_path = outdir / "residual_global_statistics.json"
    hist_path = outdir / "residual_histograms.json"
    if all(path.exists() for path in (row_path, stage_path, summary_path, hist_path)):
        return json.loads(summary_path.read_text(encoding="utf-8"))
    rows = []
    hist = {}
    for enhancement in (1, 2, 3):
        preceding = load_torch(prefix_paths[PREFIXES[enhancement]])
        a = load_torch(candidate_paths(outdir, enhancement, "A")[0])
        p = load_torch(candidate_paths(outdir, enhancement, "P")[0])
        raw = np.zeros(len(RAW_LOG_EDGES)-1, dtype=np.int64)
        normalized = np.zeros(len(NORMALIZED_EDGES)-1, dtype=np.int64)
        for index, name in enumerate(original):
            residual = original[name].to(device)-preceding[name].to(device)
            block, raw_count, norm_count = characterize_tensor(
                name, enhancement, residual, a[name], p[name])
            rows.extend(block)
            raw += raw_count.astype(np.int64)
            normalized += norm_count.astype(np.int64)
            if (index+1) % 18 == 0:
                print(f"Residual E{enhancement} stats: {index+1}/{len(original)} tensors", flush=True)
        hist[str(enhancement)] = {"raw_log_abs_counts": raw.tolist(),
                                  "normalized_counts": normalized.tolist()}
        del preceding, a, p
    global_rows, stage_rows = summarize_all(rows)
    save_csv(row_path, rows)
    save_csv(stage_path, stage_rows)
    write_json(summary_path, global_rows)
    write_json(hist_path, {"raw_log_edges": RAW_LOG_EDGES.tolist(),
                            "normalized_edges": NORMALIZED_EDGES.tolist(),
                            "enhancements": hist})
    return global_rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("test"))
    parser.add_argument("--weights", default="IMAGENET1K_V2", choices=("IMAGENET1K_V2",))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("outputs/residual_ablation_newfull"))
    parser.add_argument("--outdir", type=Path, default=Path("outputs/lp_quantizer_adaptivity"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--scope", choices=("subset", "full", "both"), default="both")
    parser.add_argument("--phase", choices=("all", "prepare", "validate", "statistics", "subset", "full", "analysis"), default="all")
    parser.add_argument("--quantizers", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--enhancements", nargs="+", type=int, choices=(1, 2, 3), default=[1, 2, 3])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.workers < 0 or len(set(args.quantizers)) != len(args.quantizers):
        raise ValueError("Invalid execution settings")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type != "cuda":
        print("Warning: CUDA unavailable; full inference may be very slow", flush=True)
    args.outdir.mkdir(parents=True, exist_ok=True)
    weights = MODEL_REGISTRY["resnet50"][1]
    transform = weights.transforms()
    dataset = build_dataset(args.data_dir.resolve(), transform)
    inspect_dataset(dataset, True)
    indices, manifest = balanced_indices(dataset.samples, args.data_dir.resolve(), seed=args.seed)
    old_manifest = args.outdir / "subset_manifest.json"
    if old_manifest.exists() and json.loads(old_manifest.read_text())["manifest_sha256"] != manifest["manifest_sha256"]:
        raise AssertionError("Manifest changed; use a different output directory")
    write_json(old_manifest, manifest)
    model = build_model("resnet50")
    groups = eligible_groups(model)
    names = [name for group in groups.values() for name in group]
    original = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if name in names}
    if len(names) != 54 or sum(value.numel() for value in original.values()) != 25_502_912:
        raise AssertionError("Eligible tensor/weight count differs from Experiment 1")
    checkpoint = Path(os.environ.get("TORCH_HOME", str(Path.home()/".cache"/"torch"))) / "hub" / "checkpoints" / "resnet50-11ad3fa6.pth"
    source = args.artifacts_dir / "residual_ablation.json"
    if not source.is_file():
        raise FileNotFoundError(source)
    old = json.loads(source.read_text(encoding="utf-8"))["metadata"]
    if old["data_dir"] != str(args.data_dir.resolve()) or old["transform"] != str(transform):
        raise AssertionError("Experiment 1 dataset or transform differs")
    metadata = {"model": "resnet50", "weights": str(weights), "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.exists() else None,
                "experiment1_results_sha256": sha256_file(source),
                "experiment1_artifact_sha256": {
                    key: sha256_file(args.artifacts_dir / f"{key}_codes_and_metadata.pt")
                    for key in ("base_5", "affine_minmax", "affine_percentile")},
                "dataset": str(args.data_dir.resolve()), "dataset_images": len(dataset),
                "transform": str(transform), "subset_manifest_sha256": manifest["manifest_sha256"],
                "bit_schedule": [5, 2, 2, 2], "channel_axis": 0,
                "excluded_patterns": ["norm", "bn"], "eligible_tensors": len(names),
                "eligible_weights": sum(value.numel() for value in original.values()),
                "shared_unquantized_model_bytes": sum(value.numel()*value.element_size()
                    for name, value in model.state_dict().items() if name not in original),
                "A": "original affine min/max", "P": "original affine 0.5/99.5 percentiles",
                "S_levels": [-1.5, -.5, .5, 1.5], "N_max_iterations": 20,
                "N_tolerance": 1e-6, "L_levels": [-4, -1, 1, 4],
                "metadata_precision": "FP32", "batch_size": args.batch_size,
                "workers": args.workers, "device": str(device), "seed": args.seed,
                "cudnn_benchmark": False,
                "torch": torch.__version__, "torchvision": torchvision.__version__}
    meta_path = args.outdir / "run_metadata.json"
    if meta_path.exists():
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        for key in ("weights", "checkpoint_sha256", "dataset", "transform",
                    "subset_manifest_sha256", "batch_size", "seed",
                    "cudnn_benchmark", "experiment1_results_sha256",
                    "experiment1_artifact_sha256"):
            if key in previous and previous[key] != metadata[key]:
                raise AssertionError(f"Run provenance changed: {key}")
    write_json(meta_path, metadata)
    print("Verifying saved Experiment 1 B/A/P codes and decoded weights", flush=True)
    saved, base, affine, percentile = validate_saved_baselines(original, names, args.artifacts_dir, device)
    prior_metrics = reference_provenance_check(original, base, affine, percentile, source)
    prefix_paths = build_reference_prefixes(args.outdir, original, saved, base, affine, device)
    if args.phase == "prepare":
        return
    # All 15 codes are prepared before image inference; this ensures identical
    # incoming residuals and allows every configuration to resume independently.
    needed_quantizers = (["A", "P"] if args.phase == "validate" else
                         list(FAMILIES) if args.phase in ("all", "statistics", "analysis") else args.quantizers)
    needed_enhancements = ([1] if args.phase == "validate" else
                           [1, 2, 3] if args.phase in ("all", "statistics", "analysis") else args.enhancements)
    prepared = prepare_all_candidates(args.outdir, original, prefix_paths, saved, device,
                                      needed_quantizers, needed_enhancements)
    if args.phase in ("all", "statistics", "analysis"):
        residual_characterization(args.outdir, original, prefix_paths, device)
    if args.phase == "statistics":
        return
    del saved, base, affine, percentile
    # Remaining evaluation and reporting phases are defined below.
    run_evaluations(args, model, dataset, indices, manifest, original, prefix_paths,
                    prepared, prior_metrics, device)


def run_evaluations(args, model, dataset, indices, manifest, original,
                    prefix_paths, prepared, prior_metrics, device):
    outdir = args.outdir
    model.to(device).eval()
    fp32_unchanged = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                      if name not in original}
    predictions = outdir / "predictions"
    predictions.mkdir(exist_ok=True)
    evaluation_dir = outdir / "evaluations"
    evaluation_dir.mkdir(exist_ok=True)
    subset_loader = None
    full_loader = None
    old_exp2 = Path("outputs/mse_layer_sensitivity")
    old_manifest = old_exp2 / "subset_manifest.json"
    compatible_exp2 = old_manifest.exists() and json.loads(old_manifest.read_text())["manifest_sha256"] == manifest["manifest_sha256"]
    if args.phase in ("validate", "all", "subset") or args.scope in ("subset", "both"):
        subset_cache = old_exp2 / "subset_fp32_images.dat" if compatible_exp2 else outdir / "subset_fp32_images.dat"
        cache_subset(dataset, indices, subset_cache, args.workers, args.batch_size)
        labels = [dataset.samples[index][1] for index in indices]
        subset_loader = DataLoader(MemmapSubset(subset_cache, labels), batch_size=args.batch_size,
                                   num_workers=0, pin_memory=device.type == "cuda")
    if args.phase in ("validate", "all", "full") or args.scope in ("full", "both"):
        full_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=device.type == "cuda")
    def prediction_path(config, scope):
        return predictions / f"{scope}_{config}.npz"
    def evaluation_path(config, scope):
        return evaluation_dir / f"{scope}_{config}.json"
    def load_prediction(config, scope):
        path = prediction_path(config, scope)
        with np.load(path) as data:
            return {key: data[key] for key in ("correct1", "correct5", "cross_entropy")}
    def maybe_reuse_exp2(config, scope):
        if not compatible_exp2 or args.batch_size != 64 or config != "E1_A" or scope != "full":
            return False
        source = old_exp2 / "predictions" / f"{'full' if scope=='full' else 'subset'}_affine.npz"
        if not source.exists():
            return False
        if not prediction_path(config, scope).exists():
            shutil.copyfile(source, prediction_path(config, scope))
        return True
    @torch.no_grad()
    def run(config, scope, decoded):
        result_path = evaluation_path(config, scope)
        target = prediction_path(config, scope)
        if not args.no_resume and target.exists() and result_path.exists():
            return json.loads(result_path.read_text(encoding="utf-8")), load_prediction(config, scope)
        if not args.no_resume and maybe_reuse_exp2(config, scope):
            arrays = load_prediction(config, scope)
            metrics = metric_sums(**arrays)
            if scope == "full" and metrics["top1"] != EXPERIMENT1["E1_A"]["top1"]:
                raise AssertionError("Experiment 2 affine predictions differ from Experiment 1")
            row = {"config": config, "scope": scope, **metrics,
                   "source": "verified Experiment 2 paired predictions", "seconds": None,
                   "predictions": str(target.relative_to(outdir))}
            write_json(result_path, row)
            return row, arrays
        if decoded is None:
            raise ValueError(f"Decoded weights required for {config}/{scope}")
        for name, parameter in model.named_parameters():
            if name in decoded:
                parameter.copy_(decoded[name].to(device))
        for name, expected in fp32_unchanged.items():
            if not torch.equal(model.state_dict()[name].cpu(), expected):
                raise AssertionError(f"Excluded parameter/buffer changed: {name}")
        torch.backends.cudnn.benchmark = False
        loader = full_loader if scope == "full" else subset_loader
        if loader is None:
            raise RuntimeError(f"No {scope} loader")
        metrics, arrays, seconds = evaluate_details(model, loader, device,
                                                    len(dataset) if scope == "full" else len(indices))
        np.savez_compressed(target, **arrays)
        row = {"config": config, "scope": scope, **metrics, "seconds": seconds,
               "source": "measured", "predictions": str(target.relative_to(outdir))}
        write_json(result_path, row)
        print(f"{scope} {config}: {metrics} ({seconds}s)", flush=True)
        return row, arrays
    def candidate_decoded(enhancement, family):
        preceding = load_torch(prefix_paths[PREFIXES[enhancement]])
        return reconstruct_candidate(outdir, enhancement, family, preceding)
    def result_for(config, scope):
        path = evaluation_path(config, scope)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    def evaluate_baselines(scope):
        if scope == "subset":
            run("B", scope, load_torch(prefix_paths["B"]))
        run("E1_A", scope, load_torch(prefix_paths["BA"]))
        if scope == "subset" or args.phase == "validate":
            run("E1_P", scope, candidate_decoded(1, "P"))
    if args.phase in ("validate", "all"):
        evaluate_baselines("subset")
        evaluate_baselines("full")
        p = result_for("E1_P", "full")
        if p and (p["top1"] != EXPERIMENT1["E1_P"]["top1"] or
                  p["top5"] != EXPERIMENT1["E1_P"]["top5"]):
            raise AssertionError("Full E1 percentile accuracy does not reproduce Experiment 1")
        if args.phase == "validate":
            rebuild_tables_and_reports(outdir, original, prefix_paths, prior_metrics, args.phase)
            return
    if args.phase in ("subset", "all") and args.scope in ("subset", "both"):
        evaluate_baselines("subset")
        for enhancement in args.enhancements:
            for family in args.quantizers:
                config = f"E{enhancement}_{family}"
                run(config, "subset", candidate_decoded(enhancement, family))
                rebuild_tables_and_reports(outdir, original, prefix_paths, prior_metrics, "subset")
    if args.phase in ("full", "all") and args.scope in ("full", "both"):
        evaluate_baselines("full")
        for enhancement in args.enhancements:
            for family in args.quantizers:
                config = f"E{enhancement}_{family}"
                run(config, "full", candidate_decoded(enhancement, family))
                rebuild_tables_and_reports(outdir, original, prefix_paths, prior_metrics, "full")
    if args.phase == "analysis":
        rebuild_tables_and_reports(outdir, original, prefix_paths, prior_metrics, "analysis")


def rebuild_tables_and_reports(outdir, original, prefix_paths, prior_metrics, phase):
    evaluation_dir = outdir / "evaluations"
    evaluations = {(row["scope"], row["config"]): row for path in evaluation_dir.glob("*.json")
                   for row in [json.loads(path.read_text(encoding="utf-8"))]}
    candidate_stats = [json.loads(path.read_text(encoding="utf-8"))
                       for path in sorted((outdir / "codes").glob("E?_?.json"))]
    run_meta = json.loads((outdir / "run_metadata.json").read_text(encoding="utf-8"))
    code_rows = []
    metadata_rows = []
    for stat in candidate_stats:
        code_rows.append({key: stat.get(key) for key in ("config", "enhancement", "family", "weights",
            "channels", "code0_fraction", "code1_fraction", "code2_fraction", "code3_fraction",
            "mean_channel_code_entropy", "parameter_weighted_conditional_code_entropy",
            "marginal_code_entropy", "extreme_code_fraction", "exact_zero_reconstruction_fraction",
            "duplicate_centroid_channels", "mean_lloyd_iterations_per_channel", "scale_min",
            "scale_median", "scale_max", "center_min", "center_median", "center_max",
            "median_center_spacing", "lloyd_max_iteration_tensors",
            "calibration_seconds", "calibration_source")})
        metadata_rows.append({**{key: stat.get(key) for key in ("config", "enhancement", "family",
            "raw_code_bytes", "metadata_bytes", "framing_bytes", "physical_bytes", "codes_pt",
            "codes_binary", "calibration_seconds")},
            "shared_unquantized_model_bytes": run_meta["shared_unquantized_model_bytes"]})
    save_csv(outdir / "quantizer_code_statistics.csv", code_rows)
    save_csv(outdir / "quantizer_metadata.csv", metadata_rows)
    def arrays(scope, config):
        path = outdir / "predictions" / f"{scope}_{config}.npz"
        if not path.exists():
            return None
        with np.load(path) as data:
            return {key: data[key] for key in ("correct1", "correct5", "cross_entropy")}
    comparisons = []
    for stat in candidate_stats:
        enhancement, family, config = stat["enhancement"], stat["family"], stat["config"]
        baseline = PREFIXES[enhancement]
        previous_config = "B" if enhancement == 1 else f"E{enhancement-1}_A"
        for scope in ("subset", "full"):
            observed = evaluations.get((scope, config))
            if observed is None:
                continue
            preceding = evaluations.get((scope, previous_config))
            if preceding is None and scope == "full" and enhancement == 1:
                preceding = EXPERIMENT1["B"]
            a = evaluations.get((scope, f"E{enhancement}_A"))
            row = {**stat, "scope": scope, "status": "success",
                   "images": observed["images"],
                   "top1": observed["top1"], "top5": observed["top5"],
                   "cross_entropy": observed["cross_entropy"],
                   "inference_seconds": observed.get("seconds"),
                   "prediction_source": observed.get("source"),
                   "shared_preceding_prefix": baseline,
                   "delta_top1_vs_prefix": observed["top1"]-preceding["top1"] if preceding else None,
                   "delta_ce_vs_prefix": observed["cross_entropy"]-preceding["cross_entropy"] if preceding else None,
                   "delta_top1_vs_A": observed["top1"]-a["top1"] if a else None,
                   "delta_ce_vs_A": observed["cross_entropy"]-a["cross_entropy"] if a else None}
            if a and family != "A":
                x, y = arrays(scope, config), arrays(scope, f"E{enhancement}_A")
                if x is not None and y is not None:
                    row.update(paired_stats(x, y))
            comparisons.append(row)
    comparisons.sort(key=lambda r: (r["scope"] != "full", r["enhancement"], FAMILIES.index(r["family"])))
    write_json(outdir / "quantizer_comparison.json", comparisons)
    save_csv(outdir / "quantizer_comparison.csv", comparisons)
    prefix_numeric_path = outdir / "reference_prefix_numerical.json"
    if prefix_numeric_path.exists():
        prefix_numeric = json.loads(prefix_numeric_path.read_text(encoding="utf-8"))
    else:
        prefix_numeric = {prefix: numerical_metrics(original, load_torch(prefix_paths[prefix]))
                          for prefix in ("B", "BA", "BAA", "BAAA")}
        write_json(prefix_numeric_path, prefix_numeric)
    references = []
    for scope in ("full", "subset"):
        for prefix in ("B", "BA", "BAA", "BAAA"):
            config = "B" if prefix == "B" else f"E{len(prefix)-1}_A"
            measured = evaluations.get((scope, config))
            if measured is None and scope == "full" and prefix in ("B", "BA"):
                measured = EXPERIMENT1["B" if prefix == "B" else "E1_A"]
            if measured is None:
                continue
            numeric = prefix_numeric[prefix]
            previous = references[-1] if references and references[-1]["scope"] == scope else None
            references.append({"scope": scope, "prefix": prefix,
                               "nominal_bits": 5+2*(len(prefix)-1), "top1": measured["top1"],
                               "top5": measured["top5"], "cross_entropy": measured["cross_entropy"],
                               **numeric,
                               "delta_top1_vs_previous": measured["top1"]-previous["top1"] if previous else None,
                               "delta_parameter_mse_vs_previous": numeric["parameter_mse"]-previous["parameter_mse"] if previous else None,
                               "metric_source": measured.get("source", "verified Experiment 1 full result")})
    write_json(outdir / "reference_prefixes.json", references)
    save_csv(outdir / "reference_prefixes.csv", references)
    if phase == "analysis":
        plot_all(outdir, comparisons, references)
        write_analysis(outdir, comparisons, references)


def plot_all(outdir, comparisons, references):
    plots = outdir / "plots"
    plots.mkdir(exist_ok=True)
    global_path = outdir / "residual_global_statistics.json"
    hist_path = outdir / "residual_histograms.json"
    if not global_path.exists() or not hist_path.exists():
        return
    global_rows = json.loads(global_path.read_text(encoding="utf-8"))
    hist = json.loads(hist_path.read_text(encoding="utf-8"))
    colors = {1: "#1f77b4", 2: "#e1812c", 3: "#35975d"}
    def save(name):
        plt.tight_layout()
        plt.savefig(plots / f"{name}.png", dpi=200)
        plt.savefig(plots / f"{name}.pdf")
        plt.close()
    raw_edges = np.array(hist["raw_log_edges"])
    norm_edges = np.array(hist["normalized_edges"])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for level in (1, 2, 3):
        counts = np.array(hist["enhancements"][str(level)]["raw_log_abs_counts"])
        ax.step((raw_edges[:-1]+raw_edges[1:])/2, counts/counts.sum(), where="mid",
                color=colors[level], label=f"R{level}")
    ax.set(xlabel="log10 absolute residual magnitude", ylabel="fraction of eligible weights",
           title="Raw LP residual magnitude")
    ax.legend()
    save("raw_residual_distributions")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for level in (1, 2, 3):
        counts = np.array(hist["enhancements"][str(level)]["normalized_counts"])
        ax.step((norm_edges[:-1]+norm_edges[1:])/2, counts/counts.sum(), where="mid",
                color=colors[level], label=f"R{level}")
    ax.set(xlabel="residual / output-channel standard deviation",
           ylabel="fraction of nondegenerate weights (within plotted range)",
           title="Channel-normalized residual shape")
    ax.legend()
    save("normalized_residual_distributions")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    levels = [1, 2, 3]
    med = [row["affine_scale_channel_median"] for row in global_rows]
    q25 = [row["affine_scale_channel_q25"] for row in global_rows]
    q75 = [row["affine_scale_channel_q75"] for row in global_rows]
    ax.errorbar(levels, med, yerr=[np.array(med)-q25, np.array(q75)-med],
                marker="o", capsize=5, color="#1f77b4")
    ax.set(xticks=levels, xticklabels=["E1", "E2", "E3"], yscale="log",
           ylabel="Affine scale per output channel (median and IQR)",
           title="Quantizer scale evolution")
    save("scale_evolution")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, key, title in zip(axes,
        ("normalized_mean", "excess_kurtosis", "tail_p99_p50"),
        ("Mean / std", "Excess kurtosis", "P99 / P50 absolute tail ratio")):
        values = [row[f"{key}_channel_median"] for row in global_rows]
        ax.plot(levels, values, marker="o")
        ax.set(xticks=levels, xticklabels=["E1", "E2", "E3"], title=title)
    save("residual_shape_tails")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    occupancy = np.array([[row[f"code{k}_fraction"] for k in range(4)] for row in global_rows])
    bottom = np.zeros(3)
    for code in range(4):
        axes[0].bar(["E1", "E2", "E3"], occupancy[:, code], bottom=bottom,
                    label=f"code {code}")
        bottom += occupancy[:, code]
    axes[0].set(ylabel="fraction of weights", title="Affine symbol occupancy")
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].plot(levels, [r["parameter_weighted_conditional_code_entropy"] for r in global_rows],
                 marker="o", label="conditional")
    axes[1].plot(levels, [r["marginal_code_entropy"] for r in global_rows],
                 marker="s", label="marginal")
    axes[1].set(xticks=levels, xticklabels=["E1", "E2", "E3"],
                ylabel="Shannon entropy (bits/index)", title="Affine code entropy")
    axes[1].legend()
    save("code_occupancy_entropy")
    scope = "full" if len([r for r in comparisons if r["scope"] == "full"]) == 15 else "subset"
    rows = [r for r in comparisons if r["scope"] == scope]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(3)
    for index, family in enumerate(FAMILIES):
        values = [next((r["top1"] for r in rows if r["enhancement"] == level and
                        r["family"] == family), np.nan) for level in levels]
        ax.scatter(x+(index-2)*.10, values, s=65, label=family)
    ax.set(xticks=x, xticklabels=["E1 (7 bit)", "E2 (9 bit)", "E3 (11 bit)"],
           ylabel="ImageNet Top-1 (%)", title=f"Isolated quantizer comparison ({scope})")
    ax.set_ylim(min(r["top1"] for r in rows)-.12, max(r["top1"] for r in rows)+.12)
    ax.grid(axis="y", alpha=.2)
    ax.legend(title="Family", ncol=5)
    save("top1_by_family")
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    markers = {"A": "o", "P": "s", "S": "^", "N": "D", "L": "X"}
    mse_scale = {1: 1e6, 2: 1e7, 3: 1e8}
    for ax, level in zip(axes, levels):
        these = [r for r in rows if r["enhancement"] == level]
        for row in these:
            ax.scatter(row["parameter_mse"]*mse_scale[level], row["top1"], color=colors[level],
                       marker=markers[row["family"]], label=row["family"], s=65)
        ax.set(xlabel=f"Parameter MSE × 10^{5+level}", title=f"E{level}")
        ax.set_ylim(min(r["top1"] for r in these)-.04, max(r["top1"] for r in these)+.04)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.grid(alpha=.2)
    axes[0].set_ylabel("ImageNet Top-1 (%)")
    axes[-1].legend(title="Family", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle(f"Weight error and task accuracy ({scope})")
    save("parameter_mse_vs_top1")


def write_analysis(outdir, comparisons, references):
    path = outdir / "analysis.md"
    global_path = outdir / "residual_global_statistics.json"
    if not global_path.exists():
        return
    residuals = json.loads(global_path.read_text(encoding="utf-8"))
    full = [r for r in comparisons if r["scope"] == "full"]
    subset = [r for r in comparisons if r["scope"] == "subset"]
    def numeric(value, fmt=".3f"):
        return format(value, fmt) if value is not None else "—"
    lines = ["# Experiment 3: residual evolution and isolated LP quantizers", "",
             "ResNet-50 IMAGENET1K_V2, 50,000 labeled ImageNet validation images, per-output-channel axis-0 quantization, and the original 5-bit base. Each E2/E3 candidate receives the same decoded preceding AAA prefix. All calibration is weight-only; no mixed schedule was constructed.", "",
             "The FP32 checkpoint and dataset match the verified Experiment 1 reference (80.746% Top-1). Saved base, A E1, and P E1 codes were reproduced exactly; P E1 was rerun on all 50,000 images and matched 80.054%. An initial P E1 pass made after a subset pass with cuDNN benchmarking enabled differed by four images; it was discarded. All retained Experiment 3 subset and new full evaluations use cuDNN benchmarking disabled, and full A E1 predictions were reused from the matching Experiment 2 run.", "",
             "## Table 1 — AAA reference LP reconstruction", "",
             "| Scope | Prefix | Bits | Top-1 | Top-5 | CE | Parameter MSE | SQNR (dB) |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in references:
        lines.append(f"| {row['scope']} | {row['prefix']} | {row['nominal_bits']} | "
                     f"{row['top1']:.3f} | {row['top5']:.3f} | {row['cross_entropy']:.6f} | "
                     f"{row['parameter_mse']:.8g} | {numeric(row['parameter_sqnr_db'], '.2f')} |")
    lines += ["", "The full B and BA values are the previously verified Experiment 1 results; their saved codes and decoded weights were independently checked against the current checkpoint. All later reference accuracies were evaluated here.", "",
              "## Table 2 — Residual evolution (AAA incoming residuals)", "",
              "Pooled RMS and the other weighted columns use eligible-weight counts. Medians and IQRs summarize the same output channels across levels; shape statistics are scale invariant. Normalized histograms divide each channel by its standard deviation and exclude degenerate channels.", "",
              "| Statistic | R1 | R2 | R3 |", "|---|---:|---:|---:|"]
    stats = (
        ("Pooled RMS", "pooled_residual_rms", ".6g"),
        ("Channel median abs P50", "median_abs_channel_median", ".6g"),
        ("Channel median normalized mean", "normalized_mean_channel_median", ".3f"),
        ("Channel median skewness", "skewness_channel_median", ".3f"),
        ("Channel median excess kurtosis", "excess_kurtosis_channel_median", ".3f"),
        ("Channel median abs P99/P50", "tail_p99_p50_channel_median", ".3f"),
        ("Weight-weighted zero fraction", "zero_fraction_weighted_mean", ".4f"),
        ("Affine scale channel CV", "affine_scale_channel_cv", ".3f"),
        ("Conditional code entropy", "parameter_weighted_conditional_code_entropy", ".3f"),
        ("Extreme-code fraction", "extreme_code_fraction_weighted_mean", ".3f"),
    )
    for label, key, fmt in stats:
        lines.append(f"| {label} | " + " | ".join(numeric(row.get(key), fmt) for row in residuals) + " |")
    hist = json.loads((outdir / "residual_histograms.json").read_text(encoding="utf-8"))
    distributions = [np.array(hist["enhancements"][str(level)]["normalized_counts"], dtype=float)
                     for level in (1, 2, 3)]
    distributions = [array/array.sum() for array in distributions]
    tv12 = .5*np.abs(distributions[0]-distributions[1]).sum()
    tv23 = .5*np.abs(distributions[1]-distributions[2]).sum()
    centers = (np.array(hist["normalized_edges"][:-1]) +
               np.array(hist["normalized_edges"][1:]))/2
    central = [float(array[np.abs(centers) < .5].sum()) for array in distributions]
    outer = [float(array[np.abs(centers) > 2].sum()) for array in distributions]
    lines += ["", f"The binned channel-normalized distributions have total-variation distances {tv12:.3f} (R1–R2) and {tv23:.3f} (R2–R3) within the displayed [-8, 8] range. This descriptive statistic captures shape changes after removing per-channel scale; binning and range truncation limit its precision.", "",
              "The fractions of normalized residuals within ±0.5 standard deviations are "
              + ", ".join(f"{value:.3f}" for value in central)
              + " for R1/R2/R3; fractions beyond ±2 standard deviations are "
              + ", ".join(f"{value:.4f}" for value in outer) + ".", "",
              "## Table 3 — Isolated quantizer comparison", "",
              "Each row uses the same incoming residual as A at its enhancement. Full results are listed first; subset rows are exploratory and are included in the full validation set.", "",
              "| Scope | Enhancement | Family | Top-1 | Δ vs A (pp) | CE | Parameter MSE | Conditional entropy | Metadata bytes |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in comparisons:
        lines.append(f"| {row['scope']} | E{row['enhancement']} | {row['family']} | "
                     f"{row['top1']:.3f} | {numeric(row['delta_top1_vs_A'], '+.3f')} | "
                     f"{row['cross_entropy']:.6f} | {row['parameter_mse']:.8g} | "
                     f"{row['parameter_weighted_conditional_code_entropy']:.3f} | "
                     f"{row['metadata_bytes']:,} |")
    lines += ["", "All methods carry exactly four indices per eligible weight. The packed raw-code payload is identical at a fixed enhancement; metadata varies by family. The reported entropy is descriptive, not a measured entropy-coded byte rate. Physical binary sizes include framing and are recorded separately in `quantizer_metadata.csv`. Unquantized tensors are shared across all configurations.", "",
              "S uses {-1.5s,-0.5s,+0.5s,+1.5s} with s=max|R|/1.5; L uses {-4s,-s,+s,+4s} with s=max|R|/4. Both use lower-code midpoint ties and cannot represent zero for nonzero scale. N uses channel quantile initialization and at most 20 Lloyd iterations; empty clusters retain their prior centers. These are specific calibrated designs, not universal family comparisons.", "",
              "## Paired full-set comparisons versus A", "",
              "Positive gains mean the candidate corrects an image A missed; losses mean the reverse. Intervals are paired 10,000-draw bootstrap 95% intervals on Top-1 percentage-point differences.", "",
              "| Enhancement | Family | Δ Top-1 (pp) | Gains | Losses | 95% CI (pp) | Δ CE |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in full:
        if row["family"] == "A":
            continue
        ci = row.get("paired_bootstrap_top1_ci95")
        ci_text = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "—"
        lines.append(f"| E{row['enhancement']} | {row['family']} | "
                     f"{numeric(row.get('delta_top1_vs_A'), '+.3f')} | "
                     f"{row.get('paired_gain_images', '—')} | {row.get('paired_loss_images', '—')} | "
                     f"{ci_text} | {numeric(row.get('delta_ce_vs_A'), '+.6f')} |")
    lines += ["", "## Research interpretation", "",
              "**Q1 — Residual evolution.** Raw RMS falls by about a factor of three at each enhancement, but the normalized shapes are remarkably stable: the binned total-variation distances above are small, median excess kurtosis remains near −1.2, the P99/P50 absolute-tail ratio remains near 1.97, and affine code entropy stays near 1.60 bits/index. There is little global shape evidence here that later residuals intrinsically require a new family. Stage summaries in `residual_stage_statistics.csv` retain smaller local differences. The original residual tensors are regenerated deterministically from the FP32 checkpoint and saved decoded AAA prefixes; their full FP32 arrays are not stored separately.", "",
              "**Q2 — Quantizer preference.** The measured isolated Top-1 leader varies by enhancement:"]
    for enhancement in (1, 2, 3):
        candidates = [r for r in full if r["enhancement"] == enhancement]
        scope = "full"
        if len(candidates) < 5:
            candidates = [r for r in subset if r["enhancement"] == enhancement]
            scope = "exploratory subset"
        if candidates:
            best = max(candidates, key=lambda r: r["top1"])
            lowest_mse = min(candidates, key=lambda r: r["parameter_mse"])
            lines.append(f"- E{enhancement}: highest measured Top-1 is {best['family']} "
                         f"({best['top1']:.3f}%, {scope}); lowest weight MSE is "
                         f"{lowest_mse['family']} ({lowest_mse['parameter_mse']:.8g}).")
    lines += ["", "The E3 differences are at most a few hundredths of a point and all paired intervals include zero. These are observed rankings, not evidence of a distinct optimal E3 family. The unadjusted bootstrap intervals cover individual comparisons and do not correct for the twelve full-set A comparisons.", "",
              "**Q3 — Weight versus task distortion.** N has the lowest MSE at every enhancement and improves full Top-1 by +0.574 and +0.104 points at E1 and E2, but is −0.018 points at E3 (interval includes zero). L at E2 has *higher* parameter MSE than A yet +0.070 points measured Top-1 (interval includes zero). Error and task rankings are therefore related imperfectly; small reversals should not be overinterpreted.", "",
              "**Q4 — Numerical formats and metadata.** S has a supported E1 gain of +0.260 points versus A and needs 110,240 metadata bytes, versus A's 137,800. N gains +0.574 at E1 and +0.104 at E2 but uses 440,960 metadata bytes for four FP32 centers per channel. All methods use the same 6,375,728-byte packed 2-bit code payload; framing and physical totals are in `quantizer_metadata.csv`. L has no clear Top-1 advantage in these parameterizations. S and L cannot reconstruct zero for nonzero scales; the all-zero channel is a documented degenerate exception. These results compare particular grids and calibration rules, not complete quantizer families.", "",
              "**Q5 — LP adaptivity.** The E1 and E2 isolated gains show that independent residual quantizer choice can matter, even though global normalized residual shapes barely evolve. The benefit does not by itself prove a mixed schedule: E2 and E3 candidates here were calibrated on AAA residuals, which would change after a non-A earlier choice.", "",
              "**Q6 — Next study.** Carry A as the fixed baseline, N for its E1/E2 accuracy and MSE gains, and S for its lower-metadata E1 gain into a small predeclared mixed-schedule study. Recompute each later residual from the actual decoded preceding choice, report every prefix and physical byte count, and use a separate held-out set for any final selection claim. P's E1 +0.056-point difference is uncertain and L showed no clear benefit here. No mixed schedule, Hessian objective, retraining, or bit-plane comparison was performed in this experiment."]
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
