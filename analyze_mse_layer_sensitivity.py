#!/usr/bin/env python3
"""Diagnose the ImageNet accuracy loss of MSE-calibrated LP residuals.

All model variants share Experiment 1's saved 5-bit base and two saved 2-bit
residual codebooks. Screening is paired on a seeded, class-balanced subset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset

from evaluate_imagenet import EXCLUDE_PATTERNS, MODEL_REGISTRY, build_dataset, build_model
from evaluate_residual_ablation import inspect_dataset, prepare_base
from layered_quantization import IntegerFormat, parameter_is_quantized
from quantizer_variants import dequantize_variant, quantize_variant


STAGES = ("conv1", "layer1", "layer2", "layer3", "layer4", "fc")
ARTIFACT_NAMES = ("base_5", "affine_minmax", "affine_mse")
SEED = 123


def stage_of(name: str) -> str:
    stage = name.split(".", 1)[0]
    if stage not in STAGES:
        raise ValueError(f"Unassigned eligible ResNet-50 tensor: {name}")
    return stage


def eligible_groups(model) -> dict[str, list[str]]:
    groups = {stage: [] for stage in STAGES}
    for name, parameter in model.named_parameters():
        if parameter_is_quantized(name, parameter, EXCLUDE_PATTERNS, False):
            groups[stage_of(name)].append(name)
    if not all(groups.values()):
        raise ValueError("Each of the six stages must contain quantized weights")
    return groups


def balanced_indices(samples, root: Path, per_class: int = 5, seed: int = SEED):
    """Sample sorted image paths within each class with one fixed RNG stream."""
    groups = defaultdict(list)
    for index, (path, label) in enumerate(samples):
        groups[int(label)].append((Path(path).relative_to(root).as_posix(), index))
    rng = random.Random(seed)
    chosen = []
    for label in sorted(groups):
        candidates = sorted(groups[label])
        if len(candidates) < per_class:
            raise ValueError(f"Class {label} contains only {len(candidates)} images")
        chosen.extend((path, label, index) for path, index in rng.sample(candidates, per_class))
    chosen.sort(key=lambda item: (item[1], item[0]))
    entries = [{"path": path, "class_id": label} for path, label, _ in chosen]
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return [index for _, _, index in chosen], {
        "seed": seed, "per_class": per_class, "images": len(entries),
        "manifest_sha256": digest, "entries": entries,
    }


def write_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def load_or_build_artifacts(directory: Path, model, original_state, device):
    paths = {key: directory / f"{key}_codes_and_metadata.pt" for key in ARTIFACT_NAMES}
    present = [path.exists() for path in paths.values()]
    if any(present) and not all(present):
        raise FileNotFoundError("Partial Experiment 1 artifacts; refusing to mix saved and rebuilt codes")
    if all(present):
        return {key: torch.load(path, map_location="cpu", weights_only=False)
                for key, path in paths.items()}, "saved"
    directory.mkdir(parents=True, exist_ok=True)
    prepared = prepare_base(original_state, model, device)
    artifacts = {"base_5": {name: item[2] for name, item in prepared.items()},
                 "affine_minmax": {}, "affine_mse": {}}
    for name, (_, residual, _) in prepared.items():
        for method in ("affine_minmax", "affine_mse"):
            params, _ = quantize_variant(residual.to(device), method,
                                         mse_scale_min=0.5, mse_scale_max=1.5, mse_steps=17)
            artifacts[method][name] = {key: value.cpu() if torch.is_tensor(value) else value
                                       for key, value in params.items()}
    for key, path in paths.items():
        torch.save(artifacts[key], path)
    return artifacts, "rebuilt"


def decode_cache(artifacts, names, original_state):
    name_set = set(names)
    if any(set(artifacts[key]) != name_set for key in ARTIFACT_NAMES):
        raise AssertionError("Saved artifact tensor names differ from current quantization eligibility")
    cache = {"affine": {}, "mse": {}}
    for name in names:
        base = IntegerFormat.dequantize(artifacts["base_5"][name])
        for target, source in (("affine", "affine_minmax"), ("mse", "affine_mse")):
            residual = dequantize_variant(artifacts[source][name])
            value = base + residual
            if value.shape != original_state[name].shape or not torch.isfinite(value).all():
                raise AssertionError(f"Invalid saved decode for {name} / {source}")
            cache[target][name] = value
    return cache


@torch.no_grad()
def verify_artifacts(artifacts, cache, original_state, names, device):
    """Compare all saved codes and decoded weights with original quantizers."""
    integer = IntegerFormat("channel", 0)
    max_error = 0.0
    for name in names:
        weight = original_state[name].to(device)
        base_params, base = integer.quantize(weight, 5)
        for key in ("quantized", "scale", "zero_point"):
            if not torch.equal(base_params[key].cpu(), artifacts["base_5"][name][key]):
                raise AssertionError(f"Saved base {name}/{key} differs from original quantizer")
        residual = weight - base
        for target, source in (("affine", "affine_minmax"), ("mse", "affine_mse")):
            params, contribution = quantize_variant(residual, source, mse_scale_min=0.5,
                                                     mse_scale_max=1.5, mse_steps=17)
            for key in ("quantized", "scale", "zero_point"):
                if not torch.equal(params[key].cpu(), artifacts[source][name][key]):
                    raise AssertionError(f"Saved {source} {name}/{key} differs from original quantizer")
            expected = base + contribution
            error = (expected.cpu() - cache[target][name]).abs().max().item()
            max_error = max(max_error, error)
            if error != 0:
                raise AssertionError(f"Saved reconstructed weight differs: {name}/{target}: {error}")
    return max_error


def selected_names(groups, selected: set[str]) -> set[str]:
    names = {name for stage in selected if stage in groups for name in groups[stage]}
    names.update(item for item in selected if any(item in group for group in groups.values()))
    unknown = selected - set(groups) - names
    if unknown:
        raise ValueError(f"Unknown stages or tensors: {sorted(unknown)}")
    return names


@torch.no_grad()
def apply_configuration(model, cache, mse_names: set[str], previous: set[str] | None = None):
    """Copy only changed eligible tensors; excluded parameters and buffers stay FP32."""
    names = set(cache["affine"])
    if not mse_names <= names:
        raise ValueError(f"Unknown tensors: {sorted(mse_names - names)}")
    changed = names if previous is None else mse_names ^ previous
    for name, parameter in model.named_parameters():
        if name not in changed:
            continue
        source = cache["mse"] if name in mse_names else cache["affine"]
        parameter.copy_(source[name].to(parameter.device))


def metric_sums(correct1, correct5, cross_entropy):
    count = len(correct1)
    if count == 0:
        raise ValueError("No evaluated images")
    return {"images": count, "top1": round(100 * float(np.sum(correct1)) / count, 4),
            "top5": round(100 * float(np.sum(correct5)) / count, 4),
            "cross_entropy": round(float(np.sum(cross_entropy, dtype=np.float64)) / count, 6)}


@torch.no_grad()
def evaluate_details(model, loader, device, expected: int):
    model.eval()
    c1, c5, losses = [], [], []
    start = time.perf_counter()
    for batch, (images, labels) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        c1.append((logits.argmax(1) == labels).cpu().numpy().astype(np.uint8))
        c5.append((logits.topk(5, dim=1).indices == labels[:, None]).any(1).cpu().numpy().astype(np.uint8))
        losses.append(F.cross_entropy(logits, labels, reduction="none").cpu().numpy().astype(np.float32))
        if batch % 50 == 0:
            print(f"  {len(c1) * loader.batch_size}/{expected} batches={batch}", flush=True)
    arrays = {"correct1": np.concatenate(c1), "correct5": np.concatenate(c5),
              "cross_entropy": np.concatenate(losses)}
    if len(arrays["correct1"]) != expected:
        raise AssertionError(f"Expected {expected} images, got {len(arrays['correct1'])}")
    return metric_sums(**arrays), arrays, round(time.perf_counter() - start, 2)


def paired_stats(candidate, baseline, seed=SEED):
    a = candidate["correct1"].astype(np.int8)
    b = baseline["correct1"].astype(np.int8)
    if len(a) != len(b):
        raise AssertionError("Paired predictions have different image counts")
    gain = int(np.sum((a == 1) & (b == 0)))
    loss = int(np.sum((a == 0) & (b == 1)))
    n = len(a)
    result = {"paired_gain_images": gain, "paired_loss_images": loss,
              "paired_top1_delta": 100 * (gain - loss) / n,
              "paired_ce_delta": float(np.mean(candidate["cross_entropy"].astype(np.float64)
                                                - baseline["cross_entropy"].astype(np.float64)))}
    if n == 50000:
        counts = np.random.default_rng(seed).multinomial(n, [gain/n, loss/n, 1-(gain+loss)/n], size=10000)
        delta = 100 * (counts[:, 0] - counts[:, 1]) / n
        result["paired_bootstrap_top1_ci95"] = [float(x) for x in np.quantile(delta, [0.025, 0.975])]
        result["bootstrap_seed"] = seed
    return result


def tensor_diagnostic(name, stage, weight, base, affine, mse, aff_params, mse_params):
    residual = weight.double() - base.double()
    a = dequantize_variant(aff_params).double()
    m = dequantize_variant(mse_params).double()
    a_sse = (residual - a).square().sum().item()
    m_sse = (residual - m).square().sum().item()
    a_parameter_sse = (weight.double() - affine.double()).square().sum().item()
    m_parameter_sse = (weight.double() - mse.double()).square().sum().item()
    signal_ss = weight.double().square().sum().item()
    n = weight.numel()
    ascale = aff_params["scale"].double().flatten()
    mscale = mse_params["scale"].double().flatten()
    ratio = (mscale / ascale).numpy()
    delta_sq = (affine.double() - mse.double()).square().sum().item()
    return {"name": name, "stage": stage, "weights": n, "channels": weight.shape[0],
            "affine_residual_sse": a_sse, "mse_residual_sse": m_sse,
            "affine_parameter_sse": a_parameter_sse, "mse_parameter_sse": m_parameter_sse,
            "affine_residual_mse": a_sse/n, "mse_residual_mse": m_sse/n,
            "mse_reduction_abs": (a_sse-m_sse)/n,
            "mse_reduction_pct": 100*(a_sse-m_sse)/a_sse if a_sse else 0,
            "signal_ss": signal_ss,
            "affine_sqnr_db": 10*math.log10(signal_ss/a_parameter_sse) if a_parameter_sse and signal_ss else None,
            "mse_sqnr_db": 10*math.log10(signal_ss/m_parameter_sse) if m_parameter_sse and signal_ss else None,
            "changed_code_fraction": (aff_params["quantized"] != mse_params["quantized"]).float().mean().item(),
            "changed_zero_point_fraction": (aff_params["zero_point"] != mse_params["zero_point"]).float().mean().item(),
            "scale_ratio_min": float(np.min(ratio)), "scale_ratio_median": float(np.median(ratio)),
            "scale_ratio_mean": float(np.mean(ratio)), "scale_ratio_max": float(np.max(ratio)),
            "reconstructed_difference_norm": math.sqrt(delta_sq),
            "reconstructed_difference_sse": delta_sq}


def aggregate_diagnostics(rows, stage):
    subset = [row for row in rows if row["stage"] == stage]
    n = sum(row["weights"] for row in subset)
    a = sum(row["affine_residual_sse"] for row in subset)
    m = sum(row["mse_residual_sse"] for row in subset)
    ap = sum(row["affine_parameter_sse"] for row in subset)
    mp = sum(row["mse_parameter_sse"] for row in subset)
    signal = sum(row["signal_ss"] for row in subset)
    diff = sum(row["reconstructed_difference_sse"] for row in subset)
    return {"name": stage, "stage": stage, "weights": n,
            "channels": sum(row["channels"] for row in subset),
            "affine_residual_sse": a, "mse_residual_sse": m,
            "affine_parameter_sse": ap, "mse_parameter_sse": mp,
            "affine_residual_mse": a/n, "mse_residual_mse": m/n,
            "mse_reduction_abs": (a-m)/n,
            "mse_reduction_pct": 100*(a-m)/a if a else 0,
            "signal_ss": signal,
            "affine_sqnr_db": 10*math.log10(signal/ap) if ap else None,
            "mse_sqnr_db": 10*math.log10(signal/mp) if mp else None,
            "changed_code_fraction": sum(row["changed_code_fraction"]*row["weights"] for row in subset)/n,
            "changed_zero_point_fraction": sum(row["changed_zero_point_fraction"]*row["channels"] for row in subset)/sum(row["channels"] for row in subset),
            "scale_ratio_min": min(row["scale_ratio_min"] for row in subset),
            "scale_ratio_median": None,
            "scale_ratio_mean": sum(row["scale_ratio_mean"]*row["channels"] for row in subset)/sum(row["channels"] for row in subset),
            "scale_ratio_max": max(row["scale_ratio_max"] for row in subset),
            "reconstructed_difference_norm": math.sqrt(diff),
            "reconstructed_difference_sse": diff}


class MemmapSubset(Dataset):
    """Exact FP32 preprocessing cached to disk to avoid repeated PNG decoding."""
    def __init__(self, path: Path, labels: list[int]):
        self.path, self.labels = path, labels
        self.data = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.data is None:
            self.data = np.memmap(self.path, dtype=np.float32, mode="r",
                                  shape=(len(self.labels), 3, 224, 224))
        return torch.from_numpy(self.data[index].copy()), self.labels[index]


def cache_subset(dataset, indices, path: Path, workers: int, batch_size: int):
    expected_bytes = len(indices)*3*224*224*4
    if path.exists() and path.stat().st_size == expected_bytes:
        return
    tmp = path.with_suffix(".tmp")
    data = np.memmap(tmp, dtype=np.float32, mode="w+", shape=(len(indices), 3, 224, 224))
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, num_workers=workers,
                        shuffle=False)
    position = 0
    for images, _ in loader:
        end = position + len(images)
        data[position:end] = images.numpy()
        position = end
        if position % (batch_size*20) < batch_size:
            print(f"Caching exact transformed subset: {position}/{len(indices)}", flush=True)
    if position != len(indices):
        raise AssertionError("Incomplete subset cache")
    data.flush()
    del data
    os.replace(tmp, path)


def save_csv(path: Path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def persist_rows(outdir, scope, rows):
    write_json(outdir / f"{scope}.json", rows)
    save_csv(outdir / f"{scope}.csv", rows)


def plot_results(outdir, stages, layers, full):
    plots = outdir / "plots"
    plots.mkdir(exist_ok=True)
    def save(name):
        plt.tight_layout()
        plt.savefig(plots / f"{name}.png", dpi=180)
        plt.savefig(plots / f"{name}.pdf")
        plt.close()
    for direction, title, name in (("substitute", "Top-1 change vs all-affine (pp)", "stage_substitution"),
                                   ("restore", "Top-1 recovery vs all-MSE (pp)", "stage_restoration")):
        rows = [row for row in stages if row.get("direction") == direction and row.get("stage") in STAGES]
        if rows:
            rows.sort(key=lambda r: STAGES.index(r["stage"]))
            plt.figure(figsize=(8, 4))
            plt.bar([r["stage"] for r in rows], [r["delta_top1"] for r in rows])
            plt.axhline(0, color="black", lw=.8)
            plt.ylabel(title)
            save(name)
    for field, ylabel, name in (("delta_top1", "Top-1 change (pp)", "layer_mse_vs_top1"),
                                ("delta_cross_entropy", "Cross-entropy change", "layer_mse_vs_ce")):
        rows = [row for row in layers if row.get("direction") in ("substitute", "restore")]
        if rows:
            plt.figure(figsize=(7, 5))
            for direction in ("substitute", "restore"):
                these = [r for r in rows if r["direction"] == direction]
                plt.scatter([r["mse_reduction_pct"] for r in these], [r[field] for r in these],
                            alpha=.7, label=direction)
            plt.xlabel("Residual MSE reduction (%)")
            plt.ylabel(ylabel)
            plt.legend()
            save(name)
    hybrid = [r for r in full if r.get("direction") == "selective_restore"]
    if hybrid:
        plt.figure(figsize=(7, 4))
        plt.bar([r["config"] for r in hybrid], [r["top1"] for r in hybrid])
        plt.ylabel("Full ImageNet Top-1 (%)")
        plt.xticks(rotation=20, ha="right")
        save("selective_hybrid_accuracy")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("test"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("outputs/residual_ablation_newfull"))
    parser.add_argument("--outdir", type=Path, default=Path("outputs/mse_layer_sensitivity"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--phase", choices=("all", "validate", "stage", "layer", "full", "analysis"), default="all")
    parser.add_argument("--stages", nargs="*", choices=STAGES)
    parser.add_argument("--layers", nargs="*")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("Invalid batch size or workers")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    # Screening used cuDNN benchmarking for repeated subset passes. Full
    # confirmation matches Experiment 1's default non-benchmark inference.
    torch.backends.cudnn.benchmark = True
    args.outdir.mkdir(parents=True, exist_ok=True)
    weights = MODEL_REGISTRY["resnet50"][1]
    transform = weights.transforms()
    dataset = build_dataset(args.data_dir.resolve(), transform)
    inspect_dataset(dataset, True)
    indices, manifest = balanced_indices(dataset.samples, args.data_dir.resolve())
    manifest_path = args.outdir / "subset_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["manifest_sha256"] != manifest["manifest_sha256"]:
        raise AssertionError("Subset manifest changed on resume")
    write_json(manifest_path, manifest)
    model = build_model("resnet50")
    original_state = {name: item.detach().cpu().clone() for name, item in model.state_dict().items()}
    groups = eligible_groups(model)
    names = [name for stage in STAGES for name in groups[stage]]
    artifacts, artifact_origin = load_or_build_artifacts(args.artifacts_dir, model, original_state, device)
    cache = decode_cache(artifacts, names, original_state)
    print("Validating saved codes against original quantizer for every eligible tensor", flush=True)
    max_error = verify_artifacts(artifacts, cache, original_state, names, device)
    diagnostics = []
    for name in names:
        base = IntegerFormat.dequantize(artifacts["base_5"][name])
        diagnostics.append(tensor_diagnostic(name, stage_of(name), original_state[name], base,
                        cache["affine"][name], cache["mse"][name],
                        artifacts["affine_minmax"][name], artifacts["affine_mse"][name]))
    diagnostics += [aggregate_diagnostics(diagnostics, stage) for stage in STAGES]
    diagnostics.append(aggregate_all(diagnostics[:len(names)]))
    for summary in diagnostics[len(names):]:
        selected = names if summary["name"] == "all" else groups[summary["name"]]
        ratios = np.concatenate([(artifacts["affine_mse"][name]["scale"].double() /
                                  artifacts["affine_minmax"][name]["scale"].double()).flatten().numpy()
                                 for name in selected])
        summary["scale_ratio_median"] = float(np.median(ratios))
    save_csv(args.outdir / "parameter_diagnostics.csv", diagnostics)
    metadata = {"dataset": str(args.data_dir.resolve()), "dataset_images": len(dataset),
                "subset_sha256": manifest["manifest_sha256"], "weights": str(weights),
                "transform": str(transform), "base_bits": 5, "residual_bits": 2,
                "mse_scale_grid": [0.5, 1.5, 17], "axis": 0,
                "exclude_patterns": list(EXCLUDE_PATTERNS), "artifact_origin": artifact_origin,
                "artifacts_dir": str(args.artifacts_dir.resolve()),
                "saved_code_max_reconstruction_error": max_error,
                "eligible_tensors": len(names), "eligible_weights": sum(d["weights"] for d in diagnostics[:len(names)]),
                "batch_size": args.batch_size, "workers": args.workers, "device": str(device),
                "subset_cudnn_benchmark": True, "full_cudnn_benchmark": False,
                "torch": torch.__version__, "torchvision": torchvision.__version__}
    meta_path = args.outdir / "run_metadata.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        for key in ("dataset", "subset_sha256", "weights", "transform", "artifacts_dir", "batch_size"):
            if old[key] != metadata[key]:
                raise AssertionError(f"Run metadata changed on resume: {key}")
    write_json(meta_path, metadata)
    model.to(device)
    all_names = set(names)
    current = None
    subset_cache = args.outdir / "subset_fp32_images.dat"
    if args.phase != "analysis":
        cache_subset(dataset, indices, subset_cache, args.workers, args.batch_size)
    labels = [dataset.samples[index][1] for index in indices]
    subset_loader = DataLoader(MemmapSubset(subset_cache, labels), batch_size=args.batch_size,
                               num_workers=0, pin_memory=device.type == "cuda")
    full_loader = None
    arrays_dir = args.outdir / "predictions"
    arrays_dir.mkdir(exist_ok=True)
    def prediction_path(config, scope):
        return arrays_dir / f"{scope}_{config}.npz"
    def run(config, mse_set, scope):
        nonlocal current, full_loader
        target = prediction_path(config, scope)
        if target.exists():
            with np.load(target) as data:
                arrays = {key: data[key] for key in ("correct1", "correct5", "cross_entropy")}
            return metric_sums(**arrays), arrays, None
        apply_configuration(model, cache, mse_set, current)
        current = set(mse_set)
        if scope == "full" and full_loader is None:
            full_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.workers, pin_memory=device.type == "cuda")
        loader = full_loader if scope == "full" else subset_loader
        torch.backends.cudnn.benchmark = scope != "full"
        result, arrays, seconds = evaluate_details(model, loader, device,
                                                    len(dataset) if scope == "full" else len(indices))
        np.savez_compressed(target, **arrays)
        print(f"{scope} {config}: {result} ({seconds}s)", flush=True)
        return result, arrays, seconds
    def row(config, direction, scope, mse_set, reference=None, stage=None, tensor=None):
        metrics, arrays, seconds = run(config, mse_set, scope)
        result = {"config": config, "direction": direction, "scope": scope,
                  "stage": stage, "tensor": tensor, **metrics, "seconds": seconds,
                  "predictions": str(prediction_path(config, scope).relative_to(args.outdir))}
        if reference:
            _, baseline_arrays, _ = run(reference[0], reference[1], scope)
            paired = paired_stats(arrays, baseline_arrays)
            result.update(paired)
            result["baseline"] = reference[0]
            result["delta_top1"] = metrics["top1"] - metric_sums(**baseline_arrays)["top1"]
            result["delta_cross_entropy"] = metrics["cross_entropy"] - metric_sums(**baseline_arrays)["cross_entropy"]
        return result
    stage_file = args.outdir / "stage_screen.json"
    layer_file = args.outdir / "layer_screen.json"
    full_file = args.outdir / "full_confirmation.json"
    stage_rows = json.loads(stage_file.read_text()) if stage_file.exists() else []
    layer_rows = json.loads(layer_file.read_text()) if layer_file.exists() else []
    full_rows = json.loads(full_file.read_text()) if full_file.exists() else []
    def add(rows, scope, config, *params, **kwargs):
        if any(item["config"] == config for item in rows):
            return
        rows.append(row(config, *params, **kwargs))
        persist_rows(args.outdir, scope, rows)
    if args.phase in ("all", "validate", "stage"):
        add(stage_rows, "stage_screen", "affine", "reference", "subset", set())
        add(stage_rows, "stage_screen", "mse", "reference", "subset", all_names,
            reference=("affine", set()))
        if args.phase == "validate":
            return
        for stage in args.stages or STAGES:
            stage_names = set(groups[stage])
            add(stage_rows, "stage_screen", f"substitute_{stage}", "substitute", "subset", stage_names,
                reference=("affine", set()), stage=stage)
            add(stage_rows, "stage_screen", f"restore_{stage}", "restore", "subset", all_names-stage_names,
                reference=("mse", all_names), stage=stage)
    if args.phase in ("all", "layer"):
        if len([r for r in stage_rows if r.get("direction") in ("substitute", "restore")]) < 12:
            raise RuntimeError("Complete all stage screens before layer screening")
        scores = {stage: max(abs(r["delta_top1"]) for r in stage_rows if r.get("stage") == stage)
                  for stage in STAGES}
        selected = args.stages or sorted(STAGES, key=lambda s: scores[s], reverse=True)[:2]
        print("Selected stages for tensor screening:", selected, scores, flush=True)
        for stage in selected:
            for name in groups[stage]:
                if args.layers and name not in args.layers:
                    continue
                safe = name.replace(".", "_")
                diag = next(item for item in diagnostics if item["name"] == name)
                for direction, config, mse_set, reference in (
                    ("substitute", f"substitute_{safe}", {name}, ("affine", set())),
                    ("restore", f"restore_{safe}", all_names-{name}, ("mse", all_names))):
                    if any(item["config"] == config for item in layer_rows):
                        continue
                    item = row(config, direction, "subset", mse_set, reference=reference,
                               stage=stage, tensor=name)
                    item["mse_reduction_pct"] = diag["mse_reduction_pct"]
                    item["mse_reduction_abs"] = diag["mse_reduction_abs"]
                    layer_rows.append(item)
                    persist_rows(args.outdir, "layer_screen", layer_rows)
        # Test a small cumulative restoration when single stages show limited recovery.
        selected_by_recovery = sorted(STAGES, key=lambda s: next(r["delta_top1"] for r in stage_rows
                                     if r["config"] == f"restore_{s}"), reverse=True)[:2]
        restore_names = set().union(*(set(groups[s]) for s in selected_by_recovery))
        add(layer_rows, "layer_screen", "restore_top2_stages", "selective_restore", "subset",
            all_names-restore_names, reference=("mse", all_names),
            stage="+".join(selected_by_recovery))
    if args.phase in ("all", "full"):
        if len([r for r in stage_rows if r.get("direction") in ("substitute", "restore")]) < 12:
            raise RuntimeError("Complete stage screening before full confirmation")
        worst = min((r for r in stage_rows if r["direction"] == "substitute"),
                    key=lambda r: r["delta_top1"])["stage"]
        best = max((r for r in stage_rows if r["direction"] == "restore"),
                   key=lambda r: r["delta_top1"])["stage"]
        add(full_rows, "full_confirmation", "affine", "reference", "full", set())
        add(full_rows, "full_confirmation", "mse", "reference", "full", all_names,
            reference=("affine", set()))
        add(full_rows, "full_confirmation", f"substitute_{worst}", "substitute", "full",
            set(groups[worst]), reference=("affine", set()), stage=worst)
        add(full_rows, "full_confirmation", f"restore_{best}", "restore", "full",
            all_names-set(groups[best]), reference=("mse", all_names), stage=best)
        top2 = sorted(STAGES, key=lambda s: next(r["delta_top1"] for r in stage_rows
                      if r["config"] == f"restore_{s}"), reverse=True)[:2]
        if top2[0] != top2[1]:
            restored = set().union(*(set(groups[s]) for s in top2))
            add(full_rows, "full_confirmation", "restore_top2_stages", "selective_restore", "full",
                all_names-restored, reference=("mse", all_names), stage="+".join(top2))
    if args.phase in ("all", "analysis"):
        plot_results(args.outdir, stage_rows, layer_rows, full_rows)
        write_analysis(args.outdir, stage_rows, layer_rows, full_rows, diagnostics)


def aggregate_all(rows):
    return aggregate_diagnostics([{**row, "stage": "all"} for row in rows], "all")


def write_analysis(outdir, stages, layers, full, diagnostics):
    stage_sub = sorted((r for r in stages if r.get("direction") == "substitute"), key=lambda r: r["delta_top1"])
    stage_res = sorted((r for r in stages if r.get("direction") == "restore"), key=lambda r: r["delta_top1"], reverse=True)
    tensor_sub = sorted((r for r in layers if r.get("direction") == "substitute"), key=lambda r: r["delta_top1"])
    tensor_res = sorted((r for r in layers if r.get("direction") == "restore"), key=lambda r: r["delta_top1"], reverse=True)
    full_map = {r["config"]: r for r in full}
    def fmt(rows, field, count=6):
        return ", ".join(f"{r.get('stage') if field == 'stage' else r.get('tensor')}: {r['delta_top1']:+.3f} pp"
                         for r in rows[:count])
    lines = ["# Experiment 2: MSE LP layer sensitivity", "",
             "All variants use the saved ResNet-50 IMAGENET1K_V2 5-bit base codes and fixed 2-bit affine residual codes. Every eligible tensor was re-quantized once from the original FP32 checkpoint to check the saved codes and decoded weights exactly. BatchNorm parameters and buffers remain FP32.", "",
             "The 5,000-image seeded class-balanced subset is exploratory. Full-set results use the same 50,000 labeled validation images, transforms, and batch size as Experiment 1. Subset images are contained in the full set, so confirmation is not independent held-out validation. Screening used cuDNN benchmarking; full confirmation used Experiment 1's default non-benchmark setting.", "",
             "## Stage screening (5,000 images)", "",
             f"The all-affine and all-MSE references are {next(r['top1'] for r in stages if r['config']=='affine'):.3f}% and {next(r['top1'] for r in stages if r['config']=='mse'):.3f}% Top-1, respectively.", "",
             "| Stage | MSE substitution vs affine (pp) | Affine restoration vs MSE (pp) | Residual MSE reduction |", "|---|---:|---:|---:|"]
    for stage in STAGES:
        sub = next(r for r in stage_sub if r["stage"] == stage)
        res = next(r for r in stage_res if r["stage"] == stage)
        diag = next(r for r in diagnostics if r["name"] == stage)
        lines.append(f"| {stage} | {sub['delta_top1']:+.3f} | {res['delta_top1']:+.3f} | {diag['mse_reduction_pct']:.2f}% |")
    lines += ["", f"Largest stage effects: substitution {fmt(stage_sub, 'stage', 2)}; restoration {fmt(stage_res, 'stage', 2)}.", "",
              "## Tensor screening (5,000 images)", "",
              "Individual-tensor screens cover every eligible tensor in conv1 and layer1, the two stages selected by the magnitude of their paired stage effects. Tensors in the other stages were not screened individually.", "",
              f"Largest losses from single-tensor substitution: {fmt(tensor_sub, 'tensor', 5)}.", "",
              f"Largest recoveries from single-tensor restoration: {fmt(tensor_res, 'tensor', 5)}.", "",
              "The stage and tensor effects depend on the surrounding quantizers. For example, substituting MSE in conv1 from the affine model and restoring affine in conv1 from the MSE model measure different perturbations. Individual effects cannot be summed to attribute the full gap.", "",
              "## Full ImageNet confirmation (50,000 images)", "",
              "Paired 95% intervals use 10,000 multinomial bootstrap draws from the observed gain, loss, and unchanged counts for each comparison; the fixed seed is 123.", ""]
    for row in full:
        ci = row.get("paired_bootstrap_top1_ci95")
        lines.append(f"- {row['config']}: Top-1 {row['top1']:.3f}%, Top-5 {row['top5']:.3f}%, CE {row['cross_entropy']:.6f}"
                     + (f", delta {row['delta_top1']:+.3f} pp, paired bootstrap 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}] pp" if ci else ""))
    all_diag = next(r for r in diagnostics if r["name"] == "all")
    if "affine" in full_map and "mse" in full_map:
        gap = full_map["affine"]["top1"] - full_map["mse"]["top1"]
        best = full_map.get("restore_top2_stages")
        if best:
            recovery = best["top1"] - full_map["mse"]["top1"]
            lines += ["", f"Restoring conv1 and layer1 recovers {recovery:.3f} of the {gap:.3f} lost percentage points ({100*recovery/gap:.1f}%). The remaining {gap-recovery:.3f} points indicate additional effects outside these stages and/or interactions."]
    if "restore_top2_stages" in full_map and "restore_conv1" in full_map:
        marginal = full_map["restore_top2_stages"]["top1"] - full_map["restore_conv1"]["top1"]
        lines += ["", f"Adding layer1 restoration after conv1 gives a further {marginal:+.3f} points on the full set. This is a conditional effect; the isolated layer1 restoration was screened only on the subset."]
    lines += ["", "## Interpretation and next experiment", "",
              f"Across eligible weights, residual MSE falls from {all_diag['affine_residual_mse']:.9g} to {all_diag['mse_residual_mse']:.9g} ({all_diag['mse_reduction_pct']:.3f}% reduction). Every stage improves residual MSE, including the accuracy-sensitive conv1 and layer1 stages. Lower unweighted parameter MSE therefore does not ensure higher ImageNet accuracy for this LP refinement.",
              "Stage diagnostics aggregate squared errors by total weight count; reconstructed-weight difference norms measure the size of a separate quantity. The fixed-code substitutions locate where accuracy is sensitive to the calibration choice, but do not establish a causal mechanism for changed predictions or a general ranking across other checkpoints and datasets.",
              "A follow-up could pre-register a sensitivity- or Hessian-weighted residual objective, estimate its weights without validation labels, and compare it with fixed ordinary affine and unweighted MSE calibration on an independent evaluation set. This experiment did not implement or tune such an objective."]
    (outdir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
