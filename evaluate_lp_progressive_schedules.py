#!/usr/bin/env python3
"""Experiment 4: weight-only uniform calibration and recursive LP schedules.

Each unique prefix is calibrated from its own decoded parent, cached, and
evaluated once on the complete ImageNet validation set. The analysis and
serialized storage are produced by ``lp_progressive_storage.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from analyze_mse_layer_sensitivity import eligible_groups, evaluate_details, metric_sums, paired_stats, write_json
from evaluate_imagenet import MODEL_REGISTRY, build_dataset, build_model
from evaluate_residual_ablation import inspect_dataset
from evaluate_lp_quantizer_adaptivity import (
    EXPERIMENT1, code_behavior, cpu_params, load_torch, numerical_metrics,
    save_csv, sha256_file, validate_saved_baselines,
)
from lp_quantizer_families import dequantize_family, quantize_family
from lp_uniform_calibration import ALPHAS, dequantize_uniform, quantize_uniform

SCHEDULES = ("AAA", "UUU", "NNN", "UAA", "NAA", "AUA", "ANA", "AAU", "AAN", "NNA", "NUA", "UNA")
REFERENCE_PARENTS = {1: "B", 2: "BA", 3: "BAA"}
OLD_REFERENCE = {"B": "B", "BA": "E1_A", "BAA": "E2_A", "BAAA": "E3_A",
                 "BN": "E1_N", "BAN": "E2_N", "BAAN": "E3_N"}
ISOLATED = tuple((f"E{stage}_{family}", REFERENCE_PARENTS[stage], family)
                 for stage in (1, 2, 3) for family in ("A", "S", "U0", "U", "N"))


def all_prefixes() -> list[str]:
    return ["B"] + sorted({"B" + schedule[:n] for schedule in SCHEDULES for n in (1, 2, 3)},
                          key=lambda prefix: (len(prefix), prefix))


def decode(params: dict, family: str) -> torch.Tensor:
    return dequantize_uniform(params) if family in ("U0", "U") else dequantize_family(params, family)


def _old_code_path(old: Path, prefix: str) -> Path | None:
    if prefix in OLD_REFERENCE and prefix != "B":
        return old / "codes" / (OLD_REFERENCE[prefix] + ".pt")
    return None


def _old_weight_path(old: Path, prefix: str) -> Path | None:
    return old / "decoded_prefixes" / (prefix + "_eligible_weights.pt") if prefix in ("B", "BA", "BAA", "BAAA") else None


def _uniform_behavior(artifacts: dict, family: str) -> dict:
    counts = np.zeros(4, dtype=np.int64)
    alpha_counts = np.zeros(len(ALPHAS), dtype=np.int64)
    scales = []
    entropy_weighted = 0.0
    channels = weights = 0
    for params in artifacts.values():
        codes = params["quantized"].reshape(params["quantized"].shape[0], -1)
        per = torch.stack([(codes == k).sum(1) for k in range(4)], 1).double()
        counts += per.sum(0).long().numpy()
        probability = per / codes.shape[1]
        ent = -(probability * probability.clamp_min(1e-300).log2()).sum(1)
        entropy_weighted += float(ent.sum()) * codes.shape[1]
        channels += codes.shape[0]
        weights += codes.numel()
        scales.extend(params["scale"].reshape(-1).tolist())
        alpha_counts += np.bincount(params["alpha_index"].numpy(), minlength=len(ALPHAS))
    proportions = counts / weights
    marginal = -sum(float(p * np.log2(p)) for p in proportions if p)
    return {"family": family, "weights": weights, "channels": channels,
            **{f"code{k}_fraction": float(proportions[k]) for k in range(4)},
            "parameter_weighted_conditional_code_entropy": entropy_weighted / weights,
            "marginal_code_entropy": marginal,
            "scale_min": min(scales), "scale_median": float(np.median(scales)),
            "scale_max": max(scales), "alpha_distribution":
            {str(ALPHAS[k]): int(n) for k, n in enumerate(alpha_counts)}}


class Study:
    def __init__(self, args):
        self.args = args
        self.out = args.outdir.resolve()
        self.old = args.saved_artifacts.resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        for directory in ("codes", "decoded_prefixes", "evaluations", "predictions"):
            (self.out / directory).mkdir(exist_ok=True)
        self.device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                                   else "cpu" if args.device == "auto" else args.device)
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(123)
        weights = MODEL_REGISTRY["resnet50"][1]
        self.transform = weights.transforms()
        self.dataset = build_dataset(args.data_dir.resolve(), self.transform)
        inspect_dataset(self.dataset, True)
        self.model = build_model("resnet50")
        names = [name for group in eligible_groups(self.model).values() for name in group]
        self.original = {name: p.detach().cpu().clone() for name, p in self.model.named_parameters() if name in names}
        if len(names) != 54 or sum(x.numel() for x in self.original.values()) != 25_502_912:
            raise AssertionError("ResNet-50 eligible tensor count changed")
        old_meta = json.loads((self.old / "run_metadata.json").read_text())
        checkpoint = Path(os.environ.get("TORCH_HOME", str(Path.home() / ".cache" / "torch"))) / "hub/checkpoints/resnet50-11ad3fa6.pth"
        old_base = args.artifacts_dir / "base_5_codes_and_metadata.pt"
        identity = {"checkpoint_sha256": sha256_file(checkpoint), "base_sha256": sha256_file(old_base),
                    "uniform_quantizer_sha256": sha256_file(Path(__file__).resolve().parent / "lp_uniform_calibration.py"),
                    "original_quantizers_sha256": sha256_file(Path(__file__).resolve().parent / "lp_quantizer_families.py"),
                    "dataset": str(args.data_dir.resolve()), "dataset_images": len(self.dataset),
                    "transform": str(self.transform), "bit_schedule": [5, 2, 2, 2],
                    "channel_axis": 0, "alpha_grid": list(ALPHAS),
                    "cudnn_benchmark": False, "batch_size": args.batch_size,
                    "workers": args.workers, "device": str(self.device),
                    "torch": torch.__version__, "torchvision": torchvision.__version__,
                    "eligible_tensors": len(names), "eligible_weights": 25_502_912,
                    "shared_unquantized_model_bytes": old_meta["shared_unquantized_model_bytes"]}
        for key in ("checkpoint_sha256", "dataset", "transform", "dataset_images"):
            if identity[key] != old_meta[key]:
                raise AssertionError(f"Experiment 3 provenance mismatch: {key}")
        if identity["base_sha256"] != old_meta["experiment1_artifact_sha256"]["base_5"]:
            raise AssertionError("5-bit base artifact hash differs from Experiment 3")
        meta = self.out / "run_metadata.json"
        if meta.exists():
            previous = json.loads(meta.read_text())
            for key in ("checkpoint_sha256", "base_sha256", "dataset", "transform", "batch_size", "alpha_grid",
                        "uniform_quantizer_sha256", "original_quantizers_sha256"):
                if key in previous and previous[key] != identity[key]:
                    raise AssertionError(f"Resume identity changed: {key}")
        write_json(meta, identity)
        self.identity = identity
        digest = hashlib.sha256()
        for path, label in self.dataset.samples:
            digest.update(str(Path(path).relative_to(args.data_dir.resolve())).encode() + b"\0" + str(label).encode() + b"\n")
        manifest = {"image_count": len(self.dataset), "ordered_samples_sha256": digest.hexdigest(),
                    "data_dir": str(args.data_dir.resolve()), "transform": str(self.transform),
                    "ordering": "torchvision NumericImageFolder, shuffle=False"}
        manifest_path = self.out / "evaluation_manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise AssertionError("Full evaluation ordering changed")
        write_json(manifest_path, manifest)
        self.model.to(self.device).eval()
        self.fixed = {name: x.cpu().clone() for name, x in self.model.state_dict().items() if name not in self.original}
        self.loader = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=self.device.type == "cuda")
        self.saved_base, base, affine, _ = validate_saved_baselines(self.original, names, args.artifacts_dir, self.device)
        for prefix, expected in (("B", base), ("BA", affine)):
            source = _old_weight_path(self.old, prefix)
            if source and source.exists():
                actual = load_torch(source)
                if any(not torch.equal(actual[name], expected[name]) for name in names):
                    raise AssertionError(f"Saved {prefix} reconstruction changed")
        self.base = base

    def code_path(self, prefix: str) -> Path:
        old = _old_code_path(self.old, prefix)
        return old if old is not None else self.out / "codes" / (prefix + ".pt")

    def weights(self, prefix: str) -> dict:
        if prefix == "B":
            return self.base
        old = _old_weight_path(self.old, prefix)
        path = old if old is not None else self.out / "decoded_prefixes" / (prefix + ".pt")
        if path.exists():
            return load_torch(path)
        parent = self.weights(prefix[:-1])
        artifacts = load_torch(self.code_path(prefix))
        family = prefix[-1]
        result = {name: parent[name] + decode(artifacts[name], family) for name in self.original}
        torch.save(result, path)
        return result

    def prepare(self, prefix: str, *, parent: str | None = None, family: str | None = None) -> dict:
        parent = parent or prefix[:-1]
        family = family or prefix[-1]
        cache_identity = {"checkpoint_sha256": self.identity["checkpoint_sha256"],
                          "base_sha256": self.identity["base_sha256"],
                          "prefix": prefix, "parent": parent, "family": family,
                          "bit_schedule": self.identity["bit_schedule"],
                          "uniform_quantizer_sha256": self.identity["uniform_quantizer_sha256"],
                          "original_quantizers_sha256": self.identity["original_quantizers_sha256"],
                          "alpha_grid": list(ALPHAS) if family == "U" else None,
                          "channel_axis": 0}
        cache_key = hashlib.sha256(json.dumps(cache_identity, sort_keys=True).encode()).hexdigest()
        stats_path = self.out / "codes" / (prefix + ".json")
        code_path = self.code_path(prefix) if family != "U0" else self.out / "codes" / (prefix + ".pt")
        if stats_path.exists() and code_path.exists() and not self.args.no_resume:
            prior = json.loads(stats_path.read_text())
            if prior.get("cache_key") not in (None, cache_key):
                raise AssertionError(f"Prefix cache identity changed: {prefix}")
            if prior.get("cache_key") is None:
                prior["cache_key"] = cache_key
                write_json(stats_path, prior)
            return prior
        parent_weights = self.weights(parent)
        started = time.perf_counter()
        if _old_code_path(self.old, prefix) is not None and family in ("A", "N"):
            artifacts = load_torch(code_path)
            calibration_seconds = json.loads((self.old / "codes" / (OLD_REFERENCE[prefix] + ".json")).read_text()).get("calibration_seconds")
            source = "verified Experiment 3 codes"
        else:
            artifacts = {}
            for number, name in enumerate(self.original, 1):
                residual = self.original[name].to(self.device) - parent_weights[name].to(self.device)
                params, decoded = (quantize_uniform(residual, family) if family in ("U0", "U")
                                   else quantize_family(residual, family))
                if not torch.equal(decoded, decode(params, family)):
                    raise AssertionError(f"Decoder mismatch {prefix}/{name}")
                artifacts[name] = cpu_params(params)
                if number % 18 == 0:
                    print(f"Calibrating {prefix}: {number}/54", flush=True)
            torch.save(artifacts, code_path)
            calibration_seconds = round(time.perf_counter() - started, 3)
            source = "weight-only calibration in Experiment 4"
        incoming_sse = residual_sse = 0.0
        decoded_weights = {}
        for name in self.original:
            contribution = decode(artifacts[name], family)
            residual = self.original[name] - parent_weights[name]
            incoming_sse += residual.double().square().sum().item()
            residual_sse += (residual.double() - contribution.double()).square().sum().item()
            decoded_weights[name] = parent_weights[name] + contribution
        numeric = numerical_metrics(self.original, decoded_weights)
        old_config = OLD_REFERENCE.get(prefix)
        if old_config and prefix != "B":
            old_stats = json.loads((self.old / "codes" / (old_config + ".json")).read_text())
            if abs(numeric["parameter_mse"] - old_stats["parameter_mse"]) > 1e-12:
                raise AssertionError(f"Saved {prefix} MSE changed")
        n = numeric["eligible_weights"]
        behavior = _uniform_behavior(artifacts, family) if family in ("U0", "U") else code_behavior(artifacts, family)
        stats = {"prefix": prefix, "parent": parent, "family": family, "nominal_bits": 5+2*(len(parent)-1)+2,
                 "cache_key": cache_key,
                 "incoming_residual_mse": incoming_sse/n, "residual_mse": residual_sse/n,
                 **numeric, **behavior, "calibration_seconds": calibration_seconds,
                 "calibration_source": source, "code_artifact": str(code_path)}
        write_json(stats_path, stats)
        if prefix.startswith("B") and family != "U0" and _old_weight_path(self.old, prefix) is None:
            torch.save(decoded_weights, self.out / "decoded_prefixes" / (prefix + ".pt"))
        print(f"Prepared {prefix}: MSE={numeric['parameter_mse']:.9g}", flush=True)
        return stats

    def evaluation(self, prefix: str, *, weights: dict | None = None, old_config: str | None = None) -> dict:
        path = self.out / "evaluations" / (prefix + ".json")
        prediction = self.out / "predictions" / (prefix + ".npz")
        if path.exists() and (prediction.exists() or prefix == "B") and not self.args.no_resume:
            return json.loads(path.read_text())
        old_config = old_config or OLD_REFERENCE.get(prefix)
        old_eval = self.old / "evaluations" / (f"full_{old_config}.json" if old_config else "")
        old_pred = self.old / "predictions" / (f"full_{old_config}.npz" if old_config else "")
        if old_config and old_config != "B" and old_eval.exists() and old_pred.exists():
            source = json.loads(old_eval.read_text())
            with np.load(old_pred) as arrays:
                metrics = metric_sums(**{key: arrays[key] for key in ("correct1", "correct5", "cross_entropy")})
            if any(metrics[key] != source[key] for key in ("top1", "top5", "cross_entropy")):
                raise AssertionError(f"Saved {prefix} evaluation differs from predictions")
            shutil.copyfile(old_pred, prediction)
            row = {"prefix": prefix, **metrics, "source": "verified Experiment 3 full predictions",
                   "seconds": source.get("seconds")}
        elif prefix == "B":
            row = {"prefix": "B", "images": 50000, **EXPERIMENT1["B"],
                   "source": "verified Experiment 1 full result", "seconds": None}
        else:
            weights = weights or self.weights(prefix)
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    if name in weights:
                        parameter.copy_(weights[name].to(self.device))
            for name, fixed in self.fixed.items():
                if not torch.equal(self.model.state_dict()[name].cpu(), fixed):
                    raise AssertionError(f"Excluded state changed: {name}")
            metrics, arrays, seconds = evaluate_details(self.model, self.loader, self.device, len(self.dataset))
            np.savez_compressed(prediction, **arrays)
            row = {"prefix": prefix, **metrics, "source": "measured Experiment 4 full images", "seconds": seconds}
            print(f"Evaluated {prefix}: Top-1={metrics['top1']:.3f}% CE={metrics['cross_entropy']:.6f} ({seconds}s)", flush=True)
        write_json(path, row)
        return row

    def run_phase_a(self):
        for stage in (1, 2, 3):
            parent = REFERENCE_PARENTS[stage]
            for family in ("A", "S", "U0", "U", "N"):
                key = f"E{stage}_{family}"
                prefix = parent + family if family != "U0" else key
                if family in ("A", "S", "N"):
                    old_stats = json.loads((self.old / "codes" / f"E{stage}_{family}.json").read_text())
                    old_eval = json.loads((self.old / "evaluations" / f"full_E{stage}_{family}.json").read_text())
                    row = {**old_stats, "prefix": prefix, "parent": parent, "nominal_bits": 5+2*stage,
                           "top1": old_eval["top1"], "top5": old_eval["top5"],
                           "cross_entropy": old_eval["cross_entropy"], "images": old_eval["images"],
                           "inference_seconds": old_eval.get("seconds"), "source": "verified Experiment 3"}
                else:
                    stats = self.prepare(prefix, parent=parent, family=family)
                    if family == "U0":
                        preceding = self.weights(parent)
                        artifacts = load_torch(self.out / "codes" / (prefix + ".pt"))
                        decoded = {name: preceding[name] + dequantize_uniform(artifacts[name])
                                   for name in self.original}
                        evaluated = self.evaluation(prefix, weights=decoded)
                    else:
                        evaluated = self.evaluation(prefix)
                    row = {**stats, **evaluated, "inference_seconds": evaluated.get("seconds")}
                yield row

    def run_phase_b(self):
        for prefix in all_prefixes():
            if prefix == "B":
                numeric = numerical_metrics(self.original, self.base)
                evaluated = self.evaluation("B")
                yield {"prefix": "B", "nominal_bits": 5, **numeric, **evaluated}
                continue
            stats = self.prepare(prefix)
            evaluated = self.evaluation(prefix)
            yield {**stats, **evaluated}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("test"))
    parser.add_argument("--saved-artifacts", type=Path, default=Path("outputs/lp_quantizer_adaptivity"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("outputs/residual_ablation_newfull"))
    parser.add_argument("--outdir", type=Path, default=Path("outputs/lp_progressive_quantizer_study"))
    parser.add_argument("--weights", choices=("IMAGENET1K_V2",), default="IMAGENET1K_V2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--phase", choices=("A", "B", "C", "all"), default="all")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("Invalid batch or worker count")
    if args.phase == "C":
        finalize(args)
        return
    study = Study(args)
    if args.phase in ("A", "all"):
        rows = list(study.run_phase_a())
        write_json(study.out / "isolated_quantizer_comparison.json", rows)
        save_csv(study.out / "isolated_quantizer_comparison.csv", rows)
    if args.phase in ("B", "all"):
        rows = list(study.run_phase_b())
        write_json(study.out / "progressive_prefixes.json", rows)
        save_csv(study.out / "progressive_prefixes.csv", rows)
    if args.phase == "all":
        finalize(args)


def finalize(args):
    root = Path(__file__).resolve().parent
    shared = ["--outdir", str(args.outdir), "--saved-artifacts", str(args.saved_artifacts)]
    subprocess.check_call([sys.executable, str(root / "lp_progressive_storage.py"),
                           *shared, "--artifacts-dir", str(args.artifacts_dir)])
    subprocess.check_call([sys.executable, str(root / "report_lp_progressive.py"), *shared])


if __name__ == "__main__":
    main()
