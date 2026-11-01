#!/usr/bin/env python3
"""Experiment 5 full-ImageNet base/enhancement factorial and recursive schedules."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from analyze_mse_layer_sensitivity import evaluate_details, metric_sums, write_json
from evaluate_lp_progressive_schedules import Study
from evaluate_lp_quantizer_adaptivity import (cpu_params, load_torch, numerical_metrics,
                                               save_csv, sha256_file)
from layered_quantization import IntegerFormat
from lp_base_quantizers import FAMILIES, dequantize_base, quantize_base
from lp_quantizer_families import dequantize_family, quantize_family
from lp_uniform_calibration import dequantize_uniform, quantize_uniform

SCHEMES = ("AAA", "UUU", "NNN")


def base_key(family: str) -> str:
    return f"B_{family}"


def extend(key: str, family: str) -> str:
    return f"{key}_E{key.count('_E')+1}_{family}"


def family_at(key: str) -> str:
    return key.rsplit("_", 1)[1]


def parent_key(key: str) -> str | None:
    return key.rsplit("_E", 1)[0] if "_E" in key else None


def unique_progressive_keys() -> list[str]:
    keys = []
    for base in FAMILIES:
        root = base_key(base)
        for scheme in SCHEMES:
            key = root
            for family in scheme:
                key = extend(key, family)
                if key not in keys:
                    keys.append(key)
    return keys


def decode(params: dict, family: str, base: bool) -> torch.Tensor:
    if base:
        return dequantize_base(params)
    if family == "U":
        return dequantize_uniform(params)
    return dequantize_family(params, family)


def artifact_metrics(artifacts: dict, family: str, bits: int) -> dict:
    counts = np.zeros(1 << bits, dtype=np.int64)
    entropy = 0.0
    weights = channels = 0
    spacings = []
    bias = 0.0
    for params in artifacts.values():
        codes = params["quantized"].reshape(params["quantized"].shape[0], -1).long()
        per = torch.stack([(codes == i).sum(1) for i in range(1 << bits)], 1).double()
        p = per / codes.shape[1]
        entropy += float((-(p*p.clamp_min(1e-300).log2()).sum(1)*codes.shape[1]).sum())
        counts += per.sum(0).long().numpy()
        weights += codes.numel()
        channels += codes.shape[0]
        if bits == 5:
            if family == "N":
                spacings.extend((params["centers"][:, 1:]-params["centers"][:, :-1]).flatten().tolist())
            elif family in ("A", "P", "U", "L"):
                spacings.extend(params["scale"].reshape(-1).tolist())
    p = counts / weights
    marginal = -sum(float(x*np.log2(x)) for x in p if x)
    result = {"eligible_weights": weights, "channels": channels,
              "conditional_code_entropy": entropy/weights,
              "marginal_code_entropy": marginal,
              "code_occupancy": counts.tolist(),
              "median_step_or_spacing": float(np.median(spacings)) if spacings else None}
    if family == "N" and bits == 5:
        result["duplicate_centroid_channels"] = sum(x.get("duplicate_centroid_channels", 0) for x in artifacts.values())
        result["max_iteration_tensors"] = sum(not x.get("converged", False) for x in artifacts.values())
    return result


class Experiment:
    def __init__(self, args):
        self.args = args
        self.out = args.outdir.resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        for directory in ("codes", "decoded_prefixes", "evaluations", "predictions"):
            (self.out / directory).mkdir(exist_ok=True)
        # Study validates checkpoint, base artifact, dataset ordering, exclusions,
        # and the official transform against Experiments 1/3/4.
        self.ctx = Study(args)
        self.old3 = args.saved_artifacts.resolve()
        self.old4 = args.experiment4_artifacts.resolve()
        if json.loads((self.old4 / "evaluation_manifest.json").read_text()) != json.loads(
                (self.out / "evaluation_manifest.json").read_text()):
            raise AssertionError("Image order or preprocessing differs from Experiment 4")
        meta = json.loads((self.out / "run_metadata.json").read_text())
        meta.update({"experiment": 5, "base_families": FAMILIES, "schemes": SCHEMES,
                     "base_quantizer_sha256": sha256_file(Path(__file__).parent / "lp_base_quantizers.py"),
                     "five_bit_gamma_grid": [float(g) for g in __import__("lp_base_quantizers").GAMMAS],
                     "two_bit_uniform_grid": "Experiment 4 alpha=0.35:0.0125:0.70 (29 points); preserved",
                     "log_ratio_5bit": 2**.5, "seed": 123})
        write_json(self.out / "run_metadata.json", meta)
        self.original = self.ctx.original
        self.fixed = self.ctx.fixed
        self.model = self.ctx.model
        self.device = self.ctx.device
        self.loader = self.ctx.loader

    def code_file(self, key):
        return self.out / "codes" / (key + ".pt")

    def weight_file(self, key):
        return self.out / "decoded_prefixes" / (key + ".pt")

    def source(self, key: str):
        if key == "B_A":
            return self.args.artifacts_dir / "base_5_codes_and_metadata.pt", None
        if not key.startswith("B_A_E1_"):
            return None, None
        families = key.split("_E")
        seq = [part.rsplit("_", 1)[-1] for part in families[1:]]
        if len(seq) == 1:
            family = seq[0]
            if family == "U":
                return self.old4 / "codes/BU.pt", self.old4 / "predictions/BU.npz"
            if family in ("A", "P", "N", "L"):
                return self.old3 / "codes" / f"E1_{family}.pt", self.old3 / "predictions" / f"full_E1_{family}.npz"
        if len(seq) in (2, 3) and len(set(seq)) == 1 and seq[0] in ("A", "U", "N"):
            old_key = "B" + seq[0]*len(seq)
            if old_key in ("BAA", "BAAA", "BNN", "BNNN"):
                config = {"BAA":"E2_A", "BAAA":"E3_A"}.get(old_key)
                if config:
                    return self.old3 / "codes" / (config + ".pt"), self.old3 / "predictions" / (f"full_{config}.npz")
            return self.old4 / "codes" / (old_key + ".pt"), self.old4 / "predictions" / (old_key + ".npz")
        return None, None

    def weights(self, key: str) -> dict:
        path = self.weight_file(key)
        if path.exists():
            return load_torch(path)
        self.prepare(key)
        return load_torch(path)

    def prepare(self, key: str) -> dict:
        stats_file = self.out / "codes" / (key + ".json")
        code_file = self.code_file(key)
        weight_file = self.weight_file(key)
        parent = parent_key(key)
        family = family_at(key)
        identity = {"checkpoint": self.ctx.identity["checkpoint_sha256"],
                    "base_artifact": self.ctx.identity["base_sha256"],
                    "base_quantizer": sha256_file(Path(__file__).parent / "lp_base_quantizers.py"),
                    "two_bit_uniform": self.ctx.identity["uniform_quantizer_sha256"],
                    "two_bit_families": self.ctx.identity["original_quantizers_sha256"],
                    "key": key, "parent": parent}
        cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if (not self.args.no_resume and stats_file.exists() and code_file.exists() and weight_file.exists()):
            saved = json.loads(stats_file.read_text())
            if saved["cache_key"] != cache_key:
                raise AssertionError(f"Calibration identity differs: {key}")
            return saved
        started = time.perf_counter()
        source, _ = self.source(key)
        if source and source.exists():
            artifacts = load_torch(source)
            if parent is None:
                artifacts = {name: {**params, "family": family} for name, params in artifacts.items()}
            source_name = str(source)
        else:
            artifacts = {}
            previous = self.weights(parent) if parent else None
            for i, name in enumerate(self.original, 1):
                tensor = self.original[name].to(self.device)
                if previous:
                    tensor = tensor - previous[name].to(self.device)
                if parent is None:
                    params, result = quantize_base(tensor, family)
                elif family == "U":
                    params, result = quantize_uniform(tensor, "U")
                else:
                    params, result = quantize_family(tensor, family)
                decoded = decode(params, family, parent is None)
                if not torch.equal(result, decoded):
                    raise AssertionError(f"Quantizer reconstruction mismatch: {key}/{name}")
                artifacts[name] = cpu_params(params)
                if i % 18 == 0:
                    print(f"Calibrating {key}: {i}/54", flush=True)
            source_name = "weight-only Experiment 5 calibration"
        if set(artifacts) != set(self.original):
            raise AssertionError(f"Eligible tensors changed: {key}")
        previous = self.weights(parent) if parent else None
        decoded = {}
        for name in self.original:
            params = artifacts[name]
            codes = params["quantized"]
            if codes.min() < 0 or codes.max() >= (32 if parent is None else 4):
                raise AssertionError(f"Code range differs: {key}/{name}")
            contribution = decode(params, family, parent is None)
            decoded[name] = contribution if parent is None else previous[name] + contribution
        if key == "B_A" and any(not torch.equal(decoded[name], self.ctx.base[name]) for name in decoded):
            raise AssertionError("Original 5-bit base reconstruction changed")
        if key == "B_A_E1_A_E2_A_E3_A":
            expected = load_torch(self.old3 / "decoded_prefixes/BAAA_eligible_weights.pt")
            if any(not torch.equal(decoded[name], expected[name]) for name in decoded):
                raise AssertionError("Original AAA reconstruction changed")
        torch.save(artifacts, code_file)
        torch.save(decoded, weight_file)
        metrics = numerical_metrics(self.original, decoded)
        behavior = artifact_metrics(artifacts, family, 5 if parent is None else 2)
        incoming_sse = sum((self.original[name].double() - previous[name].double()).square().sum().item()
                           for name in self.original) if previous else None
        stats = {"key": key, "parent": parent, "family": family,
                 "nominal_bits": 5+2*key.count("_E"), "cache_key": cache_key,
                 "calibration_source": source_name,
                 "calibration_seconds": round(time.perf_counter()-started, 3) if not source else None,
                 "incoming_residual_mse": incoming_sse/25_502_912 if incoming_sse is not None else None,
                 **metrics, **behavior}
        write_json(stats_file, stats)
        print(f"Prepared {key}: MSE={metrics['parameter_mse']:.9g}", flush=True)
        return stats

    def evaluate(self, key: str) -> dict:
        path = self.out / "evaluations" / (key + ".json")
        pred = self.out / "predictions" / (key + ".npz")
        if path.exists() and pred.exists() and not self.args.no_resume:
            return json.loads(path.read_text())
        _, old_pred = self.source(key)
        if old_pred and old_pred.exists():
            shutil.copyfile(old_pred, pred)
            with np.load(pred) as arrays:
                metrics = metric_sums(**{item: arrays[item] for item in
                                         ("correct1", "correct5", "cross_entropy")})
            source = str(old_pred)
            seconds = None
        else:
            weights = self.weights(key)
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    if name in weights:
                        parameter.copy_(weights[name].to(self.device))
            for name, value in self.fixed.items():
                if not torch.equal(self.model.state_dict()[name].cpu(), value):
                    raise AssertionError(f"Excluded state changed: {name}")
            metrics, arrays, seconds = evaluate_details(self.model, self.loader, self.device,
                                                         len(self.ctx.dataset))
            np.savez_compressed(pred, **arrays)
            source = "measured Experiment 5 full ImageNet"
        row = {"key": key, **metrics, "source": source, "seconds": seconds}
        write_json(path, row)
        print(f"Evaluated {key}: Top-1={row['top1']:.3f}% CE={row['cross_entropy']:.6f}", flush=True)
        return row

    def row(self, key: str) -> dict:
        return {**self.prepare(key), **self.evaluate(key)}

    def bases(self):
        rows = [self.row(base_key(family)) for family in FAMILIES]
        write_json(self.out / "base_quantizer_results.json", rows)
        save_csv(self.out / "base_quantizer_results.csv", rows)

    def factorial(self):
        rows = []
        for base in FAMILIES:
            root = base_key(base)
            self.prepare(root)
            for family in FAMILIES:
                rows.append(self.row(extend(root, family)))
                write_json(self.out / "base_e1_factorial.json", rows)
                save_csv(self.out / "base_e1_factorial.csv", rows)

    def progressive(self):
        rows = []
        for key in unique_progressive_keys():
            rows.append(self.row(key))
            write_json(self.out / "progressive_prefix_results.json", rows)
            save_csv(self.out / "progressive_prefix_results.csv", rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("test"))
    parser.add_argument("--saved-artifacts", type=Path, default=Path("outputs/lp_quantizer_adaptivity"))
    parser.add_argument("--experiment4-artifacts", type=Path, default=Path("outputs/lp_progressive_quantizer_study"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("outputs/residual_ablation_newfull"))
    parser.add_argument("--outdir", type=Path, default=Path("outputs/lp_base_enhancement_study"))
    parser.add_argument("--weights", choices=("IMAGENET1K_V2",), default="IMAGENET1K_V2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--phase", choices=("bases", "factorial", "progressive", "all"), default="all")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    experiment = Experiment(args)
    if args.phase in ("bases", "all"):
        experiment.bases()
    if args.phase in ("factorial", "all"):
        experiment.factorial()
    if args.phase in ("progressive", "all"):
        experiment.progressive()


if __name__ == "__main__":
    main()
