"""Incremental, resumable conversion of a sharded Qwen3 checkpoint."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from . import FORMAT_VERSION, QWEN3_MODEL_ID, QWEN3_REVISION
from .diagnostics import elapsed, peak_rss_bytes, started
from .model_adapter import SelectionPolicy, inventory, read_tensor
from .quantize import encode_layers
from .quantize import AUTOROUND_RTN_SOURCE
from .rtn_reproduction import AUTOROUND_COMMIT, MODES
from .storage import SIDECARS, atomic_json, atomic_safetensors, copy_sidecars, sha256, storage_report


def _revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _versions() -> dict:
    versions = {"python": sys.version.split()[0], "torch": torch.__version__}
    for package in ("numpy", "safetensors", "PyYAML", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _code_fingerprint() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent.parent
    for name in ("layered_quantization.py", "lp_llm/model_adapter.py", "lp_llm/quantize.py",
                 "lp_llm/rtn_reproduction.py",
                 "lp_llm/storage.py", "lp_llm/encode.py"):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def _valid(root: Path, record: dict) -> bool:
    try:
        paths = ([record["shared"]] if "shared" in record else
                 [p for layer in record["layers"] for p in (layer["code"], layer["metadata"])])
        if not all((root / p).is_file() for p in paths):
            return False
        if "shared" in record:
            return sha256(root / record["shared"]) == record["shared_sha256"]
        return all(sha256(root / layer["code"]) == layer["code_sha256"] and
                   sha256(root / layer["metadata"]) == layer["metadata_sha256"]
                   for layer in record["layers"])
    except (KeyError, OSError):
        return False


def encode(source: Path, output: Path, settings: dict, *, stop_after: int | None = None) -> dict | None:
    source, output = Path(source), Path(output)
    bits = [int(x) for x in settings["bits"]]
    if not bits or any(x < 1 or x > 16 for x in bits):
        raise ValueError("bits must contain widths in [1, 16]")
    method = settings.get("method", "lp")
    if method not in ("lp", "monolithic"):
        raise ValueError("method must be lp or monolithic")
    if method == "monolithic" and len(bits) != 1:
        raise ValueError("Each monolithic run must specify exactly one bit width")
    granularity = settings.get("granularity", "channel")
    group_size = int(settings.get("group_size", 128))
    if granularity not in ("channel", "group") or group_size < 1:
        raise ValueError("Invalid granularity or group size")
    quantizer_family = settings.get("quantizer_family", "affine_asymmetric")
    if quantizer_family not in ("affine_asymmetric", "symmetric_rtn", "autoround_rtn_sym", *MODES):
        raise ValueError(f"Unknown quantizer_family: {quantizer_family}")
    if quantizer_family in ("symmetric_rtn", "autoround_rtn_sym") and any(b < 2 for b in bits):
        raise ValueError(f"{quantizer_family} requires all layer widths >= 2")
    if quantizer_family in MODES and (bits != [2] or granularity != "group" or group_size != 128):
        raise ValueError("Phase 1 modes require bits=[2], group granularity, group_size=128")
    compute_device = settings.get("compute_device", "cpu")
    if quantizer_family in MODES and compute_device not in ("cpu", "cuda"):
        raise ValueError("Phase 1 compute_device must be cpu or cuda")
    if quantizer_family == "autoround_opt_fp16":
        expected_opt = {"iters": 0, "disable_opt_rtn": False,
                        "enable_neuqi": False, "imatrix": None, "search_grid": 181}
        if settings.get("opt_rtn_settings", expected_opt) != expected_opt:
            raise ValueError("Unsupported optimized RTN settings")
    policy = SelectionPolicy(tuple(settings.get("projections", SelectionPolicy().projections)))
    config, mapping = inventory(source, policy)
    identity = {"format": FORMAT_VERSION, "model_id": settings.get("model_id", QWEN3_MODEL_ID),
                "revision": settings.get("revision", QWEN3_REVISION), "method": method,
                "bits": bits, "granularity": granularity, "group_size": group_size,
                "quantizer_family": quantizer_family,
                "projections": list(policy.projections),
                "index_sha256": sha256(source / "model.safetensors.index.json"),
                "config_sha256": sha256(source / "config.json"),
                "shard_stats": {shard: {"bytes": (source / shard).stat().st_size,
                                        "mtime_ns": (source / shard).stat().st_mtime_ns}
                                for shard in sorted(set(mapping.values()))},
                "software": _versions(), "implementation_sha256": _code_fingerprint(),
                "sidecar_sha256": {name: sha256(source / name) for name in sorted(SIDECARS)
                                   if (source / name).is_file()}}
    if quantizer_family == "autoround_rtn_sym":
        identity["quantizer_spec"] = {
            "implementation": AUTOROUND_RTN_SOURCE, "scale_dtype": "float16",
            "scale_threshold": 1e-5, "rounding": "torch.round_ties_to_even",
            "signed_code_mapping": "unsigned=signed+2**(bits-1)",
            "padding": "right_zero_per_row"}
    if quantizer_family in MODES:
        algorithm, dtype = MODES[quantizer_family]
        identity["experiment_id"] = settings.get("experiment_id", quantizer_family)
        identity["quantizer_spec"] = {
            "implementation": f"intel/auto-round@{AUTOROUND_COMMIT}",
            "algorithm": algorithm, "source_dtype": "bfloat16",
            "quantization_input_dtype": str(dtype).removeprefix("torch."),
            "statistics_dtype": str(dtype).removeprefix("torch."),
            "scale_calculation_dtype": str(dtype).removeprefix("torch."),
            "stored_scale_dtype": "float16",
            "rounding_dtype": "float32" if dtype == torch.bfloat16 else "float16",
            "qdq_multiply_dtype": "float32" if dtype == torch.bfloat16 else "float16",
            "in_memory_reconstruction_dtype": str(dtype).removeprefix("torch."),
            "checkpoint_dtype": "bfloat16", "checkpoint_materialization": "qdq_dtype_then_bfloat16",
            "evaluation_dtype": "float16", "compute_device": compute_device,
            "group_size": 128, "scale_threshold": 1e-5, "rounding": "torch.round_ties_to_even",
            "signed_code_mapping": "unsigned=signed+2", "padding": "right_zero_per_row"}
        if quantizer_family == "autoround_opt_fp16":
            identity["quantizer_spec"]["opt_rtn_settings"] = expected_opt
    output.mkdir(parents=True, exist_ok=True)
    run_path = output / "run.json"
    if run_path.exists():
        if json.loads(run_path.read_text())["identity"] != identity:
            raise ValueError("Output directory belongs to a different experiment")
    else:
        if any(output.iterdir()):
            raise ValueError("Nonempty output directory has no run identity")
        atomic_json(run_path, {"identity": identity, "code_revision": _revision()})
    sidecars = copy_sidecars(source, output)
    complete_path, manifest_path = output / "COMPLETE.json", output / "manifest.json"
    if complete_path.exists() and manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if (json.loads(complete_path.read_text()).get("manifest_sha256") == sha256(manifest_path)
                and previous.get("identity") == identity
                and all(_valid(output, r) for r in previous["tensors"])):
            return previous
    records = []
    begun = started()
    with torch.no_grad():
        for tensor_id, name in enumerate(sorted(mapping)):
            marker = f"markers/{tensor_id:04d}.json"
            path = output / marker
            if path.exists():
                record = json.loads(path.read_text())
                if record.get("name") == name and _valid(output, record):
                    records.append(record)
                    continue
            original = read_tensor(source, mapping, name)
            record = {"name": name, "tensor_id": tensor_id, "shape": list(original.shape),
                      "numel": original.numel(), "original_dtype": str(original.dtype),
                      "marker": marker, "eligible": policy.eligible(name)}
            if not record["eligible"]:
                target = output / f"shared/{tensor_id:04d}.safetensors"
                atomic_safetensors(target, {name: original})
                record.update(shared=str(target.relative_to(output)).replace("\\", "/"),
                              shared_sha256=sha256(target))
                del original
            else:
                from .storage import write_layer
                quantization_source = original if quantizer_family in MODES else original.float()
                del original
                record["layers"] = []
                record["metrics"] = []
                schedules = bits
                nominal = 0
                for layer, (params, metrics, prefix) in enumerate(
                        encode_layers(quantization_source, schedules, granularity, group_size,
                                      quantizer_family=quantizer_family,
                                      compute_device=compute_device)):
                    stream = write_layer(output, tensor_id, layer, params)
                    record["layers"].append(stream)
                    nominal += schedules[layer]
                    metrics["nominal_bits"] = nominal
                    record["metrics"].append(metrics)
                    del params, prefix
                del quantization_source
            atomic_json(path, record)  # completion marker is written last
            records.append(record)
            print(f"[{len(records)}/{len(mapping)}] {name} ({'eligible' if record['eligible'] else 'shared'})", flush=True)
            gc.collect()
            if stop_after is not None and len(records) >= stop_after:
                return None
    manifest = {"format": FORMAT_VERSION, "identity": identity, "method": method,
                "quantizer_family": quantizer_family,
                "bits": bits, "sidecars": sidecars,
                "software": identity["software"], "code_revision": _revision(),
                "tensors": records, "eligible_parameters": sum(r["numel"] for r in records if r["eligible"]),
                "excluded_parameters": sum(r["numel"] for r in records if not r["eligible"]),
                "processing_seconds": elapsed(begun), "peak_rss_bytes": peak_rss_bytes()}
    eligible = [r for r in records if r["eligible"]]
    count = manifest["eligible_parameters"]
    if count:
        signal = sum(r["numel"] * r["metrics"][0]["residual_energy_fp32"] for r in eligible) / count
        manifest["prefix_diagnostics"] = []
        for depth in range(len(bits)):
            metrics = {key: sum(r["numel"] * r["metrics"][depth][key] for r in eligible) / count
                       for key in ("parameter_mse_fp32", "parameter_mse_after_bf16_cast",
                                   "residual_energy_fp32", "residual_mse_fp32")}
            mse = metrics["parameter_mse_fp32"]
            metrics.update(depth=depth + 1, nominal_bits=sum(bits[:depth + 1]),
                           sqnr_db_fp32=(10 * math.log10(signal / mse) if mse and signal
                                         else None if not mse else 0.0))
            manifest["prefix_diagnostics"].append(metrics)
        if quantizer_family in MODES:
            for diagnostic in manifest["prefix_diagnostics"]:
                depth = diagnostic["depth"] - 1
                diagnostic["qdq_decode_max_abs_before_cast"] = max(
                    r["metrics"][depth]["qdq_decode_max_abs_before_cast"] for r in eligible)
                diagnostic["qdq_decode_max_abs_after_bf16_cast"] = max(
                    r["metrics"][depth]["qdq_decode_max_abs_after_bf16_cast"] for r in eligible)
                for key in ("qdq_decode_rmse_before_cast", "qdq_decode_rmse_after_bf16_cast"):
                    diagnostic[key] = math.sqrt(sum(
                        r["numel"] * r["metrics"][depth][key] ** 2 for r in eligible) / count)
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "COMPLETE.json", {"manifest_sha256": sha256(output / "manifest.json")})
    report = storage_report(output, manifest)
    atomic_json(output / "storage_report.json", report)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bits", help="Comma-separated override, e.g. 2,2,2,2 or 4")
    parser.add_argument("--granularity", choices=("channel", "group"))
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--method", choices=("lp", "monolithic"))
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    if args.bits:
        settings["bits"] = [int(item) for item in args.bits.split(",")]
    for key in ("granularity", "group_size", "method"):
        value = getattr(args, key)
        if value is not None:
            settings[key] = value
    encode(args.source, args.output, settings)


if __name__ == "__main__":
    main()
