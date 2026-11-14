"""Isolated signed G128 LP hierarchy with a fixed one-half correction."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .progressive_lp import (_files, _read_chunk, _slice, encode_residual,
                             identity as lp_identity, validate_repaired_base, verified_layers, ROWS,
                             TOTAL_PARAMETERS)
from .rrq_pr2308_compare import _eligible, _weight_map
from .rrq_reference_eval import check_revision, corpus_identity, run as evaluate_matgptq
from .storage import SIDECARS, atomic_json, pack_codes, sha256


ALPHA = 0.5
EXPERIMENT = "signed_rtn_g128_alpha050_v1"
S01_PRODUCER = "78b67800d786ba8bf03534818ae77137066c6ad8"
TOKEN_SHA = "8ffcee84dd4b684ec631cb52845e18c9c181ce6fdb8b26d65a4ffdc3d35fbda3"


def scaled_identity(branch: str, source: Path, base: Path) -> dict:
    if branch not in ("s01", "r02"):
        raise ValueError("Unknown scaled LP branch")
    return {**lp_identity(branch, source, base, "autoround_signed_rtn"),
            "experiment": EXPERIMENT, "alpha": ALPHA,
            "checkpoint_dtype": "float16_eligible_source_exact_excluded"}


def _external_s01(root: Path, source: Path, base: Path, experiment_identity: dict) -> dict:
    """Validate and record the existing signed E1 without claiming new authorship."""
    external = root.parent.parent / "ablations/s01_autoround_signed_rtn"
    marker = external / "layer1/encoding_complete.json"
    if not marker.is_file():
        raise ValueError("Existing S01 signed E1 is unavailable")
    with patch.dict(os.environ, {"FOLLOWUP_CODE_SHA": S01_PRODUCER}):
        manifest = verified_layers(external, 1,
            lp_identity("s01", source, base, "autoround_signed_rtn"))[0]
    if manifest.get("code_sha") != S01_PRODUCER or manifest.get("projection_count") != 252:
        raise ValueError("Existing S01 signed E1 provenance differs")
    reference = {"status": "external_layer_validated", "identity": experiment_identity,
                 "layer": 1, "path": str((external / "layer1").resolve()),
                 "manifest_sha256": sha256(marker), "producer_sha": S01_PRODUCER,
                 "validating_sha": os.environ.get("FOLLOWUP_CODE_SHA"),
                 "coefficient": ALPHA, "code_bytes": manifest["code_bytes"],
                 "metadata_bytes": manifest["metadata_bytes"]}
    path = root / "layer1_reference.json"
    if path.exists():
        if json.loads(path.read_text()) != reference:
            raise ValueError("Existing external E1 reference differs")
    else:
        atomic_json(path, reference)
    return {"directory": external / "layer1", "manifest": manifest, "reference": reference}


def _own_layer(root: Path, depth: int, ident: dict) -> dict:
    directory = root / f"layer{depth}"
    marker = directory / "encoding_complete.json"
    record = json.loads(marker.read_text())
    if (record.get("status") != "completed_validated" or record.get("identity") != ident or
            record.get("depth") != depth or record.get("alpha") != ALPHA or
            record.get("code_sha") != os.environ.get("FOLLOWUP_CODE_SHA") or
            record.get("projection_count") != 252 or
            record.get("chunk_count") != len(record.get("chunks", []))):
        raise ValueError(f"Scaled layer identity differs: {depth}")
    by_name = {}
    for chunk in record["chunks"]:
        if chunk.get("quantizer") != "autoround_signed_rtn":
            raise ValueError("Scaled chunk quantizer differs")
        _read_chunk(directory, chunk)
        by_name.setdefault(chunk["name"], []).append(chunk)
    if len(by_name) != 252 or set(by_name) != set(record.get("tensor_rows", {})):
        raise ValueError("Scaled layer projection inventory differs")
    for name, chunks in by_name.items():
        end = 0
        for chunk in sorted(chunks, key=lambda item: item["first"]):
            if chunk["first"] != end or chunk["stop"] <= end:
                raise ValueError("Scaled layer row coverage differs")
            end = chunk["stop"]
        if end != record["tensor_rows"][name]:
            raise ValueError("Scaled layer final row coverage differs")
    return {"directory": directory, "manifest": record}


def layers(branch: str, depth: int, source: Path, base: Path, root: Path,
           ident: dict) -> list[dict]:
    if depth not in (1, 2, 3):
        raise ValueError("Scaled LP depth must be 1..3")
    result = []
    if branch == "s01":
        result.append(_external_s01(root, source, base, ident))
        start = 2
    else:
        start = 1
    for number in range(start, depth + 1):
        result.append(_own_layer(root, number, ident))
    return result


def _contribution(layer: dict, name: str, first: int, stop: int) -> torch.Tensor:
    matches = [chunk for chunk in layer["manifest"]["chunks"]
               if chunk["name"] == name and chunk["first"] == first and chunk["stop"] == stop]
    if len(matches) != 1:
        raise ValueError("Scaled predecessor row partition differs")
    return _read_chunk(layer["directory"], matches[0])


def reconstruct(base_rows: torch.Tensor, prior: list[dict], name: str,
                first: int, stop: int) -> torch.Tensor:
    value = base_rows.float()
    for layer in prior:
        value = value + ALPHA * _contribution(layer, name, first, stop)
    return value


def encode(branch: str, depth: int, source: Path, base: Path, root: Path,
           *, rows: int = ROWS) -> dict:
    if rows != ROWS or depth not in (1, 2, 3) or (branch == "s01" and depth == 1):
        raise ValueError("Invalid scaled encoding stage or row partition")
    validate_repaired_base(branch, source, base)
    ident = scaled_identity(branch, source, base)
    previous = layers(branch, depth - 1, source, base, root, ident) if depth > 1 else []
    target = root / f"layer{depth}"
    partial = root / f"layer{depth}.partial"
    if partial.exists():
        raise FileExistsError("Interrupted scaled layer requires inspection")
    if target.exists():
        return _own_layer(root, depth, ident)["manifest"]
    source_map, base_map = _weight_map(source), _weight_map(base)
    names = [name for name in sorted(source_map) if _eligible(name)]
    if len(names) != 252 or set(source_map) != set(base_map):
        raise ValueError("Scaled source/base projection inventory differs")
    started = time.monotonic()
    partial.mkdir(parents=True)
    atomic_json(partial / "run.json", {"status": "started", "identity": ident,
                                       "depth": depth, "code_sha": os.environ.get("FOLLOWUP_CODE_SHA")})
    chunks, tensor_rows = [], {}
    error_sum = signal_sum = cast_error_sum = 0.0
    elements = 0
    for number, name in enumerate(names, 1):
        with safe_open(source / source_map[name], framework="pt", device="cpu") as handle:
            shape = handle.get_slice(name).get_shape()
        tensor_rows[name] = shape[0]
        for first in range(0, shape[0], rows):
            stop = min(first + rows, shape[0])
            original = _slice(source, source_map, name, first, stop).float()
            base_rows = _slice(base, base_map, name, first, stop)
            prefix = reconstruct(base_rows, previous, name, first, stop)
            params, decoded = encode_residual(original - prefix, "autoround_signed_rtn")
            value = prefix + ALPHA * decoded
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"Nonfinite scaled prefix: {name}")
            stem = f"chunks/{number:03d}-{first:05d}"
            code, meta = partial / (stem + ".bin"), partial / (stem + ".safetensors")
            code.parent.mkdir(parents=True, exist_ok=True)
            pack_codes(code, params["codes"], 2)
            save_file({"scale": params["scale"].contiguous()}, meta, metadata={"format": "pt"})
            chunk = {"name": name, "first": first, "stop": stop,
                     "shape": list(params["codes"].shape), "quantizer": "autoround_signed_rtn",
                     "code": code.relative_to(partial).as_posix(),
                     "metadata": meta.relative_to(partial).as_posix(),
                     "code_sha256": sha256(code), "metadata_sha256": sha256(meta),
                     "code_bytes": code.stat().st_size, "metadata_bytes": meta.stat().st_size,
                     "scale_payload_bytes": params["scale"].numel() * params["scale"].element_size()}
            if not torch.equal(_read_chunk(partial, chunk), decoded):
                raise ValueError("Scaled chunk round trip differs")
            chunks.append(chunk)
            error_sum += float((original - value).double().square().sum())
            cast_error_sum += float((original - value.to(torch.float16).float()).double().square().sum())
            signal_sum += float(original.double().square().sum())
            elements += original.numel()
    report = {"status": "completed_validated", "identity": ident, "depth": depth,
              "alpha": ALPHA, "code_sha": os.environ.get("FOLLOWUP_CODE_SHA"),
              "projection_count": len(tensor_rows), "tensor_rows": tensor_rows,
              "chunk_count": len(chunks), "chunks": chunks, "encoded_elements": elements,
              "mse_fp32": error_sum / elements, "mse_after_fp16_cast": cast_error_sum / elements,
              "sqnr_db_fp32": 10 * math.log10(signal_sum / error_sum) if error_sum else None,
              "code_bytes": sum(item["code_bytes"] for item in chunks),
              "metadata_bytes": sum(item["metadata_bytes"] for item in chunks),
              "runtime_seconds": time.monotonic() - started}
    atomic_json(partial / "encoding_complete.json", report)
    os.replace(partial, target)
    return report


def materialize(branch: str, depth: int, source: Path, base: Path, root: Path,
                target: Path, *, historical_layer_sha: str | None = None) -> dict:
    ident = scaled_identity(branch, source, base)
    if historical_layer_sha is None:
        prior = layers(branch, depth, source, base, root, ident)
    else:
        if len(historical_layer_sha) != 40:
            raise ValueError("Historical LP producer SHA is invalid")
        with patch.dict(os.environ, {"FOLLOWUP_CODE_SHA": historical_layer_sha}):
            prior = layers(branch, depth, source, base, root, ident)
    partial = target.with_name(target.name + ".partial")
    if target.exists() or partial.exists():
        raise FileExistsError("Scaled checkpoint or partial already exists")
    source_map, base_map = _weight_map(source), _weight_map(base)
    if set(source_map) != set(base_map):
        raise ValueError("Scaled checkpoint tensor inventory differs")
    partial.mkdir(parents=True)
    for sidecar in SIDECARS:
        if (source / sidecar).is_file():
            shutil.copy2(source / sidecar, partial / sidecar)
    mapping, payload = {}, 0
    error_sum = signal_sum = cast_error_sum = 0.0
    eligible_elements = 0
    for number, name in enumerate(sorted(source_map), 1):
        with safe_open(source / source_map[name], framework="pt", device="cpu") as handle:
            shape = tuple(handle.get_slice(name).get_shape())
        if _eligible(name):
            value = torch.empty(shape, dtype=torch.float16)
            for first in range(0, shape[0], ROWS):
                stop = min(first + ROWS, shape[0])
                base_rows = _slice(base, base_map, name, first, stop)
                reconstructed = reconstruct(base_rows, prior, name, first, stop)
                if not bool(torch.isfinite(reconstructed).all()):
                    raise ValueError("Nonfinite scaled checkpoint projection")
                original = _slice(source, source_map, name, first, stop).float()
                cast = reconstructed.to(torch.float16)
                error_sum += float((original - reconstructed).double().square().sum())
                cast_error_sum += float((original - cast.float()).double().square().sum())
                signal_sum += float(original.double().square().sum())
                eligible_elements += original.numel()
                value[first:stop] = cast
        else:
            with safe_open(source / source_map[name], framework="pt", device="cpu") as handle:
                value = handle.get_tensor(name)
            with safe_open(base / base_map[name], framework="pt", device="cpu") as handle:
                if not torch.equal(value, handle.get_tensor(name)):
                    raise ValueError("Scaled excluded tensor differs from source")
        filename = f"model-{number:05d}-of-{len(source_map):05d}.safetensors"
        save_file({name: value.contiguous()}, partial / filename, metadata={"format": "pt"})
        mapping[name] = filename
        payload += value.numel() * value.element_size()
        del value
    atomic_json(partial / "model.safetensors.index.json", {
        "metadata": {"total_size": payload}, "weight_map": mapping})
    report = {"status": "completed_validated", "identity": ident, "depth": depth,
              "code_sha": os.environ.get("FOLLOWUP_CODE_SHA"), "projection_count": 252,
              "tensor_count": len(mapping), "tensor_payload_bytes": payload,
              "eligible_elements": eligible_elements, "mse_fp32": error_sum / eligible_elements,
              "mse_after_fp16_cast": cast_error_sum / eligible_elements,
              "sqnr_db_fp32": 10 * math.log10(signal_sum / error_sum) if error_sum else None,
              "shard_sha256": {name: sha256(partial / name) for name in mapping.values()}}
    atomic_json(partial / "encoding_complete.json", report)
    os.replace(partial, target)
    return report


def storage(branch: str, depth: int, source: Path, base: Path, root: Path) -> dict:
    ident = scaled_identity(branch, source, base)
    prior = layers(branch, depth, source, base, root, ident)
    base_bytes = sum(path.stat().st_size for path in base.iterdir() if path.is_file())
    layer_bytes = []
    for layer in prior:
        directory, manifest = layer["directory"], layer["manifest"]
        layer_bytes.append({"depth": manifest["depth"],
            "code_bytes": sum((directory / chunk["code"]).stat().st_size for chunk in manifest["chunks"]),
            "metadata_bytes": sum((directory / chunk["metadata"]).stat().st_size
                                  for chunk in manifest["chunks"]),
            "manifest_bytes": (directory / "encoding_complete.json").stat().st_size,
            "external": "reference" in layer})
    reference_bytes = ((root / "layer1_reference.json").stat().st_size if branch == "s01" else 0)
    enhancement_bytes = sum(item["code_bytes"] + item["metadata_bytes"] + item["manifest_bytes"]
                            for item in layer_bytes) + reference_bytes
    return {"base_external_file_bytes": base_bytes, "base_counted_once": True,
            "layers": layer_bytes, "coefficient_reference_bytes": reference_bytes,
            "physical_reconstruction_bytes": base_bytes + enhancement_bytes,
            "nominal_projection_bits": 2 + depth * 2,
            "effective_bits_per_total_parameter":
                8 * (base_bytes + enhancement_bytes) / TOTAL_PARAMETERS,
            "basis": "measured_fake_qdq_base_plus_measured_encoded_enhancements"}


def evaluate(branch: str, depth: int, source: Path, base: Path, root: Path,
             result: Path, matgptq: Path) -> dict:
    check_revision(matgptq)
    validate_repaired_base(branch, source, base)
    ident = scaled_identity(branch, source, base)
    layers(branch, depth, source, base, root, ident)
    full = result / "full.json"
    if full.is_file():
        record = json.loads(full.read_text())
        if (record.get("status") != "completed_validated" or record.get("identity") != ident or
                record.get("code_sha") != os.environ.get("FOLLOWUP_CODE_SHA") or
                sha256(result / "evaluation.log") != record.get("evaluation_log_sha256") or
                (result / "checkpoint").exists()):
            raise ValueError("Existing scaled evaluation differs")
        return record
    if result.exists() and any(result.iterdir()):
        raise FileExistsError("Incomplete scaled evaluation requires inspection")
    result.mkdir(parents=True)
    atomic_json(result / "run.json", {"status": "started", "identity": ident,
                                      "depth": depth, "code_sha": os.environ.get("FOLLOWUP_CODE_SHA")})
    checkpoint = result / "checkpoint"
    materialization = materialize(branch, depth, source, base, root, checkpoint)
    corpus = corpus_identity(checkpoint)
    if corpus.get("token_sha256") != TOKEN_SHA or corpus.get("scored_tokens") != 298862:
        raise ValueError("Scaled MatGPTQ corpus identity differs")
    measurement = evaluate_matgptq(matgptq, checkpoint, result / "evaluation.log")
    if not math.isfinite(measurement["perplexity"]):
        raise ValueError("Nonfinite scaled perplexity")
    report = {"status": "completed_validated", "identity": ident, "depth": depth,
              "code_sha": os.environ.get("FOLLOWUP_CODE_SHA"),
              "materialization": materialization, "corpus": corpus,
              "evaluation": measurement, "storage": storage(branch, depth, source, base, root),
              "evaluation_log_sha256": sha256(result / "evaluation.log")}
    atomic_json(full, report)
    shutil.rmtree(checkpoint)  # only this job's generated checkpoint, after durable result
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("reference", "encode", "evaluate"))
    parser.add_argument("--branch", choices=("s01", "r02"), required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--matgptq", type=Path)
    args = parser.parse_args()
    if args.action == "reference":
        if args.branch != "s01" or args.depth != 1:
            parser.error("Only S01 E1 can be externally referenced")
        result = _external_s01(args.root, args.source, args.base,
                               scaled_identity(args.branch, args.source, args.base))["reference"]
    elif args.action == "encode":
        result = encode(args.branch, args.depth, args.source, args.base, args.root)
    else:
        if args.result is None or args.matgptq is None:
            parser.error("Evaluation requires --result and --matgptq")
        result = evaluate(args.branch, args.depth, args.source, args.base,
                          args.root, args.result, args.matgptq)
    print(json.dumps({"status": result["status"], "branch": args.branch,
                      "depth": args.depth}, indent=2))


if __name__ == "__main__":
    main()
