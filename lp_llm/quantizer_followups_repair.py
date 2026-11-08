"""Separately identify and materialize an S01 checkpoint after FP16 load loss.

The original AutoRound export is read only. This path is enabled only when all
excluded-weight changes exactly match a BF16 -> FP16 -> export-dtype round trip.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
from pathlib import Path

from .quantizer_followups import (PR_COMMIT, TOKEN_SHA256, _read_marker, audit_source, identity, load_config,
                                  validate_checkpoint, validate_resolved, verify_original_checkpoint)
from .quantizer_followups_audit import audit
from .rrq_autoround import validate_output
from .rrq_reference_eval import check_revision, corpus_identity, run as evaluate_matgptq
from .rrq_w2 import MATGPTQ_COMMIT
from .storage import atomic_json, sha256


REPAIR_ID = "s01_fp16_load_repair"


def _tensor_region(path: Path, name: str) -> tuple[int, int, str, tuple[int, ...]]:
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if length > 100_000_000:
            raise ValueError(f"Implausible Safetensors header: {path}")
        header = json.loads(stream.read(length))
    entry = header[name]
    start, end = entry["data_offsets"]
    if not 0 <= start < end <= path.stat().st_size - 8 - length:
        raise ValueError(f"Invalid tensor offsets: {name}")
    return 8 + length + start, end - start, entry["dtype"], tuple(entry["shape"])


def _copy_tensor_bytes(source: Path, source_name: str, target: Path, target_name: str) -> None:
    source_at, source_size, source_dtype, source_shape = _tensor_region(source, source_name)
    target_at, target_size, target_dtype, target_shape = _tensor_region(target, target_name)
    if (source_size, source_dtype, source_shape) != (target_size, target_dtype, target_shape):
        raise ValueError(f"Source/export dtype or shape differs for {source_name}")
    with source.open("rb") as reader, target.open("r+b") as writer:
        reader.seek(source_at)
        writer.seek(target_at)
        remaining = source_size
        while remaining:
            chunk = reader.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"Truncated source tensor: {source_name}")
            writer.write(chunk)
            remaining -= len(chunk)
        writer.flush()
        os.fsync(writer.fileno())


def _tensor_hash(path: Path, name: str) -> str:
    at, remaining, _, _ = _tensor_region(path, name)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(at)
        while remaining:
            chunk = stream.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"Truncated tensor: {name}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _map(directory: Path) -> dict[str, str]:
    return json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]


def verify_projection_bytes(raw: Path, repaired: Path, names: list[str]) -> dict[str, str]:
    raw_map, repaired_map = _map(raw), _map(repaired)
    hashes = {}
    for name in names:
        before = _tensor_hash(raw / raw_map[name], name)
        after = _tensor_hash(repaired / repaired_map[name], name)
        if before != after:
            raise ValueError(f"Quantized projection bytes changed during repair: {name}")
        hashes[name] = before
    return hashes


def repair(source: Path, raw_result: Path, target_result: Path, upstream: Path,
           matgptq: Path, stage: str = "all") -> dict:
    config = load_config("s01")
    check_revision(upstream, PR_COMMIT)
    check_revision(matgptq, MATGPTQ_COMMIT)
    original_identity = identity(config, source, upstream)
    verify_original_checkpoint(original_identity["source"], raw_result)
    if json.loads((raw_result / "run.json").read_text()).get("identity") != original_identity:
        raise ValueError("Original S01 run identity differs")
    if json.loads((raw_result / "quantization_process.json").read_text()).get("returncode") != 0:
        raise ValueError("Original S01 quantization did not exit successfully")
    if not (raw_result / "quantization.log").is_file():
        raise ValueError("Original S01 quantization log is missing")
    audit_source(upstream, "s01")
    validate_resolved(config, raw_result / "resolved_config.json", upstream)
    raw_info = validate_output(raw_result / "export", source)
    raw = Path(raw_info["resolved_checkpoint_path"])
    evidence = audit(source, raw_result / "export")
    if len(evidence["quantized_projection_names"]) != 252:
        raise ValueError("S01 export does not contain 252 eligible projections")
    changed = [entry for entry in evidence["excluded_tensors"] if entry["changed"]]
    if not changed or any(entry["fp16_roundtrip_mismatches"] for entry in evidence["excluded_tensors"]):
        raise ValueError("Excluded changes are not fully explained by FP16 load round-trip")
    if any(entry["source_dtype"] != entry["export_dtype"] for entry in changed):
        raise ValueError("Repair requires identical source and export dtypes")
    repair_identity = {"id": REPAIR_ID, "original_run_identity": original_identity,
                       "raw_checkpoint": {key: raw_info[key] for key in
                                          ("config_sha256", "index_sha256", "shard_sha256")},
                       "repair_rule": "restore excluded source bytes after verified FP16 round-trip",
                       "quantized_projection_count": 252}
    marker_path = target_result / "encoding_complete.json"
    run_path = target_result / "run.json"
    if run_path.is_file():
        if json.loads(run_path.read_text()).get("identity") != repair_identity:
            raise ValueError("Repaired result has another identity")
    elif target_result.exists() and any(target_result.iterdir()):
        raise ValueError("Nonempty unowned repaired result directory")
    else:
        target_result.mkdir(parents=True, exist_ok=True)
        atomic_json(run_path, {"status": "started", "identity": repair_identity})
    repaired = target_result / "export"
    if not repaired.exists():
        partial = target_result / "export.partial"
        if partial.exists():
            raise ValueError("Interrupted repair copy exists; inspect before retry")
        shutil.copytree(raw, partial)
        source_map, repaired_map = _map(source), _map(partial)
        for entry in changed:
            name = entry["name"]
            _copy_tensor_bytes(source / source_map[name], name,
                               partial / repaired_map[name], name)
        os.replace(partial, repaired)
    projection_hashes = verify_projection_bytes(raw, repaired, evidence["quantized_projection_names"])
    checkpoint = validate_checkpoint(config, source, repaired)
    marker = _read_marker(marker_path, repair_identity)
    current = {"status": "completed_validated", "identity": repair_identity,
               "checkpoint_validation": checkpoint, "projection_tensor_sha256": projection_hashes,
               "restored_excluded_tensors": [entry["name"] for entry in changed],
               "original_quantization_log_sha256": sha256(raw_result / "quantization.log")}
    if marker:
        if marker != current:
            raise ValueError("Repaired checkpoint differs from completion marker")
    else:
        atomic_json(marker_path, current)
    if stage == "encode":
        return current
    corpus = corpus_identity(repaired)
    if corpus["token_sha256"] != TOKEN_SHA256 or corpus["scored_tokens"] != 298862:
        raise ValueError("MatGPTQ corpus identity differs")
    full_path = target_result / "full.json"
    existing = _read_marker(full_path, repair_identity)
    if existing:
        if existing["checkpoint_validation"] != checkpoint or sha256(target_result / "evaluation.log") != existing["evaluation_log_sha256"]:
            raise ValueError("Repaired evaluation identity differs")
        return existing
    if (target_result / "evaluation.log").exists():
        raise ValueError("Unmarked repaired evaluation log exists; inspect before retry")
    evaluation = evaluate_matgptq(matgptq, repaired, target_result / "evaluation.log", python=sys.executable)
    full = {"status": "completed_validated", "identity": repair_identity,
            "checkpoint_validation": checkpoint, "corpus": corpus, "evaluation": evaluation,
            "evaluation_log_sha256": sha256(target_result / "evaluation.log")}
    atomic_json(full_path, full)
    return full


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--raw-result", type=Path, required=True)
    parser.add_argument("--repaired-result", type=Path, required=True)
    parser.add_argument("--upstream-source", type=Path, required=True)
    parser.add_argument("--matgptq-source", type=Path, required=True)
    parser.add_argument("--stage", choices=("encode", "all"), default="all")
    args = parser.parse_args()
    value = repair(args.source, args.raw_result, args.repaired_result,
                   args.upstream_source, args.matgptq_source, args.stage)
    print(json.dumps({"status": value["status"], "result": str(args.repaired_result),
                      "stage": args.stage}, indent=2))


if __name__ == "__main__":
    main()
