"""Materialize and evaluate verified S01/R02 LP prefixes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .progressive_lp import (BRANCHES, ROWS, _files, _slice, encode_layer,
                             identity, reconstruct_block, storage_report, validate_repaired_base,
                             verified_layers)
from .rrq_pr2308_compare import _eligible, _weight_map
from .rrq_reference_eval import check_revision, corpus_identity, run as evaluate_matgptq
from .rrq_w2 import MATGPTQ_COMMIT
from .storage import SIDECARS, atomic_json, sha256


TOKEN_SHA = "8ffcee84dd4b684ec631cb52845e18c9c181ce6fdb8b26d65a4ffdc3d35fbda3"


def _chunk_for(manifest: dict, name: str, first: int, stop: int) -> dict:
    matches = [item for item in manifest["chunks"] if
               item["name"] == name and item["first"] == first and item["stop"] == stop]
    if len(matches) != 1:
        raise ValueError("LP chunk partition or tensor coverage differs")
    return matches[0]


def materialize(branch: str, depth: int, source: Path, base: Path, root: Path,
                target: Path, *, rows: int = ROWS) -> dict:
    """One tensor per shard; never overwrite a checkpoint or interrupted partial."""
    if depth not in (0, 1, 2, 3):
        raise ValueError("LP materialization depth must be 0..3")
    ident = identity(branch, source, base)
    layers = verified_layers(root, depth, ident)
    partial = target.with_name(target.name + ".partial")
    if target.exists() or partial.exists():
        raise FileExistsError("LP checkpoint or partial materialization exists")
    source_map, base_map = _weight_map(source), _weight_map(base)
    partial.mkdir(parents=True)
    for sidecar in SIDECARS:
        if (source / sidecar).is_file():
            shutil.copy2(source / sidecar, partial / sidecar)
    mapping, total = {}, 0
    for number, name in enumerate(sorted(source_map), 1):
        with safe_open(source / source_map[name], framework="pt", device="cpu") as handle:
            shape = tuple(handle.get_slice(name).get_shape())
        if _eligible(name):
            if depth == 0:
                with safe_open(base / base_map[name], framework="pt", device="cpu") as handle:
                    value = handle.get_tensor(name)
            else:
                value = torch.empty(shape, dtype=torch.float16)
                for first in range(0, shape[0], rows):
                    stop = min(first + rows, shape[0])
                    base_rows = _slice(base, base_map, name, first, stop)
                    entries = [(root / f"layer{index}", _chunk_for(manifest, name, first, stop))
                               for index, manifest in enumerate(layers, 1)]
                    reconstructed = reconstruct_block(base_rows, entries)
                    if not bool(torch.isfinite(reconstructed).all()):
                        raise ValueError(f"Nonfinite LP checkpoint projection: {name}")
                    value[first:stop] = reconstructed.to(torch.float16)
        else:
            with safe_open(source / source_map[name], framework="pt", device="cpu") as handle:
                value = handle.get_tensor(name)
            with safe_open(base / base_map[name], framework="pt", device="cpu") as handle:
                if not torch.equal(value, handle.get_tensor(name)):
                    raise ValueError(f"Excluded W2 tensor differs from source: {name}")
        filename = f"model-{number:05d}-of-{len(source_map):05d}.safetensors"
        save_file({name: value.contiguous()}, partial / filename, metadata={"format": "pt"})
        with safe_open(partial / filename, framework="pt", device="cpu") as handle:
            if not torch.equal(handle.get_tensor(name), value):
                raise ValueError(f"Materialized tensor round trip differs: {name}")
        mapping[name] = filename
        total += value.numel() * value.element_size()
        del value
    atomic_json(partial / "model.safetensors.index.json", {
        "metadata": {"total_size": total}, "weight_map": mapping})
    checked = _weight_map(partial)
    if checked != mapping:
        raise ValueError("Materialized checkpoint index differs")
    report = {"status": "completed_validated", "identity": ident, "depth": depth,
              "code_sha": os.environ.get("FOLLOWUP_CODE_SHA"),
              "source": _files(source), "base": _files(base), "tensor_count": len(mapping),
              "projection_count": 252, "checkpoint_dtype": "base_exact" if depth == 0 else
              "float16_eligible_mixed_source_excluded",
              "tensor_payload_bytes": total,
              "shard_sha256": {name: sha256(partial / name)
                                for name in sorted(set(mapping.values()))}}
    atomic_json(partial / "encoding_complete.json", report)
    os.replace(partial, target)
    return report


def evaluate(branch: str, depth: int, source: Path, base: Path, root: Path,
             matgptq: Path, result: Path, *, python: str = sys.executable) -> dict:
    if depth not in (1, 2, 3):
        raise ValueError("Invalid LP evaluation depth")
    check_revision(matgptq, MATGPTQ_COMMIT)
    validate_repaired_base(branch, source, base)
    ident = identity(branch, source, base)
    verified_layers(root, depth, ident)
    full = result / "full.json"
    if full.is_file():
        if (result / "checkpoint").exists() or (result / "checkpoint.partial").exists():
            raise FileExistsError("Completed LP evaluation retains a temporary checkpoint")
        record = json.loads(full.read_text())
        if (record.get("identity") != ident or record.get("status") != "completed_validated" or
                (os.environ.get("FOLLOWUP_CODE_SHA") and
                 record.get("code_sha") != os.environ["FOLLOWUP_CODE_SHA"]) or
                sha256(result / "evaluation.log") != record.get("evaluation_log_sha256")):
            raise ValueError("Existing LP evaluation differs")
        return record
    if result.exists() and any(result.iterdir()):
        raise FileExistsError("Incomplete LP evaluation requires inspection")
    result.mkdir(parents=True)
    atomic_json(result / "run.json", {"status": "started", "identity": ident, "depth": depth,
                                      "code_sha": os.environ.get("FOLLOWUP_CODE_SHA")})
    checkpoint = result / "checkpoint"
    try:
        materialization = materialize(branch, depth, source, base, root, checkpoint)
        corpus = corpus_identity(checkpoint)
        if corpus["token_sha256"] != TOKEN_SHA or corpus["scored_tokens"] != 298862:
            raise ValueError("LP MatGPTQ corpus identity differs")
        measurement = evaluate_matgptq(matgptq, checkpoint, result / "evaluation.log", python=python)
        record = {"status": "completed_validated", "identity": ident, "depth": depth,
                  "code_sha": os.environ.get("FOLLOWUP_CODE_SHA"),
                  "materialization": materialization, "corpus": corpus,
                  "evaluation": measurement,
                  "storage": storage_report(branch, source, base, root, depth),
                  "evaluation_log_sha256": sha256(result / "evaluation.log")}
        atomic_json(full, record)
        # Only this job's generated checkpoint is removed after durable full.json.
        shutil.rmtree(checkpoint)
        return record
    except Exception as error:
        atomic_json(result / "failure.json", {"status": "failed", "error": str(error),
                                               "error_type": type(error).__name__, "identity": ident})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("encode", "evaluate"))
    parser.add_argument("--branch", choices=tuple(BRANCHES), required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--matgptq", type=Path)
    parser.add_argument("--pilot-tensors", type=int)
    args = parser.parse_args()
    if args.action == "encode":
        try:
            validate_repaired_base(args.branch, args.source, args.base)
            report = encode_layer(args.branch, args.depth, args.source, args.base, args.root,
                                  limit_tensors=args.pilot_tensors)
            (args.root / f"failure-layer{args.depth}.json").unlink(missing_ok=True)
        except Exception as error:
            atomic_json(args.root / f"failure-layer{args.depth}.json", {
                "status": "failed", "branch": args.branch, "depth": args.depth,
                "error_type": type(error).__name__, "error": str(error)})
            raise
    else:
        if args.result is None or args.matgptq is None:
            parser.error("evaluate requires --result and --matgptq")
        report = evaluate(args.branch, args.depth, args.source, args.base, args.root,
                          args.matgptq, args.result)
    print(json.dumps({"status": report["status"], "depth": args.depth,
                      "branch": args.branch}, indent=2))


if __name__ == "__main__":
    main()
