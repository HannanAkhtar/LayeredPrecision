"""Validate AutoRound RTN batch stages before a Slurm runner skips them.

Exit 0 for valid, 3 for absent/incomplete, and 2 for present but invalid.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from . import QWEN3_MODEL_ID, QWEN3_REVISION
from .encode import _code_fingerprint, _valid
from .evaluate import _identity
from .model_adapter import SelectionPolicy
from .quantize import AUTOROUND_RTN_SOURCE
from .storage import atomic_json, sha256


def encoding(artifact: Path, source: Path) -> bool:
    complete = artifact / "COMPLETE.json"
    if not complete.exists():
        return False
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if json.loads(complete.read_text())["manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Encoding completion hash differs from manifest")
    identity = manifest["identity"]
    if (manifest["method"] != "lp" or manifest["bits"] != [2, 2, 2, 2]
            or manifest["quantizer_family"] != "autoround_rtn_sym"
            or identity["model_id"] != QWEN3_MODEL_ID
            or identity["revision"] != QWEN3_REVISION
            or identity["projections"] != list(SelectionPolicy().projections)
            or identity["quantizer_spec"]["implementation"] != AUTOROUND_RTN_SOURCE
            or identity["granularity"] != "group" or identity["group_size"] != 128
            or identity["implementation_sha256"] != _code_fingerprint()):
        raise ValueError("Encoding family, settings, or implementation differs")
    if json.loads((artifact / "run.json").read_text())["identity"] != identity:
        raise ValueError("Encoding run identity differs")
    if (sha256(source / "model.safetensors.index.json") != identity["index_sha256"]
            or sha256(source / "config.json") != identity["config_sha256"]):
        raise ValueError("Original checkpoint identity differs")
    for shard, stats in identity["shard_stats"].items():
        current = (source / shard).stat()
        if current.st_size != stats["bytes"] or current.st_mtime_ns != stats["mtime_ns"]:
            raise ValueError(f"Original shard changed: {shard}")
    for name, digest in identity["sidecar_sha256"].items():
        if sha256(source / name) != digest or sha256(artifact / "hf" / name) != digest:
            raise ValueError(f"Source or copied sidecar changed: {name}")
    if not manifest["tensors"] or any(not _valid(artifact, r) for r in manifest["tensors"]):
        raise ValueError("Missing or corrupt tensor streams")
    for record in manifest["tensors"]:
        if not record["eligible"]:
            continue
        if len(record["layers"]) != 4 or any(
                layer.get("family") != "autoround_rtn_sym" or layer.get("bits") != 2
                or layer.get("granularity") != "group" or layer.get("group_size") != 128
                or layer.get("scale_dtype") != "float16"
                or layer.get("implementation") != AUTOROUND_RTN_SOURCE
                for layer in record["layers"]):
            raise ValueError(f"Invalid AutoRound layer metadata: {record['name']}")
    return True


def reconstruction(artifact: Path, checkpoint: Path, depth: int) -> bool:
    if not (checkpoint / "COMPLETE.json").exists():
        return False
    identity = _identity(checkpoint, "autoround-stage-validation")
    if (identity["kind"] != "lp_reconstructed" or
            identity["reconstruction"] != {"manifest_sha256": sha256(artifact / "manifest.json"),
                                            "depth": depth}):
        raise ValueError("Reconstruction belongs to a different prefix")
    return True


def _evaluation_identity(artifact: Path, checkpoint: Path, evaluator: Path, depth: int) -> dict:
    return {"manifest_sha256": sha256(artifact / "manifest.json"), "depth": depth,
            "checkpoint_index_sha256": sha256(checkpoint / "model.safetensors.index.json"),
            "evaluator_sha256": sha256(evaluator), "sequence_length": 2048,
            "dtype": "float16", "protocol": "matgptq_nonoverlap_wikitext2"}


def evaluation(artifact: Path, checkpoint: Path, result: Path,
               evaluator: Path, depth: int, mark: bool = False) -> bool:
    marker = result.with_name(result.name + ".validated.json")
    if not result.exists() and not marker.exists():
        return False
    if not result.exists():
        raise ValueError("Evaluation marker exists without result")
    if not reconstruction(artifact, checkpoint, depth):
        raise ValueError("Checkpoint reconstruction is incomplete")
    value = json.loads(result.read_text())
    if not isinstance(value, dict):
        raise ValueError("Evaluation output must be a JSON object")
    ppl = value.get("perplexity", value.get("ppl"))
    if isinstance(ppl, bool) or not isinstance(ppl, (float, int)) or not math.isfinite(ppl) or ppl <= 0:
        raise ValueError("Evaluation result lacks finite positive perplexity")
    identity = _evaluation_identity(artifact, checkpoint, evaluator, depth)
    payload = {"identity": identity, "result_sha256": sha256(result)}
    if mark:
        if marker.exists() and json.loads(marker.read_text()) != payload:
            raise ValueError("Existing evaluation marker differs")
        atomic_json(marker, payload)
    elif not marker.exists() or json.loads(marker.read_text()) != payload:
        raise ValueError("Evaluation result is unmarked or has different identity/content")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("encoding", "reconstruction", "evaluation", "mark-evaluation"))
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--evaluator", type=Path)
    parser.add_argument("--depth", type=int)
    args = parser.parse_args()
    try:
        if args.stage == "encoding":
            done = encoding(args.artifact, args.source)
        elif args.stage == "reconstruction":
            done = reconstruction(args.artifact, args.checkpoint, args.depth)
        else:
            done = evaluation(args.artifact, args.checkpoint, args.result, args.evaluator,
                              args.depth, args.stage == "mark-evaluation")
    except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"INVALID {args.stage}: {exc}", file=sys.stderr)
        return 2
    if not done:
        print(f"INCOMPLETE {args.stage}")
        return 3
    print(f"VALID {args.stage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
