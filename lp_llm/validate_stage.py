"""Validate completed monolithic batch stages without changing their artifacts.

Exit 0 for a valid completed stage, 3 for an absent completion artifact, and
2 for a present but invalid artifact. The batch runner never skips exit 2.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from .encode import _code_fingerprint, _valid
from .evaluate import _existing, _identity, _package_version
from .storage import sha256


def encoding(artifact: Path, source: Path, bits: int, granularity: str) -> bool:
    complete = artifact / "COMPLETE.json"
    if not complete.exists():
        return False
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if json.loads(complete.read_text())["manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Encoding completion hash differs from manifest")
    identity = manifest["identity"]
    if (manifest["method"] != "monolithic" or manifest["bits"] != [bits]
            or identity["bits"] != [bits] or identity["granularity"] != granularity
            or identity["group_size"] != 128 or identity["implementation_sha256"] != _code_fingerprint()):
        raise ValueError("Completed encoding has different experiment settings or code")
    if json.loads((artifact / "run.json").read_text())["identity"] != identity:
        raise ValueError("Encoding run identity differs from manifest")
    if (identity["index_sha256"] != sha256(source / "model.safetensors.index.json")
            or identity["config_sha256"] != sha256(source / "config.json")):
        raise ValueError("Original checkpoint index or configuration changed")
    for shard, stats in identity["shard_stats"].items():
        current = (source / shard).stat()
        if current.st_size != stats["bytes"] or current.st_mtime_ns != stats["mtime_ns"]:
            raise ValueError(f"Original checkpoint shard changed: {shard}")
    for name, digest in identity["sidecar_sha256"].items():
        if sha256(source / name) != digest or sha256(artifact / "hf" / name) != digest:
            raise ValueError(f"Source or copied sidecar changed: {name}")
    records = manifest["tensors"]
    if not records or any(not _valid(artifact, record) for record in records):
        raise ValueError("Encoding tensor stream is missing or corrupt")
    return True


def reconstruction(artifact: Path, checkpoint: Path) -> bool:
    complete = checkpoint / "COMPLETE.json"
    if not complete.exists():
        return False
    identity = _identity(checkpoint, "monolithic-batch-validation")
    if (identity["kind"] != "lp_reconstructed" or
            identity["reconstruction"] != {"manifest_sha256": sha256(artifact / "manifest.json"),
                                           "depth": 1}):
        raise ValueError("Reconstruction does not match this monolithic artifact")
    return True


def evaluation(checkpoint: Path, result: Path) -> bool:
    if not result.exists():
        return False
    identity = _identity(checkpoint, "wikitext-2-raw-v1") | {
        "context": 2048, "stride": 512, "datasets": _package_version("datasets"),
        "join": "\\n\\n", "add_special_tokens": False}
    value = _existing(result, identity)
    if (value is None or value.get("corpus_tokens", 0) < 2
            or value.get("scored_tokens", 0) < 1
            or not math.isfinite(value.get("perplexity", float("nan")))):
        raise ValueError("Completed WikiText result is incomplete or invalid")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    encode_parser = sub.add_parser("encoding")
    encode_parser.add_argument("--artifact", type=Path, required=True)
    encode_parser.add_argument("--source", type=Path, required=True)
    encode_parser.add_argument("--bits", type=int, required=True)
    encode_parser.add_argument("--granularity", choices=("channel", "group"), required=True)
    reconstruct_parser = sub.add_parser("reconstruction")
    reconstruct_parser.add_argument("--artifact", type=Path, required=True)
    reconstruct_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser = sub.add_parser("evaluation")
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.stage == "encoding":
            done = encoding(args.artifact, args.source, args.bits, args.granularity)
        elif args.stage == "reconstruction":
            done = reconstruction(args.artifact, args.checkpoint)
        else:
            done = evaluation(args.checkpoint, args.result)
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
