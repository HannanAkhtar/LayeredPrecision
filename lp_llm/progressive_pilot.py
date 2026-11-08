"""Seven full-size projections from one real Qwen3 layer, two E2 stages."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from .progressive_lp import (BRANCHES, encode_layer, identity,
                             validate_repaired_base, verified_layers)
from .storage import atomic_json


def run(branch: str, source: Path, base: Path, root: Path, report_path: Path,
        code_sha: str) -> dict:
    import resource
    if branch not in BRANCHES or len(code_sha) != 40:
        raise ValueError("Invalid pilot branch or code SHA")
    if report_path.exists():
        record = json.loads(report_path.read_text())
        if (record.get("status") != "pilot_validated" or record.get("code_sha") != code_sha or
                record.get("identity") != identity(branch, source, base)):
            raise ValueError("Existing pilot report identity differs")
        verified_layers(root, 2, record["identity"], pilot=True)
        return record
    started = time.monotonic()
    validate_repaired_base(branch, source, base)
    first = encode_layer(branch, 1, source, base, root, limit_tensors=7)
    second = encode_layer(branch, 2, source, base, root, limit_tensors=7)
    names = {item["name"] for item in first["chunks"]}
    if (first["status"] != "pilot_validated" or second["status"] != "pilot_validated" or
            len(names) != 7 or any(not name.startswith("model.layers.0.") for name in names) or
            "model.layers.0.mlp.down_proj.weight" not in names):
        raise ValueError("One-layer LP pilot did not validate all seven projections")
    ident = identity(branch, source, base)
    verified_layers(root, 2, ident, pilot=True)
    report = {"status": "pilot_validated", "branch": branch, "code_sha": code_sha,
              "identity": ident, "source_layer": 0, "projection_names": sorted(names),
              "base_checkpoint_source_exact_exclusions": True,
              "base_zero_enhancement_exact": True,
              "first": {key: first[key] for key in ("projection_count", "encoded_elements",
                       "mse_fp32", "normalized_mse_fp32", "code_bytes", "metadata_bytes")},
              "second": {key: second[key] for key in ("projection_count", "encoded_elements",
                       "mse_fp32", "normalized_mse_fp32", "code_bytes", "metadata_bytes")},
              "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "host_allocation_bytes": int(os.environ.get("SLURM_MEM_PER_NODE", "0")) * 1024**2,
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
              "runtime_seconds": time.monotonic() - started,
              "artifact_bytes": sum(path.stat().st_size for path in root.rglob("*") if path.is_file())}
    if (report["host_allocation_bytes"] and
            report["process_peak_rss_kib"] * 1024 >= 0.9 * report["host_allocation_bytes"]):
        raise RuntimeError("LP pilot lacks 10% host-memory headroom")
    atomic_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=tuple(BRANCHES), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.branch, args.source, args.base, args.root,
                         args.report, args.code_sha), indent=2))


if __name__ == "__main__":
    main()
