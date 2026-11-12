"""Summarize only verified S01/R02 scaled LP and control results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from .storage import atomic_json, sha256


TOKEN_SHA = "8ffcee84dd4b684ec631cb52845e18c9c181ce6fdb8b26d65a4ffdc3d35fbda3"


def _row(name: str, branch: str, bits: int, path: Path, kind: str) -> dict:
    if not path.is_file():
        return {"id": name, "branch": branch, "nominal_bits": bits, "kind": kind,
                "status": "missing", "perplexity": None, "result_path": str(path)}
    full = json.loads(path.read_text())
    ppl = full.get("evaluation", {}).get("perplexity")
    log = path.parent / "evaluation.log"
    corpus = full.get("corpus", {})
    if (full.get("status") != "completed_validated" or not isinstance(ppl, (int, float)) or
            not math.isfinite(ppl) or corpus.get("token_sha256") != TOKEN_SHA or
            corpus.get("scored_tokens") != 298862 or not log.is_file() or
            sha256(log) != full.get("evaluation_log_sha256")):
        return {"id": name, "branch": branch, "nominal_bits": bits, "kind": kind,
                "status": "invalid_report", "perplexity": None, "result_path": str(path)}
    storage = full.get("storage", {})
    check = full.get("checkpoint_validation", {})
    return {"id": name, "branch": branch, "nominal_bits": bits, "kind": kind,
            "status": "completed_validated_reported", "perplexity": ppl,
            "mse_fp32": (full.get("materialization", {}) or {}).get("mse_fp32"),
            "sqnr_db_fp32": (full.get("materialization", {}) or {}).get("sqnr_db_fp32"),
            "physical_reconstruction_bytes": storage.get("physical_reconstruction_bytes") or
                check.get("actual_checkpoint_file_bytes"),
            "storage_basis": storage.get("basis", "fake_qdq_checkpoint"),
            "result_path": str(path)}


def aggregate(result_root: Path) -> dict:
    progressive = result_root / "progressive_llm"
    rows = [{"id": "original_fp16", "branch": "reference", "nominal_bits": 16,
             "kind": "reference", "status": "historical_reported", "perplexity": 7.01}]
    for branch, w2 in (("s01", 12.47), ("r02", 32.22)):
        base = result_root / f"quantizer_followups/{branch}_fp16_load_repair/full.json"
        base_row = _row(f"{branch}_w2", branch, 2, base, "base")
        if base_row["status"] == "completed_validated_reported" and round(base_row["perplexity"], 2) != w2:
            raise ValueError(f"Stored {branch} W2 baseline differs")
        if base_row["status"] == "completed_validated_reported":
            export = base.parent / "export"
            base_row["physical_reconstruction_bytes"] = sum(
                path.stat().st_size for path in export.rglob("*") if path.is_file())
        rows.append(base_row)
        for depth in (1, 2, 3):
            rows.append(_row(f"{branch}_lp{2 + depth * 2}", branch, 2 + depth * 2,
                progressive / f"scaled_signed_v1/{branch}/evaluation/depth{depth}/full.json", "scaled_lp"))
        for bits in (4, 6, 8):
            path = progressive / f"controls/{branch}_w{bits}/full.json"
            if branch == "s01" and bits == 4 and not path.is_file():
                path = progressive / "scaled_signed_v1/control_recovery/s01_w4/evaluation/full.json"
            rows.append(_row(f"{branch}_w{bits}", branch, bits, path, "monolithic_control"))
    return {"status": "reported_results_only", "note": "Physical bytes and modeled packed bytes are not interchangeable",
            "rows": rows}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--result-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = aggregate(a.result_root)
    if a.output.exists():
        raise FileExistsError("Aggregate output already exists")
    atomic_json(a.output, report)
    with a.output.with_suffix(".csv").open("w", newline="") as handle:
        fields = ("id", "branch", "kind", "nominal_bits", "status", "perplexity",
                  "mse_fp32", "sqnr_db_fp32", "physical_reconstruction_bytes",
                  "storage_basis", "result_path")
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(report["rows"])
    print(json.dumps({"rows": len(report["rows"]), "output": str(a.output)}, indent=2))


if __name__ == "__main__":
    main()
