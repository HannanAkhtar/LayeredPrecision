"""Read-only PR #2308 status and result table; published values stay separate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .rrq_pr2308 import RESULT_SUBDIR
from .storage import atomic_json


def collect(result_root: Path) -> dict:
    root = result_root / RESULT_SUBDIR
    rows = []
    for experiment_id in ("pr01", "pr02", "pr03", "pr04"):
        directory = root / experiment_id
        row = {"id": experiment_id, "status": "not_started", "perplexity": None,
               "source": "official_PR_2308"}
        if directory.is_dir() and any(directory.iterdir()):
            row["status"] = "ready"
        if experiment_id == "pr01" and (directory / "comparison.json").is_file():
            audit = json.loads((directory / "comparison.json").read_text())
            row["status"] = ("completed_validated" if audit.get("status") == "completed_validated" and
                             audit.get("pr", {}).get("revision") ==
                             "7985c1b0f4eef65fb717511668fda7e347e584c3" else "failed")
        if experiment_id == "pr02" and (directory / "parity.json").is_file():
            parity = json.loads((directory / "parity.json").read_text())
            row["status"] = parity.get("status", "failed")
        if (directory / "run.json").is_file():
            row["status"] = "running"
        if (directory / "quantization_complete.json").is_file():
            marker = json.loads((directory / "quantization_complete.json").read_text())
            row["status"] = ("quantized_not_evaluated" if marker.get("status") == "completed_validated"
                             else "failed")
            row["quantized_module_count"] = marker.get("checkpoint", {}).get("quantized_module_count")
        if experiment_id == "pr04" and (directory / "qdq_complete.json").is_file():
            row["status"] = "quantized_not_evaluated"
        if (directory / "failure.json").is_file() and not (
                experiment_id == "pr03" and (directory / "quantization_complete.json").is_file()):
            row["status"] = "failed"
            row["failure"] = json.loads((directory / "failure.json").read_text())
        if (directory / "full.json").is_file():
            full = json.loads((directory / "full.json").read_text())
            if full.get("status") == "completed_validated":
                row["status"] = "completed_validated"
                row["perplexity"] = full.get("score", {}).get("perplexity")
                row["token_sha256"] = full.get("corpus", {}).get("token_sha256")
            else:
                row["status"] = "failed"
        rows.append(row)
    return {"experiments": rows, "existing_results_reference_only": {
        "j01": 7.01, "j02": 9928.00, "j03": 481.00, "j04": 9928.00, "j05": 65.62},
        "published_paper_reference_not_measured": {"perplexity": 16.75}}


def markdown(report: dict) -> str:
    lines = ["| ID | Status | WikiText-2 PPL |", "|---|---|---:|"]
    for row in report["experiments"]:
        ppl = "n/a" if row["perplexity"] is None else str(row["perplexity"])
        lines.append(f"| {row['id']} | {row['status']} | {ppl} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = collect(args.result_root)
    atomic_json(args.output, report)
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))


if __name__ == "__main__":
    main()
