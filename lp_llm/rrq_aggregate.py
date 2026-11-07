"""Aggregate measured RRQ W2 results without filling missing values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .rrq_w2 import MODES, load_config
from .storage import atomic_json


def aggregate(result_root: Path) -> dict:
    root = Path(result_root) / "rrq_w2_reproduction"
    rows = []
    def add_row(experiment_id: str, config: dict, job: Path,
                reference_variant: str = "current") -> None:
        diagnostic_path, full_path = job / "diagnostic.json", job / "full.json"
        encoding_path = job / "encoding_complete.json"
        diagnostic = json.loads(diagnostic_path.read_text()) if diagnostic_path.exists() else None
        full = json.loads(full_path.read_text()) if full_path.exists() else None
        encoding = json.loads(encoding_path.read_text()) if encoding_path.exists() else None
        for path, value in ((diagnostic_path, diagnostic), (full_path, full)):
            if value and (value.get("status") != "complete" or
                          value.get("configuration", {}).get("config_sha256") != config["config_sha256"]):
                raise ValueError(f"Incomplete or inconsistent result: {path}")
            if value and config["kind"] == "autoround" and value.get("identity", {}).get(
                    "autoround_variant") != reference_variant:
                raise ValueError(f"Wrong AutoRound reference variant: {path}")
        if encoding and (encoding.get("status") != "complete" or
                         encoding.get("identity", {}).get("config_sha256") != config["config_sha256"] or
                         encoding.get("identity", {}).get("autoround_variant") != reference_variant):
            raise ValueError(f"Incomplete or inconsistent encoding marker: {encoding_path}")
        report = (diagnostic or full or {}).get("construction", {}) or encoding or {}
        restored_names = report.get("identity", {}).get("restored_names")
        configured_names = report.get("configured_module_selection", {}).get("selected_module_names")
        changed_names = report.get("weight_change_diagnostic", {}).get("changed_tensor_names")
        eligible_names = report.get("eligible_tensor_names")
        coverage = (f"{len(restored_names)} restored" if restored_names is not None else
                    f"{len(configured_names)} configured" if configured_names is not None else
                    f"{len(changed_names)} changed (selection unavailable)" if changed_names is not None else
                    f"{len(eligible_names)} LP eligible" if eligible_names is not None else None)
        rows.append({"id": experiment_id if reference_variant == "current" else
                     f"{experiment_id}/{reference_variant}",
                     "reference_variant": reference_variant if config["kind"] == "autoround" else None,
                     "quantizer": ("original checkpoint" if experiment_id == "j01" else
                                   "B3 optimized RTN + BF16 restoration" if config["kind"] == "hybrid" else
                                   config["mode"]),
                     "bits": None if experiment_id == "j01" else
                             "mixed (2/BF16)" if config["kind"] == "hybrid" else config["bits"],
                     "group": None if experiment_id == "j01" else config["group_size"],
                     "coverage": coverage,
                     "effective_bpp": report.get("estimated_effective_bits_per_total_parameter",
                                                 report.get("effective_bits_per_total_parameter")),
                     "diagnostic_ppl": diagnostic["score"]["perplexity"] if diagnostic else None,
                     "full_ppl": full["score"]["perplexity"] if full else None,
                     "status": "unavailable" if config["kind"] == "unavailable" else
                     "full_complete" if full else "screen_complete" if diagnostic else
                     "encoding_complete" if encoding else
                     "export_unverified" if config["kind"] == "autoround" and
                     (job / "checkpoint").exists() else
                     "attempted_incomplete" if (job / "run.json").exists() or
                     (job / "autoround.log").exists() else "pending"})
    for experiment_id in MODES:
        config = load_config(experiment_id)
        job = root / experiment_id
        add_row(experiment_id, config, job)
        historical = job / "preprint_candidate"
        if config["kind"] == "autoround" and historical.exists():
            add_row(experiment_id, config, historical, "preprint_candidate")
    value = {"measured_results_only": True, "rows": rows}
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "aggregate.json", value)
    columns = ("ID", "Quantizer", "Bits", "Group", "Coverage", "Effective bpp (hybrid estimate)",
               "Diagnostic PPL", "Full PPL", "Status")
    markdown = "| " + " | ".join(columns) + " |\n|" + "|".join(["---"] * len(columns)) + "|\n"
    for row in rows:
        values = [row[k] for k in ("id", "quantizer", "bits", "group", "coverage", "effective_bpp",
                                   "diagnostic_ppl", "full_ppl", "status")]
        markdown += "| " + " | ".join("" if value is None else str(value) for value in values) + " |\n"
    (root / "aggregate.md").write_text(markdown, encoding="utf-8")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.result_root), indent=2))


if __name__ == "__main__":
    main()
