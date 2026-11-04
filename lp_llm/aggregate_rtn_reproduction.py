"""Combine B0's established result with B1–B3 Phase 1 JSON reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .storage import atomic_json


IDS = ("b1_plain_fp16", "b2_plain_bf16", "b3_opt_fp16")


def _read(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def aggregate(result_root: Path) -> dict:
    result_root = Path(result_root)
    b0_artifact = result_root / "lp_autoround_rtn_g128_2_2_2_2"
    b0_manifest = _read(b0_artifact / "manifest.json")
    b0_storage = _read(b0_artifact / "storage_report.json")
    b0_full = _read(result_root / "matgptq_protocol" / "lp_autoround_rtn_g128_2.json")
    rows = [{"experiment_id": "b0_lp_plain_fp32", "algorithm": "plain RTN",
             "checkpoint_dtype": "bfloat16",
             "quantization_input_dtype": "float32",
             "parameter_mse_fp32": (b0_manifest["prefix_diagnostics"][0]["parameter_mse_fp32"]
                                    if b0_manifest else None),
             "sqnr_db_fp32": (b0_manifest["prefix_diagnostics"][0]["sqnr_db_fp32"]
                              if b0_manifest else None),
             "effective_bits_per_total_parameter": (b0_storage["prefixes"][0]["effective_bits_per_total_parameter"]
                                                    if b0_storage else None),
             "diagnostic_perplexity": None,
             "full_perplexity": (b0_full.get("perplexity", b0_full.get("ppl"))
                                 if b0_full else 9730.84365),
             "full_perplexity_provenance": ("existing_result_file" if b0_full else "user_reported_established_baseline") }]
    for experiment_id in IDS:
        report = _read(result_root / "rtn_reproduction" / experiment_id / "result.json")
        if report is None:
            rows.append({"experiment_id": experiment_id, "status": "not_run"})
            continue
        spec = report.get("quantizer") or {}
        diagnostic = report.get("diagnostic") or {}
        storage = report.get("physical_storage") or {}
        rows.append({"experiment_id": experiment_id, "status": report["validation_status"],
                     "algorithm": spec.get("algorithm"),
                     "checkpoint_dtype": "bfloat16",
                     "quantization_input_dtype": spec.get("quantization_input_dtype"),
                     "parameter_mse_fp32": report.get("parameter_mse_fp32"),
                     "sqnr_db_fp32": report.get("sqnr_db_fp32"),
                     "effective_bits_per_total_parameter": storage.get("effective_bits_per_total_parameter"),
                     "diagnostic_perplexity": diagnostic.get("perplexity"),
                     "full_perplexity": report.get("full_wikitext2_perplexity")})
        if experiment_id == "b3_opt_fp16":
            materialization = _read(result_root / "rtn_reproduction" / experiment_id /
                                    "materialization_fp16" / "comparison.json")
            if materialization:
                rows.append({"experiment_id": "b3_opt_fp16_fp16_checkpoint",
                             "algorithm": spec.get("algorithm"), "checkpoint_dtype": "float16",
                             "quantization_input_dtype": spec.get("quantization_input_dtype"),
                             "parameter_mse_fp32": report.get("parameter_mse_fp32"),
                             "sqnr_db_fp32": report.get("sqnr_db_fp32"),
                             "effective_bits_per_total_parameter": storage.get("effective_bits_per_total_parameter"),
                             "diagnostic_perplexity": materialization["diagnostic"]["fp16_checkpoint"]["perplexity"],
                             "full_perplexity": None})
    return {"protocol": "MatGPTQ-style WikiText-2, nonoverlap 2048, FP16 inference for full results",
            "rows": rows}


def markdown(value: dict) -> str:
    columns = ("experiment_id", "algorithm", "quantization_input_dtype", "checkpoint_dtype", "parameter_mse_fp32",
               "sqnr_db_fp32", "effective_bits_per_total_parameter", "diagnostic_perplexity",
               "full_perplexity")
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in value["rows"]:
        lines.append("| " + " | ".join(str(row.get(column, "")) if row.get(column) is not None else ""
                                       for column in columns) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    value = aggregate(args.result_root)
    root = args.result_root / "rtn_reproduction"
    atomic_json(root / "comparison.json", value)
    (root / "comparison.md").write_text(markdown(value), encoding="utf-8")
    print(markdown(value))


if __name__ == "__main__":
    main()
