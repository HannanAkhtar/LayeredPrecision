"""Summarize completed AutoRound RTN prefixes and optional local controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .storage import atomic_json


def _json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def report(artifact: Path, result_root: Path, asymmetric_artifact: Path | None = None,
           asymmetric_result_prefix: str | None = None) -> dict:
    manifest = json.loads((artifact / "manifest.json").read_text())
    storage = json.loads((artifact / "storage_report.json").read_text())
    if manifest.get("quantizer_family") != "autoround_rtn_sym":
        raise ValueError("Expected an AutoRound RTN artifact")
    controls = {
        "asymmetric_g128": (asymmetric_artifact, asymmetric_result_prefix),
        "ternary_g128": (result_root / "lp_symmetric_rtn_g128_2_2_2_2", "lp_symmetric_rtn_g128"),
    }
    rows = []
    for depth, (metrics, bytes_) in enumerate(zip(manifest["prefix_diagnostics"], storage["prefixes"]), 1):
        bits = depth * 2
        checkpoint = result_root / "decoded" / f"lp_autoround_rtn_g128_{bits}bit"
        eval_result = _json(result_root / "matgptq_protocol" / f"lp_autoround_rtn_g128_{bits}.json")
        reconstruction = _json(checkpoint / "COMPLETE.json")
        comparison = {}
        for label, (control_artifact, result_prefix) in controls.items():
            control_manifest = _json(control_artifact / "manifest.json") if control_artifact else None
            control_eval = (_json(result_root / "matgptq_protocol" / f"{result_prefix}_{bits}.json")
                            if result_prefix else None)
            control_metrics = control_manifest.get("prefix_diagnostics", []) if control_manifest else []
            comparison[label] = {
                "parameter_mse_fp32": (control_metrics[depth - 1].get("parameter_mse_fp32")
                                         if len(control_metrics) >= depth else None),
                "perplexity": (control_eval.get("perplexity", control_eval.get("ppl"))
                               if control_eval else None),
            }
        rows.append({
            "nominal_bits": bits,
            "effective_bits_per_total_parameter": bytes_["effective_bits_per_total_parameter"],
            "packed_code_bytes": bytes_["packed_code_bytes"],
            "metadata_and_framing_bytes": (bytes_["scale_zero_metadata_bytes"]
                                           + bytes_["other_decoder_framing_bytes"]),
            "parameter_mse_fp32": metrics["parameter_mse_fp32"],
            "sqnr_db_fp32": metrics["sqnr_db_fp32"],
            "perplexity": (eval_result.get("perplexity", eval_result.get("ppl"))
                           if eval_result else None),
            "quantization_seconds_full_hierarchy": manifest["processing_seconds"],
            "reconstruction_seconds": reconstruction.get("seconds") if reconstruction else None,
            "comparison": comparison,
        })
    return {"protocol": "matgptq_nonoverlap_wikitext2_fp16_2048",
            "note": "Null comparison fields mean local control artifacts/results were unavailable.",
            "prefixes": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--asymmetric-artifact", type=Path)
    parser.add_argument("--asymmetric-result-prefix")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = report(args.artifact, args.result_root, args.asymmetric_artifact,
                   args.asymmetric_result_prefix)
    atomic_json(args.output, value)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
