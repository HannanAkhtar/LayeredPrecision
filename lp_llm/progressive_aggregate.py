"""Measured and explicitly modeled LP/control quality versus total-model storage."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from .progressive_lp import BRANCHES, TOTAL_PARAMETERS, storage_report
from .progressive_plan import points
from .progressive_resources import estimate
from .storage import atomic_json


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _new_status(point: dict, root: Path, full: dict | None) -> str:
    if full:
        return "completed"
    path = Path(point["output"])
    if point["kind"] == "lp":
        branch_root = root / point["branch"]
        depth = point["depth"]
        if (branch_root / f"failure-layer{depth}.json").is_file() or (path / "failure.json").is_file():
            return "failed"
        if (branch_root / f"layer{depth}.partial").exists() or (path / "checkpoint.partial").exists():
            return "incomplete"
        if (branch_root / f"layer{depth}/encoding_complete.json").is_file():
            return "encoded"
    else:
        if (path / "failure.json").is_file():
            return "failed"
        if (path / "export.partial").exists():
            return "incomplete"
        if (path / "encoding_complete.json").is_file():
            return "encoded"
        if (path / "quantization_process.json").is_file():
            return "attempted"
    return "pending"


def modeled_compact(source: Path, branch: str, bits: int) -> dict:
    sizes = estimate(source)
    codes = math.ceil(sizes["projected_elements"] * bits / 8)
    scales = sizes["projection_group_count"] * 2  # pinned AutoRound FP16 scale
    zeros = 0 if branch == "s01" else sizes["projection_group_count"]  # modeled uint8 integer zp
    excluded = sizes["source_dense_payload_bytes"] - sizes["projected_elements"] * 2
    return {"modeled_code_bytes": codes, "modeled_scale_bytes": scales,
            "modeled_zero_point_bytes": zeros, "source_exact_excluded_bytes": excluded,
            "modeled_total_bytes": codes + scales + zeros + excluded,
            "assumption": "hypothetical packed codes, FP16 scales, uint8 asymmetric zero points; "
                          "official fake-QDQ export supplies no compact bitstream"}


def rows(source: Path, result_root: Path) -> list[dict]:
    root = result_root / "progressive_llm"
    items = []
    existing = [("j01", None, None, 7.01, result_root / "rrq_w2_reproduction/j01/full.json"),
                ("s01", "s01", 2, 12.47, result_root / "quantizer_followups/s01_fp16_load_repair/full.json"),
                ("r02", "r02", 2, 32.22, result_root / "quantizer_followups/r02_fp16_load_repair/full.json"),
                ("r01", "r01", 2, 32.53, result_root / "quantizer_followups/r01_fp16_load_repair/full.json"),
                ("j05", None, 2, 65.62, result_root / "rrq_w2_reproduction/j05/full.json")]
    for name, branch, bits, established, path in existing:
        full = _read(path)
        ppl = full.get("evaluation", {}).get("perplexity") if full else None
        if ppl is None and full:
            ppl = full.get("perplexity")
        modeled = modeled_compact(source, branch, 2) if branch in BRANCHES else None
        base_dir = path.parent / "export" if branch in BRANCHES else None
        physical_base = (sum(item.stat().st_size for item in base_dir.iterdir() if item.is_file())
                         if base_dir is not None and base_dir.is_dir() else None)
        items.append({"id": name, "kind": "existing", "method": name,
                      "base": branch, "layers": 0, "nominal_projection_bits": bits,
                      "status": "completed" if full else "unavailable",
                      "perplexity": ppl, "established_rounded_ppl": established,
                      "actual_payload_bytes": None, "physical_reconstruction_bytes": physical_base,
                      "modeled_compact_total_bytes": modeled["modeled_total_bytes"] if modeled else None,
                      "effective_total_model_bpp": (8 * modeled["modeled_total_bytes"] / TOTAL_PARAMETERS)
                      if modeled else None, "storage_basis": "modeled_compact" if modeled else "unavailable",
                      "result_path": str(path)})
    for point in points(result_root, 2):
        branch = point["branch"]
        full = _read(Path(point["output"]) / "full.json")
        if point["kind"] == "lp":
            depth = point["depth"]
            base = result_root / "quantizer_followups" / BRANCHES[branch] / "export"
            try:
                stored = storage_report(branch, source, base, root / branch, depth)["prefixes"][-1]
            except (FileNotFoundError, ValueError):
                stored = None
            base_model = modeled_compact(source, branch, 2)
            residual = (stored["total_reconstruction_bytes"] - stored["base_dense_external_bytes"]
                        if stored else None)
            compact = base_model["modeled_total_bytes"] + residual if residual is not None else None
            items.append({"id": point["id"], "kind": "lp", "method": "common_signed_e2",
                "base": branch, "layers": depth, "nominal_projection_bits": 2 + depth * 2,
                "status": _new_status(point, root, full),
                "perplexity": full.get("evaluation", {}).get("perplexity") if full else None,
                "established_rounded_ppl": None,
                "actual_payload_bytes": stored["residual_code_payload_bytes"] if stored else None,
                "physical_reconstruction_bytes": stored["total_reconstruction_bytes"] if stored else None,
                "modeled_compact_total_bytes": compact,
                "effective_total_model_bpp": 8 * compact / TOTAL_PARAMETERS if compact else None,
                "storage_basis": "modeled_base_plus_measured_residual" if compact else "unavailable",
                "result_path": point["output"]})
        else:
            modeled = modeled_compact(source, branch, point["bits"])
            physical = (full.get("checkpoint_validation", {}).get("actual_checkpoint_file_bytes")
                        if full else None)
            items.append({"id": point["id"], "kind": "control",
                "method": "signround_v2" if branch == "s01" else "optimized_rtn_asym_neuqi",
                "base": branch, "layers": 0, "nominal_projection_bits": point["bits"],
                "status": _new_status(point, root, full),
                "perplexity": full.get("evaluation", {}).get("perplexity") if full else None,
                "established_rounded_ppl": None, "actual_payload_bytes": None,
                "physical_reconstruction_bytes": physical,
                "modeled_compact_total_bytes": modeled["modeled_total_bytes"],
                "effective_total_model_bpp": 8 * modeled["modeled_total_bytes"] / TOTAL_PARAMETERS,
                "storage_basis": "modeled_compact_no_bitstream",
                "result_path": point["output"]})
    for row in items:
        row.update({"code_sha": None, "mse_fp32": None,
                    "normalized_mse_fp32": None, "sqnr_db_fp32": None,
                    "encoding_runtime_seconds": None, "encoding_peak_rss_kib": None,
                    "encoding_peak_gpu_used_mb": None,
                    "evaluation_runtime_seconds": None,
                    "evaluation_peak_rss_mb": None,
                    "evaluation_peak_gpu_used_mb": None})
        if row["kind"] == "existing":
            continue
        if row["kind"] == "lp":
            layer = _read(root / row["base"] /
                          f"layer{row['layers']}/encoding_complete.json")
            if layer:
                row.update({"code_sha": layer.get("code_sha"),
                            "mse_fp32": layer.get("mse_fp32"),
                            "normalized_mse_fp32": layer.get("normalized_mse_fp32"),
                            "sqnr_db_fp32": layer.get("sqnr_db_fp32"),
                            "encoding_runtime_seconds": layer.get("runtime_seconds"),
                            "encoding_peak_rss_kib": layer.get("process_peak_rss_kib")})
        else:
            result = Path(row["result_path"])
            process = _read(result / "quantization_process.json")
            marker = _read(result / "encoding_complete.json")
            if marker:
                row["code_sha"] = marker.get("identity", {}).get("code_sha")
            if process:
                row.update({"encoding_runtime_seconds": process.get("runtime_seconds"),
                            "encoding_peak_rss_kib": (
                                process["sampled_peak_process_tree_rss_mb"] * 1024
                                if process.get("sampled_peak_process_tree_rss_mb") is not None else None),
                            "encoding_peak_gpu_used_mb": process.get("peak_gpu_memory_used_mb")})
        full = _read(Path(row["result_path"]) / "full.json")
        execution = full.get("evaluation", {}).get("execution", {}) if full else {}
        row.update({"evaluation_runtime_seconds": execution.get("runtime_seconds"),
                    "evaluation_peak_rss_mb": execution.get("sampled_peak_process_tree_rss_mb"),
                    "evaluation_peak_gpu_used_mb": execution.get("peak_gpu_memory_used_mb")})
    return items


def write(source: Path, result_root: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    records = rows(source, result_root)
    report = {"status": "aggregated", "rows": records,
              "model_parameter_count": TOTAL_PARAMETERS,
              "storage_warning": "control and W2 compact bytes are modeled; LP residual files are measured; "
                                 "dense checkpoint bytes are separate; physical simulcast stores W2 and "
                                 "all three control checkpoints independently, without excluded-weight deduplication"}
    atomic_json(output / "comparison.json", report)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    return report


def plots(report: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    measured = [row for row in report["rows"] if row["perplexity"] is not None and
                row["effective_total_model_bpp"] is not None]
    for filename, title, selected in (
            ("perplexity_vs_total_bpp.png", "WikiText-2 perplexity versus modeled total-model bpp", measured),
            ("lp_vs_monolithic.png", "LP and monolithic controls", [row for row in measured
                if row["kind"] in ("lp", "control")])):
        figure, axis = plt.subplots(figsize=(8, 5))
        for row in selected:
            axis.scatter(row["effective_total_model_bpp"], row["perplexity"],
                         marker="o" if row["kind"] == "lp" else "s")
            axis.annotate(row["id"], (row["effective_total_model_bpp"], row["perplexity"]),
                          fontsize=7)
        axis.set(xlabel="Modeled effective bits per total parameter", ylabel="WikiText-2 perplexity",
                 title=title)
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output / filename, dpi=160)
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 5))
    for branch in BRANCHES:
        layered = [row for row in report["rows"] if row["base"] == branch and row["kind"] == "lp"
                   and row["physical_reconstruction_bytes"] is not None]
        controls = [row for row in report["rows"] if row["base"] == branch and row["kind"] == "control"
                    and row["physical_reconstruction_bytes"] is not None]
        base = next((row for row in report["rows"] if row["id"] == branch), None)
        if layered:
            axis.bar(branch + " LP hierarchy", max(row["physical_reconstruction_bytes"] for row in layered) / 2**30)
        if len(controls) == 3 and base and base["physical_reconstruction_bytes"] is not None:
            simulcast = base["physical_reconstruction_bytes"] + sum(
                row["physical_reconstruction_bytes"] for row in controls)
            axis.bar(branch + " control simulcast", simulcast / 2**30)
    axis.set(ylabel="Measured file bytes (GiB)", title="Multi-rate storage: hierarchy versus simulcast")
    figure.tight_layout()
    figure.savefig(output / "multirate_storage.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = write(args.source, args.result_root, args.output)
    plots(report, args.output)
    print(json.dumps({"rows": len(report["rows"]), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
