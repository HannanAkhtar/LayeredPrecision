"""Fast read-only AUS inventory; presence is never numerical validation."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .storage import atomic_json


def _read(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) if root.is_dir() else 0


def inspect(name: str, root: Path, *, checkpoint: Path | None = None,
            marker: Path | None = None, evaluation: Path | None = None) -> dict:
    checkpoint = checkpoint or root
    marker = marker or root / "encoding_complete.json"
    evaluation = evaluation or root / "full.json"
    completion, full = _read(marker), _read(evaluation)
    index = _read(checkpoint / "model.safetensors.index.json")
    mapping = index.get("weight_map", {}) if index else {}
    shards = sorted(set(mapping.values())) if isinstance(mapping, dict) else []
    missing = [item for item in shards if not (checkpoint / item).is_file()]
    partial = root / "export.partial"
    process = _read(root / "quantization_process.json")
    requires_index = name in ("original", "s01_w2", "r02_w2") or \
        name in {f"{branch}_w{bits}" for branch in ("s01", "r02") for bits in (4, 6, 8)}
    if not root.exists():
        state = "missing"
    elif requires_index and index is None:
        state = "incomplete"
    elif name == "original" and not missing:
        state = "source_present_unverified"
    elif name == "s01_signed_e1" and completion and completion.get("status") == "completed_validated" and not missing:
        state = "numerically_complete_resource_uncertified"
    elif full and full.get("status") == "completed_validated" and not missing:
        state = "complete_validated_reported"
    elif completion and completion.get("status") in ("completed_validated", "external_layer_validated") and not missing:
        state = "encoded_not_evaluated"
    elif partial.exists() and process and process.get("returncode") == 0:
        state = "recoverable_candidate"
    elif partial.exists() or process or root.exists():
        state = "incomplete"
    else:
        state = "missing"
    if missing:
        state = "incomplete"
    return {"name": name, "path": str(root), "exists": root.exists(), "state": state,
            "checkpoint_path": str(checkpoint), "checkpoint_index_exists": index is not None,
            "indexed_tensor_count": len(mapping), "indexed_shards": shards,
            "missing_indexed_shards": missing, "completion_marker": str(marker),
            "completion_status": completion.get("status") if completion else None,
            "producing_code_sha": (completion or {}).get("code_sha"),
            "recorded_identity": (completion or full or {}).get("identity"),
            "partial_exists": partial.exists(),
            "quantization_returncode": process.get("returncode") if process else None,
            "evaluation_path": str(evaluation),
            "evaluation_status": full.get("status") if full else None,
            "perplexity": (full or {}).get("evaluation", {}).get("perplexity"),
            "provenance_files": {item: (root / item).is_file() for item in
                ("run.json", "resolved_config.json", "quantization_process.json",
                 "excluded_repair.json", "integrity_report.json")},
            "physical_file_bytes": _bytes(root)}


def inventory(source: Path, result_root: Path) -> dict:
    progressive = result_root / "progressive_llm"
    items = [inspect("original", source),
             inspect("s01_w2", result_root / "quantizer_followups/s01_fp16_load_repair",
                     checkpoint=result_root / "quantizer_followups/s01_fp16_load_repair/export"),
             inspect("r02_w2", result_root / "quantizer_followups/r02_fp16_load_repair",
                     checkpoint=result_root / "quantizer_followups/r02_fp16_load_repair/export"),
             inspect("s01_signed_e1", progressive / "ablations/s01_autoround_signed_rtn",
                     marker=progressive / "ablations/s01_autoround_signed_rtn/layer1/encoding_complete.json",
                     evaluation=progressive / "ablations/s01_autoround_signed_rtn/evaluation/full.json")]
    for branch in ("s01", "r02"):
        scaled = progressive / "scaled_signed_v1" / branch
        for depth in (1, 2, 3):
            items.append(inspect(f"{branch}_lp{2 + depth * 2}", scaled,
                marker=(scaled / "layer1_reference.json" if branch == "s01" and depth == 1
                        else scaled / f"layer{depth}/encoding_complete.json"),
                evaluation=scaled / f"evaluation/depth{depth}/full.json"))
        for bits in (4, 6, 8):
            control = progressive / "controls" / f"{branch}_w{bits}"
            checkpoint = control / ("export" if (control / "export").exists() else "export.partial")
            if checkpoint.is_dir() and not (checkpoint / "model.safetensors.index.json").is_file():
                nested = [path for path in checkpoint.iterdir() if path.is_dir() and
                          (path / "model.safetensors.index.json").is_file()]
                if len(nested) == 1:
                    checkpoint = nested[0]
            items.append(inspect(f"{branch}_w{bits}", control, checkpoint=checkpoint))
    usage = shutil.disk_usage(result_root if result_root.exists() else source.parent)
    return {"status": "read_only_inventory", "source": str(source),
            "result_root": str(result_root), "filesystem_free_bytes": usage.free,
            "quota_status": "unknown", "quota_free_bytes": None,
            "warning": "File presence and recorded status do not establish numerical validity",
            "artifacts": items}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--result-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = inventory(args.source, args.result_root)
    if args.output.exists():
        raise FileExistsError("Inventory output already exists; choose a new path")
    atomic_json(args.output, report)
    for item in report["artifacts"]:
        print(f"{item['name']:18} {item['state']:42} {item['physical_file_bytes']:>14} bytes")
    print(f"filesystem_free_bytes={report['filesystem_free_bytes']} quota=unknown")


if __name__ == "__main__":
    main()
