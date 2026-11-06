"""Restore selected original BF16 tensors into the existing B3 QDQ checkpoint."""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file

from .evaluate import _identity
from .model_adapter import SelectionPolicy, inventory, read_tensor
from .storage import atomic_json, atomic_safetensors, sha256


def selected_names(manifest: dict, mode: str) -> set[str]:
    eligible = {r["name"] for r in manifest["tensors"] if r["eligible"]}
    if mode in ("first_third", "final_third"):
        blocks = sorted({int(re.search(r"model\.layers\.(\d+)\.", name).group(1)) for name in eligible})
        count = math.ceil(len(blocks) / 3)
        selected = set(blocks[:count] if mode == "first_third" else blocks[-count:])
        return {name for name in eligible if int(re.search(r"model\.layers\.(\d+)\.", name).group(1)) in selected}
    projections = set(mode.split(","))
    if not projections <= set(SelectionPolicy().projections):
        raise ValueError(f"Unknown projection selection: {mode}")
    return {name for name in eligible if name.rsplit(".", 2)[-2] in projections}


def build(source: Path, artifact: Path, base: Path, output: Path, mode: str) -> dict:
    source, artifact, base, output = map(Path, (source, artifact, base, output))
    resolved_output = output.resolve()
    if any(path.resolve() == resolved_output or path.resolve().is_relative_to(resolved_output)
           for path in (source, artifact, base)):
        raise ValueError("Hybrid output may not overwrite source, artifact, or B3 base")
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("quantizer_family") != "autoround_opt_fp16" or manifest.get("bits") != [2]
            or json.loads((artifact / "COMPLETE.json").read_text())["manifest_sha256"] != sha256(manifest_path)):
        raise ValueError("Expected completed existing B3 optimized 2-bit artifact")
    config, source_map = inventory(source)
    base_identity = _identity(base, "rrq-b3-base")
    if base_identity["reconstruction"] != {"manifest_sha256": sha256(manifest_path),
                                            "depth": 1, "checkpoint_dtype": "float16"}:
        raise ValueError("Existing B3 FP16 checkpoint does not match artifact")
    names = selected_names(manifest, mode)
    if not names:
        raise ValueError("Hybrid selects no original tensors")
    identity = {"mode": mode, "manifest_sha256": sha256(manifest_path),
                "source_identity": _identity(source, "rrq-hybrid-source"),
                "base_identity": base_identity,
                "restored_names": sorted(names)}
    output.mkdir(parents=True, exist_ok=True)
    run_path = output / "hybrid_run.json"
    if run_path.exists():
        if json.loads(run_path.read_text()) != identity:
            raise ValueError("Hybrid output belongs to another experiment")
    elif any(output.iterdir()):
        raise ValueError("Nonempty hybrid output has no identity")
    else:
        atomic_json(run_path, identity)
    base_index = json.loads((base / "model.safetensors.index.json").read_text())
    if len(set(base_index["weight_map"].values())) != len(base_index["weight_map"]):
        raise ValueError("B3 FP16 checkpoint must have one tensor per Safetensors shard")
    for sidecar in manifest["sidecars"]:
        target = output / Path(sidecar).name
        if not target.exists():
            shutil.copyfile(base / target.name, target)
    for record in manifest["tensors"]:
        name = record["name"]
        filename = base_index["weight_map"][name]
        target = output / filename
        marker = output / (filename + ".done.json")
        if marker.exists() or target.exists():
            if (not marker.exists() or not target.exists() or
                    json.loads(marker.read_text()).get("sha256") != sha256(target)):
                raise ValueError(f"Incomplete or corrupt hybrid tensor: {name}")
            continue
        if name in names:
            original = read_tensor(source, source_map, name)
            if original.dtype != torch.bfloat16:
                raise ValueError(f"Original restoration dtype changed: {name}")
            atomic_safetensors(target, {name: original})
            if not torch.equal(load_file(str(target))[name], original):
                raise ValueError(f"Original tensor restoration failed: {name}")
        else:
            if sha256(base / filename) != json.loads((base / f".done-{record['tensor_id']:04d}.json").read_text())["sha256"]:
                raise ValueError(f"Corrupt B3 base tensor: {name}")
            shutil.copyfile(base / filename, target)
        atomic_json(marker, {"name": name, "sha256": sha256(target), "restored": name in names})
    index_path = output / "model.safetensors.index.json"
    atomic_json(index_path, base_index)
    restored = sum(r["numel"] for r in manifest["tensors"] if r["name"] in names)
    total = sum(r["numel"] for r in manifest["tensors"])
    compact_base = json.loads((artifact / "storage_report.json").read_text())["prefixes"][0]["total_physical_bytes"]
    replaced_stream_bytes = sum((artifact / layer[file]).stat().st_size
                                for r in manifest["tensors"] if r["name"] in names
                                for layer in r["layers"][:1] for file in ("code", "metadata"))
    compact_manifest = {"format": "rrq_w2_hybrid_reference_v1", "identity": identity,
                        "b3_manifest": str(manifest_path),
                        "restored_safetensors": {name: base_index["weight_map"][name] for name in sorted(names)},
                        "remaining_B3_names": sorted(r["name"] for r in manifest["tensors"]
                                                     if r["eligible"] and r["name"] not in names)}
    compact_manifest_path = output / "hybrid_compact_manifest.json"
    if compact_manifest_path.exists():
        if json.loads(compact_manifest_path.read_text()) != compact_manifest:
            raise ValueError("Existing compact hybrid manifest differs")
    else:
        atomic_json(compact_manifest_path, compact_manifest)
    restored_safetensors_bytes = sum((output / base_index["weight_map"][name]).stat().st_size for name in names)
    compact = (compact_base - replaced_stream_bytes + restored_safetensors_bytes +
               compact_manifest_path.stat().st_size)
    value = {"status": "complete", "identity": identity, "restored_parameters": restored,
             "restored_fraction": restored / total, "remaining_2bit_parameters": manifest["eligible_parameters"] - restored,
             "remaining_2bit_tensor_names": sorted(r["name"] for r in manifest["tensors"]
                                                   if r["eligible"] and r["name"] not in names),
             "excluded_tensor_names": sorted(r["name"] for r in manifest["tensors"] if not r["eligible"]),
             "hybrid_compact_storage_bytes": compact, "effective_bits_per_total_parameter": 8 * compact / total,
             "removed_B3_stream_bytes": replaced_stream_bytes,
             "restored_BF16_bytes": restored * 2,
             "restored_safetensors_bytes": restored_safetensors_bytes,
             "compact_manifest_bytes": compact_manifest_path.stat().st_size,
             "qdq_checkpoint_safetensors_bytes": sum((output / filename).stat().st_size
                                                   for filename in set(base_index["weight_map"].values())),
             "index_sha256": sha256(index_path), "restored_checkpoint_dtype": "bfloat16",
             "base_checkpoint_dtype": "float16", "inference_dtype": "float16"}
    complete = output / "HYBRID_COMPLETE.json"
    if complete.exists() and json.loads(complete.read_text()) != value:
        raise ValueError("Existing hybrid completion differs")
    if not complete.exists():
        atomic_json(complete, value)
    return value
