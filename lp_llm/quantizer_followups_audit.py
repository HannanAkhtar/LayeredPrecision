"""Read-only, chunked audit of an existing S01 fake-QDQ export."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

from .rrq_autoround import discover_checkpoint
from .rrq_pr2308_compare import _eligible, _weight_map
from .storage import atomic_json, sha256


BOUNDS = (0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def _tensor_chunks(directory: Path, mapping: dict[str, str], name: str, rows: int = 256):
    with safe_open(directory / mapping[name], framework="pt", device="cpu") as handle:
        view = handle.get_slice(name)
        shape = tuple(view.get_shape())
        if not shape:
            yield (), handle.get_tensor(name)
        else:
            for start in range(0, shape[0], rows):
                yield (start, min(start + rows, shape[0])), view[start:min(start + rows, shape[0])]


def compare_tensor(source: Path, source_map: dict[str, str], output: Path,
                   output_map: dict[str, str], name: str) -> dict:
    left = _tensor_chunks(source, source_map, name)
    right = _tensor_chunks(output, output_map, name)
    count = changed = nonfinite = fp16_roundtrip_mismatches = 0
    source_nonzero_to_export_zero = 0
    sum_sq = signed_sum = max_abs = 0.0
    buckets = [0] * (len(BOUNDS) + 1)
    samples = []
    source_hash, output_hash = hashlib.sha256(), hashlib.sha256()
    source_dtype = output_dtype = None
    for (left_span, original), (right_span, actual) in zip(left, right, strict=True):
        if left_span != right_span or original.shape != actual.shape:
            raise ValueError(f"Tensor slice mismatch: {name}")
        source_dtype, output_dtype = str(original.dtype), str(actual.dtype)
        expected = original.to(actual.dtype)
        fp16_roundtrip = original.to(torch.float16).to(actual.dtype)
        fp16_roundtrip_mismatches += int((fp16_roundtrip != actual).sum())
        source_nonzero_to_export_zero += int(((expected != 0) & (actual == 0)).sum())
        source_hash.update(original.contiguous().view(torch.uint8).numpy().tobytes())
        output_hash.update(actual.contiguous().view(torch.uint8).numpy().tobytes())
        delta = actual.float() - expected.float()
        absolute = delta.abs()
        finite = torch.isfinite(actual)
        nonfinite += int((~finite).sum())
        if not bool(finite.all()):
            raise ValueError(f"Nonfinite tensor in export: {name}")
        count += actual.numel()
        unequal = actual != expected
        changed += int(unequal.sum())
        sum_sq += float(torch.sum(delta.double().square()))
        signed_sum += float(torch.sum(delta.double()))
        max_abs = max(max_abs, float(absolute.max()))
        # Index 0 is exact equality; later bins use absolute-error boundaries.
        buckets[0] += int((absolute == 0).sum())
        for index, (low, high) in enumerate(zip(BOUNDS, BOUNDS[1:]), start=1):
            buckets[index] += int(((absolute > low) & (absolute <= high)).sum())
        buckets[-1] += int((absolute > BOUNDS[-1]).sum())
        if len(samples) < 8 and bool(unequal.any()):
            positions = unequal.nonzero(as_tuple=False)[:8 - len(samples)]
            for position in positions:
                local = tuple(int(v) for v in position)
                global_index = ((left_span[0] + local[0],) + local[1:]
                                if left_span else local)
                samples.append({"index": global_index,
                                "source": float(original[local]),
                                "source_cast": float(expected[local]),
                                "export": float(actual[local])})
    if not count:
        raise ValueError(f"Empty tensor: {name}")
    return {"name": name, "source_dtype": source_dtype, "export_dtype": output_dtype,
            "elements": count, "changed": changed, "changed_fraction": changed / count,
            "fp16_roundtrip_mismatches": fp16_roundtrip_mismatches,
            "source_nonzero_to_export_zero": source_nonzero_to_export_zero,
            "max_abs_error_after_cast": max_abs, "rmse_after_cast": math.sqrt(sum_sq / count),
            "mean_signed_error_after_cast": signed_sum / count,
            "nonfinite": nonfinite, "absolute_error_histogram": buckets,
            "histogram_boundaries": BOUNDS, "first_mismatches": samples,
            "source_tensor_sha256": source_hash.hexdigest(),
            "export_tensor_sha256": output_hash.hexdigest()}


def compare_pair(directory: Path, mapping: dict[str, str], left: str, right: str) -> dict | None:
    if left not in mapping or right not in mapping:
        return None
    with safe_open(directory / mapping[left], framework="pt", device="cpu") as handle:
        left_shape = tuple(handle.get_slice(left).get_shape())
    with safe_open(directory / mapping[right], framework="pt", device="cpu") as handle:
        right_shape = tuple(handle.get_slice(right).get_shape())
    if left_shape != right_shape:
        return {"same_shape": False, "left_shape": left_shape, "right_shape": right_shape}
    changed = count = 0
    for (_, a), (_, b) in zip(_tensor_chunks(directory, mapping, left),
                             _tensor_chunks(directory, mapping, right), strict=True):
        count += a.numel()
        changed += int((a != b).sum())
    return {"same_shape": True, "different_elements": changed, "elements": count}


def audit(source: Path, export_root: Path) -> dict:
    checkpoint = discover_checkpoint(export_root)
    source_map, output_map = _weight_map(source), _weight_map(checkpoint)
    if set(source_map) != set(output_map):
        raise ValueError("Export tensor inventory differs from source")
    excluded = [name for name in sorted(source_map) if not _eligible(name)]
    comparisons = [compare_tensor(source, source_map, checkpoint, output_map, name)
                   for name in excluded]
    head = next((entry for entry in comparisons if entry["name"] == "lm_head.weight"), None)
    if head is None:
        raise ValueError("Expected lm_head.weight in the indexed source and export")
    return {"status": "read_only_audit", "source": str(source.resolve()),
            "export": str(checkpoint.resolve()),
            "source_config_sha256": sha256(source / "config.json"),
            "export_config_sha256": sha256(checkpoint / "config.json"),
            "quantized_projection_names": [name for name in sorted(source_map) if _eligible(name)],
            "excluded_tensor_count": len(excluded),
            "changed_excluded_names": [entry["name"] for entry in comparisons if entry["changed"]],
            "lm_head": head,
            "head_vs_embeddings": {
                "source": compare_pair(source, source_map, "lm_head.weight", "model.embed_tokens.weight"),
                "export": compare_pair(checkpoint, output_map, "lm_head.weight", "model.embed_tokens.weight")},
            "excluded_tensors": comparisons,
            "interpretation": "Differences are measured after casting each source tensor to its exported dtype; "
                              "the report alone does not identify whether a difference arose during load, "
                              "quantization, or export."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = audit(args.source, args.export_root)
    atomic_json(args.output, value)
    print(json.dumps({"output": str(args.output), "changed_excluded_names": value["changed_excluded_names"],
                      "changed_excluded_summary": [{key: entry[key] for key in (
                          "name", "changed", "max_abs_error_after_cast", "rmse_after_cast",
                          "fp16_roundtrip_mismatches", "source_nonzero_to_export_zero")}
                          for entry in value["excluded_tensors"] if entry["changed"]],
                      "lm_head": {key: value["lm_head"][key] for key in (
                          "source_dtype", "export_dtype", "elements", "changed",
                          "changed_fraction", "max_abs_error_after_cast", "rmse_after_cast")},
                      "head_vs_embeddings": value["head_vs_embeddings"]}, indent=2))


if __name__ == "__main__":
    main()
