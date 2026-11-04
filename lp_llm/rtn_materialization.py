"""Compare B3 packed and saved weights with pinned upstream FP16 QDQ."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file

from .evaluate import _identity
from .model_adapter import inventory, read_tensor
from .storage import atomic_json, decode_prefix, sha256
from .upstream_reference import direct_qdq


def compare_weights(source: Path, artifact: Path, bf16_checkpoint: Path,
                    fp16_checkpoint: Path, device: str = "cuda:0") -> dict:
    manifest = json.loads((artifact / "manifest.json").read_text())
    if manifest["quantizer_family"] != "autoround_opt_fp16" or manifest["bits"] != [2]:
        raise ValueError("Expected the existing B3 2-bit artifact")
    _, mapping = inventory(source)
    bf16_index = json.loads((bf16_checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    fp16_index = json.loads((fp16_checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    totals = {key: {"max_abs_error": 0.0, "squared_error_sum": 0.0, "exact": True}
              for key in ("packed_fp32_cast_fp16", "bf16_saved", "fp16_saved", "bf16_cast_prediction")}
    count = 0
    for record in manifest["tensors"]:
        if not record["eligible"]:
            continue
        name = record["name"]
        original = read_tensor(source, mapping, name).to(torch.float16).to(device)
        upstream, _ = direct_qdq(original, "optimized")
        upstream = upstream.cpu()
        packed = decode_prefix(artifact, record, 1).to(torch.float16)
        bf16 = load_file(str(bf16_checkpoint / bf16_index[name]))[name]
        fp16 = load_file(str(fp16_checkpoint / fp16_index[name]))[name]
        if bf16.dtype != torch.bfloat16 or fp16.dtype != torch.float16:
            raise ValueError(f"Wrong saved dtype for {name}")
        candidates = {"packed_fp32_cast_fp16": packed,
                      "bf16_saved": bf16,
                      "fp16_saved": fp16,
                      "bf16_cast_prediction": upstream.to(torch.bfloat16)}
        for key, candidate in candidates.items():
            difference = (candidate.float() - upstream.float()).abs()
            row = totals[key]
            row["max_abs_error"] = max(row["max_abs_error"], float(difference.max()))
            row["squared_error_sum"] += float(difference.square().sum())
            row["exact"] = row["exact"] and torch.equal(candidate.float(), upstream.float())
        if not torch.equal(bf16, upstream.to(torch.bfloat16)):
            raise ValueError(f"BF16 checkpoint differs from the upstream BF16 cast: {name}")
        count += upstream.numel()
    if count == 0:
        raise ValueError("No eligible B3 weights")
    for row in totals.values():
        row["rmse"] = math.sqrt(row.pop("squared_error_sum") / count)
    if not totals["fp16_saved"]["exact"]:
        raise ValueError("FP16 checkpoint does not exactly match upstream FP16 QDQ")
    return {"status": "complete", "quantizer_family": "autoround_opt_fp16",
            "artifact_manifest_sha256": sha256(artifact / "manifest.json"),
            "source_index_sha256": sha256(source / "model.safetensors.index.json"),
            "bf16_checkpoint_identity": _identity(bf16_checkpoint, "b3-materialization-bf16"),
            "fp16_checkpoint_identity": _identity(fp16_checkpoint, "b3-materialization-fp16"),
            "eligible_parameters": count, "comparisons": totals,
            "checkpoint_dtypes": {"bf16": "bfloat16", "fp16": "float16"},
            "evaluation_model_dtype": "float16"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--bf16-checkpoint", type=Path, required=True)
    parser.add_argument("--fp16-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    value = compare_weights(args.source, args.artifact, args.bf16_checkpoint,
                            args.fp16_checkpoint, args.device)
    atomic_json(args.output, value)
    print(json.dumps(value["comparisons"], indent=2))


if __name__ == "__main__":
    main()
