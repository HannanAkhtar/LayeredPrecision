"""Compare existing B3 BF16 checkpoint with a separate FP16 materialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .evaluate import _identity
from .reconstruct import reconstruct
from .rtn_diagnostic_eval import evaluate as diagnostic_evaluate
from .rtn_logits import compare as compare_logits
from .rtn_materialization import compare_weights
from .storage import atomic_json, sha256


def run(source: Path, result_root: Path, device: str = "cuda:0") -> dict:
    root = result_root / "rtn_reproduction" / "b3_opt_fp16"
    artifact, bf16_checkpoint = root / "artifact", root / "checkpoint"
    fp16_root = root / "materialization_fp16"
    fp16_checkpoint = fp16_root / "checkpoint"
    if not (artifact / "COMPLETE.json").is_file() or not (bf16_checkpoint / "COMPLETE.json").is_file():
        raise ValueError("Existing B3 artifact and BF16 checkpoint must be complete")
    manifest = json.loads((artifact / "manifest.json").read_text())
    if manifest["quantizer_family"] != "autoround_opt_fp16" or manifest["bits"] != [2]:
        raise ValueError("Existing artifact is not the B3 2-bit base")
    _identity(bf16_checkpoint, "b3-existing-bf16")
    if not (fp16_checkpoint / "COMPLETE.json").exists():
        reconstruct(artifact, fp16_checkpoint, 1, checkpoint_dtype="float16")
    fp16_identity = _identity(fp16_checkpoint, "b3-new-fp16")
    if (fp16_identity["reconstruction"].get("checkpoint_dtype") != "float16" or
            fp16_identity["reconstruction"].get("manifest_sha256") != sha256(artifact / "manifest.json")):
        raise ValueError("FP16 checkpoint identity is missing its materialization dtype")
    weight_path = fp16_root / "weights.json"
    if weight_path.exists():
        weights = json.loads(weight_path.read_text())
        if (weights.get("artifact_manifest_sha256") != sha256(artifact / "manifest.json") or
                weights.get("bf16_checkpoint_identity") != _identity(bf16_checkpoint, "b3-materialization-bf16") or
                weights.get("fp16_checkpoint_identity") != _identity(fp16_checkpoint, "b3-materialization-fp16")):
            raise ValueError("Existing B3 weight comparison has a different identity")
    else:
        weights = compare_weights(source, artifact, bf16_checkpoint, fp16_checkpoint, device)
        atomic_json(weight_path, weights)
    if (weights.get("status") != "complete" or
            not weights.get("comparisons", {}).get("fp16_saved", {}).get("exact")):
        raise ValueError("B3 FP16 checkpoint failed exact upstream weight validation")
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    logit_path = fp16_root / "logits.json"
    if logit_path.exists():
        logit_values = json.loads(logit_path.read_text())
        if (logit_values.get("artifact_manifest_sha256") != sha256(artifact / "manifest.json") or
                logit_values.get("checkpoint_identity") != _identity(bf16_checkpoint, "rtn-logit-parity") or
                logit_values.get("fp16_checkpoint_identity") != _identity(fp16_checkpoint, "rtn-logit-parity-fp16")):
            raise ValueError("Existing B3 logit comparison has a different identity")
    else:
        logit_values = compare_logits(source, artifact, bf16_checkpoint, tokens, device,
                                      fp16_checkpoint=fp16_checkpoint)
        atomic_json(logit_path, logit_values)
    bf16_diag = diagnostic_evaluate(bf16_checkpoint, root / "diagnostic.json", device=device)
    fp16_diag = diagnostic_evaluate(fp16_checkpoint, fp16_root / "diagnostic.json", device=device)
    if (bf16_diag["selected_token_sha256"] != fp16_diag["selected_token_sha256"] or
            bf16_diag["scored_tokens"] != fp16_diag["scored_tokens"] or
            bf16_diag["identity"]["block_length"] != fp16_diag["identity"]["block_length"] or
            bf16_diag["identity"]["block_limit"] != fp16_diag["identity"]["block_limit"]):
        raise ValueError("Diagnostic token blocks differ between materializations")
    storage = json.loads((artifact / "storage_report.json").read_text())["prefixes"][0]
    checkpoint_files = [r["name"] for r in manifest["tensors"]]
    bf16_index = json.loads((bf16_checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    fp16_index = json.loads((fp16_checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    comparison = {"status": "complete", "algorithm": "optimized_search_scales",
                  "artifact_manifest_sha256": sha256(artifact / "manifest.json"),
                  "weight_comparisons": weights["comparisons"],
                  "logit_parity": {key: logit_values[key] for key in (
                      "upstream_in_memory", "lp_in_memory", "saved_reloaded", "saved_fp16_reloaded")},
                  "diagnostic": {
                      "bf16_checkpoint": {"perplexity": bf16_diag["perplexity"],
                                          "mean_cross_entropy": bf16_diag["mean_cross_entropy"],
                                          "result": str(root / "diagnostic.json")},
                      "fp16_checkpoint": {"perplexity": fp16_diag["perplexity"],
                                          "mean_cross_entropy": fp16_diag["mean_cross_entropy"],
                                          "result": str(fp16_root / "diagnostic.json")},
                      "selected_token_sha256": fp16_diag["selected_token_sha256"],
                      "scored_tokens": fp16_diag["scored_tokens"]},
                  "storage": {"artifact": storage,
                              "checkpoint_parameter_bytes": sum(r["numel"] for r in manifest["tensors"]) * 2,
                              "bf16_checkpoint_safetensors_bytes": sum(
                                  (bf16_checkpoint / bf16_index[name]).stat().st_size for name in checkpoint_files),
                              "fp16_checkpoint_safetensors_bytes": sum(
                                  (fp16_checkpoint / fp16_index[name]).stat().st_size for name in checkpoint_files),
                              "bf16_element_bytes": 2, "fp16_element_bytes": 2},
                  "model_loading": {"bf16_checkpoint_load_dtype": "bfloat16",
                                    "fp16_checkpoint_load_dtype": "float16",
                                    "logit_and_diagnostic_inference_dtype": "float16"},
                  "bf16_checkpoint_identity": _identity(bf16_checkpoint, "b3-comparison-bf16"),
                  "fp16_checkpoint_identity": fp16_identity}
    comparison_path = fp16_root / "comparison.json"
    if comparison_path.exists():
        if json.loads(comparison_path.read_text()) != comparison:
            raise ValueError("Existing B3 materialization comparison differs")
    else:
        atomic_json(comparison_path, comparison)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    value = run(args.source, args.result_root, args.device)
    print(json.dumps(value["diagnostic"], indent=2))


if __name__ == "__main__":
    main()
