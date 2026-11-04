"""Fixed-token logit parity for pinned upstream, packed LP, and reloaded models."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .evaluate import _identity
from .model_adapter import SelectionPolicy
from .rtn_reproduction import MODES
from .storage import atomic_json, decode_prefix, sha256
from .upstream_reference import direct_qdq


def logit_stats(reference: torch.Tensor, candidate: torch.Tensor, tokens: torch.Tensor) -> dict:
    delta = (candidate.float() - reference.float()).abs()
    predicted_reference = reference.argmax(-1)
    predicted_candidate = candidate.argmax(-1)
    return {"max_abs_logit_error": float(delta.max()),
            "mean_abs_logit_error": float(delta.mean()),
            "logit_rmse": float(torch.sqrt(delta.square().mean())),
            "predicted_token_agreement": float((predicted_reference == predicted_candidate).float().mean()),
            "mean_token_cross_entropy": float(F.cross_entropy(
                candidate[:, :-1].float().reshape(-1, candidate.shape[-1]),
                tokens[:, 1:].reshape(-1)))}


def compare(source: Path, artifact: Path, checkpoint: Path, tokens: torch.Tensor,
            device: str = "cuda:0") -> dict:
    from transformers import AutoModelForCausalLM

    manifest = json.loads((artifact / "manifest.json").read_text())
    family = manifest["quantizer_family"]
    algorithm, qdq_dtype = MODES[family]
    reference_fn = "plain" if algorithm == "plain" else "optimized"
    policy = SelectionPolicy()
    records = {record["name"]: record for record in manifest["tensors"]}
    results = {}
    for variant in ("upstream_in_memory", "lp_in_memory", "saved_reloaded"):
        path = checkpoint if variant == "saved_reloaded" else source
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                       low_cpu_mem_usage=True, device_map=device)
        if variant != "saved_reloaded":
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if not policy.eligible(name):
                        continue
                    if variant == "upstream_in_memory":
                        qdq, _ = direct_qdq(param.data.to(qdq_dtype), reference_fn)
                    else:
                        qdq = decode_prefix(artifact, records[name], 1).to(qdq_dtype)
                    param.data = qdq.to(param.device)
        model.to(dtype=torch.float16)
        model.eval()
        with torch.inference_mode():
            logits = model(tokens.to(device)).logits.float().cpu()
        results[variant] = logits
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    reference = results["upstream_in_memory"]
    output = {"input_tokens": tokens.tolist(), "evaluation_dtype": "float16",
              "upstream_in_memory": logit_stats(reference, reference, tokens),
              "lp_in_memory": logit_stats(reference, results["lp_in_memory"], tokens),
              "saved_reloaded": logit_stats(reference, results["saved_reloaded"], tokens),
              "saved_vs_lp_in_memory": logit_stats(results["lp_in_memory"], results["saved_reloaded"], tokens),
              "source_index_sha256": sha256(source / "model.safetensors.index.json"),
              "artifact_manifest_sha256": sha256(artifact / "manifest.json"),
              "checkpoint_identity": _identity(checkpoint, "rtn-logit-parity")}
    if any(not math.isfinite(v["mean_token_cross_entropy"]) for v in
           (output["upstream_in_memory"], output["lp_in_memory"], output["saved_reloaded"])):
        raise ValueError("Nonfinite logit diagnostic")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    tokens = torch.tensor([[int(x) for x in args.tokens.split(",")]], dtype=torch.long)
    if tokens.shape[1] < 2:
        raise ValueError("Need at least two input tokens")
    value = compare(args.source, args.artifact, args.checkpoint, tokens, args.device)
    atomic_json(args.output, value)
    print(json.dumps({key: value[key] for key in
                      ("lp_in_memory", "saved_reloaded", "saved_vs_lp_in_memory")}, indent=2))


if __name__ == "__main__":
    main()
