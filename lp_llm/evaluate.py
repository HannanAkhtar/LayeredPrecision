"""WikiText-2 sliding-window perplexity and lm-evaluation-harness zero shot."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

from .diagnostics import elapsed, peak_rss_bytes, started
from .storage import SIDECARS, atomic_json, sha256


TASKS = ("arc_easy", "arc_challenge", "piqa", "winogrande", "hellaswag")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _identity(checkpoint: Path, task: str) -> dict:
    """Content identity for either a completed LP decode or an ordinary HF checkpoint."""
    checkpoint = checkpoint.resolve()
    config_path = checkpoint / "config.json"
    index_path = checkpoint / "model.safetensors.index.json"
    config = json.loads(config_path.read_text())
    index = json.loads(index_path.read_text())
    if not isinstance(config, dict) or not isinstance(config.get("model_type"), str):
        raise ValueError("Invalid Hugging Face model configuration")
    mapping = index.get("weight_map")
    if not isinstance(mapping, dict) or not mapping or not all(
            isinstance(name, str) and name and isinstance(shard, str) and shard
            for name, shard in mapping.items()):
        raise ValueError("Invalid Safetensors checkpoint index")
    shards = {}
    for shard in sorted(set(mapping.values())):
        path = (checkpoint / shard).resolve()
        if not path.is_relative_to(checkpoint) or not path.is_file():
            raise ValueError(f"Missing or invalid checkpoint shard: {shard}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            assigned = {name for name, filename in mapping.items() if filename == shard}
            if set(handle.keys()) != assigned:
                raise ValueError(f"Safetensors index differs from shard {shard}")
        shards[shard] = sha256(path)
    sidecars = {name: sha256(checkpoint / name) for name in sorted(SIDECARS)
                if name != "config.json" and (checkpoint / name).is_file()}
    identity = {"checkpoint": str(checkpoint), "kind": "huggingface_safetensors",
                "config_sha256": sha256(config_path), "index_sha256": sha256(index_path),
                "shard_sha256": shards, "sidecar_sha256": sidecars, "task": task,
                "torch": torch.__version__, "transformers": _package_version("transformers")}
    complete_path = checkpoint / "COMPLETE.json"
    if complete_path.exists():
        complete = json.loads(complete_path.read_text())
        if complete.get("index_sha256") != identity["index_sha256"]:
            raise ValueError("LP reconstruction index integrity check failed")
        reconstruction = json.loads((checkpoint / "reconstruction.json").read_text())
        if complete.get("identity") != reconstruction.get("identity"):
            raise ValueError("LP reconstruction completion identity mismatch")
        markers = list(checkpoint.glob(".done-*.json"))
        if len(markers) != len(mapping):
            raise ValueError("LP reconstruction completion markers are incomplete")
        for marker in markers:
            done = json.loads(marker.read_text())
            filename = mapping.get(done.get("name"))
            if filename is None or done.get("sha256") != shards[filename]:
                raise ValueError(f"LP reconstruction shard integrity check failed: {marker}")
        identity["kind"] = "lp_reconstructed"
        identity["reconstruction"] = complete["identity"]
    return identity


def _existing(output: Path, identity: dict):
    if output.exists():
        value = json.loads(output.read_text())
        if value.get("identity") == identity and value.get("status") == "complete":
            return value
        raise ValueError(f"Existing result has a different identity: {output}")
    return None


def sliding_nll(model, tokens: torch.Tensor, context: int, stride: int, device: str) -> tuple[float, int]:
    if context < 2 or stride < 1 or stride > context:
        raise ValueError("Require context >= 2 and 1 <= stride <= context")
    nll_sum, scored_tokens = 0.0, 0
    total_tokens = tokens.size(1)
    for i in range(0, total_tokens, stride):
        end = min(i + stride, total_tokens)
        begin = max(end - context, 0)
        input_ids = tokens[:, begin:end].to(device)
        labels = input_ids.clone()
        target_length = end - i
        labels[:, :-target_length] = -100
        valid = int((labels[:, 1:] != -100).sum().item())
        if valid > 0:
            with torch.inference_mode():
                loss = model(input_ids, labels=labels).loss
            nll_sum += loss.item() * valid
            scored_tokens += valid
    return nll_sum, scored_tokens


def wikitext(checkpoint: Path, output: Path, context: int = 2048, stride: int = 512):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint, output = Path(checkpoint), Path(output)
    identity = _identity(checkpoint, "wikitext-2-raw-v1") | {
        "context": context, "stride": stride, "datasets": _package_version("datasets"),
        "join": "\\n\\n", "add_special_tokens": False}
    old = _existing(output, identity)
    if old:
        return old
    begun = started()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16,
                                                  low_cpu_mem_usage=True, device_map="cuda:0")
    model.eval()
    total, scored = sliding_nll(model, tokens, context, stride, "cuda")
    value = {"identity": identity, "status": "complete", "corpus_tokens": tokens.size(1),
             "scored_tokens": scored, "negative_log_likelihood": total,
             "perplexity": math.exp(total / scored), "seconds": elapsed(begun),
             "dataset": "Salesforce/wikitext", "dataset_config": "wikitext-2-raw-v1",
             "dataset_split": "test", "context": context, "stride": stride,
             "corpus_join": "\\n\\n", "add_special_tokens": False,
             "peak_rss_bytes": peak_rss_bytes(),
             "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated()}
    atomic_json(output, value)
    return value


def zeroshot(checkpoint: Path, output: Path):
    from lm_eval import simple_evaluate
    from lm_eval.utils import handle_non_serializable

    checkpoint, output = Path(checkpoint), Path(output)
    model_args = f"pretrained={checkpoint},dtype=bfloat16"
    identity = _identity(checkpoint, "zeroshot-lm-eval") | {
        "harness_version": _package_version("lm_eval"), "tasks": list(TASKS),
        "num_fewshot": 0, "batch_size": 1, "model_args": model_args}
    old = _existing(output, identity)
    if old:
        return old
    begun = started()
    raw = simple_evaluate(model="hf", model_args=model_args,
                          tasks=list(TASKS), num_fewshot=0, batch_size=1,
                          log_samples=False)
    if not isinstance(raw, dict) or "results" not in raw:
        raise ValueError("LM Evaluation Harness returned no task results")
    raw = json.loads(json.dumps(raw, default=handle_non_serializable),
                     parse_constant=lambda value: value)
    # Keep the harness's complete result, including stderr, versions, and configs.
    value = {"identity": identity, "status": "complete", "results": raw["results"],
             "harness": raw, "task_versions": raw.get("versions"),
             "settings": {"model": "hf", "model_args": model_args,
                          "tasks": list(TASKS), "num_fewshot": 0, "batch_size": 1,
                          "log_samples": False, "apply_chat_template": False},
             "seconds": elapsed(begun), "peak_rss_bytes": peak_rss_bytes()}
    atomic_json(output, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", choices=("wikitext", "zeroshot"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    args = parser.parse_args()
    result = (wikitext(args.checkpoint, args.output, args.context, args.stride)
              if args.protocol == "wikitext" else zeroshot(args.checkpoint, args.output))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
