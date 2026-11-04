"""Deterministic WikiText-2 non-overlapping block screen for Phase 1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from .evaluate import _existing, _identity, _package_version
from .storage import atomic_json


DATASET = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"


def blocks(tokens: torch.Tensor, length: int = 2048, limit: int = 8):
    if length < 2 or limit < 1 or tokens.ndim != 2 or tokens.shape[0] != 1:
        raise ValueError("Expected one token sequence, length >= 2, and positive block limit")
    for start in range(0, min(tokens.shape[1] // length, limit) * length, length):
        yield tokens[:, start:start + length]


def score_blocks(model, token_blocks, device: str) -> dict:
    rows, nll_sum, scored = [], 0.0, 0
    for index, block in enumerate(token_blocks):
        valid = block.shape[1] - 1
        if valid < 1:
            continue
        with torch.inference_mode():
            loss = model(block.to(device), labels=block.to(device)).loss
        cross_entropy = float(loss.item())
        if not math.isfinite(cross_entropy):
            raise ValueError(f"Nonfinite cross entropy in block {index}")
        nll_sum += cross_entropy * valid
        scored += valid
        rows.append({"block": index, "input_tokens": block.shape[1],
                     "scored_tokens": valid, "mean_cross_entropy": cross_entropy,
                     "negative_log_likelihood": cross_entropy * valid})
    if not scored:
        raise ValueError("No complete diagnostic block was available")
    mean = nll_sum / scored
    return {"blocks": rows, "scored_tokens": scored,
            "negative_log_likelihood": nll_sum,
            "mean_cross_entropy": mean, "perplexity": math.exp(mean)}


def evaluate(checkpoint: Path, output: Path, *, block_length: int = 2048,
             block_limit: int = 8, device: str = "cuda:0") -> dict:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint, output = Path(checkpoint), Path(output)
    identity = _identity(checkpoint, "rtn-diagnostic-wikitext2") | {
        "dataset": DATASET, "dataset_config": DATASET_CONFIG, "split": "test",
        "join": "\\n\\n", "add_special_tokens": False,
        "block_length": block_length, "block_limit": block_limit,
        "evaluation_dtype": "float16", "datasets_version": _package_version("datasets")}
    old = _existing(output, identity)
    if old:
        return old
    text = "\n\n".join(load_dataset(DATASET, DATASET_CONFIG, split="test")["text"])
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    selected = list(blocks(tokens, block_length, block_limit))
    token_hash = hashlib.sha256(torch.cat(selected, dim=1).numpy().tobytes()).hexdigest()
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float16,
                                                  low_cpu_mem_usage=True, device_map=device)
    model.eval()
    score = score_blocks(model, selected, device)
    value = {"status": "complete", "identity": identity,
             "corpus_tokens": tokens.shape[1], "selected_input_tokens": sum(x.numel() for x in selected),
             "selected_token_sha256": token_hash, "block_construction": "nonoverlap_full_blocks_from_start",
             **score}
    atomic_json(output, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-length", type=int, default=2048)
    parser.add_argument("--block-limit", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.checkpoint, args.output,
                              block_length=args.block_length, block_limit=args.block_limit,
                              device=args.device), indent=2))


if __name__ == "__main__":
    main()
