"""WikiText-2 sliding-window perplexity and lm-evaluation-harness zero shot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .diagnostics import elapsed, peak_rss_bytes, started
from .storage import atomic_json, sha256


TASKS = ("arc_easy", "arc_challenge", "piqa", "winogrande", "hellaswag")


def _identity(checkpoint: Path, task: str) -> dict:
    complete = json.loads((checkpoint / "COMPLETE.json").read_text())
    return {"checkpoint": str(checkpoint.resolve()), "index_sha256": complete["index_sha256"],
            "task": task}


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
    total, scored, previous_end = 0.0, 0, 0
    with torch.inference_mode():
        for begin in range(0, tokens.size(1), stride):
            end = min(begin + context, tokens.size(1))
            newly_scored = end - previous_end
            inputs = tokens[:, begin:end].to(device)
            labels = inputs.clone()
            labels[:, :-newly_scored] = -100
            result = model(inputs, labels=labels)
            count = newly_scored - (1 if begin == 0 else 0)
            total += float(result.loss) * count
            scored += count
            previous_end = end
            if end == tokens.size(1):
                break
    return total, scored


def wikitext(checkpoint: Path, output: Path, context: int = 2048, stride: int = 512):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint, output = Path(checkpoint), Path(output)
    identity = _identity(checkpoint, "wikitext-2-raw-v1") | {"context": context, "stride": stride}
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
             "peak_rss_bytes": peak_rss_bytes(),
             "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated()}
    atomic_json(output, value)
    return value


def zeroshot(checkpoint: Path, output: Path):
    from lm_eval import simple_evaluate

    checkpoint, output = Path(checkpoint), Path(output)
    identity = _identity(checkpoint, "zeroshot-lm-eval-0.4.13")
    old = _existing(output, identity)
    if old:
        return old
    begun = started()
    raw = simple_evaluate(model="hf", model_args=f"pretrained={checkpoint},dtype=bfloat16",
                          tasks=list(TASKS), num_fewshot=0, batch_size=1)
    results = {}
    for task in TASKS:
        row = raw["results"][task]
        results[task] = {key: row.get(key + ",none") for key in ("acc", "acc_norm")
                         if key + ",none" in row}
    value = {"identity": identity, "status": "complete", "results": results,
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
