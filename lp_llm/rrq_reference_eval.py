"""Run the pinned, unmodified MatGPTQ WikiText-2 evaluator."""

from __future__ import annotations

import hashlib
import math
import re
import subprocess
import sys
from pathlib import Path

import torch

from .rrq_w2 import MATGPTQ_COMMIT
from .rrq_process import run_logged
from .rrq_runtime import probe


def check_revision(checkout: Path, expected: str = MATGPTQ_COMMIT) -> str:
    actual = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if actual != expected:
        raise ValueError(f"Upstream checkout {checkout} is {actual}; expected {expected}")
    if subprocess.run(["git", "-C", str(checkout), "diff", "--quiet", "HEAD"], check=False).returncode:
        raise ValueError(f"Upstream checkout has modified tracked files: {checkout}")
    return actual


def command(checkout: Path, checkpoint: Path, python: str | None = None) -> list[str]:
    python = python or sys.executable
    return [python, str(checkout / "eval_ppl.py"), "--model_name_or_path", str(checkpoint),
            "--sequence_length", "2048", "--eval_datasets", "wikitext2",
            "--eval_batch_size", "1", "--dtype", "float16"]


def blocks(tokens: torch.Tensor, length: int = 2048):
    if tokens.ndim != 2 or tokens.shape[0] != 1 or length < 2:
        raise ValueError("Expected one token sequence and length >= 2")
    for start in range(0, tokens.shape[1] // length * length, length):
        yield tokens[:, start:start + length]


def aggregate(losses: list[float], length: int = 2048) -> dict:
    if not losses or any(not math.isfinite(v) for v in losses):
        raise ValueError("No finite block losses")
    scored = len(losses) * (length - 1)
    cross_entropy = sum(v * (length - 1) for v in losses) / scored
    return {"sequence_count": len(losses), "scored_tokens": scored,
            "cross_entropy": cross_entropy, "perplexity": math.exp(cross_entropy)}


def local_score(model, token_blocks, device: str = "cuda:0") -> dict:
    """Independent matched FP16 score used only for J01 protocol parity."""
    nll_running = None
    processed = 0
    sequences = 0
    length = None
    for block in token_blocks:
        if length is None:
            length = block.shape[1]
        elif block.shape[1] != length:
            raise ValueError("Local evaluation requires equal-length full blocks")
        block = block.to(device)
        with torch.inference_mode():
            logits = model(block).logits
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
                block[:, 1:].contiguous().view(-1))
        valid = block.shape[1] - 1
        nll_running = (loss if nll_running is None else
                       (valid / (processed + valid)) * loss +
                       (processed / (processed + valid)) * nll_running)
        processed += valid
        sequences += 1
    if nll_running is None or not torch.isfinite(nll_running):
        raise ValueError("No finite local MatGPTQ block losses")
    return {"sequence_count": sequences, "scored_tokens": processed,
            "cross_entropy": float(nll_running.item()),
            "perplexity": float(nll_running.exp().item())}


def local_evaluate(checkpoint: Path) -> dict:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
    tokens = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt",
                       add_special_tokens=False)["input_ids"]
    token_sha = hashlib.sha256(tokens.numpy().tobytes()).hexdigest()
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float16,
                                                  low_cpu_mem_usage=True, device_map="cuda:0")
    model.eval()
    return {"implementation": "independent_matched_fp16", "token_sha256": token_sha,
            **local_score(model, blocks(tokens))}


def corpus_identity(checkpoint: Path) -> dict:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception as error:
        raise RuntimeError("Official WikiText-2 raw test dataset is unavailable") from error
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
    tokens = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt",
                       add_special_tokens=False)["input_ids"]
    if tokens.ndim != 2 or tokens.shape[0] != 1:
        raise ValueError("Unexpected MatGPTQ token sequence")
    return {"dataset": "wikitext/wikitext-2-raw-v1", "dataset_split": "test",
            "dataset_revision": "unpinned_official_default", "dataset_version": str(dataset.info.version),
            "dataset_fingerprint": dataset._fingerprint, "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_fast": bool(tokenizer.is_fast), "add_special_tokens": False,
            "join": "\\n\\n", "token_sha256": hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
            "corpus_tokens": int(tokens.numel()), "sequence_count": tokens.shape[1] // 2048,
            "scored_tokens": (tokens.shape[1] // 2048) * 2047,
            "dropped_tail_tokens": tokens.shape[1] % 2048}


def parse_perplexity(stdout: str) -> float:
    matches = re.findall(r"^wikitext2\s*:\s*([0-9.eE+-]+)\s*$",
                         stdout, re.IGNORECASE | re.MULTILINE)
    if not matches:
        matches = re.findall(r"(?:wikitext2|wikitext-2)[^\n]*?(?:ppl|perplexity)\s*[:=]\s*([0-9.eE+-]+)",
                         stdout, re.IGNORECASE)
    if not matches:
        matches = re.findall(r"(?:ppl|perplexity)\s*[:=]\s*([0-9.eE+-]+)", stdout, re.IGNORECASE)
    if len(matches) != 1:
        raise ValueError(f"Could not identify unique MatGPTQ perplexity in output: {stdout[-1000:]}")
    value = float(matches[0])
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Invalid MatGPTQ perplexity")
    return value


def run(checkout: Path, checkpoint: Path, log: Path, *, python: str | None = None) -> dict:
    check_revision(checkout)
    if not (checkout / "eval_ppl.py").is_file():
        raise FileNotFoundError(checkout / "eval_ppl.py")
    cmd = command(checkout, checkpoint, python)
    runtime = probe(cmd[0], ["torch", "transformers", "datasets", "accelerate", "safetensors", "numpy"])
    execution = run_logged(cmd, log, cwd=checkout)
    if execution["returncode"]:
        raise RuntimeError(f"MatGPTQ evaluator failed ({execution['returncode']}); see {log}")
    ppl = parse_perplexity(log.read_text(encoding="utf-8"))
    return {"implementation": "official_matgptq", "revision": MATGPTQ_COMMIT,
            "runtime_dependency_mode": "official_source_with_existing_LP_environment",
            "source_checkout": str(checkout.resolve()), "runtime_imports": runtime,
            "command": cmd, "log": str(log), "execution": execution, "perplexity": ppl,
            "cross_entropy": math.log(ppl), "reported_precision": "perplexity rounded to 2 decimal places by upstream script",
            "cross_entropy_precision": "derived from rounded upstream perplexity"}
