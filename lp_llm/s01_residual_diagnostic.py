"""Read-only S01 baseline and exact-residual diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors import safe_open

from .progressive_lp import ROWS, _read_chunk, _slice, identity, validate_repaired_base, verified_layers
from .rrq_pr2308_compare import _eligible, _weight_map, same_hf_contents
from .rrq_reference_eval import check_revision, corpus_identity, run as evaluate_matgptq
from .storage import atomic_json, sha256


TOKEN_SHA = "8ffcee84dd4b684ec631cb52845e18c9c181ce6fdb8b26d65a4ffdc3d35fbda3"
SCORED = 298862


def _records(result_root: Path) -> dict:
    paths = {
        "w2": result_root / "quantizer_followups/s01_fp16_load_repair/full.json",
        "ternary_e2": result_root / "progressive_llm/evaluation/s01_lp_2p2/full.json",
        "signed_e2": result_root /
            "progressive_llm/ablations/s01_autoround_signed_rtn/evaluation/full.json",
    }
    result = {}
    for key, path in paths.items():
        record = json.loads(path.read_text())
        corpus = record.get("corpus", {})
        if (record.get("status") != "completed_validated" or
                corpus.get("token_sha256") != TOKEN_SHA or
                corpus.get("scored_tokens") != SCORED or
                not math.isfinite(record.get("evaluation", {}).get("perplexity", float("nan")))):
            raise ValueError(f"Stored {key} evaluation protocol differs")
        result[key] = {"path": str(path), "sha256": sha256(path),
                       "code_sha": record.get("code_sha"),
                       "identity": record.get("identity"),
                       "corpus": corpus, "perplexity": record["evaluation"]["perplexity"]}
    if len({row["corpus"]["token_sha256"] for row in result.values()}) != 1:
        raise ValueError("S01 test corpus differs across records")
    return result


def baseline(source: Path, base: Path, result_root: Path, matgptq: Path,
             output: Path, code_sha: str) -> dict:
    """Run the pinned official test evaluator on the original checkpoint once."""
    check_revision(matgptq)
    validate_repaired_base("s01", source, base)
    ident = identity("s01", source, base)
    prior = _records(result_root)
    from .rrq_pr2308 import source_identity
    if (not same_hf_contents(prior["w2"]["identity"]["original_run_identity"]["source"],
                             source_identity(source)) or
            prior["ternary_e2"]["identity"].get("source") != ident["source"] or
            prior["ternary_e2"]["identity"].get("repaired_base") != ident["repaired_base"] or
            prior["signed_e2"]["identity"].get("source") != ident["source"] or
            prior["signed_e2"]["identity"].get("repaired_base") != ident["repaired_base"]):
        raise ValueError("Stored S01 evaluation source or base identity differs")
    if output.exists():
        saved = json.loads((output / "full.json").read_text())
        if (saved.get("identity") != ident or saved.get("code_sha") != code_sha or
                saved.get("prior") != prior or
                sha256(output / "evaluation.log") != saved.get("evaluation_log_sha256")):
            raise ValueError("Existing original baseline differs")
        return saved
    output.mkdir(parents=True, exist_ok=False)
    corpus = corpus_identity(source)
    if corpus.get("token_sha256") != TOKEN_SHA or corpus.get("scored_tokens") != SCORED:
        raise ValueError("Original baseline token protocol differs")
    measured = evaluate_matgptq(matgptq, source, output / "evaluation.log")
    report = {"status": "completed_validated", "identity": ident, "code_sha": code_sha,
              "original_model_path": str(source.resolve()), "prior": prior,
              "corpus": corpus, "evaluation": measured,
              "evaluation_log_sha256": sha256(output / "evaluation.log")}
    atomic_json(output / "full.json", report)
    return report


def exact_endpoint(source: Path, base: Path, output: Path, code_sha: str,
                   *, rows: int = ROWS) -> dict:
    """Stream FP32 residuals and require FP16 endpoint parity for every weight."""
    if rows < 1:
        raise ValueError("Row block must be positive")
    validate_repaired_base("s01", source, base)
    ident = identity("s01", source, base)
    if output.exists():
        saved = json.loads(output.read_text())
        if saved.get("status") != "completed_validated" or saved.get("identity") != ident or \
                saved.get("code_sha") != code_sha:
            raise ValueError("Existing exact endpoint report differs")
        return saved
    source_map, base_map = _weight_map(source), _weight_map(base)
    if set(source_map) != set(base_map):
        raise ValueError("Source and base inventory differs")
    count = differing = fp32_differing = excluded = projections = 0
    base_error = exact_error = cast_error = signal = 0.0
    maximum = fp32_maximum = 0.0
    by_block = {}
    by_family = {}
    for name in sorted(source_map):
        if not _eligible(name):
            with safe_open(source / source_map[name], framework="pt", device="cpu") as handle:
                original = handle.get_tensor(name)
            with safe_open(base / base_map[name], framework="pt", device="cpu") as handle:
                other = handle.get_tensor(name)
            if not torch.equal(original, other):
                raise ValueError(f"Excluded tensor differs from source: {name}")
            excluded += 1
            continue
        projections += 1
        with safe_open(source / source_map[name], framework="pt", device="cpu") as handle:
            shape = handle.get_slice(name).get_shape()
        group = name.split(".")[2]
        block = by_block.setdefault(group, {"elements": 0, "base_error_sum": 0.0,
                                            "exact_error_sum": 0.0, "cast_error_sum": 0.0})
        family = "attention" if ".self_attn." in name else "mlp"
        family_entry = by_family.setdefault(family, {"elements": 0, "base_error_sum": 0.0,
                                                     "exact_error_sum": 0.0, "cast_error_sum": 0.0})
        for first in range(0, shape[0], rows):
            stop = min(first + rows, shape[0])
            original = _slice(source, source_map, name, first, stop).float()
            w2 = _slice(base, base_map, name, first, stop).float()
            if original.shape != w2.shape or not bool(torch.isfinite(original).all()) or \
                    not bool(torch.isfinite(w2).all()):
                raise ValueError(f"Invalid original or W2 projection: {name}")
            endpoint = w2 + (original - w2)
            cast = endpoint.to(torch.float16)
            expected = original.to(torch.float16)
            if not bool(torch.isfinite(endpoint).all()) or not bool(torch.isfinite(cast).all()):
                raise ValueError(f"Nonfinite exact endpoint: {name}")
            delta = endpoint - original
            cast_delta = cast.float() - expected.float()
            n = original.numel()
            b = float((w2 - original).double().square().sum())
            e = float(delta.double().square().sum())
            c = float(cast_delta.double().square().sum())
            count += n
            base_error += b; exact_error += e; cast_error += c
            signal += float(original.double().square().sum())
            block["elements"] += n
            block["base_error_sum"] += b
            block["exact_error_sum"] += e
            block["cast_error_sum"] += c
            family_entry["elements"] += n
            family_entry["base_error_sum"] += b
            family_entry["exact_error_sum"] += e
            family_entry["cast_error_sum"] += c
            fp32_differing += int((endpoint != original).sum())
            fp32_maximum = max(fp32_maximum, float(delta.abs().max()))
            differing += int((cast != expected).sum())
            maximum = max(maximum, float(cast_delta.abs().max()))
    if projections != 252 or differing or cast_error:
        raise ValueError(f"Exact residual endpoint fails FP16 parity: {projections=} {differing=} {maximum=}")
    report = {"status": "completed_validated", "code_sha": code_sha, "identity": ident,
              "projection_count": projections, "excluded_count": excluded,
              "eligible_elements": count, "fp16_differing_elements": differing,
              "fp32_differing_elements": fp32_differing,
              "fp32_max_abs_difference": fp32_maximum,
              "fp16_max_abs_difference": maximum, "w2_mse_fp32": base_error / count,
              "exact_mse_fp32": exact_error / count,
              "exact_mse_after_fp16_cast": cast_error / count,
              "w2_sqnr_db": 10 * math.log10(signal / base_error),
              "by_block": {key: {"elements": item["elements"],
                  "w2_mse_fp32": item["base_error_sum"] / item["elements"],
                  "exact_mse_fp32": item["exact_error_sum"] / item["elements"],
                  "exact_mse_after_fp16_cast": item["cast_error_sum"] / item["elements"]}
                  for key, item in by_block.items()},
              "by_family": {key: {"elements": item["elements"],
                  "w2_mse_fp32": item["base_error_sum"] / item["elements"],
                  "exact_mse_fp32": item["exact_error_sum"] / item["elements"],
                  "exact_mse_after_fp16_cast": item["cast_error_sum"] / item["elements"]}
                  for key, item in by_family.items()}}
    atomic_json(output, report)
    return report


def validation_sweep(source: Path, base: Path, result_root: Path, output: Path,
                     code_sha: str, *, blocks: int = 8) -> dict:
    """Score fixed WikiText validation blocks with one loaded FP16 W2 model."""
    if blocks not in (8, 32):
        raise ValueError("Use the fixed 8-block screen or 32-block confirmation")
    diagnostics = result_root / "progressive_llm/diagnostics/s01_residual_behavior" / code_sha
    exact_path = diagnostics / "exact_endpoint.json"
    exact = json.loads(exact_path.read_text())
    ident = identity("s01", source, base)
    if (exact.get("status") != "completed_validated" or exact.get("identity") != ident or
            exact.get("code_sha") != code_sha or exact.get("projection_count") != 252 or
            exact.get("fp16_differing_elements") != 0):
        raise ValueError("Validated exact-residual endpoint required before sweep")
    baseline = json.loads((diagnostics / "original_baseline/full.json").read_text())
    if (baseline.get("status") != "completed_validated" or
            baseline.get("identity") != ident or baseline.get("code_sha") != code_sha or
            baseline.get("corpus", {}).get("token_sha256") != TOKEN_SHA):
        raise ValueError("Comparable original test baseline required")
    signed_root = result_root / "progressive_llm/ablations/s01_autoround_signed_rtn"
    signed_marker = json.loads((signed_root / "layer1/encoding_complete.json").read_text())
    signed_sha = signed_marker.get("code_sha")
    if not isinstance(signed_sha, str) or len(signed_sha) != 40:
        raise ValueError("Signed E2 producing commit unavailable")
    with patch.dict(os.environ, {"FOLLOWUP_CODE_SHA": signed_sha}):
        signed = verified_layers(signed_root, 1,
            identity("s01", source, base, "autoround_signed_rtn"))[0]
    signed_resource_path = result_root / "progressive_llm/job_resources/s01_e2_signed_encode-56172.json"
    signed_resource = json.loads(signed_resource_path.read_text())
    if (signed_resource.get("job_id") != "56172" or
            signed_resource.get("host_memory_within_allocation") is not False):
        raise ValueError("Signed E2 job 56172 resource failure evidence differs")
    if output.exists():
        saved = json.loads(output.read_text())
        if (saved.get("status") != "completed_validated" or saved.get("code_sha") != code_sha or
                saved.get("exact_report_sha256") != sha256(exact_path) or
                saved.get("signed_manifest_sha256") != sha256(signed_root / "layer1/encoding_complete.json")):
            raise ValueError("Existing validation sweep differs")
        return saved
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=False)
    tokens = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt",
                       add_special_tokens=False)["input_ids"]
    if tokens.shape[1] < blocks * 2048:
        raise ValueError("WikiText validation subset is too short")
    subset = tokens[:, :blocks * 2048].contiguous()
    token_sha = hashlib.sha256(subset.numpy().tobytes()).hexdigest()
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16,
                                                   low_cpu_mem_usage=True, device_map="cuda:0")
    model.eval()
    model.config.use_cache = False
    parameters = dict(model.named_parameters())
    source_map, base_map = _weight_map(source), _weight_map(base)
    chunks = {}
    for chunk in signed["chunks"]:
        chunks[(chunk["name"], chunk["first"], chunk["stop"])] = chunk
    candidates = [(kind, alpha) for kind in ("exact", "signed")
                  for alpha in (0.0, 0.25, 0.5, 0.75, 1.0)]
    results = []
    for kind, alpha in candidates:
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        error_sum = cast_error_sum = signal_sum = 0.0
        count = changed = 0
        for name in sorted(source_map):
            if not _eligible(name):
                continue
            param = parameters.get(name)
            if param is None or param.is_meta or param.dtype != torch.float16:
                raise ValueError(f"Missing materialized FP16 projection: {name}")
            changed += 1
            shape = param.shape
            for first in range(0, shape[0], ROWS):
                stop = min(first + ROWS, shape[0])
                original = _slice(source, source_map, name, first, stop).float()
                w2 = _slice(base, base_map, name, first, stop).float()
                if kind == "exact":
                    correction = original - w2
                else:
                    chunk = chunks.get((name, first, stop))
                    if chunk is None:
                        raise ValueError("Signed E2 row partition differs")
                    correction = _read_chunk(signed_root / "layer1", chunk)
                result = w2 + alpha * correction
                if not bool(torch.isfinite(result).all()):
                    raise ValueError("Nonfinite validation candidate")
                param.data[first:stop].copy_(result.to(torch.float16).to(param.device))
                error_sum += float((result - original).double().square().sum())
                cast_error_sum += float((result.to(torch.float16).float() - original).double().square().sum())
                signal_sum += float(original.double().square().sum())
                count += original.numel()
        if changed != 252:
            raise ValueError("Validation sweep did not cover all projections")
        losses = []
        for token_block in subset.split(2048, dim=1):
            block = token_block.to("cuda:0")
            with torch.inference_mode():
                logits = model(block, use_cache=False).logits
                loss = torch.nn.functional.cross_entropy(logits[:, :-1].contiguous().reshape(-1, logits.shape[-1]),
                                                         block[:, 1:].reshape(-1))
            losses.append(float(loss))
            del logits
        from .rrq_reference_eval import aggregate
        metric = aggregate(losses)
        import resource
        results.append({"kind": kind, "alpha": alpha, "mse_fp32": error_sum / count,
                        "mse_after_fp16_cast": cast_error_sum / count,
                        "sqnr_db_fp32": 10 * math.log10(signal_sum / error_sum) if error_sum else None,
                        "validation": metric, "runtime_seconds": time.monotonic() - started,
                        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    report = {"status": "completed_validated", "code_sha": code_sha,
              "source_base_identity": ident, "signed_producing_sha": signed_sha,
              "signed_encoding_job_id": "56172",
              "signed_encoding_resource_sha256": sha256(signed_resource_path),
              "signed_encoding_resource_certified": False,
              "signed_manifest_sha256": sha256(signed_root / "layer1/encoding_complete.json"),
              "exact_report_sha256": sha256(exact_path), "dataset": "wikitext-2-raw-v1",
              "split": "validation", "join": "\\n\\n", "add_special_tokens": False,
              "sequence_length": 2048, "subset_blocks": blocks, "token_sha256": token_sha,
              "results": results}
    atomic_json(output, report)
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("baseline", "exact", "sweep"))
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--result-root", type=Path, required=True)
    p.add_argument("--matgptq", type=Path)
    p.add_argument("--code-sha", required=True)
    p.add_argument("--blocks", type=int, default=8)
    args = p.parse_args()
    root = args.result_root / "progressive_llm/diagnostics/s01_residual_behavior" / args.code_sha
    if args.action == "exact":
        result = exact_endpoint(args.source, args.base, root / "exact_endpoint.json", args.code_sha)
    elif args.action == "sweep":
        result = validation_sweep(args.source, args.base, args.result_root,
                                  root / f"validation_sweep_{args.blocks}.json",
                                  args.code_sha, blocks=args.blocks)
    else:
        if args.matgptq is None:
            p.error("baseline requires --matgptq")
        result = baseline(args.source, args.base, args.result_root, args.matgptq,
                          root / "original_baseline", args.code_sha)
    print(json.dumps({"status": result["status"], "code_sha": args.code_sha}, indent=2))


if __name__ == "__main__":
    main()
