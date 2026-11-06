"""Execute one configured RRQ W2 investigation in an isolated result directory."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import torch
from safetensors import safe_open

from .evaluate import _identity
from .rrq_autoround import run as run_autoround, validate_output
from .rrq_hybrid import build as build_hybrid
from .rrq_reference_eval import (check_revision, corpus_identity, local_evaluate,
                                 run as run_reference)
from .rrq_w2 import AUTOROUND_COMMIT, MATGPTQ_COMMIT, completion_matches, identity, load_config
from .storage import atomic_json, sha256


def _git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _checkpoint_dtypes(checkpoint: Path) -> dict[str, int]:
    mapping = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    counts: dict[str, int] = {}
    for shard in sorted(set(mapping.values())):
        with safe_open(checkpoint / shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                dtype = str(handle.get_slice(name).get_dtype())
                counts[dtype] = counts.get(dtype, 0) + 1
    return counts


def _checkpoint(config: dict, source: Path, result_root: Path, job: Path,
                autoround_source: Path) -> tuple[Path, dict]:
    kind, mode = config["kind"], config["mode"]
    if kind == "evaluation":
        if config["id"] == "j01":
            checkpoint = source
        elif config["id"] == "j02":
            checkpoint = result_root / "rtn_reproduction/b1_plain_fp16/checkpoint"
        else:
            checkpoint = result_root / "rtn_reproduction/b3_opt_fp16/checkpoint"
        details = {"origin": "existing_checkpoint"}
        if config["id"] != "j01":
            artifact = result_root / "rtn_reproduction" / ("b1_plain_fp16" if config["id"] == "j02" else "b3_opt_fp16") / "artifact"
            storage = json.loads((artifact / "storage_report.json").read_text())["prefixes"][0]
            manifest = json.loads((artifact / "manifest.json").read_text())
            details["effective_bits_per_total_parameter"] = storage["effective_bits_per_total_parameter"]
            details["packed_representation_bytes"] = storage["total_physical_bytes"]
            details["eligible_tensor_names"] = sorted(r["name"] for r in
                                                     manifest["tensors"] if r["eligible"])
            details["excluded_tensor_names"] = sorted(r["name"] for r in manifest["tensors"] if not r["eligible"])
            details["quantizer_spec"] = manifest["identity"].get("quantizer_spec")
        return checkpoint, details
    if kind == "autoround":
        checkpoint = job / "checkpoint"
        marker = job / "encoding_complete.json"
        expected = {"config_sha256": config["config_sha256"], "source": _identity(source, "rrq-original"),
                    "autoround_commit": AUTOROUND_COMMIT}
        old = completion_matches(marker, expected)
        if old:
            if old.get("checkpoint") != validate_output(checkpoint):
                raise ValueError("Previously completed AutoRound checkpoint changed")
            return checkpoint, old
        if checkpoint.exists() and any(checkpoint.iterdir()):
            raise ValueError(f"Incomplete official AutoRound output requires manual inspection: {checkpoint}")
        output = run_autoround(config, source, checkpoint, autoround_source, job / "autoround.log")
        value = {"status": "complete", "identity": expected, **output}
        atomic_json(marker, value)
        return checkpoint, value
    if kind == "hybrid":
        base_dir = result_root / "rtn_reproduction/b3_opt_fp16"
        checkpoint = job / "checkpoint"
        value = build_hybrid(source, base_dir / "artifact",
                             base_dir / "materialization_fp16/checkpoint", checkpoint, mode)
        return checkpoint, {"origin": "original_BF16_restoration_into_B3_FP16", **value}
    raise ValueError(f"{config['id']} is unavailable: {config.get('reason')}")


def run(experiment_id: str, source: Path, result_root: Path, matgptq_source: Path,
        autoround_source: Path, *, stage: str = "screen") -> dict:
    if stage not in ("screen", "full"):
        raise ValueError("stage must be screen or full")
    config = load_config(experiment_id)
    if config["kind"] == "unavailable":
        raise RuntimeError(f"{experiment_id} unavailable: {config['reason']}")
    source, result_root = Path(source), Path(result_root)
    job = result_root / "rrq_w2_reproduction" / experiment_id
    job.mkdir(parents=True, exist_ok=True)
    begun = time.monotonic()
    base_identity = identity(config, source)
    run_path = job / "run.json"
    if run_path.exists():
        if json.loads(run_path.read_text()) != base_identity:
            raise ValueError(f"Experiment directory has different identity: {job}")
    elif any(job.iterdir()):
        raise ValueError(f"Nonempty experiment directory lacks identity: {job}")
    else:
        atomic_json(run_path, base_identity)
    check_revision(matgptq_source, MATGPTQ_COMMIT)
    if config["kind"] == "autoround":
        check_revision(autoround_source, AUTOROUND_COMMIT)
    checkpoint, construction = _checkpoint(config, source, result_root, job, autoround_source)
    eval_identity = identity(config, source, checkpoint)
    eval_identity["stage"] = stage
    result_path = job / ("diagnostic.json" if config["kind"] == "hybrid" and stage == "screen"
                         else "full.json")
    old = completion_matches(result_path, eval_identity)
    if old:
        if (old.get("checkpoint_identity") != _identity(checkpoint, "rrq-result") or
                not math.isfinite(old.get("score", {}).get("perplexity", float("nan"))) or
                not math.isfinite(old.get("score", {}).get("cross_entropy", float("nan")))):
            raise ValueError(f"Previously completed evaluation is incomplete or corrupt: {result_path}")
        return old
    if config["kind"] == "hybrid" and stage == "screen":
        from .rtn_diagnostic_eval import evaluate
        evaluation = evaluate(checkpoint, job / "diagnostic_detail.json")
        score = {"perplexity": evaluation["perplexity"],
                 "cross_entropy": evaluation["mean_cross_entropy"],
                 "scored_tokens": evaluation["scored_tokens"],
                 "token_sha256": evaluation["selected_token_sha256"],
                 "evaluation": "eight_block_diagnostic", "detail": evaluation}
    else:
        corpus = corpus_identity(checkpoint)
        official = run_reference(matgptq_source, checkpoint, job / "matgptq.log")
        score = {**corpus, **official, "evaluation": "full_matgptq"}
        if experiment_id == "j01":
            local = local_evaluate(checkpoint)
            if (local["scored_tokens"] != corpus["scored_tokens"] or
                    local["token_sha256"] != corpus["token_sha256"]):
                raise ValueError("J01 local and official-protocol token identities differ")
            score["local_matched_evaluation"] = local
            score["perplexity_delta"] = local["perplexity"] - official["perplexity"]
            score["cross_entropy_delta"] = local["cross_entropy"] - official["cross_entropy"]
    if not math.isfinite(score["perplexity"]) or not math.isfinite(score["cross_entropy"]):
        raise ValueError("Nonfinite evaluation result")
    value = {"status": "complete", "identity": eval_identity, "configuration": config,
             "source_model_revision": config["model_revision"], "git_commit": _git_revision(),
             "checkpoint_path": str(checkpoint), "checkpoint_identity": _identity(checkpoint, "rrq-result"),
             "construction": construction, "configured_checkpoint_dtype": config["checkpoint_dtype"],
             "checkpoint_tensor_dtypes": _checkpoint_dtypes(checkpoint),
             "inference_dtype": config["inference_dtype"], "score": score,
             "runtime_seconds": time.monotonic() - begun,
             "runner_peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None}
    atomic_json(result_path, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--matgptq-source", type=Path, required=True)
    parser.add_argument("--autoround-source", type=Path, required=True)
    parser.add_argument("--stage", choices=("screen", "full"), default="screen")
    args = parser.parse_args()
    print(json.dumps(run(args.experiment_id, args.source, args.result_root,
                         args.matgptq_source, args.autoround_source, stage=args.stage), indent=2))


if __name__ == "__main__":
    main()
