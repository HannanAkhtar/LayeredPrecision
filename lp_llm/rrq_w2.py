"""Explicit configuration and identity for the 14 RRQ W2 investigations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from . import QWEN3_MODEL_ID, QWEN3_REVISION


MATGPTQ_COMMIT = "0e5bd93bc53381f7195db870f5408ae04906bb3c"
AUTOROUND_COMMIT = "6c824f741be358983baba87aff83f9c5aa0a01e7"
AUTOROUND_HISTORICAL_COMMIT = "24bfce00bb3bd053498232f0e8395118fe10a906"
AUTOROUND_REVISIONS = {"current": AUTOROUND_COMMIT, "preprint_candidate": AUTOROUND_HISTORICAL_COMMIT}
CONFIG_ROOT = Path(__file__).resolve().parent.parent / "configs" / "rrq_w2"
MODES = {
    "j01": ("evaluation", "original", 128),
    "j02": ("evaluation", "b1_plain_fp16", 128),
    "j03": ("evaluation", "b3_opt_fp16_bf16", 128),
    "j04": ("autoround", "plain", 128),
    "j05": ("autoround", "optimized", 128),
    "j06": ("autoround", "optimized", 64),
    "j07": ("autoround", "optimized", 32),
    "j08": ("unavailable", "signroundv2_unverified", 128),
    "j09": ("hybrid", "q_proj,k_proj", 128),
    "j10": ("hybrid", "v_proj,o_proj", 128),
    "j11": ("hybrid", "gate_proj,up_proj", 128),
    "j12": ("hybrid", "down_proj", 128),
    "j13": ("hybrid", "first_third", 128),
    "j14": ("hybrid", "final_third", 128),
}


def load_config(experiment_id: str) -> dict:
    if experiment_id not in MODES:
        raise ValueError(f"Unknown RRQ W2 experiment: {experiment_id}")
    path = CONFIG_ROOT / f"{experiment_id}.yaml"
    raw = path.read_bytes()
    value = yaml.safe_load(raw)
    kind, mode, group = MODES[experiment_id]
    if (value.get("id") != experiment_id or value.get("kind") != kind or
            value.get("mode") != mode or value.get("group_size") != group or
            value.get("bits") != 2 or value.get("symmetric") is not True or
            value.get("model_id") != QWEN3_MODEL_ID or
            value.get("model_revision") != QWEN3_REVISION or
            value.get("evaluation_protocol") != "matgptq_wikitext2_2048_slow" or
            value.get("inference_dtype") != "float16"):
        raise ValueError(f"Invalid or inconsistent experiment configuration: {path}")
    value["config_sha256"] = hashlib.sha256(raw).hexdigest()
    if kind == "autoround":
        value["quantizer_configuration"] = {
            "implementation": "official_auto_round_cli", "revision": AUTOROUND_COMMIT,
            "algorithm": "plain_rtn" if mode == "plain" else "optimized_rtn_scale_search",
            "iterations": 0, "gradient_updates": False,
            "configured_nsamples": 1 if mode == "plain" else 128,
            "observed_calibration_examples_prepared": (0 if experiment_id == "j04" else
                                                        128 if experiment_id == "j05" else None),
            "observed_calibration_data_used": (False if experiment_id == "j04" else
                                               True if experiment_id == "j05" else None),
            "observed_calibration_examples_consumed": 0 if experiment_id == "j04" else None,
            "source_audited_expected_calibration_examples": 0 if mode == "plain" else 128,
            "calibration_evidence": ("J04 zero-shot RTN without calibration" if experiment_id == "j04" else
                                     "J05 log: NeelNanda/pile-10k, 128 prepared samples, cached block "
                                     "inputs, OptimizedRTNQuantizer" if experiment_id == "j05" else
                                     "optimized recipe configured 128; verify when this job runs"),
            "group_size": group, "symmetric": True, "bits": 2,
            "scheme": "W2A16" if group == 128 else f"W2A16G{group}",
            "format": "fake", "model_dtype_argument": "float16",
            "model_load_dtype": "float16_explicit_CLI",
            "optimized_2bit_scale_search": {"index_min": -90, "index_max": 90, "step": 0.01,
                                            "zero_index_skipped": True} if mode != "plain" else None,
            "module_selection": "upstream_default_no_ignore_layers_no_lm_head"}
    elif kind == "hybrid":
        value["quantizer_configuration"] = {"implementation": "existing_B3_optimized_RTN",
                                                   "new_quantization": False,
                                                   "base_artifact": "rtn_reproduction/b3_opt_fp16/artifact"}
    return value


def identity(config: dict, source: Path, checkpoint: Path | None = None,
             *, autoround_variant: str = "current") -> dict:
    from .evaluate import _identity
    result = {"experiment_id": config["id"], "config_sha256": config["config_sha256"],
              "model_id": config["model_id"], "revision": config["model_revision"],
              "source_checkpoint": _identity(source, "rrq-source"),
              "matgptq_commit": MATGPTQ_COMMIT}
    if config["kind"] == "autoround":
        if autoround_variant not in AUTOROUND_REVISIONS:
            raise ValueError(f"Unknown AutoRound reference variant: {autoround_variant}")
        result["autoround_variant"] = autoround_variant
        result["autoround_commit"] = AUTOROUND_REVISIONS[autoround_variant]
    if checkpoint is not None:
        result["checkpoint"] = _identity(checkpoint, "rrq-evaluation")
    return result


def completion_matches(path: Path, expected: dict) -> dict | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value.get("status") != "complete" or value.get("identity") != expected:
        raise ValueError(f"Existing result has another identity: {path}")
    return value
