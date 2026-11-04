"""Configuration-driven, restart-safe 2-bit RTN reproduction workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from .encode import encode, _revision
from .evaluate import _identity
from .reconstruct import reconstruct
from .rtn_audit import audit
from .rtn_diagnostic_eval import evaluate as diagnostic_evaluate
from .rtn_logits import compare as compare_logits
from .rtn_reproduction import AUTOROUND_COMMIT, MODES, quantize_base
from .storage import atomic_json, sha256
from .upstream_reference import direct_qdq
from .validate_autoround_stage import evaluation as validate_full_evaluation


def _read_completed(path: Path, identity: dict):
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value.get("status") != "complete" or value.get("identity") != identity:
        raise ValueError(f"Existing stage has a different identity or is incomplete: {path}")
    return value


def _settings(config: Path) -> tuple[dict, str]:
    raw = config.read_bytes()
    settings = yaml.safe_load(raw)
    if settings.get("quantizer_family") not in MODES or settings.get("bits") != [2]:
        raise ValueError("Runner accepts only B1–B3 single-base configurations")
    expected = {"b1_plain_fp16": "autoround_plain_fp16",
                "b2_plain_bf16": "autoround_plain_bf16",
                "b3_opt_fp16": "autoround_opt_fp16"}
    if settings.get("experiment_id") not in expected:
        raise ValueError("Unknown Phase 1 experiment ID")
    if settings["quantizer_family"] != expected[settings["experiment_id"]]:
        raise ValueError("Experiment ID and quantizer family disagree")
    if (settings.get("method") != "lp" or settings.get("granularity") != "group"
            or settings.get("group_size") != 128):
        raise ValueError("Phase 1 configuration must encode one LP G128 base")
    return settings, hashlib.sha256(raw).hexdigest()


def numerical(settings: dict, config_sha: str, output: Path) -> dict:
    identity = {"experiment_id": settings["experiment_id"], "config_sha256": config_sha,
                "autoround_commit": AUTOROUND_COMMIT, "torch": torch.__version__}
    old = _read_completed(output, identity)
    if old:
        return old
    family = settings["quantizer_family"]
    algorithm, dtype = MODES[family]
    torch.manual_seed(331)
    x = torch.randn(6, 259, dtype=dtype)
    x[0] = 0
    x[1] = 2.5
    x[2] = -2.5
    x[3] = x[3].abs() + 0.1
    x[4] = -x[4].abs() - 0.1
    x[5, 0], x[5, 1] = 50, -100
    qdq, scale = direct_qdq(x.to(settings["compute_device"]),
                            "plain" if algorithm == "plain" else "optimized")
    params, ours = quantize_base(x, family, compute_device=settings["compute_device"])
    pad = (-x.shape[1]) % 128
    grouped = torch.nn.functional.pad(x.to(settings["compute_device"]), (0, pad)).reshape(-1, 128)
    reference_codes = (grouped.div(scale).round_().clamp_(-2, 1).int() + 2)
    reference_codes = reference_codes.reshape(x.shape[0], -1)[:, :x.shape[1]].cpu()
    qdq = qdq.cpu()
    scale = scale.reshape_as(params["scale"]).cpu()
    delta = (qdq.float() - ours.float()).abs()
    scale_match = torch.equal(scale, params["scale"])
    qdq_match = torch.equal(qdq, ours)
    codes_match = torch.equal(reference_codes, params["codes"].int())
    value = {"status": "complete" if scale_match and qdq_match and codes_match else "discrepancy",
             "identity": identity, "scale_exact": scale_match, "codes_exact": codes_match,
             "qdq_exact": qdq_match,
             "maximum_absolute_error": float(delta.max()),
             "rmse": float(torch.sqrt(delta.square().mean())),
             "scale_signs_equal": torch.equal(torch.sign(scale), torch.sign(params["scale"])),
             "input_dtype": str(dtype).removeprefix("torch."),
             "compute_device": settings["compute_device"]}
    atomic_json(output, value)
    if value["status"] != "complete":
        raise ValueError(f"Pinned upstream numerical parity failed: {output}")
    return value


def _paths(result_root: Path, experiment_id: str) -> dict:
    root = result_root / "rtn_reproduction" / experiment_id
    return {"root": root, "artifact": root / "artifact", "checkpoint": root / "checkpoint",
            "numerical": root / "numerical.json", "audit": root / "coverage.json",
            "logits": root / "logits.json", "diagnostic": root / "diagnostic.json",
            "full": root / "full_wikitext2.json", "report": root / "result.json"}


def prepare(source: Path, settings: dict, config_sha: str, paths: dict):
    if not paths["numerical"].exists():
        raise ValueError("Run numerical parity before checkpoint generation")
    _read_completed(paths["numerical"], {"experiment_id": settings["experiment_id"],
                                         "config_sha256": config_sha,
                                         "autoround_commit": AUTOROUND_COMMIT,
                                         "torch": torch.__version__})
    coverage = audit(source)
    for key, expected in (("quantized_parameters", settings.get("expected_quantized_parameters")),
                          ("excluded_parameters", settings.get("expected_excluded_parameters"))):
        if expected is not None and coverage[key] != expected:
            raise ValueError(f"Source checkpoint coverage differs: {key}={coverage[key]}, expected {expected}")
    if paths["audit"].exists() and json.loads(paths["audit"].read_text()) != coverage:
        raise ValueError("Existing coverage inventory differs")
    atomic_json(paths["audit"], coverage)
    encode(source, paths["artifact"], settings)
    reconstruct(paths["artifact"], paths["checkpoint"], 1)
    _identity(paths["checkpoint"], "rtn-prepare-validation")


def logits(source: Path, paths: dict, device: str):
    identity = {"source_index_sha256": sha256(source / "model.safetensors.index.json"),
                "manifest_sha256": sha256(paths["artifact"] / "manifest.json"),
                "checkpoint_index_sha256": sha256(paths["checkpoint"] / "model.safetensors.index.json"),
                "tokens": [1, 2, 3, 4, 5, 6, 7, 8], "device": device}
    old = _read_completed(paths["logits"], identity)
    if old:
        return old
    result = compare_logits(source, paths["artifact"], paths["checkpoint"],
                            torch.tensor([identity["tokens"]]), device)
    value = {"status": "complete", "identity": identity, **result}
    atomic_json(paths["logits"], value)
    return value


def full(paths: dict, evaluator: Path):
    if not evaluator.is_file():
        raise ValueError(f"Missing existing untracked MatGPTQ evaluator: {evaluator}")
    if validate_full_evaluation(paths["artifact"], paths["checkpoint"], paths["full"],
                                evaluator, 1):
        return json.loads(paths["full"].read_text())
    if paths["full"].exists():
        raise ValueError("Existing full result is unmarked or invalid")
    subprocess.run([sys.executable, "-m", "lp_llm.evaluate_matgptq",
                    "--checkpoint", str(paths["checkpoint"]), "--output", str(paths["full"]),
                    "--sequence-length", "2048", "--dtype", "float16"],
                   cwd=evaluator.parent.parent, check=True)
    validate_full_evaluation(paths["artifact"], paths["checkpoint"], paths["full"],
                             evaluator, 1, mark=True)
    return json.loads(paths["full"].read_text())


def report(settings: dict, config_sha: str, paths: dict) -> dict:
    manifest = json.loads((paths["artifact"] / "manifest.json").read_text()) if (paths["artifact"] / "manifest.json").exists() else None
    storage = json.loads((paths["artifact"] / "storage_report.json").read_text()) if (paths["artifact"] / "storage_report.json").exists() else None
    coverage = json.loads(paths["audit"].read_text()) if paths["audit"].exists() else None
    numerical_result = json.loads(paths["numerical"].read_text()) if paths["numerical"].exists() else None
    logit_result = json.loads(paths["logits"].read_text()) if paths["logits"].exists() else None
    diagnostic = json.loads(paths["diagnostic"].read_text()) if paths["diagnostic"].exists() else None
    full_result = json.loads(paths["full"].read_text()) if paths["full"].exists() else None
    spec = manifest["identity"]["quantizer_spec"] if manifest else None
    metrics = manifest["prefix_diagnostics"][0] if manifest else None
    size = storage["prefixes"][0] if storage else None
    value = {"experiment_id": settings["experiment_id"], "experiment_name": settings["quantizer_family"],
             "config_sha256": config_sha, "model_id": settings["model_id"],
             "revision": settings["revision"], "git_commit": _revision(),
             "autoround_source_revision": AUTOROUND_COMMIT,
             "source_index_sha256": coverage["index_sha256"] if coverage else None,
             "checkpoint_identity": _identity(paths["checkpoint"], "rtn-report") if manifest else None,
             "quantizer": spec, "coverage": coverage,
             "parameter_mse_fp32": metrics["parameter_mse_fp32"] if metrics else None,
             "sqnr_db_fp32": metrics["sqnr_db_fp32"] if metrics else None,
             "parameter_mse_after_bf16_cast": metrics["parameter_mse_after_bf16_cast"] if metrics else None,
             "qdq_decoder_difference": ({key: metrics[key] for key in (
                 "qdq_decode_max_abs_before_cast", "qdq_decode_rmse_before_cast",
                 "qdq_decode_max_abs_after_bf16_cast", "qdq_decode_rmse_after_bf16_cast")}
                 if metrics else None),
             "logit_parity": logit_result, "diagnostic": diagnostic,
             "full_wikitext2_perplexity": (full_result.get("perplexity", full_result.get("ppl"))
                                           if full_result else None),
             "quantization_seconds": manifest["processing_seconds"] if manifest else None,
             "physical_storage": size, "numerical_validation": numerical_result,
             "validation_status": {"numerical": bool(numerical_result and numerical_result["status"] == "complete"),
                                   "checkpoint": bool(manifest), "logits": bool(logit_result),
                                   "diagnostic": bool(diagnostic), "full": bool(full_result)}}
    atomic_json(paths["report"], value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("numerical", "prepare", "logits", "diagnostic", "full", "report", "screen"), required=True)
    parser.add_argument("--evaluator-code", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    settings, config_sha = _settings(args.config)
    paths = _paths(args.result_root, settings["experiment_id"])
    paths["root"].mkdir(parents=True, exist_ok=True)
    if args.stage in ("numerical", "screen"):
        numerical(settings, config_sha, paths["numerical"])
    if args.stage in ("prepare", "screen"):
        prepare(args.source, settings, config_sha, paths)
    if args.stage in ("logits", "screen"):
        logits(args.source, paths, args.device)
    if args.stage in ("diagnostic", "screen"):
        diagnostic_evaluate(paths["checkpoint"], paths["diagnostic"], device=args.device)
    if args.stage == "full":
        if args.evaluator_code is None:
            raise ValueError("--evaluator-code is required for full evaluation")
        if not paths["numerical"].is_file() or not paths["logits"].is_file() or not paths["diagnostic"].is_file():
            raise ValueError("Run numerical, logit, and diagnostic stages before full evaluation")
        full(paths, args.evaluator_code / "lp_llm" / "evaluate_matgptq.py")
    value = report(settings, config_sha, paths)
    print(json.dumps({"experiment_id": settings["experiment_id"], "validation_status": value["validation_status"],
                      "result": str(paths["report"])}, indent=2))


if __name__ == "__main__":
    main()
