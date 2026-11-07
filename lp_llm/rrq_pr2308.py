"""Pinned RRQ PR #2308 experiment identity, source audit and execution.

Only PR01/PR02 run locally. PR03/PR04 require an explicitly selected GPU job.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from . import QWEN3_MODEL_ID, QWEN3_REVISION
from .evaluate import _identity
from .rrq_reference_eval import check_revision, corpus_identity, run as evaluate_matgptq
from .rrq_runtime import probe, source_environment
from .rrq_process import run_logged
from .rrq_w2 import MATGPTQ_COMMIT
from .storage import atomic_json, sha256


PR_COMMIT = "7985c1b0f4eef65fb717511668fda7e347e584c3"
TOKEN_SHA256 = "8ffcee84dd4b684ec631cb52845e18c9c181ce6fdb8b26d65a4ffdc3d35fbda3"
CONFIG_ROOT = Path(__file__).resolve().parent.parent / "configs" / "rrq_pr2308"
RESULT_SUBDIR = "rrq_pr2308"
J05_RELATIVE = Path("rrq_w2_reproduction/j05/checkpoint/qwen3-8b-base-w2g128")


def lp_revision() -> str:
    return subprocess.check_output(["git", "-C", str(CONFIG_ROOT.parent.parent),
                                    "rev-parse", "HEAD"], text=True).strip()


def lp_worktree_dirty() -> bool:
    return bool(subprocess.check_output(["git", "-C", str(CONFIG_ROOT.parent.parent),
                                         "status", "--porcelain"], text=True).strip())


def config_for(experiment_id: str) -> dict:
    if experiment_id not in {"pr01", "pr02", "pr03", "pr04"}:
        raise ValueError(f"Conditional experiments require a documented trigger: {experiment_id}")
    path = CONFIG_ROOT / f"{experiment_id}.yaml"
    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    if (config.get("id") != experiment_id or config.get("model_id") != QWEN3_MODEL_ID or
            config.get("model_revision") != QWEN3_REVISION or config.get("rrq_commit") != PR_COMMIT):
        raise ValueError(f"Invalid PR experiment configuration: {path}")
    config["config_sha256"] = __import__("hashlib").sha256(raw).hexdigest()
    return config


def _source(checkout: Path, relative: str) -> str:
    return (checkout / relative).read_text(encoding="utf-8")


def source_audit(checkout: Path) -> dict:
    """Fail closed on the pinned source path used by this experiment."""
    check_revision(checkout, PR_COMMIT)
    config = _source(checkout, "auto_round/algorithms/quantization/rrq/config.py")
    quantizer = _source(checkout, "auto_round/algorithms/quantization/rrq/quantizer.py")
    entry = _source(checkout, "auto_round/autoround.py")
    exporter = _source(checkout, "auto_round/export/formats/autoround/export_to_rrq.py")
    base = _source(checkout, "auto_round/algorithms/quantization/base.py")
    integer = _source(checkout, "auto_round/data_type/int.py")
    compressor = _source(checkout, "auto_round/compressors/base.py")
    entry_tree = ast.parse(entry)
    auto_round = next(n for n in entry_tree.body if isinstance(n, ast.ClassDef) and n.name == "AutoRound")
    entry_new = next(n for n in auto_round.body if isinstance(n, ast.FunctionDef) and n.name == "__new__")
    direct_parameters = {n.arg for n in entry_new.args.args}
    expected_direct = {"model", "scheme", "alg_configs", "dataset", "low_gpu_mem_usage",
                       "device_map", "seed"}
    expected_forwarded = {"seqlen", "nsamples", "batch_size", "model_dtype", "format"}
    forwarded = {name for name in expected_forwarded if
                 f'"{name}"' in entry and (f'"{name}": "base"' in entry or
                                            f'direct_kwargs.pop("{name}"' in entry)}
    required = {
        "public_api_keywords": expected_direct <= direct_parameters and forwarded == expected_forwarded,
        "zero_iteration_config": "self.iters = int(self._rrq_iters or 0)" in config,
        "calibration_required": "self.need_calib = True" in config,
        "two_plane_option": "num_residual_planes not in (1, 3)" in config,
        "imatrix_normalized": "m.imatrix /= m.imatrix_cnt" in quantizer,
        "imatrix_uses_fp32_squares": "torch.pow(flattened, 2)" in quantizer,
        "standard_base_call": "self._quantize_layer_via_rtn(layer, disable_opt_rtn=False)" in quantizer,
        "standard_rtn_zero_iterations": "iters=0," in base,
        "single_quantize_and_save_api": "def quantize_and_save(" in compressor,
        "separate_base_export": '"base"' in exporter and '"residual"' in exporter,
        "rrq_format_dispatch": "auto_round:rrq" in entry,
        "weighted_w2_scale_search": ("if bits == 2:" in integer and "search_min = 18 * 5" in integer
                                      and "step = 0.01" in integer and "qw=imatrix" in integer),
        "default_lm_head_excluded": "quant_lm_head: bool = False" in compressor,
        "symmetric_base_uses_gptq_zero_grid": (
            'backend = backend.replace("auto_round", "auto_round:auto_gptq")' in
            _source(checkout, "auto_round/export/formats/autoround/integer.py")),
    }
    if not all(required.values()):
        raise ValueError(f"Pinned PR source contract changed: {[key for key, ok in required.items() if not ok]}")
    # Parse syntax so accidental prose matches cannot pass a corrupt checkout.
    for text in (config, quantizer, entry, exporter, base, integer, compressor):
        ast.parse(text)
    return {"revision": PR_COMMIT, "source_path": str(checkout.resolve()),
            "checks": required, "algorithm": "RRQ RTN with activation-weighted optimized RTN base",
            "iterations": 0, "gradient_updates": False, "calibration_required": True,
            "base_path": "_quantize_layer_via_rtn(disable_opt_rtn=False) in normal compressor context",
            "imatrix": "sum(input.float()**2, axis=tokens) / accumulated batch count",
            "requested_weight_dtype": "float16", "default_scale_dtype": "float16 unless overridden",
            "packing": "base/ standard AutoRound INT2; residual/ RRQ INT2",
            "selection": "upstream check_to_quantized on supported Linear modules; verify exported list",
            "selection_defaults": {"quant_lm_head": False, "ignore_layers": "", "layer_config": None},
            "expected_qwen3_projection_types": ["q_proj", "k_proj", "v_proj", "o_proj",
                                                   "gate_proj", "up_proj", "down_proj"],
            "expected_qwen3_projection_count": 36 * 7,
            "expected_exclusions": ["lm_head", "embeddings", "normalization parameters"],
            "selection_note": "Source-level default expectation; PR03 validates the actual exported module inventory",
            "scale_search": {"bits": 2, "search_index_min": -90, "search_index_max": 90,
                             "step": 0.01, "loss_weight": "imatrix"},
            "source_files": {name: sha256(checkout / name) for name in (
                "auto_round/algorithms/quantization/rrq/config.py",
                "auto_round/algorithms/quantization/rrq/quantizer.py",
                "auto_round/export/formats/autoround/export_to_rrq.py",
                "auto_round/inference/rrq_model.py",
                "auto_round_extension/torch/qlinear_torch_zp.py")}}


def source_identity(source: Path) -> dict:
    value = _identity(source, "rrq-pr2308-source")
    cfg = json.loads((source / "config.json").read_text())
    if cfg.get("model_type") != "qwen3" or cfg.get("_commit_hash") not in (None, QWEN3_REVISION):
        raise ValueError("Source is not pinned Qwen3-8B-Base")
    if cfg.get("num_hidden_layers") != 36 or cfg.get("hidden_size") != 4096:
        raise ValueError("Unexpected Qwen3-8B source architecture")
    return value


def experiment_identity(config: dict, source: Path | None, checkout: Path) -> dict:
    identity = {"id": config["id"], "config_sha256": config["config_sha256"],
                "rrq_commit": PR_COMMIT, "model_id": QWEN3_MODEL_ID,
                "model_revision": QWEN3_REVISION, "matgptq_commit": MATGPTQ_COMMIT,
                "source_checkout": str(checkout.resolve())}
    if source is not None:
        identity["source_checkpoint"] = source_identity(source)
    return identity


def _marker(path: Path, identity: dict) -> dict | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value.get("identity") != identity or value.get("status") != "completed_validated":
        raise ValueError(f"Existing experiment marker has wrong identity or status: {path}")
    return value


def pr01(checkout: Path, result: Path, *, j05_config: dict | None = None) -> dict:
    audit = source_audit(checkout)
    if j05_config is None:
        from .rrq_w2 import load_config
        j05_config = load_config("j05")
    comparison = {"status": "completed_validated", "experiment_id": "pr01",
                  "lp_git_commit": lp_revision(), "config_sha256": config_for("pr01")["config_sha256"],
                  "lp_worktree_dirty": lp_worktree_dirty(),
                  "model_id": QWEN3_MODEL_ID, "model_revision": QWEN3_REVISION,
                  "pr": audit, "j05": {
        "revision": "6c824f741be358983baba87aff83f9c5aa0a01e7",
        "algorithm": "official CLI optimized RTN W2G128", "iterations": 0,
        "calibration": "128 NeelNanda/pile-10k examples prepared and cached, per J05 log",
        "format": "fake QDQ", "requested_model_dtype": "float16",
        "actual_export_dtypes": "read existing J05 completion metadata at deployment",
        "evaluation": "MatGPTQ WikiText-2 raw, 2048 nonoverlapping blocks, FP16"},
        "unresolved": ["Exact calibration sample order and imatrix equality with J05 were not persisted",
                       "PR model selection and actual exported dtypes require full-model validation",
                       "The August paper's calibration-free recipe is not this November PR's calibrated RTN"]}
    comparison["j05"]["recorded_config"] = j05_config
    atomic_json(result / "comparison.json", comparison)
    return comparison


def _pr03_script() -> str:
    return '''import json, os, sys, torch
from pathlib import Path
from auto_round import AutoRound, RRQConfig
source, output, metadata = map(Path, sys.argv[1:4])
import auto_round
resolved = Path(auto_round.__file__).resolve()
expected = Path(os.environ["RRQ_PR_SOURCE"]).resolve()
if not resolved.is_relative_to(expected):
    raise RuntimeError(f"Wrong AutoRound imported: {resolved}")
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(source, trust_remote_code=False)
if cfg.model_type != "qwen3" or cfg.num_hidden_layers != 36 or cfg.hidden_size != 4096:
    raise RuntimeError("Wrong source architecture")
rrq_config = RRQConfig(group_size=128, sym=True, iters=0, num_residual_planes=1)
rrq = AutoRound(model=str(source), scheme="W2A16",
    alg_configs=rrq_config,
    format="auto_round:rrq", dataset="NeelNanda/pile-10k", nsamples=128,
    seqlen=2048, batch_size=8, seed=42, model_dtype="float16",
    device_map="cuda:0", low_gpu_mem_usage=True)
loaded_model = getattr(rrq, "model", None)
first_dtype = None
if isinstance(loaded_model, torch.nn.Module):
    first_param = next(loaded_model.parameters(), None)
    first_dtype = str(first_param.dtype) if first_param is not None else None
metadata.write_text(json.dumps({"imported_auto_round":str(resolved),
    "model_config_dtype":str(getattr(cfg,"torch_dtype",None)),
    "observed_first_loaded_weight_dtype":first_dtype,
    "resolved_config":{"bits":rrq_config.bits,"group_size":rrq_config.group_size,
        "sym":rrq_config.sym,"iters":rrq_config.iters,
        "need_calib":rrq_config.need_calib,"total_planes":rrq_config.total_planes,
        "scale_dtype":str(getattr(rrq_config,"scale_dtype",None))},
    "requested_calibration":{"dataset":"NeelNanda/pile-10k","nsamples":128,
        "seqlen":2048,"batch_size":8,"seed":42},
    "torch_version":torch.__version__, "rrq_type":type(rrq).__name__,
    "config_type":type(rrq.alg_configs).__name__ if hasattr(rrq,"alg_configs") else None}, indent=2))
rrq.quantize_and_save(str(output))
'''


def pr03_command(source: Path, output: Path, metadata: Path, python: str) -> list[str]:
    return [python, "-c", _pr03_script(), str(source), str(output), str(metadata)]


def run_pr03(config: dict, source: Path, checkout: Path, result: Path, *, python: str,
             overlay: Path | None = None) -> dict:
    from .rrq_pr2308_compare import same_hf_contents, validate_packed_base, validate_residual
    audit = source_audit(checkout)
    identity = experiment_identity(config, source, checkout)
    j05_run = result.parents[1] / "rrq_w2_reproduction/j05/run.json"
    if not j05_run.is_file():
        raise ValueError(f"Existing J05 source identity is unavailable: {j05_run}")
    j05_identity = json.loads(j05_run.read_text())
    if (j05_identity.get("revision") != QWEN3_REVISION or
            not same_hf_contents(j05_identity.get("source_checkpoint"),
                                 _identity(source, "rrq-source"))):
        raise ValueError("PR03 source checkpoint differs from J05's pinned original")
    marker = result / "quantization_complete.json"
    run_path = result / "run.json"
    if run_path.is_file():
        recorded = json.loads(run_path.read_text())
        if recorded.get("identity") != identity:
            raise ValueError("PR03 run directory has a different identity")
    elif result.exists() and any(result.iterdir()):
        raise ValueError("Nonempty PR03 directory lacks run identity")
    existing = _marker(marker, identity)
    export = result / "export"
    if existing is not None:
        current = validate_packed_base(export / "base", source)
        if current != existing["checkpoint"] or validate_residual(export / "residual", current) != existing["residual"]:
            raise ValueError("Completed PR03 checkpoint has changed")
        return existing
    if export.exists() and any(export.iterdir()):
        raise ValueError("Unmarked PR03 export exists; inspect it before any rerun")
    result.mkdir(parents=True, exist_ok=True)
    env = source_environment(checkout, overlay)
    env["RRQ_PR_SOURCE"] = str(checkout.resolve())
    imports = probe(python, ["auto_round", "torch", "transformers", "datasets", "safetensors"], env=env)
    imported = Path(imports["modules"]["auto_round"]["file"]).resolve()
    if not imported.is_relative_to(checkout.resolve()):
        raise ValueError(f"Wrong AutoRound import: {imported}")
    cmd = pr03_command(source, export, result / "runtime.json", python)
    atomic_json(run_path, {"status": "running", "identity": identity,
                           "command": cmd, "source_audit": audit, "imports": imports})
    execution = None
    try:
        execution = run_logged(cmd, result / "quantization.log", cwd=checkout, env=env)
        if execution["returncode"]:
            raise RuntimeError(f"PR03 official quantization failed; see {result / 'quantization.log'}")
        checkpoint = validate_packed_base(export / "base", source)
        residual = validate_residual(export / "residual", checkpoint)
        record = {"status": "completed_validated", "identity": identity, "checkpoint": checkpoint,
                  "residual": residual,
                  "lp_git_commit": lp_revision(),
                  "lp_worktree_dirty": lp_worktree_dirty(),
                  "quantization_log_sha256": sha256(result / "quantization.log"),
                  "execution": execution, "runtime": json.loads((result / "runtime.json").read_text()),
                  "imports": imports, "calibration": config["calibration"]}
        atomic_json(marker, record)
        return record
    except Exception as error:
        atomic_json(result / "failure.json", {"status": "failed", "identity": identity,
                                               "stage": "quantization_or_validation",
                                               "error_type": type(error).__name__, "error": str(error),
                                               "execution": execution})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", choices=["pr01", "pr02", "pr03", "pr04"], required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--rrq-source", type=Path, required=True)
    parser.add_argument("--matgptq-source", type=Path)
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    config = config_for(args.experiment_id)
    result = args.result_root / RESULT_SUBDIR / args.experiment_id
    if args.experiment_id == "pr01":
        result.mkdir(parents=True, exist_ok=True)
        print(json.dumps(pr01(args.rrq_source, result), indent=2))
    elif args.experiment_id == "pr02":
        from .rrq_pr2308_parity import run_parity
        result.mkdir(parents=True, exist_ok=True)
        print(json.dumps(run_parity(args.rrq_source, result), indent=2))
    else:
        if args.source is None:
            parser.error("PR03/PR04 require --source")
        if args.experiment_id == "pr03":
            print(json.dumps(run_pr03(config, args.source, args.rrq_source, result,
                                      python=args.python, overlay=args.overlay), indent=2))
        else:
            if args.matgptq_source is None:
                parser.error("PR04 requires --matgptq-source")
            from .rrq_pr2308_compare import run_pr04
            print(json.dumps(run_pr04(config, args.source, args.rrq_source, args.matgptq_source,
                                      args.result_root, python=args.python), indent=2))


if __name__ == "__main__":
    main()
