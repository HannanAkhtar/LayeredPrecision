"""Resumable, source-pinned follow-up quantization experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from safetensors import safe_open

from . import QWEN3_MODEL_ID, QWEN3_REVISION
from .rrq_autoround import validate_output, _w2_grid_check
from .rrq_pr2308_compare import _eligible, _get, _weight_map, same_hf_contents
from .rrq_reference_eval import check_revision, corpus_identity, run as evaluate_matgptq
from .rrq_runtime import probe, source_environment
from .rrq_overlay import validate_overlay
from .rrq_process import run_logged
from .rrq_pr2308 import PR_COMMIT, TOKEN_SHA256, source_identity
from .rrq_w2 import MATGPTQ_COMMIT
from .storage import SIDECARS, atomic_json, sha256


GPTQ_COMMIT = "ed96f2e04ec868b04f969733a242ce3b5bb7cb1e"
CONFIG_ROOT = Path(__file__).resolve().parent.parent / "configs/quantizer_followups"
IDS = ("s01", "r01", "r02", "r03")


def load_config(experiment_id: str) -> dict:
    if experiment_id not in IDS:
        raise ValueError(f"Unknown follow-up experiment: {experiment_id}")
    path = CONFIG_ROOT / f"{experiment_id}.yaml"
    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    if (config["id"] != experiment_id or config["model_id"] != QWEN3_MODEL_ID or
            config["model_revision"] != QWEN3_REVISION or config["bits"] != 2 or
            config["group_size"] != 128 or config["model_dtype"] != "float16" or
            config["evaluation_dtype"] != "float16"):
        raise ValueError(f"Invalid pinned experiment config: {path}")
    expected = GPTQ_COMMIT if experiment_id == "r03" else PR_COMMIT
    key = "gptqmodel_commit" if experiment_id == "r03" else "autoround_commit"
    if config[key] != expected or (experiment_id != "r03" and config.get("scale_dtype") != "fp16"):
        raise ValueError(f"Wrong upstream revision in {path}")
    config["config_sha256"] = hashlib.sha256(raw).hexdigest()
    return config


def audit_source(checkout: Path, experiment_id: str) -> dict:
    expected = GPTQ_COMMIT if experiment_id == "r03" else PR_COMMIT
    check_revision(checkout, expected)
    if experiment_id == "r03":
        files = {"config": "gptqmodel/quantization/config.py",
                 "qwen3": "gptqmodel/models/definitions/qwen3.py",
                 "export": "gptqmodel/models/auto.py",
                 "dequant": "gptqmodel/nn_modules/qlinear/torch.py",
                 "gptq": "gptqmodel/quantization/gptq.py",
                 "grid": "gptqmodel/quantization/quantizer.py"}
        required = {"config": "class QuantizeConfig", "qwen3": "class Qwen3QModel",
                    "export": 'format == "hf"', "dequant": "def dequantize_model",
                    "gptq": "blocksize=128", "grid": "def quantize(x, scale, zero, maxq"}
    else:
        files = {"rtn": "auto_round/algorithms/quantization/rtn/config.py",
                 "dispatch": "auto_round/data_type/utils.py",
                 "integer": "auto_round/data_type/int.py",
                 "neuqi": "auto_round/data_type/neuqi.py",
                 "v2_config": "auto_round/algorithms/quantization/sign_round/config.py",
                 "v2_quantizer": "auto_round/algorithms/quantization/sign_roundv2/quantizer.py"}
        required = {"rtn": "class OptimizedRTNConfig", "dispatch": "opt_rtn_int_sym_neuqi",
                    "integer": "def search_scales",
                    "neuqi": 'register_dtype("opt_rtn_int_asym")',
                    "v2_config": "class SignRoundV2Config",
                    "v2_quantizer": "class SignRoundV2Quantizer"}
    hashes = {}
    for key, relative in files.items():
        path = checkout / relative
        if required[key] not in path.read_text(encoding="utf-8"):
            raise ValueError(f"Pinned source contract changed: {path}")
        hashes[relative] = sha256(path)
    return {"revision": expected, "checkout": str(checkout.resolve()), "source_files": hashes,
            "scope": "source contract; resolved runtime and checkpoint coverage checked separately"}


def identity(config: dict, source: Path, checkout: Path) -> dict:
    return {"id": config["id"], "config_sha256": config["config_sha256"],
            "source": source_identity(source),
            "upstream_revision": GPTQ_COMMIT if config["id"] == "r03" else PR_COMMIT,
            "upstream_checkout": str(checkout.resolve()), "matgptq_commit": MATGPTQ_COMMIT}


def verify_original_checkpoint(current: dict, result: Path) -> None:
    j05_run = result.parents[1] / "rrq_w2_reproduction/j05/run.json"
    if not j05_run.is_file():
        raise ValueError(f"Prior J05 source identity unavailable: {j05_run}")
    recorded = json.loads(j05_run.read_text()).get("source_checkpoint")
    if not same_hf_contents(recorded, current):
        raise ValueError("Follow-up source differs from J05's pinned original checkpoint")


def reference_environment(checkout: Path, overlay: Path | None, python: str) -> dict[str, str]:
    env = source_environment(checkout, overlay)
    selected = Path(python).resolve().parent.parent
    current = Path(sys.executable).resolve().parent.parent
    if selected != current:
        env["CONDA_PREFIX"] = str(selected)
        current_bin = str(current / "bin")
        selected_bin = str(selected / "bin")
        env["PATH"] = os.pathsep.join([selected_bin] + [part for part in env.get("PATH", "").split(os.pathsep)
                                                       if part and Path(part) != Path(current_bin)])
        if env.get("LD_LIBRARY_PATH"):
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                part for part in env["LD_LIBRARY_PATH"].split(os.pathsep)
                if part and not Path(part).is_relative_to(current))
    return env


def _read_marker(path: Path, expected: dict) -> dict | None:
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    if record.get("status") != "completed_validated" or record.get("identity") != expected:
        raise ValueError(f"Existing marker conflicts with experiment identity: {path}")
    return record


def validate_resolved(config: dict, path: Path, checkout: Path) -> dict:
    record = json.loads(path.read_text())
    imported = Path(record["reference_module"]).resolve()
    if not imported.is_relative_to(checkout.resolve()):
        raise ValueError("Resolved quantizer was imported from the wrong checkout")
    if config["id"] == "r03":
        observed = record.get("config", {})
        expected_gptq = {"bits": "2", "group_size": "128", "sym": "True",
                         "desc_act": "False", "act_group_aware": "True",
                         "damp_percent": "0.05", "damp_auto_increment": "0.01",
                         "static_groups": "False", "true_sequential": "True",
                         "format": "gptq"}
        if (any(observed.get(key) != value for key, value in expected_gptq.items()) or
                record.get("backend") != "TORCH" or record.get("model_dtype") != "float16" or
                record.get("packed_decoder") != "CPU INT2 and official Torch non-Triton" or
                record.get("offload_mode") != "shared_job_owned" or
                record.get("calibration_sample_count") != 128 or
                record.get("materializer") !=
                "streamed source inventory, independent INT2 parity, official Torch FP16 QDQ" or
                record.get("packed_to_qdq_exact_projection_parity") is not True or
                record.get("packed_to_qdq_projection_count") != 252 or
                record.get("qdq_stream", {}).get("format") != "dense_fp16_qdq_streamed" or
                record.get("qdq_stream", {}).get("quantized_projection_count") != 252 or
                len(record.get("packed_to_qdq_hashes", {})) != 252):
            raise ValueError("GPTQ settings or packed-to-QDQ parity not verified")
    else:
        expected_class = "SignRoundV2Quantizer" if config["id"] == "s01" else "OptimizedRTNQuantizer"
        expected_config_class = "SignRoundV2Config" if config["id"] == "s01" else "OptimizedRTNConfig"
        settings = record.get("resolved_algorithm", {})
        iterations = settings.get("iters")
        # OptimizedRTNConfig deliberately has no ``iters`` attribute. Existing
        # R01/R02 resolved records therefore contain JSON null; the verified
        # zero-iteration dispatch below supplies the missing semantic value.
        if (config["id"] in ("r01", "r02") and iterations is None and
                record.get("resolved_config_class") == "OptimizedRTNConfig"):
            iterations = 0
        if (record.get("resolved_quantizer_class") != expected_class or
                record.get("resolved_config_class") != expected_config_class or
                iterations != config["iterations"] or
                settings.get("bits") != 2 or settings.get("group_size") != 128 or
                settings.get("sym") is not config["symmetric"] or
                settings.get("enable_neuqi") is not config["enable_neuqi"]):
            raise ValueError("AutoRound resolved configuration differs from pinned experiment")
        if config["id"] in ("r01", "r02"):
            expected = "opt_rtn_int_sym_neuqi" if config["id"] == "r01" else "opt_rtn_int_asym"
            if (record.get("selected_quant_function") != expected or
                    record.get("neuqi_grid") != {"coarse": 256, "fine": 64, "backend": "auto"} or
                    settings.get("disable_opt_rtn") is not False):
                raise ValueError("NeUQI optimized RTN was not selected")
    return record


def _grid(values: torch.Tensor, symmetric: bool) -> dict:
    if symmetric:
        return _w2_grid_check(values)
    unique = torch.unique(values.float())
    if unique.numel() > 4:
        raise ValueError("Asymmetric W2 group has more than four values")
    if unique.numel() < 3:
        return {"levels": int(unique.numel()), "grid_check": "underdetermined"}
    gaps = unique[1:] - unique[:-1]
    scale = float(gaps.min())
    # Missing integer levels can make observed gaps 2x or 3x the real scale.
    candidates = [scale / divisor for divisor in (1, 2, 3)]
    for candidate in candidates:
        steps = torch.round((unique - unique[0]) / candidate)
        residual = (unique - unique[0] - steps * candidate).abs()
        tolerance = 2 * torch.finfo(values.dtype).eps * float(unique.abs().max())
        if int(steps.max()) <= 3 and float(residual.max()) <= tolerance:
            return {"levels": int(unique.numel()), "scale_estimate": candidate,
                    "max_grid_residual": float(residual.max()), "tolerance": tolerance,
                    "zero_point": "not identifiable from fake QDQ alone"}
    raise ValueError("Asymmetric W2 group is not on a four-level affine grid")


def _check_all_w2_groups(weight: torch.Tensor, group_size: int = 128) -> int:
    """Check every row/group fits at most four affine levels, in bounded batches."""
    if weight.ndim != 2 or group_size < 2:
        raise ValueError("Expected a projection matrix and positive group size")
    rows, width = weight.shape
    count = 0
    tolerance_unit = torch.finfo(weight.dtype).eps
    full_width = width // group_size * group_size
    for row_start in range(0, rows, 64):
        block = weight[row_start:min(row_start + 64, rows)]
        batches = []
        if full_width:
            batches.append(block[:, :full_width].reshape(-1, group_size))
        if full_width < width:
            batches.append(block[:, full_width:])
        for group in batches:
            group = group.float()
            ordered = torch.sort(group, dim=1).values
            differences = ordered[:, 1:] - ordered[:, :-1]
            distinct = differences > 0
            levels = distinct.sum(dim=1) + 1
            if bool(torch.any(levels > 4)):
                raise ValueError(f"W2 group has more than four levels near row {row_start}")
            gaps = torch.where(distinct, differences, float("inf")).min(dim=1).values
            varying = torch.isfinite(gaps)
            if bool(varying.any()):
                values = ordered[varying]
                gap = gaps[varying].unsqueeze(1)
                relative = values - values[:, :1]
                tolerance = 4 * tolerance_unit * values.abs().amax(dim=1, keepdim=True) + 1e-7
                valid = torch.zeros(values.shape[0], dtype=torch.bool)
                for divisor in (1, 2, 3):
                    spacing = gap / divisor
                    codes = torch.round(relative / spacing)
                    residual = (relative - codes * spacing).abs().amax(dim=1, keepdim=True)
                    valid |= ((codes.amax(dim=1) <= 3) & (residual[:, 0] <= tolerance[:, 0]))
                if not bool(valid.all()):
                    raise ValueError(f"W2 group is not affine near row {row_start}")
            count += group.shape[0]
    return count


def validate_checkpoint(config: dict, source: Path, export: Path) -> dict:
    if config["id"] == "r03":
        checkpoint = export / "qdq"
        if not checkpoint.is_dir():
            raise ValueError("GPTQ QDQ export is missing")
        output_map = _weight_map(checkpoint)
        packed = export / "packed"
        packed_map = _weight_map(packed)
        packed_config = json.loads((packed / "config.json").read_text())
        qcfg = packed_config.get("quantization_config", {})
        if (qcfg.get("bits") != 2 or qcfg.get("group_size") != 128 or qcfg.get("sym") is not True):
            raise ValueError("GPTQ packed checkpoint has wrong bit, group or symmetry metadata")
        if not any(name.endswith(".qweight") for name in packed_map):
            raise ValueError("GPTQ packed checkpoint has no 2-bit code payload")
        packed_bytes = sum((packed / name).stat().st_size for name in set(packed_map.values()))
        packed_kinds = {}
        for name, shard in packed_map.items():
            kind = name.rsplit(".", 1)[-1]
            with safe_open(packed / shard, framework="pt", device="cpu") as handle:
                slice_ = handle.get_slice(name)
                elements = 1
                for size in slice_.get_shape():
                    elements *= size
                dtype_bytes = {"I32": 4, "I64": 8, "F16": 2, "BF16": 2,
                               "F32": 4, "U8": 1, "I8": 1}.get(slice_.get_dtype())
                if dtype_bytes is None:
                    raise ValueError(f"Unsupported packed GPTQ tensor dtype: {name}")
                packed_kinds[kind] = packed_kinds.get(kind, 0) + elements * dtype_bytes
        checkpoint_info = {"resolved_checkpoint_path": str(checkpoint.resolve()),
                           "packed_checkpoint_bytes": packed_bytes,
                           "packed_shard_sha256": {name: sha256(packed / name)
                                                   for name in sorted(set(packed_map.values()))},
                           "packed_tensor_bytes_by_kind": packed_kinds,
                           "packed_total_file_bytes": sum(path.stat().st_size for path in packed.rglob("*")
                                                          if path.is_file()),
                           "packed_config_sha256": sha256(packed / "config.json")}
    else:
        checkpoint_info = validate_output(export, source)
        checkpoint = Path(checkpoint_info["resolved_checkpoint_path"])
        output_map = _weight_map(checkpoint)
    source_map = _weight_map(source)
    if set(output_map) != set(source_map):
        raise ValueError("Checkpoint tensor inventory differs from original")
    if config["id"] == "r03":
        expected_bases = {name[:-7] for name in source_map if _eligible(name)}
        for suffix in ("qweight", "scales", "qzeros", "g_idx"):
            actual_bases = {name[:-(len(suffix) + 1)] for name in packed_map
                            if name.endswith("." + suffix)}
            if actual_bases != expected_bases:
                raise ValueError(f"GPTQ packed {suffix} coverage differs from 252 projections")
    source_cfg = json.loads((source / "config.json").read_text())
    output_cfg = json.loads((checkpoint / "config.json").read_text())
    for key in ("model_type", "hidden_size", "num_hidden_layers", "vocab_size",
                "intermediate_size", "num_attention_heads", "num_key_value_heads"):
        if source_cfg.get(key) != output_cfg.get(key):
            raise ValueError(f"Checkpoint architecture mismatch: {key}")
    projections, unchanged, dtype_counts, groups = [], [], {}, {}
    projected_elements = 0
    checked_groups = 0
    for name in sorted(source_map):
        original = _get(source, source_map, name)
        value = _get(checkpoint, output_map, name)
        if value.shape != original.shape or value.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"Wrong shape or dtype: {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite checkpoint tensor: {name}")
        dtype_counts[str(value.dtype)] = dtype_counts.get(str(value.dtype), 0) + 1
        if _eligible(name):
            if value.ndim != 2 or torch.equal(value, original.to(value.dtype)):
                raise ValueError(f"Unquantized eligible projection: {name}")
            if config["id"] == "r03":
                from types import SimpleNamespace
                from .quantizer_followups_worker import decode_gptq_w2_packed
                base = name[:-7]
                packed_module = SimpleNamespace(bits=2, group_size=128, sym=True,
                    pack_dtype_bits=32, in_features=value.shape[1], out_features=value.shape[0],
                    **{suffix: _get(packed, packed_map, f"{base}.{suffix}")
                       for suffix in ("qweight", "qzeros", "scales", "g_idx")})
                if value.dtype != torch.float16 or not torch.equal(value, decode_gptq_w2_packed(packed_module)):
                    raise ValueError(f"Packed GPTQ projection differs from dense QDQ: {name}")
            projections.append(name)
            projected_elements += value.numel()
            checked_groups += _check_all_w2_groups(value)
            kind = name.split(".")[-2]
            if kind not in groups:
                rows, width = value.shape
                samples = []
                for row in (0, rows // 2, rows - 1):
                    for start in (0, ((width - 1) // 128) * 128):
                        samples.append({"row": row, "start": start,
                                        **_grid(value[row, start:min(start + 128, width)],
                                                config["symmetric"])})
                groups[kind] = samples
        else:
            if not torch.equal(value, original.to(value.dtype)):
                raise ValueError(f"Excluded source tensor changed: {name}")
            unchanged.append(name)
    if len(projections) != 252 or len({name.split(".")[-2] for name in projections}) != 7:
        raise ValueError(f"Expected 252 Qwen3 projections, found {len(projections)}")
    shards = sorted(set(output_map.values()))
    checkpoint_info.update({"tensor_count": len(output_map), "quantized_projection_count": 252,
                            "quantized_projection_names": projections, "unchanged_tensor_count": len(unchanged),
                            "representative_groups": groups, "dtype_counts": dtype_counts,
                            "config_sha256": sha256(checkpoint / "config.json"),
                            "sidecar_sha256": {name: sha256(checkpoint / name) for name in sorted(SIDECARS)
                                               if name != "config.json" and (checkpoint / name).is_file()},
                            "index_sha256": sha256(checkpoint / "model.safetensors.index.json")
                            if (checkpoint / "model.safetensors.index.json").is_file() else None,
                            "shard_sha256": {name: sha256(checkpoint / name) for name in shards},
                            "qdq_checkpoint_bytes": sum((checkpoint / name).stat().st_size for name in shards),
                            "qdq_total_file_bytes": sum(path.stat().st_size for path in checkpoint.rglob("*")
                                                        if path.is_file()),
                            "quantized_projection_elements": projected_elements,
                            "checked_w2_row_groups": checked_groups,
                            "nominal_quantized_bits_per_weight": 2,
                            "scale_and_zero_point_bytes": "unavailable_in_fake_qdq" if config["id"] != "r03"
                            else "recorded_in_packed_checkpoint_files"})
    if config["id"] == "r03":
        checkpoint_info["packed_effective_total_bits_per_quantized_projection_weight"] = (
            checkpoint_info["packed_total_file_bytes"] * 8 / projected_elements)
        checkpoint_info["packed_scale_bytes"] = checkpoint_info["packed_tensor_bytes_by_kind"].get("scales", 0)
        checkpoint_info["packed_zero_bytes"] = checkpoint_info["packed_tensor_bytes_by_kind"].get("qzeros", 0)
        checkpoint_info["packed_code_bytes"] = checkpoint_info["packed_tensor_bytes_by_kind"].get("qweight", 0)
        if not all(checkpoint_info[key] for key in ("packed_scale_bytes", "packed_zero_bytes", "packed_code_bytes")):
            raise ValueError("GPTQ packed code, scale or zero metadata is missing")
    return checkpoint_info


def run_one(config: dict, source: Path, result: Path, checkout: Path, matgptq: Path,
            python: str, overlay: Path | None = None, *, stage: str = "all") -> dict:
    audit = audit_source(checkout, config["id"])
    check_revision(matgptq, MATGPTQ_COMMIT)
    ident = identity(config, source, checkout)
    verify_original_checkpoint(ident["source"], result)
    run_path = result / "run.json"
    if run_path.is_file():
        if json.loads(run_path.read_text()).get("identity") != ident:
            raise ValueError(f"Result directory has another identity: {result}")
    elif result.exists() and any(result.iterdir()):
        raise ValueError(f"Nonempty unowned result directory: {result}")
    else:
        result.mkdir(parents=True, exist_ok=True)
        atomic_json(run_path, {"status": "started", "identity": ident})
    atomic_json(result / "source_audit.json", audit)
    marker_path = result / "encoding_complete.json"
    export = result / "export"
    marker = _read_marker(marker_path, ident)
    if marker is None:
        process_path = result / "quantization_process.json"
        recovering = export.exists() and any(export.iterdir())
        if recovering:
            if not process_path.is_file() or json.loads(process_path.read_text()).get("returncode") != 0:
                raise ValueError("Unmarked export lacks recorded successful upstream exit; manual inspection required")
        else:
            if os.environ.get("FOLLOWUP_REUSE_EXPORT_ONLY") == "1":
                raise ValueError("Recovery-only mode found no existing export; quantization was not launched")
            config_path = result / "worker_config.json"
            atomic_json(config_path, config)
            env = reference_environment(checkout, overlay if config["id"] != "r03" else None, python)
            if overlay is not None and config["id"] != "r03":
                validate_overlay(overlay, python)
            modules = ["gptqmodel", "torch", "datasets", "transformers"] if config["id"] == "r03" else [
                "auto_round", "torch", "datasets", "transformers"]
            imports = probe(python, modules, env=env)
            atomic_json(result / "reference_imports.json", imports)
            upstream = "gptqmodel" if config["id"] == "r03" else "auto_round"
            imported = Path(imports["modules"][upstream]["file"]).resolve()
            if not imported.is_relative_to(checkout.resolve()):
                raise ValueError(f"Wrong upstream import: {imported}")
            command = [python, "-m", "lp_llm.quantizer_followups_worker", str(config_path),
                       str(source), str(export), str(result / "resolved_config.json")]
            execution = run_logged(command, result / "quantization.log", env=env)
            atomic_json(process_path, execution)
            if execution["returncode"]:
                raise RuntimeError(f"Quantization failed; see {result / 'quantization.log'}")
        if not (result / "resolved_config.json").is_file():
            raise ValueError("Upstream exit succeeded without resolved configuration")
        validate_resolved(config, result / "resolved_config.json", checkout)
        checkpoint = validate_checkpoint(config, source, export)
        atomic_json(result / "checkpoint_validation.json", checkpoint)
        marker = {"status": "completed_validated", "identity": ident,
                  "checkpoint_validation": checkpoint,
                  "quantization_log_sha256": sha256(result / "quantization.log"),
                  "recovered_after_successful_export": recovering}
        atomic_json(marker_path, marker)
    else:
        validate_resolved(config, result / "resolved_config.json", checkout)
        current = validate_checkpoint(config, source, export)
        if current != marker["checkpoint_validation"]:
            raise ValueError("Completed checkpoint has changed")
    if stage == "encode":
        return marker
    full_path = result / "full.json"
    existing = _read_marker(full_path, ident)
    if existing:
        if existing["checkpoint_validation"] != marker["checkpoint_validation"]:
            raise ValueError("Evaluation checkpoint identity changed")
        if sha256(result / "evaluation.log") != existing["evaluation_log_sha256"]:
            raise ValueError("Completed evaluation log has changed")
        return existing
    checkpoint_path = Path(marker["checkpoint_validation"]["resolved_checkpoint_path"])
    corpus = corpus_identity(checkpoint_path)
    if corpus["token_sha256"] != TOKEN_SHA256 or corpus["scored_tokens"] != 298862:
        raise ValueError("MatGPTQ corpus identity differs from PR04")
    evaluation_path = result / "evaluation_runtime.json"
    if evaluation_path.is_file():
        saved = json.loads(evaluation_path.read_text())
        if saved.get("identity") != ident or saved.get("checkpoint_validation") != marker["checkpoint_validation"]:
            raise ValueError("Previous evaluator run has another identity")
        if sha256(result / "evaluation.log") != saved["evaluation_log_sha256"]:
            raise ValueError("Previous evaluator log has changed")
        evaluation = saved["evaluation"]
    else:
        if (result / "evaluation.log").exists():
            raise ValueError("Unmarked evaluation log exists; inspect before rerun")
        evaluation = evaluate_matgptq(matgptq, checkpoint_path, result / "evaluation.log", python=sys.executable)
        atomic_json(evaluation_path, {"identity": ident,
                    "checkpoint_validation": marker["checkpoint_validation"],
                    "evaluation_log_sha256": sha256(result / "evaluation.log"),
                    "evaluation": evaluation})
    result_record = {"status": "completed_validated", "identity": ident,
                     "checkpoint_validation": marker["checkpoint_validation"],
                     "corpus": corpus, "evaluation": evaluation,
                     "evaluation_log_sha256": sha256(result / "evaluation.log")}
    atomic_json(full_path, result_record)
    return result_record


def aggregate(result_root: Path) -> dict:
    rows = []
    for experiment_id in IDS:
        result = result_root / experiment_id
        status = "not_started"
        if result.exists() and any(result.iterdir()):
            status = "incomplete_or_failed"
        if (result / "encoding_complete.json").is_file():
            status = "encoded" if not (result / "failure.json").is_file() else "failed_after_encoding"
        if (result / "full.json").is_file():
            full = json.loads((result / "full.json").read_text())
            status = "evaluated" if full.get("status") == "completed_validated" else "invalid_result"
        config = load_config(experiment_id)
        row = {"id": experiment_id, "method": config["method"], "symmetric": config["symmetric"],
               "group_size": 128, "status": status, "perplexity": None,
               "effective_packed_bpw": None}
        if status == "evaluated":
            full = json.loads((result / "full.json").read_text())
            row["perplexity"] = full["evaluation"]["perplexity"]
            row["checkpoint_bytes"] = full["checkpoint_validation"]["qdq_checkpoint_bytes"]
            row["effective_packed_bpw"] = full["checkpoint_validation"].get(
                "packed_effective_total_bits_per_quantized_projection_weight")
            row["quantization_runtime_seconds"] = json.loads(
                (result / "quantization_process.json").read_text()).get("runtime_seconds")
            row["evaluation_runtime_seconds"] = full["evaluation"]["execution"].get("runtime_seconds")
        rows.append(row)
    return {"prior_validated_references": [
        {"id": "j01", "method": "original_bf16", "perplexity": 7.01},
        {"id": "j04", "method": "official_plain_rtn", "perplexity": 9928.00},
        {"id": "j05", "method": "official_optimized_rtn", "perplexity": 65.62},
        {"id": "pr04", "method": "official_rrq_pr_base", "perplexity": 65.62}],
            "reference_note": "Prior AUS measurements as supplied; new runs are read from full.json only",
            "experiments": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", choices=IDS)
    parser.add_argument("--stage", choices=("all", "encode"), default="all")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--upstream-source", type=Path)
    parser.add_argument("--matgptq-source", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    root = args.result_root / "quantizer_followups"
    if args.aggregate:
        print(json.dumps(aggregate(root), indent=2))
        return
    if not all((args.experiment_id, args.source, args.upstream_source, args.matgptq_source)):
        parser.error("--experiment-id, --source, --upstream-source and --matgptq-source required")
    result = root / args.experiment_id
    try:
        record = run_one(load_config(args.experiment_id), args.source, result,
                         args.upstream_source, args.matgptq_source, args.python,
                         args.overlay, stage=args.stage)
        (result / "failure.json").unlink(missing_ok=True)
        print(json.dumps({"status": record["status"], "result": str(result)}, indent=2))
    except Exception as error:
        if (result / "run.json").is_file():
            atomic_json(result / "failure.json", {"status": "failed", "stage": args.stage,
                        "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
