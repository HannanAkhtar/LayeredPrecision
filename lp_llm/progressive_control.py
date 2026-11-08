"""Independent W4/W6/W8 AutoRound controls with source-exact exclusions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open

from .progressive_control_worker import BITS
from .progressive_lp import BRANCHES, _files
from .quantizer_followups import PR_COMMIT, TOKEN_SHA256, reference_environment
from .quantizer_followups_audit import audit
from .quantizer_followups_repair import _copy_tensor_bytes, _tensor_hash
from .rrq_autoround import validate_output
from .rrq_pr2308_compare import _eligible, _weight_map
from .rrq_reference_eval import check_revision, corpus_identity, run as evaluate_matgptq
from .rrq_process import run_logged
from .rrq_w2 import MATGPTQ_COMMIT
from .storage import atomic_json, sha256


def control_identity(branch: str, bits: int, source: Path, checkout: Path) -> dict:
    if branch not in BRANCHES or bits not in BITS:
        raise ValueError("Unknown monolithic control")
    check_revision(checkout, PR_COMMIT)
    return {"method": "signround_v2" if branch == "s01" else "optimized_rtn_asymmetric_neuqi",
            "branch": branch, "bits": bits, "group_size": 128,
            "source": _files(source), "upstream_revision": PR_COMMIT,
            "code_sha": os.environ.get("FOLLOWUP_CODE_SHA"),
            "calibration": {"dataset": "NeelNanda/pile-10k", "nsamples": 128,
                            "seqlen": 2048, "batch_size": 8, "seed": 42},
            "dtype": "float16_load_bf16_export_float16_evaluation"}


def validate_control(source: Path, export_root: Path, bits: int,
                     *, expected: dict | None = None) -> dict:
    info = validate_output(export_root, source)
    checkpoint = Path(info["resolved_checkpoint_path"])
    source_map, output_map = _weight_map(source), _weight_map(checkpoint)
    if set(source_map) != set(output_map):
        raise ValueError("Control tensor inventory differs from source")
    changed, excluded = sampled_groups = 0, 0, 0
    projection_hashes = {}
    for name in sorted(source_map):
        with safe_open(source / source_map[name], framework="pt", device="cpu") as handle:
            original = handle.get_tensor(name)
        with safe_open(checkpoint / output_map[name], framework="pt", device="cpu") as handle:
            value = handle.get_tensor(name)
        if value.shape != original.shape or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Invalid control tensor: {name}")
        if _eligible(name):
            if torch.equal(value, original.to(value.dtype)):
                raise ValueError(f"Unchanged control projection: {name}")
            if value.ndim != 2:
                raise ValueError(f"Control projection is not a matrix: {name}")
            for row in {0, value.shape[0] // 2, value.shape[0] - 1}:
                for start in {0, (value.shape[1] // 256) * 128,
                              ((value.shape[1] - 1) // 128) * 128}:
                    if start < value.shape[1]:
                        levels = torch.unique(value[row, start:start + 128]).numel()
                        if levels > 1 << bits:
                            raise ValueError(f"Control group exceeds W{bits} grid: {name}")
                        sampled_groups += 1
            changed += 1
            projection_hashes[name] = _tensor_hash(checkpoint / output_map[name], name)
        else:
            if not torch.equal(value, original):
                raise ValueError(f"Control excluded tensor changed: {name}")
            excluded += 1
    if changed != 252:
        raise ValueError(f"Control changed {changed} projections, expected 252")
    result = {"checkpoint": info, "quantized_projection_count": changed,
              "source_exact_excluded_tensor_count": excluded,
              "projection_tensor_sha256": projection_hashes,
              "sampled_group_grid_checks": sampled_groups,
              "nominal_projection_bits": bits,
              "actual_checkpoint_file_bytes": sum(path.stat().st_size for path in checkpoint.iterdir()
                                                  if path.is_file()),
              "compact_encoded_bytes": None,
              "storage_note": "official fake-QDQ export; no compact bitstream was produced"}
    if expected is not None and result != expected:
        raise ValueError("Completed control checkpoint changed")
    return result


def _repair_exclusions(source: Path, export_root: Path) -> dict:
    before = audit(source, export_root)
    if len(before["quantized_projection_names"]) != 252:
        raise ValueError("Control export omitted projections")
    if any(item["fp16_roundtrip_mismatches"] or
           item["source_dtype"] != item["export_dtype"] for item in before["excluded_tensors"]):
        raise ValueError("Excluded change is not explained by pinned FP16 load")
    checkpoint = Path(before["export"])
    source_map, output_map = _weight_map(source), _weight_map(checkpoint)
    projection_before = {name: _tensor_hash(checkpoint / output_map[name], name)
                         for name in before["quantized_projection_names"]}
    for item in before["excluded_tensors"]:
        if item["changed"]:
            _copy_tensor_bytes(source / source_map[item["name"]], item["name"],
                               checkpoint / output_map[item["name"]], item["name"])
    for name, digest in projection_before.items():
        if _tensor_hash(checkpoint / output_map[name], name) != digest:
            raise ValueError("Repair changed monolithic projection bytes")
    return {"restored_excluded_names": before["changed_excluded_names"],
            "quantized_projection_byte_identical": True,
            "raw_projection_tensor_sha256": projection_before,
            "fp16_load_roundtrip_explains_all_excluded_changes": True}


def _resolved_matches(path: Path, branch: str, bits: int, checkout: Path) -> None:
    record = json.loads(path.read_text())
    sign = branch == "s01"
    expected_calibration = {"dataset": "NeelNanda/pile-10k", "nsamples": 128,
                            "seqlen": 2048, "batch_size": 8, "seed": 42}
    resolved = record.get("resolved", {})
    if (record.get("bits") != bits or record.get("branch") != branch or
            record.get("group_size") != 128 or record.get("sym") is not sign or
            record.get("algorithm_class") !=
            ("SignRoundV2Config" if sign else "OptimizedRTNConfig") or
            record.get("quantizer_class") !=
            ("SignRoundV2Quantizer" if sign else "OptimizedRTNQuantizer") or
            record.get("selected_quant_function") !=
            (None if sign else "opt_rtn_int_asym") or
            record.get("function_name") !=
            (None if sign else "quant_tensor_opt_rtn_asym") or
            record.get("iterations") != (200 if sign else 0) or
            record.get("enable_neuqi") is not (not sign) or
            record.get("need_calib") is not True or
            record.get("data_type") != "int" or
            record.get("model_dtype") != "float16" or
            record.get("scale_dtype") != "fp16" or
            record.get("calibration") != expected_calibration or
            record.get("neuqi_grid") != (None if sign else {"coarse": 256, "fine": 64}) or
            resolved.get("bits") != bits or resolved.get("group_size") != 128 or
            resolved.get("sym") is not sign or
            bool(resolved.get("enable_neuqi", False)) is not (not sign) or
            resolved.get("iters") not in ((200,) if sign else (None, 0)) or
            not Path(record.get("reference_module", "")).resolve().is_relative_to(checkout.resolve())):
        raise ValueError("Control resolved recipe or source checkout differs")


def run(branch: str, bits: int, source: Path, result: Path, checkout: Path,
        matgptq: Path, *, python: str = sys.executable, stage: str = "evaluate") -> dict:
    check_revision(matgptq, MATGPTQ_COMMIT)
    ident = control_identity(branch, bits, source, checkout)
    run_file = result / "run.json"
    if run_file.is_file():
        if json.loads(run_file.read_text()).get("identity") != ident:
            raise ValueError("Control result identity differs")
    elif result.exists() and any(result.iterdir()):
        raise ValueError("Unowned control result directory")
    else:
        result.mkdir(parents=True)
        atomic_json(run_file, {"status": "started", "identity": ident})
    marker = result / "encoding_complete.json"
    export, partial = result / "export", result / "export.partial"
    try:
        if marker.is_file():
            if partial.exists():
                raise FileExistsError("Validated control has an unexpected partial export")
            encoded = json.loads(marker.read_text())
            if encoded.get("identity") != ident or encoded.get("status") != "completed_validated":
                raise ValueError("Control completion identity differs")
            validate_control(source, export, bits, expected=encoded["checkpoint_validation"])
        else:
            process = result / "quantization_process.json"
            repair_file = result / "excluded_repair.json"
            if not export.exists() and not partial.exists():
                if process.exists():
                    raise FileExistsError("Previous control process has no validated export; refusing requantization")
                config = {"branch": branch, "bits": bits}
                atomic_json(result / "worker_config.json", config)
                env = reference_environment(checkout, None, python)
                if branch == "r02":
                    env.update(AR_NEUQI_COARSE="256", AR_NEUQI_FINE="64", AR_NEUQI_BACKEND="auto")
                command = [python, "-m", "lp_llm.progressive_control_worker",
                           "--branch", branch, "--bits", str(bits), "--source", str(source),
                           "--export", str(partial), "--resolved", str(result / "resolved_config.json")]
                execution = run_logged(command, result / "quantization.log", env=env)
                atomic_json(process, execution)
                if execution["returncode"]:
                    raise RuntimeError("Pinned AutoRound control failed")
            if partial.exists():
                if (not process.is_file() or json.loads(process.read_text()).get("returncode") != 0 or
                        not (result / "quantization.log").is_file()):
                    raise FileExistsError("Incomplete control export has no successful process evidence")
                _resolved_matches(result / "resolved_config.json", branch, bits, checkout)
                validate_output(partial, source)
                if repair_file.is_file():
                    repair = json.loads(repair_file.read_text())
                else:
                    repair = _repair_exclusions(source, partial)
                    atomic_json(repair_file, repair)
                current = validate_control(source, partial, bits)
                if repair.get("raw_projection_tensor_sha256") != current["projection_tensor_sha256"]:
                    raise ValueError("Partial control projection hashes differ from pre-repair")
                os.replace(partial, export)
            if (not process.is_file() or json.loads(process.read_text()).get("returncode") != 0 or
                    not repair_file.is_file() or not (result / "quantization.log").is_file()):
                raise ValueError("Unmarked control export lacks successful process and repair evidence")
            _resolved_matches(result / "resolved_config.json", branch, bits, checkout)
            repair = json.loads(repair_file.read_text())
            # The resolved path changes after atomic rename; recompute identity.
            checkpoint = validate_control(source, export, bits)
            if (repair.get("raw_projection_tensor_sha256") !=
                    checkpoint["projection_tensor_sha256"] or
                    repair.get("quantized_projection_byte_identical") is not True):
                raise ValueError("Recovered control projection hashes differ from pre-repair")
            encoded = {"status": "completed_validated", "identity": ident,
                       "checkpoint_validation": checkpoint, "repair": repair,
                       "quantization_log_sha256": sha256(result / "quantization.log")}
            atomic_json(marker, encoded)
        (result / "failure.json").unlink(missing_ok=True)
        if stage == "encode":
            return encoded
        full_path = result / "full.json"
        if full_path.is_file():
            full = json.loads(full_path.read_text())
            if (full.get("identity") != ident or
                    full.get("checkpoint_validation") != encoded["checkpoint_validation"] or
                    sha256(result / "evaluation.log") != full.get("evaluation_log_sha256")):
                raise ValueError("Control evaluation identity differs")
            return full
        if (result / "evaluation.log").exists():
            raise FileExistsError("Unmarked control evaluation log requires inspection")
        checkpoint_path = Path(encoded["checkpoint_validation"]["checkpoint"]["resolved_checkpoint_path"])
        corpus = corpus_identity(checkpoint_path)
        if corpus["token_sha256"] != TOKEN_SHA256 or corpus["scored_tokens"] != 298862:
            raise ValueError("Control evaluation dataset differs")
        measurement = evaluate_matgptq(matgptq, checkpoint_path, result / "evaluation.log", python=python)
        full = {"status": "completed_validated", "identity": ident,
                "checkpoint_validation": encoded["checkpoint_validation"],
                "corpus": corpus, "evaluation": measurement,
                "evaluation_log_sha256": sha256(result / "evaluation.log")}
        atomic_json(full_path, full)
        (result / "failure.json").unlink(missing_ok=True)
        return full
    except Exception as error:
        atomic_json(result / "failure.json", {"status": "failed", "identity": ident,
                                                 "error_type": type(error).__name__,
                                                 "error": str(error)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=tuple(BRANCHES), required=True)
    parser.add_argument("--bits", type=int, choices=BITS, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--matgptq", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--stage", choices=("encode", "evaluate"), default="evaluate")
    args = parser.parse_args()
    value = run(args.branch, args.bits, args.source, args.result,
                args.checkout, args.matgptq, python=args.python, stage=args.stage)
    print(json.dumps({"status": value["status"], "result": str(args.result)}, indent=2))


if __name__ == "__main__":
    main()
