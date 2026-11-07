"""Invoke pinned official AutoRound end-to-end model quantization."""

from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path

import torch
from safetensors import safe_open

from .rrq_reference_eval import check_revision
from .rrq_overlay import validate_overlay
from .rrq_process import run_logged
from .rrq_runtime import probe, source_environment
from .rrq_w2 import AUTOROUND_COMMIT
from .storage import sha256


def command(config: dict, source: Path, output: Path, *, python: str | None = None) -> list[str]:
    if config["kind"] != "autoround":
        raise ValueError("Expected official AutoRound configuration")
    function = "run_rtn" if config["mode"] == "plain" else "run_opt_rtn"
    entry = f"from auto_round.cli.main import {function}; {function}()"
    scheme = "W2A16" if config["group_size"] == 128 else f"W2A16G{config['group_size']}"
    return [python or sys.executable, "-c", entry, "--model", str(source), "--scheme", scheme,
            "--iters", "0", "--nsamples", "1" if config["mode"] == "plain" else "128",
            "--model_dtype", "float16",
            "--format", "fake", "--output_dir", str(output)]


def verify_upstream_recipe(checkout: Path, config: dict) -> dict:
    """Fail if a selected checkout's CLI recipes or dispatch differ from the audited pins."""
    source = (checkout / "auto_round/cli/main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    recipes = next((ast.literal_eval(node.value) for node in tree.body
                    if isinstance(node, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == "RECIPES" for target in node.targets)), None)
    if not isinstance(recipes, dict):
        raise ValueError("Upstream AutoRound RECIPES declaration is unavailable")
    name = "rtn" if config["mode"] == "plain" else "opt_rtn"
    entry = "run_rtn" if name == "rtn" else "run_opt_rtn"
    function = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == entry), None)
    dispatched = bool(function and any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "start" and
        node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == name
        for statement in function.body for node in ast.walk(statement)))
    recipe = recipes.get(name)
    expected = {"iters": 0, "nsamples": 1 if name == "rtn" else 128,
                "disable_opt_rtn": name == "rtn", "seqlen": 2048}
    if not dispatched or not isinstance(recipe, dict) or any(recipe.get(key) != value
                                                             for key, value in expected.items()):
        raise ValueError(f"Upstream AutoRound {name} recipe/dispatch differs from audited revision")
    return {"entry_point": entry, "recipe_name": name, "recipe_defaults": recipe}


def configured_modules(output: Path) -> dict:
    """Read explicit upstream module selection, if the fake export retained it."""
    config = json.loads((output / "config.json").read_text())
    quant = config.get("quantization_config")
    layer = quant.get("layer_config") if isinstance(quant, dict) else None
    if not isinstance(layer, dict):
        return {"source": "not_exported_by_upstream_fake_checkpoint",
                "selected_module_names": None, "excluded_module_names": None,
                "selection_policy": "upstream_default_no_ignore_layers_no_lm_head"}
    selected = sorted(name for name, settings in layer.items()
                      if isinstance(settings, dict) and settings.get("bits", 16) < 16)
    excluded = sorted(name for name in layer if name not in selected)
    return {"source": "checkpoint_quantization_config.layer_config",
            "selected_module_names": selected, "excluded_module_names": excluded,
            "selection_policy": "explicit_upstream_layer_config"}


def discover_checkpoint(export_root: Path) -> Path:
    """Resolve exactly one direct or nested indexed export without changing its files."""
    export_root = Path(export_root).resolve()
    if not export_root.is_dir():
        raise ValueError(f"Official AutoRound export root is absent: {export_root}")
    candidates = {path.parent for path in export_root.rglob("*") if path.is_file() and
                  (path.name in {"config.json", "model.safetensors.index.json"} or
                   path.name.endswith(".safetensors"))}
    if len(candidates) != 1:
        raise ValueError(f"Expected one AutoRound checkpoint directory, found {len(candidates)}: {export_root}")
    checkpoint = candidates.pop().resolve()
    if not checkpoint.is_relative_to(export_root):
        raise ValueError(f"AutoRound checkpoint escapes export root: {checkpoint}")
    return checkpoint


def validate_output(export_root: Path, source: Path | None = None) -> dict:
    checkpoint = discover_checkpoint(export_root)
    config, index = checkpoint / "config.json", checkpoint / "model.safetensors.index.json"
    if not config.is_file() or not index.is_file():
        raise ValueError(f"Incomplete indexed Hugging Face QDQ checkpoint: {checkpoint}")
    model_config = json.loads(config.read_text())
    if not isinstance(model_config, dict) or not isinstance(model_config.get("model_type"), str):
        raise ValueError(f"Invalid model configuration: {config}")
    indexed = json.loads(index.read_text())
    mapping = indexed.get("weight_map")
    if not isinstance(mapping, dict) or not mapping or not all(
            isinstance(name, str) and name and isinstance(shard, str) and
            shard.endswith(".safetensors") and Path(shard).name == shard
            for name, shard in mapping.items()):
        raise ValueError(f"Invalid AutoRound Safetensors index: {index}")
    shards = set(mapping.values())
    if {path.name for path in checkpoint.glob("*.safetensors")} != shards or not all(
            (checkpoint / shard).is_file() for shard in shards):
        raise ValueError(f"Incomplete or unindexed AutoRound shards: {checkpoint}")
    source_shapes = None
    if source is not None:
        source = Path(source)
        source_config = json.loads((source / "config.json").read_text())
        for key in ("model_type", "hidden_size", "num_hidden_layers", "vocab_size",
                    "intermediate_size", "num_attention_heads", "num_key_value_heads"):
            if key in source_config and model_config.get(key) != source_config[key]:
                raise ValueError(f"AutoRound config differs from original model at {key}")
        source_map = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
        if set(mapping) != set(source_map):
            raise ValueError("Official output has a different tensor inventory")
        source_shapes = {}
        for shard in sorted(set(source_map.values())):
            with safe_open(source / shard, framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    source_shapes[name] = tuple(handle.get_slice(name).get_shape())
    total_bytes = 0
    for shard in sorted(shards):
        with safe_open(checkpoint / shard, framework="pt", device="cpu") as handle:
            assigned = {name for name, filename in mapping.items() if filename == shard}
            if set(handle.keys()) != assigned:
                raise ValueError(f"AutoRound Safetensors index differs from {shard}")
            for name in assigned:
                tensor = handle.get_tensor(name)
                if tensor.dtype != torch.float16 or not torch.isfinite(tensor).all():
                    raise ValueError(f"Non-FP16 or nonfinite AutoRound tensor: {name}")
                if source_shapes is not None and tuple(tensor.shape) != source_shapes[name]:
                    raise ValueError(f"AutoRound tensor shape differs from original: {name}")
                total_bytes += tensor.numel() * tensor.element_size()
    declared_size = indexed.get("metadata", {}).get("total_size")
    if declared_size is not None and declared_size != total_bytes:
        raise ValueError("AutoRound index total_size differs from tensor inventory")
    return {"resolved_checkpoint_path": str(checkpoint),
            "config_sha256": sha256(config), "index_sha256": sha256(index),
            "shard_sha256": {s: sha256(checkpoint / s) for s in sorted(shards)},
            "tensor_names": sorted(mapping), "checkpoint_dtypes": ["F16"],
            "tensor_payload_bytes": total_bytes,
            "qdq_checkpoint_safetensors_bytes": sum((checkpoint / s).stat().st_size for s in shards),
            "packed_representation_bytes": None}


def _w2_grid_check(values: torch.Tensor) -> dict:
    """Check that an FP16 QDQ group lies on one signed two-bit scale grid."""
    values = values.float()
    nonzero = torch.unique(values[values != 0].abs())
    if not nonzero.numel():
        return {"scale": 0.0, "max_grid_residual": 0.0, "code_range": [-2, 1]}
    tolerance = max(1e-7, 0.002 * float(nonzero.max()))
    best = None
    for scale in (float(value) / divisor for value in nonzero for divisor in (1, 2)):
        if scale <= 0:
            continue
        for low, high in ((-2, 1), (-1, 2)):
            codes = torch.clamp(torch.round(values / scale), low, high)
            error = float(torch.max(torch.abs(values - codes * scale)))
            if best is None or error < best[0]:
                best = (error, scale, low, high)
    assert best is not None
    if best[0] > tolerance:
        raise ValueError(f"AutoRound group is not on a signed W2 scale grid: residual={best[0]}")
    return {"scale": best[1], "max_grid_residual": best[0], "code_range": [best[2], best[3]]}


def compare_coverage(source: Path, checkpoint: Path, *, group_size: int | None = None) -> dict:
    source_map = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    output_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    if set(source_map) != set(output_map):
        raise ValueError("Official output has a different tensor inventory")
    changed, unchanged = [], []
    projections = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    eligible_seen, eligible_changed = set(), set()
    group_checks = {}
    for name in sorted(source_map):
        with safe_open(source / source_map[name], framework="pt", device="cpu") as handle:
            original = handle.get_tensor(name).to(torch.float16)
        with safe_open(checkpoint / output_map[name], framework="pt", device="cpu") as handle:
            materialized = handle.get_tensor(name)
        if original.shape != materialized.shape or not torch.isfinite(materialized).all():
            raise ValueError(f"Nonfinite official output tensor: {name}")
        different = not torch.equal(original, materialized.to(torch.float16))
        (changed if different else unchanged).append(name)
        projection = next((part for part in projections if name.endswith(f".{part}.weight")), None)
        if projection is not None:
            eligible_seen.add(projection)
            if different:
                eligible_changed.add(projection)
            if group_size is not None and projection not in group_checks:
                if materialized.ndim != 2:
                    raise ValueError(f"Expected matrix for grouped AutoRound weight: {name}")
                rows, width = materialized.shape
                starts = sorted({0, (max(1, math.ceil(width / group_size)) // 2) * group_size,
                                 ((width - 1) // group_size) * group_size})
                starts = [start for start in starts if start < width]
                samples = []
                for row in sorted({0, rows // 2, rows - 1}):
                    for start in starts:
                        end = min(start + group_size, width)
                        original_group = original[row, start:end]
                        decoded_group = materialized[row, start:end]
                        levels = int(torch.unique(decoded_group).numel())
                        if levels > 4:
                            raise ValueError(f"AutoRound W2 group has {levels} levels: {name} row={row} start={start}")
                        grid = _w2_grid_check(decoded_group)
                        samples.append({"row": row, "start": start, "length": end - start,
                                        "levels": levels, **grid,
                                        "changed": int(torch.count_nonzero(original_group != decoded_group)),
                                        "mse": float(torch.mean((original_group.float() -
                                                                 decoded_group.float()) ** 2))})
                group_checks[projection] = {"tensor": name, "samples": samples}
    if group_size is not None and (not eligible_seen or eligible_seen != eligible_changed or
                                   any(not any(sample["changed"] for sample in value["samples"])
                                       for value in group_checks.values())):
        raise ValueError("Eligible AutoRound projection weights lack observed quantization differences")
    return {"changed_tensor_names": changed, "unchanged_tensor_names": unchanged,
            "eligible_projection_types": sorted(eligible_seen),
            "changed_eligible_projection_types": sorted(eligible_changed),
            "representative_group_checks": group_checks,
            "coverage_method": "exact_FP16_comparison_to_original_not_module_selection_metadata"}


def _inspect_export(config: dict, source: Path, export_root: Path) -> tuple[Path, dict, dict]:
    checkpoint = validate_output(export_root, source)
    resolved = Path(checkpoint["resolved_checkpoint_path"])
    weight_change = compare_coverage(source, resolved, group_size=config["group_size"])
    return resolved, checkpoint, weight_change


def _recovery_log(log: Path, config: dict, source: Path, checkpoint: Path) -> dict:
    if not log.is_file():
        raise ValueError(f"AutoRound log is absent; cannot recover export: {log}")
    content = log.read_text(encoding="utf-8", errors="replace")
    if (not content.strip() or "start to quantize" not in content or
            str(source).rstrip("/") not in content or "Traceback (most recent call last)" in content):
        raise ValueError(f"AutoRound log lacks the expected model start or contains a traceback: {log}")
    if config["mode"] == "optimized":
        if "OptimizedRTNQuantizer" not in content or "NeelNanda/pile-10k" not in content:
            raise ValueError(f"AutoRound log lacks J05 optimized RTN/calibration evidence: {log}")
    elif "OptimizedRTNQuantizer" in content:
        raise ValueError(f"Plain RTN recovery log contains optimized RTN evidence: {log}")
    if (checkpoint.name != Path(source).name + f"-w2g{config['group_size']}" and
            checkpoint.resolve() != (log.parent / "checkpoint").resolve()):
        raise ValueError(f"AutoRound export directory conflicts with W2G{config['group_size']} CLI: {checkpoint}")
    return {"sha256": sha256(log), "model_start_recorded": True,
            "optimized_rtn_recorded": config["mode"] == "optimized",
            "pile_10k_recorded": "NeelNanda/pile-10k" in content,
            "upstream_exit_code": None,
            "exit_code_note": "Subprocess exited before the old runner failed, but its code was not persisted"}


def recover(config: dict, source: Path, export_root: Path, checkout: Path, log: Path,
            *, revision: str = AUTOROUND_COMMIT) -> dict:
    """Validate an existing export and log; never call the upstream quantizer."""
    check_revision(checkout, revision)
    upstream_recipe = verify_upstream_recipe(checkout, config)
    resolved, checkpoint, weight_change = _inspect_export(config, source, export_root)
    log_evidence = _recovery_log(log, config, source, resolved)
    expected_command = command(config, source, export_root)
    source_config = json.loads((source / "config.json").read_text())
    return {"implementation": "official_autoround_cli", "revision": revision,
            "reference_class": "post_preprint_current_official" if revision == AUTOROUND_COMMIT
                               else "preprint_era_candidate_not_confirmed_RRQ_revision",
            "recovered_from_existing_export": True,
            "recovery_provenance": {
                "status": "validated_previously_exported_checkpoint",
                "log": str(log), "log_evidence": log_evidence,
                "expected_cli_command": expected_command,
                "cli_verification": "run.json identity, pinned recipe, log mode/model and W2 group export name; "
                                    "original subprocess argv and exit code were not persisted"},
            "checkpoint_path": str(resolved), "checkpoint": checkpoint,
            "source_checkout": str(checkout.resolve()),
            "upstream_recipe": upstream_recipe,
            "resolved_recipe": config["quantizer_configuration"],
            "source_model_revision": config["model_revision"],
            "source_config_torch_dtype": source_config.get("torch_dtype"),
            "configured_module_selection": configured_modules(resolved),
            "weight_change_diagnostic": weight_change}


def run(config: dict, source: Path, output: Path, checkout: Path, log: Path,
        *, revision: str = AUTOROUND_COMMIT, python: str | None = None,
        overlay: Path | None = None) -> dict:
    check_revision(checkout, revision)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite existing official output: {output}")
    upstream_recipe = verify_upstream_recipe(checkout, config)
    selected_overlay = overlay if overlay is not None else checkout.parent / "extra_deps"
    env = source_environment(checkout, selected_overlay)
    cmd = command(config, source, output, python=python)
    if selected_overlay.exists():
        validate_overlay(selected_overlay, cmd[0])
    runtime = probe(cmd[0], ["auto_round", "auto_round.cli.main", "pydantic", "cpuinfo",
                             "filelock", "requests", "torch", "transformers", "datasets",
                             "accelerate", "numpy"], env=env)
    imported = runtime["modules"]["auto_round.cli.main"]["file"]
    if not imported or not Path(imported).resolve().is_relative_to(checkout.resolve()):
        raise RuntimeError(f"AutoRound import resolved outside pinned checkout: {imported}")
    execution = run_logged(cmd, log, env=env)
    if execution["returncode"]:
        raise RuntimeError(f"Official AutoRound failed ({execution['returncode']}); see {log}")
    resolved, checkpoint, weight_change = _inspect_export(config, source, output)
    source_config = json.loads((source / "config.json").read_text())
    return {"implementation": "official_autoround_cli", "revision": revision,
            "reference_class": "post_preprint_current_official" if revision == AUTOROUND_COMMIT
                               else "preprint_era_candidate_not_confirmed_RRQ_revision",
            "command": cmd, "log": str(log), "execution": execution, "checkpoint": checkpoint,
            "checkpoint_path": str(resolved), "recovered_from_existing_export": False,
            "source_checkout": str(checkout.resolve()), "dependency_overlay": str(selected_overlay.resolve()),
            "runtime_imports": runtime,
            "upstream_recipe": upstream_recipe,
            "resolved_recipe": config["quantizer_configuration"],
            "source_config_torch_dtype": source_config.get("torch_dtype"),
            "configured_module_selection": configured_modules(resolved),
            "weight_change_diagnostic": weight_change}
