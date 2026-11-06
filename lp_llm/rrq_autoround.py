"""Invoke pinned official AutoRound end-to-end model quantization."""

from __future__ import annotations

import ast
import json
import math
import os
import subprocess
from pathlib import Path

import torch
from safetensors import safe_open

from .rrq_reference_eval import check_revision
from .rrq_process import run_logged
from .rrq_w2 import AUTOROUND_COMMIT
from .storage import atomic_json, sha256


def command(config: dict, source: Path, output: Path) -> list[str]:
    if config["kind"] != "autoround":
        raise ValueError("Expected official AutoRound configuration")
    executable = "auto-round-rtn" if config["mode"] == "plain" else "auto-round-opt-rtn"
    executable = str(Path(os.environ.get("RRQ_AUTOROUND_BIN_DIR", "")) / executable)
    scheme = "W2A16" if config["group_size"] == 128 else f"W2A16G{config['group_size']}"
    return [executable, "--model", str(source), "--scheme", scheme,
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


def validate_output(output: Path) -> dict:
    config = output / "config.json"
    index = output / "model.safetensors.index.json"
    if not index.exists() and (output / "model.safetensors").is_file():
        with safe_open(output / "model.safetensors", framework="pt", device="cpu") as handle:
            names = list(handle.keys())
            widths = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1,
                      "I16": 2, "U16": 2, "I32": 4, "U32": 4, "I64": 8, "U64": 8}
            total = sum(math.prod(handle.get_slice(name).get_shape()) *
                        widths[str(handle.get_slice(name).get_dtype())] for name in names)
        atomic_json(index, {"metadata": {"total_size": total},
                            "weight_map": {name: "model.safetensors" for name in names}})
    if not config.is_file() or not index.is_file():
        raise ValueError(f"Official AutoRound did not produce indexed HF QDQ checkpoint: {output}")
    mapping = json.loads(index.read_text()).get("weight_map", {})
    if not mapping or not all((output / shard).is_file() for shard in set(mapping.values())):
        raise ValueError("Incomplete official AutoRound QDQ checkpoint")
    dtypes = set()
    for shard in sorted(set(mapping.values())):
        with safe_open(output / shard, framework="pt", device="cpu") as handle:
            assigned = {name for name, filename in mapping.items() if filename == shard}
            if set(handle.keys()) != assigned:
                raise ValueError(f"AutoRound Safetensors index differs from {shard}")
            dtypes.update(str(handle.get_slice(name).get_dtype()) for name in assigned)
    return {"config_sha256": sha256(config), "index_sha256": sha256(index),
            "shard_sha256": {s: sha256(output / s) for s in sorted(set(mapping.values()))},
            "tensor_names": sorted(mapping), "checkpoint_dtypes": sorted(dtypes),
            "qdq_checkpoint_safetensors_bytes": sum((output / s).stat().st_size for s in set(mapping.values())),
            "packed_representation_bytes": None}


def compare_coverage(source: Path, checkpoint: Path) -> dict:
    source_map = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    output_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    if set(source_map) != set(output_map):
        raise ValueError("Official output has a different tensor inventory")
    changed, unchanged = [], []
    for name in sorted(source_map):
        with safe_open(source / source_map[name], framework="pt", device="cpu") as handle:
            original = handle.get_tensor(name).to(torch.float16)
        with safe_open(checkpoint / output_map[name], framework="pt", device="cpu") as handle:
            materialized = handle.get_tensor(name)
        if not torch.isfinite(materialized).all():
            raise ValueError(f"Nonfinite official output tensor: {name}")
        (unchanged if torch.equal(original, materialized.to(torch.float16)) else changed).append(name)
    return {"changed_tensor_names": changed, "unchanged_tensor_names": unchanged,
            "coverage_method": "exact_FP16_comparison_to_original"}


def run(config: dict, source: Path, output: Path, checkout: Path, log: Path,
        *, revision: str = AUTOROUND_COMMIT) -> dict:
    check_revision(checkout, revision)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite existing official output: {output}")
    upstream_recipe = verify_upstream_recipe(checkout, config)
    cmd = command(config, source, output)
    execution = run_logged(cmd, log)
    if execution["returncode"]:
        raise RuntimeError(f"Official AutoRound failed ({execution['returncode']}); see {log}")
    checkpoint = validate_output(output)
    weight_change = compare_coverage(source, output)
    if not weight_change["changed_tensor_names"]:
        raise ValueError("Official AutoRound output changed no weights; refusing to label it quantized")
    source_config = json.loads((source / "config.json").read_text())
    return {"implementation": "official_autoround_cli", "revision": revision,
            "reference_class": "post_preprint_current_official" if revision == AUTOROUND_COMMIT
                               else "preprint_era_candidate_not_confirmed_RRQ_revision",
            "command": cmd, "log": str(log), "execution": execution, "checkpoint": checkpoint,
            "upstream_recipe": upstream_recipe,
            "resolved_recipe": config["quantizer_configuration"],
            "source_config_torch_dtype": source_config.get("torch_dtype"),
            "configured_module_selection": configured_modules(output),
            "weight_change_diagnostic": weight_change}
