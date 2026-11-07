"""Validate PR #2308 packed base, materialize QDQ, and compare to J05."""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .evaluate import _identity
from .rrq_reference_eval import check_revision, corpus_identity, run as evaluate_matgptq
from .rrq_w2 import MATGPTQ_COMMIT
from .storage import SIDECARS, atomic_json, sha256


PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def same_hf_contents(recorded: dict, current: dict) -> bool:
    """Compare immutable model files across runs without conflating package versions."""
    fields = ("checkpoint", "config_sha256", "index_sha256", "shard_sha256", "sidecar_sha256")
    return isinstance(recorded, dict) and all(recorded.get(key) == current.get(key) for key in fields)


def _weight_map(directory: Path) -> dict[str, str]:
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text()).get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"Invalid Safetensors index: {index}")
    elif (directory / "model.safetensors").is_file():
        with safe_open(directory / "model.safetensors", framework="pt", device="cpu") as handle:
            mapping = {name: "model.safetensors" for name in handle.keys()}
    else:
        raise ValueError(f"No Safetensors checkpoint: {directory}")
    if not all(isinstance(k, str) and isinstance(v, str) and Path(v).name == v and
               v.endswith(".safetensors") for k, v in mapping.items()):
        raise ValueError(f"Invalid Safetensors paths: {directory}")
    if {p.name for p in directory.glob("*.safetensors")} != set(mapping.values()):
        raise ValueError(f"Missing or unindexed shards: {directory}")
    for shard in set(mapping.values()):
        with safe_open(directory / shard, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {n for n, s in mapping.items() if s == shard}:
                raise ValueError(f"Safetensors tensor inventory mismatch: {shard}")
    return mapping


def _get(directory: Path, mapping: dict[str, str], name: str) -> torch.Tensor:
    with safe_open(directory / mapping[name], framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def _eligible(name: str) -> bool:
    return (name.startswith("model.layers.") and name.endswith(".weight") and
            any(name.endswith(f".{p}.weight") for p in PROJECTIONS))


def unpack_int2(qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor,
                *, in_features: int, out_features: int, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode qlinear_torch_zp's INT2 axis and GPTQ (zero-point minus one) grid."""
    if (qweight.dtype != torch.int32 or qzeros.dtype != torch.int32 or
            scales.dtype not in (torch.float16, torch.bfloat16, torch.float32) or
            in_features % 16 or out_features % 16 or in_features % group_size):
        raise ValueError("Unsupported or incomplete official INT2 layout")
    groups = in_features // group_size
    if (tuple(qweight.shape) != (in_features // 16, out_features) or
            tuple(scales.shape) != (groups, out_features) or
            tuple(qzeros.shape) != (groups, out_features // 16)):
        raise ValueError("Packed INT2 tensor shapes do not match the original matrix")
    shifts = (torch.arange(16, dtype=torch.int32) * 2)
    codes = ((qweight[:, None, :] >> shifts[None, :, None]) & 3).reshape(in_features, out_features)
    zp = ((qzeros[:, :, None] >> shifts[None, None, :]) & 3).reshape(groups, out_features) + 1
    # Official qlinear_torch_zp._dequantize uses scales in their stored precision.
    # Float32 comparison avoids quantization of the multiply itself.
    decoded = (codes.float() - zp.repeat_interleave(group_size, 0).float()) * \
        scales.repeat_interleave(group_size, 0).float()
    if not torch.isfinite(decoded).all():
        raise ValueError("Nonfinite official packed INT2 weight")
    return decoded.T.contiguous(), codes.T.contiguous()


def validate_packed_base(base: Path, source: Path) -> dict:
    base, source = Path(base), Path(source)
    config_path = base / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Packed base config absent: {base}")
    cfg = json.loads(config_path.read_text())
    source_cfg = json.loads((source / "config.json").read_text())
    for field in ("model_type", "hidden_size", "num_hidden_layers", "vocab_size",
                  "intermediate_size", "num_attention_heads", "num_key_value_heads"):
        if cfg.get(field) != source_cfg.get(field):
            raise ValueError(f"Packed base model differs at {field}")
    quant = cfg.get("quantization_config", {})
    if quant.get("bits") != 2 or quant.get("group_size") != 128 or quant.get("sym") is not True:
        raise ValueError("Packed base is not symmetric W2G128")
    source_map, base_map = _weight_map(source), _weight_map(base)
    eligible = sorted(n for n in source_map if _eligible(n))
    if not eligible:
        raise ValueError("No Qwen3 projection weights found")
    expected = (set(source_map) - set(eligible)) | {
        f"{n[:-7]}.{field}" for n in eligible for field in ("qweight", "scales", "qzeros")}
    if set(base_map) != expected:
        raise ValueError(f"Packed base module coverage mismatch: missing={len(expected-set(base_map))}, "
                         f"extra={len(set(base_map)-expected)}")
    module_shapes = {}
    for name in eligible:
        original_shape = tuple(_get(source, source_map, name).shape)
        if len(original_shape) != 2:
            raise ValueError(f"Expected projection matrix: {name}")
        out_features, in_features = original_shape
        if in_features % 128 or out_features % 16:
            raise ValueError(f"Unsupported packed projection dimensions: {name}")
        module_shapes[name[:-7]] = [out_features, in_features]
        for field, shape in (("qweight", (in_features // 16, out_features)),
                             ("scales", (in_features // 128, out_features)),
                             ("qzeros", (in_features // 128, out_features // 16))):
            if tuple(_get(base, base_map, f"{name[:-7]}.{field}").shape) != shape:
                raise ValueError(f"Packed shape mismatch: {name[:-7]}.{field}")
    dtype_counts: dict[str, int] = {}
    scale_dtype_counts: dict[str, int] = {}
    nonquant_dtype_counts: dict[str, int] = {}
    nonquant_changed_names = []
    for name in sorted(base_map):
        tensor = _get(base, base_map, name)
        dtype_counts[str(tensor.dtype)] = dtype_counts.get(str(tensor.dtype), 0) + 1
        if name.endswith(".scales"):
            scale_dtype_counts[str(tensor.dtype)] = scale_dtype_counts.get(str(tensor.dtype), 0) + 1
        elif name in source_map:
            nonquant_dtype_counts[str(tensor.dtype)] = nonquant_dtype_counts.get(str(tensor.dtype), 0) + 1
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite packed base tensor: {name}")
        if name.endswith((".qweight", ".qzeros")) and tensor.dtype != torch.int32:
            raise ValueError(f"Wrong packed code dtype: {name}")
        if name.endswith(".scales") and tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"Wrong packed scale dtype: {name}")
        if name in source_map:
            original = _get(source, source_map, name)
            if tensor.shape != original.shape:
                raise ValueError(f"Nonquant tensor changed shape: {name}")
            if not torch.equal(tensor, original):
                nonquant_changed_names.append(name)
    # Decode one first, middle and last eligible matrix before accepting marker.
    for name in (eligible[0], eligible[len(eligible) // 2], eligible[-1]):
        prefix = name[:-7]
        shape = _get(source, source_map, name).shape
        unpack_int2(*(_get(base, base_map, f"{prefix}.{part}") for part in
                      ("qweight", "scales", "qzeros")), in_features=shape[1], out_features=shape[0])
    return {"path": str(base.resolve()), "config_sha256": sha256(config_path),
            "shards": {s: sha256(base / s) for s in sorted(set(base_map.values()))},
            "safetensors_disk_bytes": sum((base / s).stat().st_size for s in set(base_map.values())),
            "sidecar_sha256": {name: sha256(base / name) for name in sorted(SIDECARS - {"config.json"})
                               if (base / name).is_file()},
            "index_sha256": sha256(base / "model.safetensors.index.json") if
            (base / "model.safetensors.index.json").is_file() else None,
            "module_names": [n[:-7] for n in eligible], "module_shapes": module_shapes,
            "quantized_module_count": len(eligible),
            "nonquant_tensor_count": len(source_map) - len(eligible),
            "nonquant_changed_names": nonquant_changed_names,
            "dtype_counts": dtype_counts, "scale_dtype_counts": scale_dtype_counts,
            "nonquant_dtype_counts": nonquant_dtype_counts, "packing": "qlinear_torch_zp_INT2"}


def validate_residual(residual: Path, base_info: dict) -> dict:
    """Require the one official packed residual plane promised by PR03."""
    cfg_path = residual / "config.json"
    if not cfg_path.is_file():
        raise ValueError("RRQ residual config is missing")
    cfg = json.loads(cfg_path.read_text())
    quant = cfg.get("quantization_config", {})
    if (quant.get("packing_format") != "auto_round:rrq" or
            quant.get("total_planes") != 2 or quant.get("group_size") != 128 or
            quant.get("sym") is not True):
        raise ValueError("RRQ residual configuration does not match PR03")
    mapping = _weight_map(residual)
    expected = {f"{module}.{part}_1" for module in base_info["module_names"]
                for part in ("qweight", "scales", "qzeros")}
    if set(mapping) != expected:
        raise ValueError("RRQ residual tensor inventory differs from packed base")
    dtypes = {}
    for name in sorted(mapping):
        tensor = _get(residual, mapping, name)
        module, field = name.rsplit(".", 1)
        out_features, in_features = base_info["module_shapes"][module]
        expected_shape = ((in_features // 16, out_features) if field == "qweight_1" else
                          (in_features // 128, out_features) if field == "scales_1" else
                          (in_features // 128, out_features // 16))
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"RRQ residual packed shape mismatch: {name}")
        if (name.endswith(("qweight_1", "qzeros_1")) and tensor.dtype != torch.int32 or
                name.endswith("scales_1") and tensor.dtype not in
                (torch.float16, torch.bfloat16, torch.float32)):
            raise ValueError(f"Invalid RRQ residual dtype: {name}")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite RRQ residual tensor: {name}")
        dtypes[str(tensor.dtype)] = dtypes.get(str(tensor.dtype), 0) + 1
    return {"config_sha256": sha256(cfg_path),
            "shards": {s: sha256(residual / s) for s in sorted(set(mapping.values()))},
            "safetensors_disk_bytes": sum((residual / s).stat().st_size for s in set(mapping.values())),
            "tensor_count": len(mapping), "dtype_counts": dtypes}


def materialize_qdq(base: Path, source: Path, destination: Path, *, resume: bool = False) -> dict:
    """Write a complete HF checkpoint one tensor per shard to bound host RAM."""
    if destination.exists() and not resume:
        raise ValueError(f"Refusing to overwrite existing QDQ directory: {destination}")
    validated = validate_packed_base(base, source)
    destination.mkdir(parents=True, exist_ok=resume)
    source_map, base_map = _weight_map(source), _weight_map(base)
    mapping = {}
    nonquant_equal = 0
    for index, name in enumerate(sorted(source_map), start=1):
        original = _get(source, source_map, name)
        if _eligible(name):
            prefix = name[:-7]
            decoded, _ = unpack_int2(*(_get(base, base_map, f"{prefix}.{part}") for part in
                                       ("qweight", "scales", "qzeros")),
                                     in_features=original.shape[1], out_features=original.shape[0])
            tensor = decoded.to(torch.float16)
        else:
            tensor = original
            nonquant_equal += 1
        filename = f"model-{index:05d}-of-{len(source_map):05d}.safetensors"
        target = destination / filename
        if target.exists():
            with safe_open(target, framework="pt", device="cpu") as handle:
                if set(handle.keys()) != {name} or not torch.equal(handle.get_tensor(name), tensor):
                    raise ValueError(f"Existing QDQ shard differs from expected decode: {target}")
        else:
            save_file({name: tensor.contiguous()}, target)
        mapping[name] = filename
    unexpected = {path.name for path in destination.glob("*.safetensors")} - set(mapping.values())
    if unexpected:
        raise ValueError(f"Unexpected QDQ shards in interrupted export: {sorted(unexpected)}")
    config = json.loads((source / "config.json").read_text())
    config.pop("quantization_config", None)
    config_target = destination / "config.json"
    if config_target.exists():
        if json.loads(config_target.read_text()) != config:
            raise ValueError("Interrupted QDQ configuration differs from original")
    else:
        config_target.write_text(json.dumps(config, indent=2))
    for filename in sorted(SIDECARS - {"config.json"}):
        if (source / filename).is_file():
            target = destination / filename
            if target.exists():
                if sha256(target) != sha256(source / filename):
                    raise ValueError(f"Interrupted QDQ sidecar differs from original: {filename}")
            else:
                shutil.copy2(source / filename, target)
    total_size = sum((destination / shard).stat().st_size for shard in mapping.values())
    payload_size = 0
    for name in mapping:
        tensor = _get(destination, mapping, name)
        payload_size += tensor.numel() * tensor.element_size()
    expected_index = {"metadata": {"total_size": payload_size}, "weight_map": mapping}
    index_target = destination / "model.safetensors.index.json"
    if index_target.exists():
        if json.loads(index_target.read_text()) != expected_index:
            raise ValueError("Interrupted QDQ index differs from expected tensor inventory")
    else:
        atomic_json(index_target, expected_index)
    allowed = set(mapping.values()) | {"config.json", "model.safetensors.index.json"}
    allowed |= {name for name in SIDECARS if (source / name).is_file()}
    unexpected_files = {path.name for path in destination.iterdir() if path.is_file()} - allowed
    if unexpected_files:
        raise ValueError(f"Unexpected files in QDQ checkpoint: {sorted(unexpected_files)}")
    # Hash all shards only at the completion boundary; later resume verifies them.
    identity = _identity(destination, "rrq-pr2308-qdq")
    return {"checkpoint_identity": identity, "packed_base": validated,
            "nonquant_tensors_copied_exactly": nonquant_equal,
            "safetensors_disk_bytes": total_size, "decoded_dtype": "float16",
            "nonquant_dtype": "original native tensor dtype",
            "prepacking_base_comparison": "unavailable: official one-call exporter does not persist prepacking QDQ"}


def compare_checkpoints(source: Path, j05: Path, qdq: Path) -> dict:
    source_map, j05_map, qdq_map = (_weight_map(path) for path in (source, j05, qdq))
    if set(source_map) != set(j05_map) or set(source_map) != set(qdq_map):
        raise ValueError("J05 or PR QDQ has incomplete tensor coverage")
    rows = []
    total_elements = 0
    sq_j05_pr = sq_original_j05 = sq_original_pr = 0.0
    nonquant_exact = 0
    j05_nonquant_exact = 0
    j05_nonquant_changed = []
    for name in sorted(source_map):
        original = _get(source, source_map, name)
        j = _get(j05, j05_map, name)
        p = _get(qdq, qdq_map, name)
        if original.shape != j.shape or original.shape != p.shape:
            raise ValueError(f"Shape mismatch: {name}")
        if not torch.isfinite(j).all() or not torch.isfinite(p).all():
            raise ValueError(f"Nonfinite compared tensor: {name}")
        if not _eligible(name):
            if not torch.equal(original, p):
                raise ValueError(f"PR QDQ changed unquantized tensor: {name}")
            nonquant_exact += 1
            if torch.equal(original, j):
                j05_nonquant_exact += 1
            else:
                j05_nonquant_changed.append(name)
            continue
        original_f, j_f, p_f = original.float(), j.float(), p.float()
        delta = j_f - p_f
        n = original.numel()
        pair = float((delta.double().square()).sum())
        oj = float(((original_f - j_f).double().square()).sum())
        op = float(((original_f - p_f).double().square()).sum())
        total_elements += n
        sq_j05_pr += pair
        sq_original_j05 += oj
        sq_original_pr += op
        rows.append({"tensor": name, "elements": n, "exact_fraction": float((j_f == p_f).float().mean()),
                     "mse_j05_vs_pr": pair / n, "mse_original_vs_j05": oj / n,
                     "mse_original_vs_pr": op / n, "max_abs_j05_vs_pr": float(delta.abs().max()),
                     "j05_dtype": str(j.dtype), "pr_dtype": str(p.dtype),
                     "source_dtype": str(original.dtype)})
    return {"layers": rows, "quantized_tensor_count": len(rows),
            "nonquant_pr_exact_count": nonquant_exact,
            "nonquant_j05_exact_count": j05_nonquant_exact,
            "nonquant_j05_changed_names": j05_nonquant_changed,
            "overall_weighted_mse_j05_vs_pr": sq_j05_pr / total_elements,
            "overall_weighted_mse_original_vs_j05": sq_original_j05 / total_elements,
            "overall_weighted_mse_original_vs_pr": sq_original_pr / total_elements}


def run_pr04(config: dict, source: Path, checkout: Path, matgptq: Path,
             result_root: Path, *, python: str) -> dict:
    from .rrq_pr2308 import (J05_RELATIVE, RESULT_SUBDIR, TOKEN_SHA256, config_for,
                             experiment_identity, lp_revision, lp_worktree_dirty, source_audit)
    from .rrq_autoround import validate_output
    from .rrq_w2 import AUTOROUND_COMMIT, load_config
    check_revision(matgptq, MATGPTQ_COMMIT)
    source_audit(checkout)
    pr03 = result_root / RESULT_SUBDIR / "pr03"
    pr03_marker = json.loads((pr03 / "quantization_complete.json").read_text())
    if pr03_marker.get("status") != "completed_validated" or pr03_marker.get("identity") != \
            experiment_identity(config_for("pr03"), source, checkout):
        raise ValueError("PR03 is not complete under the current identity")
    base = pr03 / "export/base"
    if validate_packed_base(base, source) != pr03_marker["checkpoint"]:
        raise ValueError("PR03 packed base changed")
    j05 = result_root / J05_RELATIVE
    j05_job = j05.parents[1]
    j05_marker_path = j05_job / "encoding_complete.json"
    if not j05_marker_path.is_file():
        raise ValueError("Validated J05 encoding completion marker is absent")
    j05_marker = json.loads(j05_marker_path.read_text())
    if (j05_marker.get("status") != "complete" or
            j05_marker.get("identity", {}).get("config_sha256") != load_config("j05")["config_sha256"] or
            j05_marker.get("identity", {}).get("autoround_commit") != AUTOROUND_COMMIT or
            not same_hf_contents(j05_marker.get("identity", {}).get("source"),
                                 _identity(source, "rrq-original"))):
        raise ValueError("J05 completion marker does not match the original experiment")
    j05_info = validate_output(j05, source)
    if Path(j05_info["resolved_checkpoint_path"]) != j05.resolve():
        raise ValueError("Unexpected J05 checkpoint location")
    if any(j05_marker.get("checkpoint", {}).get(key) != j05_info.get(key) for key in
           ("config_sha256", "index_sha256", "shard_sha256", "tensor_names")):
        raise ValueError("J05 checkpoint differs from its validated completion marker")
    j05_full_path = j05_job / "full.json"
    if not j05_full_path.is_file():
        raise ValueError("Validated J05 full evaluation is absent")
    j05_full = json.loads(j05_full_path.read_text())
    if (j05_full.get("status") != "complete" or
            j05_full.get("identity", {}).get("matgptq_commit") != MATGPTQ_COMMIT or
            j05_full.get("score", {}).get("token_sha256") != TOKEN_SHA256 or
            j05_full.get("inference_dtype") != "float16" or
            not same_hf_contents(j05_full.get("checkpoint_identity"),
                                 _identity(j05, "rrq-result"))):
        raise ValueError("J05 evaluation identity or token protocol differs")
    result = result_root / RESULT_SUBDIR / "pr04"
    result.mkdir(parents=True, exist_ok=True)
    identity = experiment_identity(config, source, checkout)
    identity.update({"pr03_checkpoint": pr03_marker["checkpoint"],
                     "j05_checkpoint": _identity(j05, "rrq-pr2308-j05")})
    existing_path = result / "full.json"
    if existing_path.is_file():
        old = json.loads(existing_path.read_text())
        if old.get("status") != "completed_validated" or old.get("identity") != identity:
            raise ValueError("PR04 completion identity mismatch")
        if old.get("qdq_checkpoint_identity") != _identity(result / "qdq_checkpoint", "rrq-pr2308-qdq"):
            raise ValueError("PR04 QDQ checkpoint changed")
        return old
    run_path = result / "run.json"
    if run_path.is_file():
        if json.loads(run_path.read_text()).get("identity") != identity:
            raise ValueError("PR04 run directory has a different identity")
    elif any(result.iterdir()):
        raise ValueError("Nonempty PR04 directory lacks run identity")
    else:
        atomic_json(run_path, {"status": "running", "identity": identity})
    stage = "materialization"
    try:
        qdq = result / "qdq_checkpoint"
        qdq_marker = result / "qdq_complete.json"
        if qdq_marker.is_file():
            materialized = json.loads(qdq_marker.read_text())
            if materialized.get("identity") != identity or materialized.get("checkpoint_identity") != \
                    _identity(qdq, "rrq-pr2308-qdq"):
                raise ValueError("PR04 materialization identity mismatch")
        else:
            materialized = {"identity": identity, **materialize_qdq(base, source, qdq,
                                                                     resume=qdq.exists())}
            atomic_json(qdq_marker, materialized)
        stage = "weight_comparison"
        comparison = compare_checkpoints(source, j05, qdq)
        atomic_json(result / "weight_comparison.json", comparison)
        stage = "corpus_identity"
        corpus = corpus_identity(qdq)
        if corpus["token_sha256"] != TOKEN_SHA256:
            raise ValueError("PR04 tokenizer fingerprint differs from J01-J05")
        stage = "evaluation"
        score = evaluate_matgptq(matgptq, qdq, result / "matgptq.log", python=python)
        value = {"status": "completed_validated", "identity": identity,
                 "lp_git_commit": lp_revision(),
                 "lp_worktree_dirty": lp_worktree_dirty(),
                 "qdq_checkpoint_identity": _identity(qdq, "rrq-pr2308-qdq"),
                 "checkpoint_materialization": materialized,
                 "j05_checkpoint_validation": j05_info,
                 "j05_evaluation_identity": j05_full["identity"],
                 "weight_comparison": comparison, "corpus": corpus, "score": score,
                 "inference_dtype": "float16", "evaluation_protocol": "official_matgptq_wikitext2_2048_nonoverlap"}
        atomic_json(existing_path, value)
        return value
    except Exception as error:
        atomic_json(result / "failure.json", {"status": "failed", "identity": identity,
                                              "stage": stage, "error_type": type(error).__name__,
                                              "error": str(error)})
        raise
