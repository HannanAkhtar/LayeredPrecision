"""S01/R02 progressive LP: FP32 signed E2, bounded row chunks, strict artifacts."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .quantize import affine_decode, symmetric_rtn_quantize
from .rrq_pr2308_compare import _eligible, _weight_map
from .storage import atomic_json, pack_codes, sha256, unpack_codes


BRANCHES = {"s01": "s01_fp16_load_repair", "r02": "r02_fp16_load_repair"}
GROUP = 128
ROWS = 512
TOTAL_PARAMETERS = 8_190_735_360
ELIGIBLE_PARAMETERS = 6_945_767_424
QUANTIZER = {"family": "symmetric_rtn", "bits": 2, "group_size": GROUP,
             "axis": "input_channels_per_output_row", "grid": "{-scale,0,+scale}",
             "code_values": {"negative": 1, "zero": 2, "positive": 3},
             "unused_code": 0, "scale_dtype": "float32", "zero_point": "implicit_2",
             "rounding": "torch.round_ties_to_even", "clipping": "absmax_per_group",
             "residual_dtype": "float32", "accumulator_dtype": "float32",
             "checkpoint_dtype": "float16_for_eligible;source_dtype_for_excluded",
             "calibration": "none"}


def _files(directory: Path) -> dict:
    from .progressive_prerequisites import verified_input_identity
    cached = verified_input_identity(directory)
    if cached is not None:
        return cached
    mapping = _weight_map(directory)
    return {"config_sha256": sha256(directory / "config.json"),
            "index_sha256": sha256(directory / "model.safetensors.index.json"),
            "shard_sha256": {name: sha256(directory / name)
                             for name in sorted(set(mapping.values()))},
            "tensor_names": sorted(mapping)}


def identity(branch: str, source: Path, base: Path) -> dict:
    if branch not in BRANCHES:
        raise ValueError("Unknown LP branch")
    source_files, base_files = _files(source), _files(base)
    if source_files["tensor_names"] != base_files["tensor_names"]:
        raise ValueError("Source and repaired base tensor inventories differ")
    eligible = [name for name in source_files["tensor_names"] if _eligible(name)]
    if len(eligible) != 252:
        raise ValueError(f"Expected 252 projections, got {len(eligible)}")
    return {"branch": branch, "source": source_files, "repaired_base": base_files,
            "residual_quantizer": QUANTIZER, "eligible_projection_count": 252}


def _slice(directory: Path, mapping: dict, name: str, first: int, stop: int) -> torch.Tensor:
    with safe_open(directory / mapping[name], framework="pt", device="cpu") as handle:
        return handle.get_slice(name)[first:stop]


def encode_residual(residual: torch.Tensor) -> tuple[dict, torch.Tensor]:
    params, encoded = symmetric_rtn_quantize(residual.float(), 2, "group", GROUP)
    decoded = affine_decode(params)
    if not torch.equal(encoded, decoded):
        raise ValueError("Encoder and independent FP32 decoder differ")
    if not bool(torch.isfinite(decoded).all()):
        raise ValueError("Nonfinite residual reconstruction")
    return params, decoded


def _read_chunk(layer_dir: Path, record: dict) -> torch.Tensor:
    code, meta = layer_dir / record["code"], layer_dir / record["metadata"]
    if (sha256(code) != record["code_sha256"] or
            sha256(meta) != record["metadata_sha256"]):
        raise ValueError("Residual chunk hash differs")
    tensors = load_file(str(meta), device="cpu")
    if set(tensors) != {"scale"} or tensors["scale"].dtype != torch.float32:
        raise ValueError("Residual scale metadata differs")
    params = {"family": "symmetric_rtn", "bits": 2, "granularity": "group",
              "group_size": GROUP, "shape": record["shape"],
              "codes": unpack_codes(code, tuple(record["shape"]), 2),
              "scale": tensors["scale"]}
    if int(params["codes"].min()) < 1 or int(params["codes"].max()) > 3:
        raise ValueError("Residual code outside signed ternary grid")
    return affine_decode(params)


def _manifest(root: Path, depth: int, expected_identity: dict,
              *, pilot: bool = False) -> dict:
    path = root / f"layer{depth}" / "encoding_complete.json"
    if not path.is_file():
        raise ValueError(f"Unvalidated LP predecessor: {path}")
    record = json.loads(path.read_text())
    expected_status = "pilot_validated" if pilot else "completed_validated"
    if (record.get("status") != expected_status or
            record.get("identity") != expected_identity or record.get("depth") != depth or
            record.get("quantizer") != QUANTIZER or
            record.get("projection_count") != (7 if pilot else 252) or
            record.get("chunk_count") != len(record.get("chunks", []))):
        raise ValueError(f"LP predecessor identity or structure differs: {path}")
    pinned_code = os.environ.get("FOLLOWUP_CODE_SHA")
    if pinned_code and record.get("code_sha") != pinned_code:
        raise ValueError("LP predecessor was encoded by a different code commit")
    by_tensor: dict[str, list[dict]] = {}
    for chunk in record["chunks"]:
        _read_chunk(root / f"layer{depth}", chunk)
        by_tensor.setdefault(chunk["name"], []).append(chunk)
    if set(by_tensor) != set(record.get("tensor_rows", {})):
        raise ValueError("LP tensor inventory is incomplete")
    for name, chunks in by_tensor.items():
        next_row = 0
        for chunk in sorted(chunks, key=lambda item: item["first"]):
            if (chunk["first"] != next_row or chunk["stop"] <= next_row or
                    chunk["shape"][0] != chunk["stop"] - next_row):
                raise ValueError(f"LP row coverage differs: {name}")
            next_row = chunk["stop"]
        if next_row != record["tensor_rows"][name]:
            raise ValueError(f"LP final row coverage differs: {name}")
    return record


def verified_layers(root: Path, depth: int, expected_identity: dict,
                    *, pilot: bool = False) -> list[dict]:
    if not 0 <= depth <= 3:
        raise ValueError("LP depth must be 0..3")
    return [_manifest(root, number, expected_identity, pilot=pilot)
            for number in range(1, depth + 1)]


def validate_repaired_base(branch: str, source: Path, base: Path) -> dict:
    """Require the original repair marker, full evaluation and current tensor audit."""
    from .progressive_prerequisites import verified_input_identity
    cached = verified_input_identity(base, branch=branch)
    if cached is not None:
        if verified_input_identity(source) is None:
            raise ValueError("Prerequisite source identity is unavailable")
        return cached
    from .quantizer_followups import load_config, validate_checkpoint

    if branch not in BRANCHES or base.parent.name != BRANCHES[branch]:
        raise ValueError("Wrong repaired W2 checkpoint directory")
    marker = json.loads((base.parent / "encoding_complete.json").read_text())
    full = json.loads((base.parent / "full.json").read_text())
    integrity = json.loads((base.parent / "integrity_report.json").read_text())
    checked = validate_checkpoint(load_config(branch), source, base)
    if (marker.get("status") != "completed_validated" or
            marker.get("checkpoint_validation") != checked or
            full.get("status") != "completed_validated" or
            full.get("checkpoint_validation") != checked or
            full.get("corpus", {}).get("token_sha256") !=
            "8ffcee84dd4b684ec631cb52845e18c9c181ce6fdb8b26d65a4ffdc3d35fbda3" or
            full.get("corpus", {}).get("scored_tokens") != 298862 or
            round(full.get("evaluation", {}).get("perplexity", -1), 2) !=
            {"s01": 12.47, "r02": 32.22}[branch] or
            integrity.get("excluded_source_exact") is not True or
            integrity.get("projection_byte_identical") is not True):
        raise ValueError("Repaired W2 checkpoint or official evaluation identity differs")
    return checked


def reconstruct_block(base: torch.Tensor, layer_records: list[tuple[Path, dict]]) -> torch.Tensor:
    if not layer_records:
        return base.clone()  # exact W2 endpoint, including original dtype
    result = base.float()
    for directory, record in layer_records:
        contribution = _read_chunk(directory, record)
        if contribution.shape != result.shape:
            raise ValueError("Residual chunk shape differs")
        result = result + contribution
    return result


def _checkpoint_bytes(base: Path) -> int:
    return sum(path.stat().st_size for path in base.iterdir() if path.is_file())


def encode_layer(branch: str, depth: int, source: Path, base: Path, root: Path,
                 *, rows: int = ROWS, limit_tensors: int | None = None) -> dict:
    """Encode one enhancement; pilot limit writes an isolated incomplete pilot."""
    started = time.monotonic()
    if depth not in (1, 2, 3) or rows < 1:
        raise ValueError("Invalid LP depth or row block")
    current_identity = identity(branch, source, base)
    prior = verified_layers(root, depth - 1, current_identity,
                            pilot=limit_tensors is not None)
    target = root / f"layer{depth}"
    partial = root / f"layer{depth}.partial"
    if partial.exists():
        raise FileExistsError(f"Interrupted LP layer requires inspection: {partial}")
    if target.exists():
        existing = _manifest(root, depth, current_identity,
                             pilot=limit_tensors is not None)
        return existing
    if limit_tensors is not None and (limit_tensors != 7 or depth > 2):
        raise ValueError("Pilot requires seven first-layer tensors and depth <= 2")
    source_map, base_map = _weight_map(source), _weight_map(base)
    names = [name for name in sorted(source_map) if _eligible(name)]
    partial.mkdir(parents=True, exist_ok=False)
    code_sha = os.environ.get("FOLLOWUP_CODE_SHA")
    atomic_json(partial / "run.json", {"status": "started", "identity": current_identity,
                                      "depth": depth, "code_sha": code_sha})
    chunks = []
    sum_sq = source_sq = 0.0
    count = 0
    layer_errors: dict[str, dict] = {}
    tensor_rows: dict[str, int] = {}
    for tensor_number, name in enumerate(names[:limit_tensors], 1):
        with safe_open(source / source_map[name], framework="pt", device="cpu") as handle:
            shape = handle.get_slice(name).get_shape()
        tensor_rows[name] = shape[0]
        for first in range(0, shape[0], rows):
            stop = min(first + rows, shape[0])
            original = _slice(source, source_map, name, first, stop).float()
            base_value = _slice(base, base_map, name, first, stop)
            if original.shape != base_value.shape or not bool(torch.isfinite(original).all()):
                raise ValueError(f"Source/base projection invalid: {name}")
            previous = base_value.float()
            for number, manifest in enumerate(prior, 1):
                matches = [chunk for chunk in manifest["chunks"] if
                           chunk["name"] == name and chunk["first"] == first and chunk["stop"] == stop]
                if len(matches) != 1:
                    raise ValueError("Predecessor row partition differs")
                previous += _read_chunk(root / f"layer{number}", matches[0])
            params, decoded = encode_residual(original - previous)
            reconstructed = previous + decoded
            if not bool(torch.isfinite(reconstructed).all()):
                raise ValueError(f"Nonfinite LP prefix: {name}")
            stem = f"chunks/{tensor_number:03d}-{first:05d}"
            code, meta = partial / (stem + ".bin"), partial / (stem + ".safetensors")
            pack_codes(code, params["codes"], 2)
            meta.parent.mkdir(parents=True, exist_ok=True)
            save_file({"scale": params["scale"].contiguous()}, meta, metadata={"format": "pt"})
            loaded = _read_chunk(partial, {"code": str(code.relative_to(partial)),
                "metadata": str(meta.relative_to(partial)), "code_sha256": sha256(code),
                "metadata_sha256": sha256(meta), "shape": list(params["codes"].shape)})
            if not torch.equal(loaded, decoded):
                raise ValueError("Saved residual code round trip differs")
            chunks.append({"name": name, "first": first, "stop": stop,
                "shape": list(params["codes"].shape), "code": str(code.relative_to(partial)).replace("\\", "/"),
                "metadata": str(meta.relative_to(partial)).replace("\\", "/"),
                "code_sha256": sha256(code), "metadata_sha256": sha256(meta),
                "code_bytes": code.stat().st_size, "metadata_bytes": meta.stat().st_size,
                "scale_payload_bytes": params["scale"].numel() * params["scale"].element_size()})
            error = original - reconstructed
            chunk_error_sq = float(error.double().square().sum())
            chunk_signal_sq = float(original.double().square().sum())
            sum_sq += chunk_error_sq
            source_sq += chunk_signal_sq
            count += original.numel()
            layer_name = name.split(".")[2]
            entry = layer_errors.setdefault(layer_name, {"squared_error": 0.0,
                                                       "signal_square": 0.0, "elements": 0})
            entry["squared_error"] += chunk_error_sq
            entry["signal_square"] += chunk_signal_sq
            entry["elements"] += original.numel()
    complete = limit_tensors is None and len({chunk["name"] for chunk in chunks}) == 252
    if not complete and limit_tensors is None:
        raise ValueError("LP layer omitted projections")
    report = {"status": "completed_validated" if complete else "pilot_validated",
              "quota_status": ("unknown_operator_authorized" if
                  os.environ.get("LP_MATRIX_ALLOW_UNKNOWN_QUOTA") == "1" and
                  not os.environ.get("LP_MATRIX_SHARED_QUOTA_FREE_GIB") else
                  "site_confirmed" if os.environ.get("LP_MATRIX_SHARED_QUOTA_FREE_GIB") else
                  "unknown_unapproved"),
              "identity": current_identity, "depth": depth, "code_sha": code_sha,
              "quantizer": QUANTIZER,
              "pilot_tensor_limit": limit_tensors,
              "tensor_rows": tensor_rows,
              "chunk_count": len(chunks), "chunks": chunks, "projection_count":
              len({chunk["name"] for chunk in chunks}), "encoded_elements": count,
              "mse_fp32": sum_sq / count, "normalized_mse_fp32": sum_sq / source_sq if source_sq else None,
              "sqnr_db_fp32": 10 * math.log10(source_sq / sum_sq) if sum_sq and source_sq else None,
              "layerwise": {layer: {"mse_fp32": value["squared_error"] / value["elements"],
                                     "normalized_mse_fp32": value["squared_error"] / value["signal_square"]
                                     if value["signal_square"] else None,
                                     "sqnr_db_fp32": 10 * math.log10(value["signal_square"] / value["squared_error"])
                                     if value["squared_error"] and value["signal_square"] else None,
                                     "elements": value["elements"]}
                            for layer, value in sorted(layer_errors.items(), key=lambda item: int(item[0]))},
              "code_bytes": sum(c["code_bytes"] for c in chunks),
              "metadata_bytes": sum(c["metadata_bytes"] for c in chunks),
              "scale_payload_bytes": sum(c["scale_payload_bytes"] for c in chunks),
              "base_checkpoint_file_bytes": _checkpoint_bytes(base)}
    report["runtime_seconds"] = time.monotonic() - started
    try:
        import resource
        report["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:  # Windows development; AUS records the Linux peak.
        report["process_peak_rss_kib"] = None
    atomic_json(partial / "encoding_complete.json", report)
    os.replace(partial, target)
    return report


def storage_report(branch: str, source: Path, base: Path, root: Path, depth: int) -> dict:
    """Measured dependent bytes; the repaired W2 base remains an external artifact."""
    ident = identity(branch, source, base)
    layers = verified_layers(root, depth, ident)
    base_bytes = _checkpoint_bytes(base)
    rows = []
    code = metadata = scales = framing = 0
    for number, layer in enumerate(layers, 1):
        directory = root / f"layer{number}"
        code += sum((directory / chunk["code"]).stat().st_size for chunk in layer["chunks"])
        metadata += sum((directory / chunk["metadata"]).stat().st_size for chunk in layer["chunks"])
        scales += sum(chunk["scale_payload_bytes"] for chunk in layer["chunks"])
        framing += (directory / "encoding_complete.json").stat().st_size
        total = base_bytes + code + metadata + framing
        rows.append({"depth": number, "nominal_projection_bits": 2 + 2 * number,
                     "base_dense_external_bytes": base_bytes,
                     "residual_code_payload_bytes": code,
                     "residual_scale_payload_bytes": scales,
                     "residual_other_metadata_and_header_bytes": metadata - scales,
                     "integrity_manifest_bytes": framing,
                     "total_reconstruction_bytes": total,
                     "effective_bits_per_total_parameter": 8 * total / TOTAL_PARAMETERS,
                     "standalone_base_requirement": "copy or package repaired W2 checkpoint"})
    return {"branch": branch, "base_file_bytes": base_bytes,
            "excluded_weights_counted_once_in_base": True,
            "base_is_external_dependency": True, "prefixes": rows,
            "run_json_bytes_not_required_for_decode": sum(
                (root / f"layer{number}" / "run.json").stat().st_size
                for number in range(1, depth + 1)),
            "all_four_rates_required_bytes": rows[-1]["total_reconstruction_bytes"] if rows else base_bytes,
            "eligible_projection_parameters": ELIGIBLE_PARAMETERS,
            "total_model_parameters": TOTAL_PARAMETERS}
