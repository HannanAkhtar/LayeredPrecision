"""Versioned raw bit streams and atomic tensor artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .quantize import AUTOROUND_RTN_SOURCE, affine_decode
from .rtn_reproduction import AUTOROUND_COMMIT, MODES


CHUNK = 1 << 20  # divisible by eight: independent chunks preserve bit alignment
SIDECARS = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
            "tokenizer.model", "chat_template.jinja"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    save_file({k: v.contiguous().cpu() for k, v in tensors.items()}, str(temp))
    os.replace(temp, path)


def pack_codes(path: Path, codes: torch.Tensor, bits: int):
    """Little bit order within each byte; stream one bounded code chunk at a time."""
    if bits < 1 or bits > 16:
        raise ValueError("Invalid bit width")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    values = codes.detach().cpu().contiguous().numpy().reshape(-1)
    with temp.open("wb") as handle:
        for start in range(0, values.size, CHUNK):
            chunk = values[start:start + CHUNK].astype(np.uint32, copy=False)
            if np.any(chunk >= (1 << bits)):
                raise ValueError("Code out of range")
            planes = ((chunk[:, None] >> np.arange(bits, dtype=np.uint32)) & 1).astype(np.uint8)
            handle.write(np.packbits(planes.reshape(-1), bitorder="little").tobytes())
    os.replace(temp, path)


def unpack_codes(path: Path, shape: tuple[int, int], bits: int) -> torch.Tensor:
    count = shape[0] * shape[1]
    expected = (count * bits + 7) // 8
    if path.stat().st_size != expected:
        raise ValueError(f"Packed stream length mismatch: {path}")
    output = np.empty(count, dtype=np.uint8 if bits <= 8 else np.uint16)
    with path.open("rb") as handle:
        for start in range(0, count, CHUNK):
            n = min(CHUNK, count - start)
            payload = handle.read((n * bits + 7) // 8)
            planes = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little")[:n * bits]
            output[start:start + n] = (planes.reshape(n, bits).astype(np.uint32) *
                                         (1 << np.arange(bits, dtype=np.uint32))).sum(1)
    return torch.from_numpy(output.reshape(shape))


def write_layer(root: Path, tensor_id: int, layer: int, params: dict) -> dict:
    stem = f"streams/{tensor_id:04d}-{layer:02d}"
    code = root / (stem + ".bin")
    meta = root / (stem + ".safetensors")
    pack_codes(code, params["codes"], params["bits"])
    family = params.get("family", "affine_asymmetric")
    if family == "affine_asymmetric":
        zero = params["zero_point"].to(torch.int32) if params["bits"] > 8 else params["zero_point"]
        metadata = {"scale": params["scale"], "zero_point": zero}
        extra_bytes = zero.numel() * zero.element_size()
    elif family in ("symmetric_rtn", "autoround_rtn_sym", *MODES):
        # The fixed signed-to-unsigned code offset is implicit; do not
        # falsely charge a per-group zero point for this symmetric format.
        metadata = {"scale": params["scale"]}
        extra_bytes = 0
    else:
        raise ValueError(f"Unknown quantizer family: {family}")
    atomic_safetensors(meta, metadata)
    record = {"code": str(code.relative_to(root)).replace("\\", "/"),
            "metadata": str(meta.relative_to(root)).replace("\\", "/"),
            "code_bytes": code.stat().st_size, "metadata_bytes": meta.stat().st_size,
            "scale_zero_bytes": params["scale"].numel() * params["scale"].element_size() +
                                extra_bytes,
            "code_sha256": sha256(code), "metadata_sha256": sha256(meta),
            "bits": params["bits"], "granularity": params["granularity"],
            "group_size": params["group_size"], "shape": params["shape"],
            "family": family, "reconstruction_dtype": "float32",
            "rounding": "torch.round_ties_to_even", "zero_in_range": True,
            "zero_scale_decodes_to": 0.0}
    if family == "autoround_rtn_sym":
        record.update(scale_dtype="float16", signed_code_offset=1 << (params["bits"] - 1),
                      signed_range=[-(1 << (params["bits"] - 1)), (1 << (params["bits"] - 1)) - 1],
                      scale_threshold=1e-5, implementation=AUTOROUND_RTN_SOURCE,
                      padding="right_zero_per_row", rounding="torch.round_ties_to_even")
    if family in MODES:
        record.update(scale_dtype="float16", signed_code_offset=2,
                      signed_range=[-2, 1], scale_threshold=1e-5,
                      implementation=f"intel/auto-round@{AUTOROUND_COMMIT}",
                      algorithm=params["algorithm"], qdq_dtype=params["qdq_dtype"],
                      compute_device=params["compute_device"],
                      checkpoint_materialization="qdq_dtype_then_bfloat16",
                      padding="right_zero_per_row", rounding="torch.round_ties_to_even")
    return record


def read_layer(root: Path, record: dict, verify: bool = True) -> dict:
    code = root / record["code"]
    meta = root / record["metadata"]
    if verify and (sha256(code) != record["code_sha256"] or
                   sha256(meta) != record["metadata_sha256"]):
        raise ValueError(f"Corrupt LP layer: {code}")
    values = load_file(str(meta), device="cpu")
    if record.get("family") == "autoround_rtn_sym":
        expected_offset = 1 << (record["bits"] - 1)
        if (record.get("scale_dtype") != "float16" or
                record.get("signed_code_offset") != expected_offset or
                record.get("implementation") != AUTOROUND_RTN_SOURCE or
                record.get("rounding") != "torch.round_ties_to_even" or
                set(values) != {"scale"} or values["scale"].dtype != torch.float16):
            raise ValueError("Invalid AutoRound RTN layer metadata")
    if record.get("family") in MODES:
        expected_algorithm, expected_dtype = MODES[record["family"]]
        if (record.get("scale_dtype") != "float16" or
                record.get("signed_code_offset") != 2 or
                record.get("implementation") != f"intel/auto-round@{AUTOROUND_COMMIT}" or
                record.get("algorithm") != expected_algorithm or
                record.get("qdq_dtype") != str(expected_dtype).removeprefix("torch.") or
                set(values) != {"scale"} or values["scale"].dtype != torch.float16):
            raise ValueError("Invalid Phase 1 RTN layer metadata")
    return {"codes": unpack_codes(code, tuple(record["shape"]), record["bits"]),
            "bits": record["bits"], "granularity": record["granularity"],
            "group_size": record["group_size"], "shape": record["shape"],
            "family": record.get("family", "affine_asymmetric"), **values}


def decode_prefix(root: Path, record: dict, depth: int) -> torch.Tensor:
    layers = record["layers"]
    if depth < 1 or depth > len(layers):
        raise ValueError("Invalid LP prefix depth")
    result = None
    for layer in layers[:depth]:
        contribution = affine_decode(read_layer(root, layer))
        result = contribution if result is None else result + contribution
    return result


def copy_sidecars(source: Path, root: Path) -> list[str]:
    result = []
    for name in sorted(SIDECARS):
        if (source / name).is_file():
            target = root / "hf" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)
            result.append(str(target.relative_to(root)).replace("\\", "/"))
    if "hf/config.json" not in result:
        raise ValueError("Source config.json is missing")
    return result


def storage_report(root: Path, manifest: dict) -> dict:
    """Measured bytes of files required for each prefix; optional reports excluded."""
    records = manifest["tensors"]
    shared = sum((root / r["shared"]).stat().st_size for r in records if "shared" in r)
    common_files = ["manifest.json", "COMPLETE.json"] + manifest["sidecars"]
    framing = sum((root / file).stat().st_size for file in common_files)
    resume_overhead = (root / "run.json").stat().st_size + sum(
        (root / r["marker"]).stat().st_size for r in records)
    total_parameters = sum(r["numel"] for r in records)
    widths = manifest["bits"]
    rows = []
    for depth in range(1, len(widths) + 1):
        layer_records = [layer for r in records if "layers" in r for layer in r["layers"][:depth]]
        codes = sum((root / layer["code"]).stat().st_size for layer in layer_records)
        metadata_files = sum((root / layer["metadata"]).stat().st_size for layer in layer_records)
        scale_zero = sum(layer["scale_zero_bytes"] for layer in layer_records)
        other = framing + metadata_files - scale_zero
        total = codes + scale_zero + other + shared
        rows.append({"depth": depth, "nominal_bits": sum(widths[:depth]),
                     "packed_code_bytes": codes, "scale_zero_metadata_bytes": scale_zero,
                     "other_decoder_framing_bytes": other, "shared_unquantized_bytes": shared,
                     "total_physical_bytes": total,
                     "effective_bits_per_total_parameter": 8 * total / total_parameters})
    return {"prefixes": rows, "cumulative_hierarchy_bytes": rows[-1]["total_physical_bytes"],
            "resume_marker_bytes_not_needed_for_decode": resume_overhead,
            "total_parameters": total_parameters}
