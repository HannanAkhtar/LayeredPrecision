"""Raw/Huffman prefix streams and exact decoder checks for Experiment 5."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import shutil
import struct
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from analyze_mse_layer_sensitivity import write_json
from evaluate_lp_base_enhancement import (FAMILIES, SCHEMES, base_key, decode, extend,
                                           parent_key, unique_progressive_keys)
from evaluate_lp_quantizer_adaptivity import load_torch, save_csv, sha256_file
from lp_progressive_storage import pack_codes, unpack_codes
from lp_base_quantizers import LOG_RATIO
from storage_utils import BitWriter, canonical_huffman_codes, decode_huffman, huffman_code_lengths

MAGIC = b"LPS5"


def all_keys() -> list[str]:
    bases = [base_key(family) for family in FAMILIES]
    e1 = [extend(base_key(base), family) for base in FAMILIES for family in FAMILIES]
    rest = [key for key in unique_progressive_keys() if key.count("_E") > 1]
    return bases + e1 + rest


def code_payload(codes: np.ndarray, bits: int, regime: str):
    raw = pack_codes(codes, bits)
    if regime == "raw":
        return raw, b"", 0
    alphabet = 1 << bits
    counts = np.bincount(codes, minlength=alphabet)
    lengths = huffman_code_lengths(Counter({i: int(count) for i, count in enumerate(counts) if count}))
    coded_bits = sum(int(counts[i])*length for i, length in lengths.items())
    if (coded_bits+7)//8 + alphabet >= len(raw):
        return raw, b"", 0
    table = bytes(lengths.get(i, 0) for i in range(alphabet))
    if len(lengths) == 1:
        return b"", table, 0
    out = io.BytesIO()
    writer = BitWriter(out)
    mapping = canonical_huffman_codes(lengths)
    for code in codes:
        writer.write(*mapping[int(code)])
    writer.flush()
    if writer.total_bits != coded_bits:
        raise AssertionError("Huffman payload length changed")
    return out.getvalue(), table, coded_bits


def _metadata(params: dict, family: str, channels: int) -> tuple[int, bytes]:
    if family in ("A", "P"):
        scale = params["scale"].reshape(-1).cpu().numpy().astype("<f4")
        zp = params["zero_point"].reshape(-1).cpu().numpy().astype("u1")
        if scale.size != channels or zp.size != channels:
            raise ValueError("Affine channel metadata mismatch")
        fallback = params.get("t_min")
        needs_fallback = (scale == 0).any() and fallback is not None and bool(
            (fallback.reshape(-1).cpu().numpy()[scale == 0] != 0).any())
        value = scale.tobytes() + zp.tobytes()
        if needs_fallback:
            value += fallback.reshape(-1).cpu().numpy().astype("<f4").tobytes()
        return int(needs_fallback), value
    if family in ("U", "L"):
        scale = params["scale"].reshape(-1).cpu().numpy().astype("<f4")
        if scale.size != channels:
            raise ValueError("Scale channel metadata mismatch")
        return 0, scale.tobytes()
    if family == "N":
        centers = params["centers"].cpu().numpy().astype("<f4")
        if centers.shape[0] != channels:
            raise ValueError("Codebook channel metadata mismatch")
        return 0, centers.tobytes()
    raise ValueError(family)


def write_stream(path: Path, artifacts: dict, family: str, bits: int, regime: str) -> dict:
    if family not in FAMILIES or bits not in (2, 5) or regime not in ("raw", "entropy"):
        raise ValueError("Unsupported stream")
    path.parent.mkdir(parents=True, exist_ok=True)
    result = {"raw_code_bytes": 0, "payload_bytes": 0, "metadata_bytes": 0,
              "huffman_table_bytes": 0, "framing_bytes": 8+(4 if family == "L" else 0),
              "huffman_tensors": 0, "raw_fallback_tensors": 0,
              "huffman_payload_bits": 0}
    with path.open("wb") as handle:
        handle.write(MAGIC + family.encode() + struct.pack("<BH", bits, len(artifacts)))
        if family == "L":
            handle.write(struct.pack("<f", LOG_RATIO if bits == 5 else 4.0))
        for name in sorted(artifacts):
            params = artifacts[name]
            codes = params["quantized"].cpu().numpy().astype(np.uint8).reshape(-1)
            shape = tuple(params["quantized"].shape)
            channels = shape[0]
            fallback, metadata = _metadata(params, family, channels)
            name_bytes = name.encode()
            frame = (struct.pack("<H", len(name_bytes)) + name_bytes +
                     struct.pack("<B", len(shape)) + struct.pack("<"+"I"*len(shape), *shape) +
                     struct.pack("<IQB", channels, codes.size, fallback))
            payload, table, payload_bits = code_payload(codes, bits, regime)
            handle.write(frame)
            handle.write(metadata)
            handle.write(struct.pack("<BQ", int(bool(table)), len(payload)))
            if table:
                handle.write(table)
                result["huffman_tensors"] += 1
                result["huffman_table_bytes"] += len(table)
                result["huffman_payload_bits"] += payload_bits
            else:
                result["raw_fallback_tensors"] += 1
            handle.write(payload)
            result["framing_bytes"] += len(frame)+9
            result["metadata_bytes"] += len(metadata)
            result["payload_bytes"] += len(payload)
            result["raw_code_bytes"] += (len(codes)*bits+7)//8
    result["physical_bytes"] = path.stat().st_size
    if result["physical_bytes"] != sum(result[key] for key in
                                       ("framing_bytes", "metadata_bytes", "payload_bytes", "huffman_table_bytes")):
        raise AssertionError("Stream byte accounting differs")
    return result


@lru_cache(maxsize=4)
def read_stream(path: Path) -> tuple[str, int, dict]:
    data = memoryview(path.read_bytes())
    if len(data) < 8 or bytes(data[:4]) != MAGIC:
        raise ValueError("Invalid LPS5 stream")
    family = bytes(data[4:5]).decode()
    bits, tensors = struct.unpack_from("<BH", data, 5)
    if family not in FAMILIES or bits not in (2, 5):
        raise ValueError("Invalid stream family or precision")
    position = 8
    if family == "L":
        encoded_ratio = struct.unpack_from("<f", data, position)[0]
        if not math.isclose(encoded_ratio, LOG_RATIO if bits == 5 else 4.0, rel_tol=1e-6):
            raise ValueError("Unsupported logarithmic ratio")
        position += 4
    result = {}
    for _ in range(tensors):
        name_len = struct.unpack_from("<H", data, position)[0]
        position += 2
        name = bytes(data[position:position+name_len]).decode()
        position += name_len
        ndim = data[position]
        position += 1
        shape = struct.unpack_from("<"+"I"*ndim, data, position)
        position += 4*ndim
        channels, count, fallback = struct.unpack_from("<IQB", data, position)
        position += 13
        if np.prod(shape) != count or shape[0] != channels:
            raise ValueError("Tensor shape mismatch")
        view = (channels,)+(1,)*(ndim-1)
        params = {"family": family, "bits": bits, "axis": 0}
        if family in ("A", "P"):
            scale = np.frombuffer(data[position:position+4*channels], dtype="<f4").copy()
            position += 4*channels
            zp = np.frombuffer(data[position:position+channels], dtype="u1").copy()
            position += channels
            if fallback:
                minimum = np.frombuffer(data[position:position+4*channels], dtype="<f4").copy()
                position += 4*channels
            else:
                minimum = np.zeros(channels, dtype=np.float32)
            params.update(scale=torch.from_numpy(scale.reshape(view)),
                          zero_point=torch.from_numpy(zp.reshape(view)),
                          t_min=torch.from_numpy(minimum.reshape(view)))
        elif family in ("U", "L"):
            scale = np.frombuffer(data[position:position+4*channels], dtype="<f4").copy()
            position += 4*channels
            params["scale"] = torch.from_numpy(scale.reshape(view))
        else:
            size = (1 << bits)*4*channels
            centers = np.frombuffer(data[position:position+size], dtype="<f4").copy()
            position += size
            params["centers"] = torch.from_numpy(centers.reshape(channels, 1 << bits))
        mode, payload_len = struct.unpack_from("<BQ", data, position)
        position += 9
        if mode == 1:
            alphabet = 1 << bits
            table = bytes(data[position:position+alphabet])
            position += alphabet
            lengths = {i: int(length) for i, length in enumerate(table) if length}
            if not lengths:
                raise ValueError("Empty Huffman table")
            payload = bytes(data[position:position+payload_len])
            codes = np.asarray(decode_huffman(payload, lengths, count,
                                              next(iter(lengths)) if len(lengths)==1 else None), dtype=np.uint8)
        elif mode == 0:
            payload = bytes(data[position:position+payload_len])
            codes = unpack_codes(payload, count, bits)
        else:
            raise ValueError("Invalid codec mode")
        position += payload_len
        params["quantized"] = torch.from_numpy(codes.reshape(shape))
        if name in result:
            raise ValueError("Duplicate tensor name")
        result[name] = params
    if position != len(data):
        raise ValueError("Trailing or truncated stream")
    return family, bits, result


def lineage(key: str) -> list[str]:
    result = [key]
    while parent_key(result[-1]):
        result.append(parent_key(result[-1]))
    return result[::-1]


def manifest(root: Path, key: str) -> Path:
    path = root / "manifests" / (key + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"format": "LPS5", "key": key,
                                "shared_fp32": "shared_fp32_state.pt",
                                "streams": ["streams/"+item+".bin" for item in lineage(key)]},
                               separators=(",", ":"), sort_keys=True), encoding="utf-8")
    return path


def read_prefix(manifest_path: Path, *, stop_key: str | None = None) -> dict:
    info = json.loads(manifest_path.read_text())
    root = manifest_path.parent.parent
    keys = lineage(info["key"])
    if stop_key is None:
        stop_key = info["key"]
    if stop_key not in keys or len(keys) != len(info["streams"]):
        raise ValueError("Invalid prefix stop or manifest")
    fixed = load_torch(root / info["shared_fp32"])
    eligible = None
    for stream in info["streams"][:keys.index(stop_key)+1]:
        family, bits, artifacts = read_stream(root / stream)
        if eligible is None:
            if bits != 5:
                raise ValueError("Missing 5-bit base")
            eligible = {name: decode(params, family, True) for name, params in artifacts.items()}
        else:
            if bits != 2 or set(eligible) != set(artifacts):
                raise ValueError("Enhancement tensor alignment differs")
            eligible = {name: eligible[name]+decode(artifacts[name], family, False) for name in eligible}
    return {**fixed, **eligible}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/lp_base_enhancement_study"))
    args = parser.parse_args()
    out = args.outdir.resolve()
    keys = all_keys()
    from evaluate_imagenet import build_model
    from analyze_mse_layer_sensitivity import eligible_groups
    model = build_model("resnet50")
    eligible = {name for group in eligible_groups(model).values() for name in group}
    fixed = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items() if name not in eligible}
    root = out / "serialized"
    shared_source = root / "shared_fp32_state.pt"
    root.mkdir(parents=True, exist_ok=True)
    if shared_source.exists():
        saved = load_torch(shared_source)
        if any(not torch.equal(saved[name], tensor) for name, tensor in fixed.items()):
            raise AssertionError("Shared FP32 state changed")
    else:
        torch.save(fixed, shared_source)
    summaries = {}
    for regime in ("raw", "entropy"):
        directory = root / regime
        directory.mkdir(parents=True, exist_ok=True)
        shared = directory / "shared_fp32_state.pt"
        if not shared.exists():
            shutil.copyfile(shared_source, shared)
        elif sha256_file(shared) != sha256_file(shared_source):
            raise AssertionError("Shared state identity differs")
        for key in keys:
            source = out / "codes" / (key + ".pt")
            target = directory / "streams" / (key + ".bin")
            stats_file = target.with_suffix(".stats.json")
            source_sha = sha256_file(source)
            cached = json.loads(stats_file.read_text()) if stats_file.exists() else None
            if not target.exists() or cached is None or cached.get("source_sha256") != source_sha:
                result = write_stream(target, load_torch(source), key.rsplit("_", 1)[1],
                                      5 if parent_key(key) is None else 2, regime)
                result["source_sha256"] = source_sha
                write_json(stats_file, result)
            else:
                result = cached
                if target.stat().st_size != result["physical_bytes"]:
                    raise AssertionError("Cached stream byte count changed")
            summaries[(regime, key)] = result
            manifest(directory, key)
            print(f"Serialized {regime}/{key}: {result['physical_bytes']:,} bytes", flush=True)
    rows = []
    for key in keys:
        for regime in ("raw", "entropy"):
            directory = root / regime
            path = directory / "manifests" / (key + ".json")
            info = json.loads(path.read_text())
            files = [directory / info["shared_fp32"], path] + [directory / name for name in info["streams"]]
            total = sum(item.stat().st_size for item in files)
            pieces = [summaries[(regime, item)] for item in lineage(key)]
            parent = parent_key(key)
            previous = next((row["total_bytes"] for row in rows if row["key"]==parent and row["regime"]==regime), None)
            row = {"key": key, "base": lineage(key)[0], "regime": regime,
                   "nominal_bits": 5+2*key.count("_E"),
                   "code_payload_bytes": sum(item["payload_bytes"] for item in pieces),
                   "raw_code_bytes": sum(item["raw_code_bytes"] for item in pieces),
                   "metadata_bytes": sum(item["metadata_bytes"] for item in pieces),
                   "huffman_table_bytes": sum(item["huffman_table_bytes"] for item in pieces),
                   "stream_framing_bytes": sum(item["framing_bytes"] for item in pieces),
                   "manifest_bytes": path.stat().st_size,
                   "shared_fp32_bytes": (directory / info["shared_fp32"]).stat().st_size,
                   "incremental_stream_bytes": pieces[-1]["physical_bytes"] if parent else 0,
                   "total_bytes": total,
                   "incremental_total_bytes": total-previous if previous is not None else total,
                   "effective_bpw": 8*total/25_502_912}
            if total != sum(row[item] for item in ("code_payload_bytes", "metadata_bytes",
                                                   "huffman_table_bytes", "stream_framing_bytes",
                                                   "manifest_bytes", "shared_fp32_bytes")):
                raise AssertionError("Prefix byte accounting differs")
            rows.append(row)
    write_json(out / "prefix_storage.json", rows)
    save_csv(out / "prefix_storage.csv", rows)
    entropy = []
    for key in keys:
        raw, coded = summaries[("raw", key)], summaries[("entropy", key)]
        entropy.append({"key": key, "raw_stream_bytes": raw["physical_bytes"],
                        "entropy_stream_bytes": coded["physical_bytes"],
                        "savings_bytes": raw["physical_bytes"]-coded["physical_bytes"],
                        "huffman_tensors": coded["huffman_tensors"],
                        "raw_fallback_tensors": coded["raw_fallback_tensors"]})
    save_csv(out / "entropy_coding_summary.csv", entropy)
    checks = []
    for regime in ("raw", "entropy"):
        directory = root / regime
        for key in keys:
            source = load_torch(out / "codes" / (key + ".pt"))
            family, bits, decoded_stream = read_stream(directory / "streams" / (key + ".bin"))
            codes_match = set(source)==set(decoded_stream) and all(
                torch.equal(source[name]["quantized"].to(torch.uint8), decoded_stream[name]["quantized"])
                for name in source)
            if not codes_match:
                raise AssertionError(f"Code mismatch {regime}/{key}")
            restored = read_prefix(directory / "manifests" / (key + ".json"))
            expected = load_torch(out / "decoded_prefixes" / (key + ".pt"))
            weights_match = all(torch.equal(restored[name], expected[name]) for name in expected)
            if not weights_match:
                raise AssertionError(f"Weight mismatch {regime}/{key}")
            checks.append({"key": key, "regime": regime, "codes_bit_exact": codes_match,
                           "weights_bit_exact": weights_match, "eligible_tensors": len(expected),
                           "shared_tensors": len(restored)-len(expected)})
            print(f"Verified {regime}/{key}", flush=True)
        for base in FAMILIES:
            for scheme in SCHEMES:
                full = base_key(base)
                for family in scheme:
                    full = extend(full, family)
                path = directory / "manifests" / (full + ".json")
                for stop in lineage(full):
                    restored = read_prefix(path, stop_key=stop)
                    expected = load_torch(out / "decoded_prefixes" / (stop + ".pt"))
                    if any(not torch.equal(restored[name], expected[name]) for name in expected):
                        raise AssertionError(f"Hierarchy stop differs: {regime}/{full}/{stop}")
    write_json(out / "decode_verification.json", checks)


if __name__ == "__main__":
    main()
