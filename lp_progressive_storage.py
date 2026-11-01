"""Prefix-decodable raw and canonical-Huffman LP streams for Experiment 4.

Each unique prefix has its own manifest referring only to the shared FP32
state and streams needed at that precision. A decoder never loads future
enhancements or the original quantized FP32 weights.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import struct
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from evaluate_lp_progressive_schedules import SCHEDULES, all_prefixes, decode
from evaluate_lp_quantizer_adaptivity import load_torch, save_csv, sha256_file
from layered_quantization import IntegerFormat
from lp_uniform_calibration import dequantize_uniform
from storage_utils import BitWriter, canonical_huffman_codes, decode_huffman, huffman_code_lengths

MAGIC = b"LPS4"


def pack_codes(codes: np.ndarray, bits: int) -> bytes:
    flat = np.asarray(codes, dtype=np.uint8).reshape(-1)
    if bits not in (2, 5) or np.any(flat >= (1 << bits)):
        raise ValueError("Invalid bit width or code")
    bit_matrix = ((flat[:, None] >> np.arange(bits, dtype=np.uint8)) & 1).astype(np.uint8)
    return np.packbits(bit_matrix.reshape(-1), bitorder="little").tobytes()


def unpack_codes(payload: bytes, count: int, bits: int) -> np.ndarray:
    if len(payload) != (count * bits + 7) // 8:
        raise ValueError("Raw payload byte count differs")
    bits_array = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little")[:count*bits]
    matrix = bits_array.reshape(count, bits)
    return (matrix.astype(np.uint16) @ (1 << np.arange(bits, dtype=np.uint16))).astype(np.uint8)


def _huffman_payload(codes: np.ndarray, alphabet: int) -> tuple[bytes, bytes, int]:
    counts = np.bincount(codes, minlength=alphabet)
    lengths = huffman_code_lengths(Counter({i: int(n) for i, n in enumerate(counts) if n}))
    total_bits = sum(int(counts[symbol]) * length for symbol, length in lengths.items())
    table = bytes([lengths.get(i, 0) for i in range(alphabet)])
    if len(lengths) == 1:
        # A one-symbol stream carries its symbol in the codebook and has no
        # payload; length 1 distinguishes it from an absent alphabet entry.
        only = next(iter(lengths))
        return b"", bytes([1 if i == only else 0 for i in range(alphabet)]), 0
    out = io.BytesIO()
    writer = BitWriter(out)
    mapping = canonical_huffman_codes(lengths)
    for code in codes:
        writer.write(*mapping[int(code)])
    writer.flush()
    if writer.total_bits != total_bits:
        raise AssertionError("Huffman bit count changed")
    return out.getvalue(), table, total_bits


def _metadata(params: dict, family: str, channels: int) -> tuple[int, bytes]:
    if family in ("B", "A"):
        scale = params["scale"].reshape(-1).detach().cpu().numpy().astype("<f4")
        zp = params["zero_point"].reshape(-1).detach().cpu().numpy().astype("u1")
        if scale.size != channels or zp.size != channels:
            raise ValueError("Affine metadata channel count differs")
        fallback = params.get("t_min")
        needs_fallback = (scale == 0).any() and fallback is not None and bool(
            (fallback.reshape(-1).detach().cpu().numpy()[scale == 0] != 0).any())
        value = scale.tobytes() + zp.tobytes()
        if needs_fallback:
            value += fallback.reshape(-1).detach().cpu().numpy().astype("<f4").tobytes()
        return int(needs_fallback), value
    if family in ("U", "S"):
        scale = params["scale"].reshape(-1).detach().cpu().numpy().astype("<f4")
        if scale.size != channels:
            raise ValueError("Uniform metadata channel count differs")
        return 0, scale.tobytes()
    if family == "N":
        centers = params["centers"].detach().cpu().numpy().astype("<f4")
        if centers.shape != (channels, 4):
            raise ValueError("Lloyd metadata channel count differs")
        return 0, centers.tobytes()
    raise ValueError(family)


def write_stream(path: Path, family: str, artifacts: dict, regime: str) -> dict:
    if family not in ("B", "A", "S", "U", "N") or regime not in ("raw", "entropy"):
        raise ValueError("Unsupported stream family/regime")
    bits = 5 if family == "B" else 2
    path.parent.mkdir(parents=True, exist_ok=True)
    accounting = {"raw_code_bytes": 0, "payload_bytes": 0, "metadata_bytes": 0,
                  "framing_bytes": 8, "huffman_table_bytes": 0,
                  "huffman_tensors": 0, "raw_fallback_tensors": 0,
                  "huffman_payload_bits": 0}
    with path.open("wb") as handle:
        handle.write(MAGIC + family.encode() + struct.pack("<BH", bits, len(artifacts)))
        for name in sorted(artifacts):
            params = artifacts[name]
            codes = params["quantized"].detach().cpu().numpy().astype(np.uint8).reshape(-1)
            shape = tuple(params["quantized"].shape)
            channels = shape[0]
            raw = pack_codes(codes, bits)
            accounting["raw_code_bytes"] += len(raw)
            flag, meta = _metadata(params, family, channels)
            name_bytes = name.encode("utf-8")
            frame = (struct.pack("<H", len(name_bytes)) + name_bytes +
                     struct.pack("<B", len(shape)) + struct.pack("<" + "I"*len(shape), *shape) +
                     struct.pack("<IQB", channels, codes.size, flag))
            handle.write(frame)
            handle.write(meta)
            accounting["framing_bytes"] += len(frame) + 9  # mode + uint64 payload length
            accounting["metadata_bytes"] += len(meta)
            payload, table, payload_bits = raw, b"", 0
            if regime == "entropy":
                counts = np.bincount(codes, minlength=1 << bits)
                lengths = huffman_code_lengths(Counter({i: int(n) for i, n in enumerate(counts) if n}))
                expected = (sum(int(counts[i])*length for i, length in lengths.items()) + 7) // 8
                if expected + (1 << bits) < len(raw):
                    payload, table, payload_bits = _huffman_payload(codes, 1 << bits)
            mode = 1 if table else 0
            handle.write(struct.pack("<BQ", mode, len(payload)))
            if table:
                handle.write(table)
                accounting["huffman_table_bytes"] += len(table)
                accounting["huffman_tensors"] += 1
                accounting["huffman_payload_bits"] += payload_bits
            else:
                accounting["raw_fallback_tensors"] += 1
            handle.write(payload)
            accounting["payload_bytes"] += len(payload)
    accounting["physical_bytes"] = path.stat().st_size
    if accounting["physical_bytes"] != sum(accounting[k] for k in
                                           ("payload_bytes", "metadata_bytes", "framing_bytes", "huffman_table_bytes")):
        raise AssertionError("Stream physical size accounting differs")
    return accounting


def read_stream(path: Path) -> tuple[str, dict[str, dict]]:
    data = memoryview(path.read_bytes())
    if len(data) < 8 or bytes(data[:4]) != MAGIC:
        raise ValueError("Invalid stream magic")
    family = bytes(data[4:5]).decode("ascii")
    bits, tensors = struct.unpack_from("<BH", data, 5)
    if bits != (5 if family == "B" else 2) or family not in ("B", "A", "S", "U", "N"):
        raise ValueError("Invalid stream header")
    position = 8
    result = {}
    for _ in range(tensors):
        name_len = struct.unpack_from("<H", data, position)[0]
        position += 2
        name = bytes(data[position:position+name_len]).decode("utf-8")
        position += name_len
        ndim = data[position]
        position += 1
        shape = struct.unpack_from("<" + "I"*ndim, data, position)
        position += 4*ndim
        channels, count, fallback = struct.unpack_from("<IQB", data, position)
        position += 13
        if math.prod(shape) != count or shape[0] != channels:
            raise ValueError("Stream shape mismatch")
        view = (channels,) + (1,)*(ndim-1)
        params = {"family": family, "shape": shape, "bits": bits, "axis": 0}
        if family in ("B", "A"):
            scale = np.frombuffer(data[position:position+4*channels], dtype="<f4").copy()
            position += 4*channels
            zp = np.frombuffer(data[position:position+channels], dtype="u1").copy()
            position += channels
            params["scale"] = torch.from_numpy(scale.reshape(view))
            params["zero_point"] = torch.from_numpy(zp.reshape(view))
            if fallback:
                minimum = np.frombuffer(data[position:position+4*channels], dtype="<f4").copy()
                position += 4*channels
                params["t_min"] = torch.from_numpy(minimum.reshape(view))
            else:
                params["t_min"] = torch.zeros(view, dtype=torch.float32)
        elif family in ("U", "S"):
            scale = np.frombuffer(data[position:position+4*channels], dtype="<f4").copy()
            position += 4*channels
            params["scale"] = torch.from_numpy(scale.reshape(view))
        else:
            centers = np.frombuffer(data[position:position+16*channels], dtype="<f4").copy()
            position += 16*channels
            params["centers"] = torch.from_numpy(centers.reshape(channels, 4))
        mode, payload_len = struct.unpack_from("<BQ", data, position)
        position += 9
        if mode == 1:
            alphabet = 1 << bits
            table = bytes(data[position:position+alphabet])
            position += alphabet
            lengths = {i: int(length) for i, length in enumerate(table) if length}
            if not lengths:
                raise ValueError("Empty Huffman codebook")
            payload = bytes(data[position:position+payload_len])
            single = next(iter(lengths)) if len(lengths) == 1 else None
            codes = np.asarray(decode_huffman(payload, lengths, count, single), dtype=np.uint8)
        elif mode == 0:
            payload = bytes(data[position:position+payload_len])
            codes = unpack_codes(payload, count, bits)
        else:
            raise ValueError("Invalid codec mode")
        position += payload_len
        params["quantized"] = torch.from_numpy(codes.reshape(shape))
        if name in result:
            raise ValueError("Duplicate tensor")
        result[name] = params
    if position != len(data):
        raise ValueError("Trailing or truncated stream bytes")
    return family, result


def read_prefix(manifest_path: Path, *, stop_prefix: str | None = None,
                stream_cache: dict | None = None,
                fixed_cache: dict | None = None) -> dict[str, torch.Tensor]:
    """Reconstruct from serialized prefix bytes, with no FP32 source weights."""
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent.parent
    target_prefix = stop_prefix or manifest["prefix"]
    if not manifest["prefix"].startswith(target_prefix):
        raise ValueError("Requested prefix is not contained in this hierarchy")
    if len(manifest["streams"]) < len(target_prefix):
        raise ValueError("Prefix manifest is missing streams")
    fixed_path = root / manifest["shared_fp32"]
    if fixed_cache is not None and fixed_path in fixed_cache:
        fixed = fixed_cache[fixed_path]
    else:
        fixed = torch.load(fixed_path, map_location="cpu", weights_only=False)
        if fixed_cache is not None:
            fixed_cache[fixed_path] = fixed
    eligible = None
    for stream_file in manifest["streams"][:len(target_prefix)]:
        stream_path = root / stream_file
        if stream_cache is not None and stream_path in stream_cache:
            family, artifacts = stream_cache[stream_path]
        else:
            family, artifacts = read_stream(stream_path)
            if stream_cache is not None:
                stream_cache[stream_path] = (family, artifacts)
        if family == "B":
            eligible = {name: IntegerFormat.dequantize(params) for name, params in artifacts.items()}
        else:
            if eligible is None or set(eligible) != set(artifacts):
                raise ValueError("Enhancement requires matching base tensors")
            eligible = {name: eligible[name] + decode(params, family) for name, params in artifacts.items()}
    if eligible is None:
        raise ValueError("Missing base")
    return {**fixed, **eligible}


def prefix_manifest(root: Path, prefix: str, regime: str) -> Path:
    directory = root / regime / "manifests"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (prefix + ".json")
    streams = ["streams/B.bin"] + ["streams/" + prefix[:n] + ".bin" for n in range(2, len(prefix)+1)]
    path.write_text(json.dumps({"format": "LPS4", "prefix": prefix, "shared_fp32": "shared_fp32_state.pt",
                                "streams": streams}, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/lp_progressive_quantizer_study"))
    parser.add_argument("--saved-artifacts", type=Path, default=Path("outputs/lp_quantizer_adaptivity"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("outputs/residual_ablation_newfull"))
    return parser.parse_args()


def main():
    args = parse_args()
    out = args.outdir.resolve()
    serialized = out / "serialized"
    prefixes = all_prefixes()
    summary = {}
    fixed_source = out / "shared_fp32_state.pt"
    from evaluate_imagenet import build_model
    from analyze_mse_layer_sensitivity import eligible_groups
    run_meta = json.loads((out / "run_metadata.json").read_text())
    checkpoint = Path(os.environ.get("TORCH_HOME", str(Path.home() / ".cache" / "torch"))) / "hub/checkpoints/resnet50-11ad3fa6.pth"
    if sha256_file(checkpoint) != run_meta["checkpoint_sha256"]:
        raise AssertionError("Storage model checkpoint differs from evaluated model")
    model = build_model("resnet50")
    eligible = {name for group in eligible_groups(model).values() for name in group}
    fixed = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items() if name not in eligible}
    if not fixed_source.exists():
        torch.save(fixed, fixed_source)
    else:
        saved_fixed = load_torch(fixed_source)
        if set(saved_fixed) != set(fixed) or any(not torch.equal(saved_fixed[name], tensor)
                                                for name, tensor in fixed.items()):
            raise AssertionError("Serialized shared FP32 state changed")
    for regime in ("raw", "entropy"):
        root = serialized / regime
        root.mkdir(parents=True, exist_ok=True)
        shared = root / "shared_fp32_state.pt"
        if not shared.exists():
            import shutil
            shutil.copyfile(fixed_source, shared)
        elif sha256_file(shared) != sha256_file(fixed_source):
            raise AssertionError(f"Shared state file differs in {regime} representation")
        for prefix in prefixes:
            source = (args.artifacts_dir / "base_5_codes_and_metadata.pt" if prefix == "B" else
                      args.saved_artifacts / "codes" /
                      ({"BA": "E1_A", "BAA": "E2_A", "BAAA": "E3_A", "BN": "E1_N", "BAN": "E2_N", "BAAN": "E3_N"}[prefix] + ".pt")
                      if prefix in ("BA", "BAA", "BAAA", "BN", "BAN", "BAAN") else
                      out / "codes" / (prefix + ".pt"))
            target = root / "streams" / (prefix + ".bin")
            stats_file = target.with_suffix(".stats.json")
            source_sha = sha256_file(source)
            cached = json.loads(stats_file.read_text()) if stats_file.exists() else None
            if not target.exists() or cached is None or cached.get("source_sha256") != source_sha:
                result = write_stream(target, "B" if prefix == "B" else prefix[-1], load_torch(source), regime)
                result["source_sha256"] = source_sha
                stats_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
            else:
                result = cached
                if result["physical_bytes"] != target.stat().st_size:
                    raise AssertionError(f"Cached stream size differs: {prefix}")
            summary[(regime, prefix)] = result
            prefix_manifest(serialized, prefix, regime)
            print(f"Serialized {regime}/{prefix}: {target.stat().st_size} bytes", flush=True)
    isolated_streams = []
    isolated_prefixes = {(1, "A"): "BA", (1, "N"): "BN", (1, "U"): "BU",
                         (2, "A"): "BAA", (2, "N"): "BAN", (2, "U"): "BAU",
                         (3, "A"): "BAAA", (3, "N"): "BAAN", (3, "U"): "BAAU"}
    for stage in (1, 2, 3):
        for family in ("A", "S", "U0", "U", "N"):
            config = f"E{stage}_{family}"
            prefix = isolated_prefixes.get((stage, family))
            if prefix:
                result = summary[("raw", prefix)]
            else:
                source = (out / "codes" / (config + ".pt") if family == "U0" else
                          args.saved_artifacts / "codes" / (config + ".pt"))
                target = serialized / "raw" / "isolated" / (config + ".bin")
                stats_file = target.with_suffix(".stats.json")
                source_sha = sha256_file(source)
                cached = json.loads(stats_file.read_text()) if stats_file.exists() else None
                if not target.exists() or cached is None or cached.get("source_sha256") != source_sha:
                    result = write_stream(target, "U" if family == "U0" else family,
                                          load_torch(source), "raw")
                    result["source_sha256"] = source_sha
                    stats_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
                else:
                    result = cached
                original = load_torch(source)
                _, restored = read_stream(target)
                if not all(torch.equal(original[name]["quantized"].to(torch.uint8),
                                       restored[name]["quantized"]) for name in original):
                    raise AssertionError(f"Isolated stream code mismatch: {config}")
            isolated_streams.append({"config": config, "prefix": prefix or config, **result})
    (out / "isolated_stream_sizes.json").write_text(json.dumps(isolated_streams, indent=2), encoding="utf-8")
    rows = []
    for prefix in prefixes:
        for regime in ("raw", "entropy"):
            root = serialized / regime
            manifest = root / "manifests" / (prefix + ".json")
            files = [root / "shared_fp32_state.pt", manifest] + [root / name for name in json.loads(manifest.read_text())["streams"]]
            total = sum(path.stat().st_size for path in files)
            layer_stats = [summary[(regime, prefix[:n])] for n in range(1, len(prefix)+1)]
            payload_bytes = sum(item["payload_bytes"] for item in layer_stats)
            metadata_bytes = sum(item["metadata_bytes"] for item in layer_stats)
            table_bytes = sum(item["huffman_table_bytes"] for item in layer_stats)
            framing_bytes = sum(item["framing_bytes"] for item in layer_stats) + manifest.stat().st_size
            if total != (payload_bytes + metadata_bytes + table_bytes + framing_bytes +
                         (root / "shared_fp32_state.pt").stat().st_size):
                raise AssertionError(f"Prefix size accounting differs: {regime}/{prefix}")
            previous = prefix[:-1] if prefix != "B" else None
            prev_total = None
            if previous:
                prev_manifest = root / "manifests" / (previous + ".json")
                prev_files = [root / "shared_fp32_state.pt", prev_manifest] + [root / name for name in json.loads(prev_manifest.read_text())["streams"]]
                prev_total = sum(path.stat().st_size for path in prev_files)
            rows.append({"prefix": prefix, "regime": regime, "nominal_bits": 5+2*(len(prefix)-1),
                         "base_code_and_metadata_bytes": (root / "streams/B.bin").stat().st_size,
                         "base_code_bytes": layer_stats[0]["payload_bytes"],
                         "base_metadata_bytes": layer_stats[0]["metadata_bytes"],
                         "enhancement_code_bytes": payload_bytes-layer_stats[0]["payload_bytes"],
                         "enhancement_metadata_bytes": metadata_bytes-layer_stats[0]["metadata_bytes"],
                         "entropy_coding_tables_bytes": table_bytes,
                         "framing_bytes": framing_bytes,
                         "incremental_code_bytes": layer_stats[-1]["payload_bytes"] if previous else layer_stats[0]["payload_bytes"],
                         "incremental_metadata_bytes": layer_stats[-1]["metadata_bytes"] if previous else layer_stats[0]["metadata_bytes"],
                         "incremental_huffman_table_bytes": layer_stats[-1]["huffman_table_bytes"] if previous else layer_stats[0]["huffman_table_bytes"],
                         "incremental_stream_framing_bytes": layer_stats[-1]["framing_bytes"] if previous else layer_stats[0]["framing_bytes"],
                         "incremental_stream_bytes": (root / "streams" / (prefix + ".bin")).stat().st_size if previous else 0,
                         "shared_fp32_bytes": (root / "shared_fp32_state.pt").stat().st_size,
                         "manifest_bytes": manifest.stat().st_size, "total_bytes": total,
                         "incremental_total_bytes": total-prev_total if previous else total,
                         "effective_bpw_excluding_shared": 8*(total-(root / "shared_fp32_state.pt").stat().st_size)/25_502_912,
                         "effective_bpw_including_shared": 8*total/25_502_912})
    from analyze_mse_layer_sensitivity import write_json
    write_json(out / "prefix_storage.json", rows)
    save_csv(out / "prefix_storage.csv", rows)
    entropy_rows = []
    for prefix in prefixes:
        raw = summary[("raw", prefix)]
        entropy = summary[("entropy", prefix)]
        entropy_rows.append({"prefix": prefix, "raw_bytes": raw["physical_bytes"],
                             "entropy_bytes": entropy["physical_bytes"],
                             "savings_bytes": raw["physical_bytes"]-entropy["physical_bytes"],
                             "huffman_tensors": entropy.get("huffman_tensors"),
                             "raw_fallback_tensors": entropy.get("raw_fallback_tensors")})
    save_csv(out / "entropy_coding_summary.csv", entropy_rows)
    verification = []
    representative = {prefix: next("B"+schedule for schedule in SCHEDULES
                                    if ("B"+schedule).startswith(prefix)) for prefix in prefixes}
    for regime in ("raw", "entropy"):
        stream_cache, fixed_cache = {}, {}
        for prefix in prefixes:
            expected_path = (args.saved_artifacts / "decoded_prefixes" / (prefix+"_eligible_weights.pt")
                             if prefix in ("B", "BA", "BAA", "BAAA") else out / "decoded_prefixes" / (prefix+".pt"))
            expected = load_torch(expected_path)
            source = (args.artifacts_dir / "base_5_codes_and_metadata.pt" if prefix == "B" else
                      args.saved_artifacts / "codes" /
                      ({"BA": "E1_A", "BAA": "E2_A", "BAAA": "E3_A", "BN": "E1_N", "BAN": "E2_N", "BAAN": "E3_N"}[prefix] + ".pt")
                      if prefix in ("BA", "BAA", "BAAA", "BN", "BAN", "BAAN") else
                      out / "codes" / (prefix + ".pt"))
            original_codes = load_torch(source)
            stream_path = serialized / regime / "streams" / (prefix + ".bin")
            decoded_stream = read_stream(stream_path)
            stream_cache[stream_path] = decoded_stream
            _, serialized_codes = decoded_stream
            codes_match = all(torch.equal(serialized_codes[name]["quantized"].to(original_codes[name]["quantized"].dtype),
                                          original_codes[name]["quantized"])
                              for name in original_codes)
            if not codes_match:
                raise AssertionError(f"Code index mismatch {regime}/{prefix}")
            state = read_prefix(serialized / regime / "manifests" / (prefix + ".json"),
                                stream_cache=stream_cache, fixed_cache=fixed_cache)
            matches = all(torch.equal(state[name], expected[name]) for name in expected)
            if not matches:
                raise AssertionError(f"Numerical prefix mismatch {regime}/{prefix}")
            full_state = read_prefix(serialized / regime / "manifests" /
                                     (representative[prefix] + ".json"),
                                     stop_prefix=prefix, stream_cache=stream_cache,
                                     fixed_cache=fixed_cache)
            full_matches = all(torch.equal(full_state[name], expected[name]) for name in expected)
            if not full_matches:
                raise AssertionError(f"Complete hierarchy stop mismatch {regime}/{prefix}")
            verification.append({"prefix": prefix, "regime": regime,
                                 "all_code_indices_bit_exact": codes_match,
                                 "all_eligible_weights_bit_exact": matches,
                                 "full_hierarchy_stop_bit_exact": full_matches,
                                 "tensors": len(expected), "shared_tensors": len(state)-len(expected)})
            print(f"Verified serialized prefix {regime}/{prefix}", flush=True)
    write_json(out / "decode_verification.json", verification)


if __name__ == "__main__":
    main()
