"""Entropy coding and complete storage accounting for LP parameter streams."""

from __future__ import annotations

import heapq
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch


def empirical_entropy_bits_per_symbol(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
        if count > 0
    )


def huffman_code_lengths(counts: Counter) -> dict[int, int]:
    items = [(int(symbol), int(count)) for symbol, count in counts.items() if count > 0]
    if not items:
        return {}
    if len(items) == 1:
        return {items[0][0]: 0}
    heap: list[tuple[int, int, Any]] = []
    uid = 0
    for symbol, count in items:
        heapq.heappush(heap, (count, uid, symbol))
        uid += 1
    while len(heap) > 1:
        count_a, _, node_a = heapq.heappop(heap)
        count_b, _, node_b = heapq.heappop(heap)
        heapq.heappush(heap, (count_a + count_b, uid, (node_a, node_b)))
        uid += 1
    lengths: dict[int, int] = {}

    def visit(node, depth):
        if isinstance(node, int):
            lengths[node] = max(depth, 1)
            return
        visit(node[0], depth + 1)
        visit(node[1], depth + 1)

    visit(heap[0][2], 0)
    return lengths


def canonical_huffman_codes(lengths: dict[int, int]) -> dict[int, tuple[int, int]]:
    if not lengths:
        return {}
    nonzero = sorted(
        ((symbol, length) for symbol, length in lengths.items() if length > 0),
        key=lambda item: (item[1], item[0]),
    )
    if not nonzero:
        symbol = next(iter(lengths))
        return {symbol: (0, 0)}
    codes: dict[int, tuple[int, int]] = {}
    code = 0
    previous_length = nonzero[0][1]
    for symbol, length in nonzero:
        if length > previous_length:
            code <<= length - previous_length
            previous_length = length
        codes[symbol] = (code, length)
        code += 1
    return codes


class BitWriter:
    def __init__(self, handle):
        self.handle = handle
        self.buffer = 0
        self.buffer_bits = 0
        self.total_bits = 0

    def write(self, code: int, length: int) -> None:
        if length == 0:
            return
        self.buffer = (self.buffer << length) | (code & ((1 << length) - 1))
        self.buffer_bits += length
        self.total_bits += length
        while self.buffer_bits >= 8:
            shift = self.buffer_bits - 8
            self.handle.write(bytes([(self.buffer >> shift) & 0xFF]))
            self.buffer_bits -= 8
            self.buffer &= (1 << shift) - 1

    def flush(self) -> None:
        if self.buffer_bits:
            self.handle.write(bytes([(self.buffer << (8 - self.buffer_bits)) & 0xFF]))
            self.buffer = 0
            self.buffer_bits = 0


def decode_huffman(
    payload: bytes,
    lengths: dict[int, int],
    num_symbols: int,
    single_symbol: int | None = None,
) -> list[int]:
    if num_symbols == 0:
        return []
    if single_symbol is not None:
        return [single_symbol] * num_symbols
    reverse = {
        (code, length): symbol
        for symbol, (code, length) in canonical_huffman_codes(lengths).items()
    }
    decoded: list[int] = []
    code = length = 0
    for byte in payload:
        for shift in range(7, -1, -1):
            code = (code << 1) | ((byte >> shift) & 1)
            length += 1
            symbol = reverse.get((code, length))
            if symbol is None:
                continue
            decoded.append(symbol)
            code = length = 0
            if len(decoded) == num_symbols:
                return decoded
    raise ValueError("Huffman payload ended before all symbols were decoded")


def _write_huffman_stream(
    symbols: Sequence[int],
    lengths: dict[int, int],
    outdir: Path,
    name: str,
    write_codebook: bool = True,
) -> dict[str, int]:
    outdir.mkdir(parents=True, exist_ok=True)
    payload_path = outdir / f"{name}_payload.bin"
    codebook_path = outdir / f"{name}_codebook.json"
    header_path = outdir / f"{name}_header.json"
    counts = Counter(symbols)
    header = {
        "num_symbols": len(symbols),
        "unique_symbols": len(counts),
        "single_symbol": (
            int(next(iter(counts))) if len(counts) == 1 else None
        ),
    }
    if write_codebook:
        codebook_path.write_text(
            json.dumps({str(key): int(value) for key, value in lengths.items()}, indent=2),
            encoding="utf-8",
        )
    header_path.write_text(json.dumps(header, indent=2), encoding="utf-8")
    payload_bits = 0
    with payload_path.open("wb") as handle:
        if symbols and len(counts) > 1:
            codes = canonical_huffman_codes(lengths)
            writer = BitWriter(handle)
            for symbol in symbols:
                writer.write(*codes[int(symbol)])
            payload_bits = writer.total_bits
            writer.flush()
    return {
        "payload_bits": payload_bits,
        "payload_bytes": payload_path.stat().st_size,
        "codebook_bytes": codebook_path.stat().st_size if write_codebook else 0,
        "header_bytes": header_path.stat().st_size,
    }


def _write_raw_stream(
    symbols: Sequence[int],
    bits_per_symbol: int,
    outdir: Path,
    name: str,
) -> dict[str, int]:
    outdir.mkdir(parents=True, exist_ok=True)
    payload_path = outdir / f"{name}_payload.bin"
    header_path = outdir / f"{name}_header.json"
    count = len(symbols)
    if count and bits_per_symbol > 0:
        values = np.asarray(symbols, dtype=np.int64) & ((1 << bits_per_symbol) - 1)
        bits = np.empty(count * bits_per_symbol, dtype=np.uint8)
        for bit in range(bits_per_symbol):
            bits[bit::bits_per_symbol] = ((values >> bit) & 1).astype(np.uint8)
        payload_path.write_bytes(np.packbits(bits, bitorder="little").tobytes())
    else:
        payload_path.write_bytes(b"")
    header_path.write_text(
        json.dumps({"num_symbols": count, "bits_per_symbol": bits_per_symbol}),
        encoding="utf-8",
    )
    return {
        "payload_bits": count * bits_per_symbol,
        "payload_bytes": payload_path.stat().st_size,
        "codebook_bytes": 0,
        "header_bytes": header_path.stat().st_size,
    }


def _temporary_or_output(base: Path | None, suffix: str) -> tuple[Path, bool]:
    if base is None:
        return Path(tempfile.mkdtemp(suffix=suffix)), True
    path = Path(str(base) + suffix)
    path.mkdir(parents=True, exist_ok=True)
    return path, False


def fixed_state_for_model(
    model: torch.nn.Module,
    per_parameter_info: dict[str, dict],
) -> dict[str, torch.Tensor]:
    """Return parameters/buffers that are not represented by quantized streams."""
    quantized = set(per_parameter_info)
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if name not in quantized
    }


def _symbol_counts(
    tensor: torch.Tensor,
    chunk_size: int = 1_000_000,
) -> Counter:
    flattened = tensor.detach().reshape(-1)
    frequencies_by_symbol: dict[int, int] = {}
    first_seen_order: list[int] = []
    for start in range(0, flattened.numel(), chunk_size):
        values = (
            flattened[start:start + chunk_size]
            .to(device="cpu", dtype=torch.int64)
            .numpy()
        )
        unique, first_indices, frequencies = np.unique(
            values, return_index=True, return_counts=True
        )
        for index in np.argsort(first_indices):
            symbol = int(unique[index])
            if symbol not in frequencies_by_symbol:
                frequencies_by_symbol[symbol] = 0
                first_seen_order.append(symbol)
            frequencies_by_symbol[symbol] += int(frequencies[index])
    return Counter({
        symbol: frequencies_by_symbol[symbol]
        for symbol in first_seen_order
    })


def _stream_data(
    per_parameter_info: dict[str, dict],
    layers_to_use: int | None,
) -> tuple[dict[int, list[torch.Tensor]], dict, dict[int, int], dict[int, Counter]]:
    streams: dict[int, list[torch.Tensor]] = defaultdict(list)
    counts: dict[int, Counter] = defaultdict(Counter)
    offsets: dict[int, int] = defaultdict(int)
    layer_bits: dict[int, int] = {}
    metadata = {"schema_version": 1, "parameters": []}
    for name, info in per_parameter_info.items():
        layers = info["layers_info"]
        depth = len(layers) if layers_to_use is None else min(layers_to_use, len(layers))
        parameter_record = {
            "name": name,
            "shape": tuple(info["original_shape"]),
            "layers": [],
        }
        for layer_index in range(depth):
            params = layers[layer_index]["quant_params"]
            quantized = params["quantized"].detach()
            num_symbols = quantized.numel()
            streams[layer_index].append(quantized)
            counts[layer_index].update(_symbol_counts(quantized))
            layer_bits[layer_index] = int(params["bits"])
            parameter_record["layers"].append({
                "layer_index": layer_index,
                "offset": offsets[layer_index],
                "num_symbols": num_symbols,
                "bits": int(params["bits"]),
                "scale": params["scale"].detach().cpu(),
                "zero_point": params["zero_point"].detach().cpu(),
                "axis": params.get("axis"),
                "granularity": params.get("selected_granularity"),
            })
            offsets[layer_index] += num_symbols
        metadata["parameters"].append(parameter_record)
    return streams, metadata, layer_bits, counts


def _text_file_size(payload: str) -> int:
    translated = payload.replace("\n", os.linesep)
    return len(translated.encode("utf-8"))


def _codebook_size(lengths: dict[int, int]) -> int:
    return _text_file_size(
        json.dumps({str(key): int(value) for key, value in lengths.items()}, indent=2)
    )


def _huffman_header(counts: Counter) -> dict[str, int | None]:
    return {
        "num_symbols": int(sum(counts.values())),
        "unique_symbols": len(counts),
        "single_symbol": int(next(iter(counts))) if len(counts) == 1 else None,
    }


def _huffman_header_size(counts: Counter) -> int:
    return _text_file_size(json.dumps(_huffman_header(counts), indent=2))


def _raw_header_size(num_symbols: int, bits_per_symbol: int) -> int:
    return _text_file_size(json.dumps({
        "num_symbols": num_symbols,
        "bits_per_symbol": bits_per_symbol,
    }))


def _iter_tensor_values(
    tensors: Sequence[torch.Tensor],
    chunk_size: int = 1_000_000,
) -> Iterable[list[int]]:
    for tensor in tensors:
        flattened = tensor.detach().reshape(-1)
        for start in range(0, flattened.numel(), chunk_size):
            yield (
                flattened[start:start + chunk_size]
                .to(device="cpu", dtype=torch.int64)
                .tolist()
            )


def _write_huffman_tensor_stream(
    tensors: Sequence[torch.Tensor],
    counts: Counter,
    lengths: dict[int, int],
    outdir: Path,
    name: str,
    write_codebook: bool = True,
) -> dict[str, int]:
    outdir.mkdir(parents=True, exist_ok=True)
    payload_path = outdir / f"{name}_payload.bin"
    codebook_path = outdir / f"{name}_codebook.json"
    header_path = outdir / f"{name}_header.json"
    if write_codebook:
        codebook_path.write_text(
            json.dumps({str(key): int(value) for key, value in lengths.items()}, indent=2),
            encoding="utf-8",
        )
    header_path.write_text(
        json.dumps(_huffman_header(counts), indent=2), encoding="utf-8"
    )
    payload_bits = sum(counts[symbol] * lengths[symbol] for symbol in counts)
    with payload_path.open("wb") as handle:
        if sum(counts.values()) and len(counts) > 1:
            codes = canonical_huffman_codes(lengths)
            writer = BitWriter(handle)
            for chunk in _iter_tensor_values(tensors):
                for symbol in chunk:
                    writer.write(*codes[symbol])
            writer.flush()
    return {
        "payload_bits": payload_bits,
        "payload_bytes": payload_path.stat().st_size,
        "codebook_bytes": codebook_path.stat().st_size if write_codebook else 0,
        "header_bytes": header_path.stat().st_size,
    }


def _write_raw_tensor_stream(
    tensors: Sequence[torch.Tensor],
    bits_per_symbol: int,
    outdir: Path,
    name: str,
) -> dict[str, int]:
    outdir.mkdir(parents=True, exist_ok=True)
    payload_path = outdir / f"{name}_payload.bin"
    header_path = outdir / f"{name}_header.json"
    num_symbols = sum(tensor.numel() for tensor in tensors)
    pending = np.empty(0, dtype=np.uint8)
    with payload_path.open("wb") as handle:
        for chunk in _iter_tensor_values(tensors):
            values = np.asarray(chunk, dtype=np.int64) & ((1 << bits_per_symbol) - 1)
            bit_array = np.empty(values.size * bits_per_symbol, dtype=np.uint8)
            for bit in range(bits_per_symbol):
                bit_array[bit::bits_per_symbol] = (
                    (values >> bit) & 1
                ).astype(np.uint8)
            if pending.size:
                bit_array = np.concatenate((pending, bit_array))
            complete = (bit_array.size // 8) * 8
            if complete:
                handle.write(
                    np.packbits(
                        bit_array[:complete], bitorder="little"
                    ).tobytes()
                )
            pending = bit_array[complete:].copy()
        if pending.size:
            handle.write(np.packbits(pending, bitorder="little").tobytes())
    header_path.write_text(
        json.dumps({
            "num_symbols": num_symbols,
            "bits_per_symbol": bits_per_symbol,
        }),
        encoding="utf-8",
    )
    return {
        "payload_bits": num_symbols * bits_per_symbol,
        "payload_bytes": payload_path.stat().st_size,
        "codebook_bytes": 0,
        "header_bytes": header_path.stat().st_size,
    }


def _write_side_information(
    outdir: Path,
    metadata: dict,
    fixed_state: dict[str, torch.Tensor] | None,
) -> tuple[int, int]:
    metadata_path = outdir / "quantizer_metadata.pt"
    torch.save(metadata, metadata_path)
    fixed_bytes = 0
    if fixed_state:
        fixed_path = outdir / "fixed_fp32_state.pt"
        torch.save(fixed_state, fixed_path)
        fixed_bytes = fixed_path.stat().st_size
    return metadata_path.stat().st_size, fixed_bytes


def measure_layered_storage(
    per_parameter_info: dict[str, dict],
    layers_to_use: int | None = None,
    fixed_state: dict[str, torch.Tensor] | None = None,
    write_path: str | Path | None = None,
) -> dict[str, float]:
    """Measure historical payload-only and complete decodable storage.

    ``storage_bytes`` fields reproduce the historical symbol-stream accounting.
    ``complete_storage_bytes`` fields additionally include quantizer metadata
    and all unquantized FP32 state required to reconstruct the model.
    """
    streams, metadata, layer_bits, counts = _stream_data(
        per_parameter_info, layers_to_use
    )
    lengths = {
        layer: huffman_code_lengths(layer_counts)
        for layer, layer_counts in counts.items()
    }
    entropy_bits = sum(
        empirical_entropy_bits_per_symbol(layer_counts) * sum(layer_counts.values())
        for layer_counts in counts.values()
    )
    global_counts = Counter()
    for layer_counts in counts.values():
        global_counts.update(layer_counts)
    global_lengths = huffman_code_lengths(global_counts)
    total_symbols = sum(sum(layer_counts.values()) for layer_counts in counts.values())

    base = Path(write_path) if write_path is not None else None
    perlayer_dir, perlayer_temp = _temporary_or_output(base, ".perlayer")
    global_dir, global_temp = _temporary_or_output(base, ".global")
    raw_dir, raw_temp = _temporary_or_output(base, ".raw")
    try:
        perlayer = {"huffman_bits": 0, "payload_bytes": 0, "codebook_bytes": 0, "header_bytes": 0}
        global_stats = {"huffman_bits": 0, "payload_bytes": 0, "codebook_bytes": 0, "header_bytes": 0}
        raw = {"raw_bits": 0, "payload_bytes": 0, "header_bytes": 0}
        keep_artifacts = base is not None
        for layer in sorted(streams):
            layer_counts = counts[layer]
            layer_huffman_bits = sum(
                layer_counts[symbol] * lengths[layer][symbol]
                for symbol in layer_counts
            )
            perlayer["huffman_bits"] += layer_huffman_bits
            perlayer["payload_bytes"] += math.ceil(layer_huffman_bits / 8)
            perlayer["codebook_bytes"] += _codebook_size(lengths[layer])
            perlayer["header_bytes"] += _huffman_header_size(layer_counts)

            layer_global_bits = sum(
                layer_counts[symbol] * global_lengths[symbol]
                for symbol in layer_counts
            )
            global_stats["huffman_bits"] += layer_global_bits
            if len(layer_counts) > 1:
                global_stats["payload_bytes"] += math.ceil(layer_global_bits / 8)
            global_stats["header_bytes"] += _huffman_header_size(layer_counts)

            num_layer_symbols = sum(layer_counts.values())
            layer_raw_bits = num_layer_symbols * layer_bits[layer]
            raw["raw_bits"] += layer_raw_bits
            raw["payload_bytes"] += math.ceil(layer_raw_bits / 8)
            raw["header_bytes"] += _raw_header_size(
                num_layer_symbols, layer_bits[layer]
            )

            if keep_artifacts:
                per_result = _write_huffman_tensor_stream(
                    streams[layer],
                    layer_counts,
                    lengths[layer],
                    perlayer_dir,
                    f"layer{layer}",
                )
                global_result = _write_huffman_tensor_stream(
                    streams[layer],
                    layer_counts,
                    global_lengths,
                    global_dir,
                    f"layer{layer}",
                    write_codebook=False,
                )
                raw_result = _write_raw_tensor_stream(
                    streams[layer],
                    layer_bits[layer],
                    raw_dir,
                    f"layer{layer}",
                )
                expected = {
                    "perlayer": (
                        math.ceil(layer_huffman_bits / 8),
                        _codebook_size(lengths[layer]),
                        _huffman_header_size(layer_counts),
                    ),
                    "global": (
                        math.ceil(layer_global_bits / 8)
                        if len(layer_counts) > 1 else 0,
                        0,
                        _huffman_header_size(layer_counts),
                    ),
                    "raw": (
                        math.ceil(layer_raw_bits / 8),
                        0,
                        _raw_header_size(num_layer_symbols, layer_bits[layer]),
                    ),
                }
                for label, result in (
                    ("perlayer", per_result),
                    ("global", global_result),
                    ("raw", raw_result),
                ):
                    observed = (
                        result["payload_bytes"],
                        result["codebook_bytes"],
                        result["header_bytes"],
                    )
                    if observed != expected[label]:
                        raise RuntimeError(
                            f"{label} stream size mismatch: {observed} != "
                            f"{expected[label]}"
                        )

        global_stats["codebook_bytes"] = _codebook_size(global_lengths)
        if keep_artifacts:
            global_codebook_path = global_dir / "global_codebook.json"
            global_codebook_path.write_text(
                json.dumps(
                    {str(key): int(value) for key, value in global_lengths.items()},
                    indent=2,
                ),
                encoding="utf-8",
            )
            if global_codebook_path.stat().st_size != global_stats["codebook_bytes"]:
                raise RuntimeError("Global Huffman codebook size mismatch")

        side_sizes = {}
        for label, directory in (
            ("perlayer", perlayer_dir),
            ("global", global_dir),
            ("raw", raw_dir),
        ):
            metadata_bytes, fixed_bytes = _write_side_information(
                directory, metadata, fixed_state
            )
            side_sizes[label] = (metadata_bytes, fixed_bytes)

        perlayer_storage = sum(
            perlayer[field]
            for field in ("payload_bytes", "codebook_bytes", "header_bytes")
        )
        global_storage = sum(
            global_stats[field]
            for field in ("payload_bytes", "codebook_bytes", "header_bytes")
        )
        raw_storage = raw["payload_bytes"] + raw["header_bytes"]
        denominator = float(total_symbols) if total_symbols else 1.0
        per_meta, per_fixed = side_sizes["perlayer"]
        global_meta, global_fixed = side_sizes["global"]
        raw_meta, raw_fixed = side_sizes["raw"]
        return {
            "entropy_bits": float(entropy_bits),
            "entropy_bpw": float(entropy_bits) / denominator,
            "huffman_bits": float(perlayer["huffman_bits"]),
            "huffman_bpw": float(perlayer["huffman_bits"]) / denominator,
            "payload_bytes": float(perlayer["payload_bytes"]),
            "codebook_bytes": float(perlayer["codebook_bytes"]),
            "header_bytes": float(perlayer["header_bytes"]),
            "storage_bytes": float(perlayer_storage),
            "metadata_bytes": float(per_meta),
            "fixed_state_bytes": float(per_fixed),
            "complete_storage_bytes": float(perlayer_storage + per_meta + per_fixed),
            "global_huffman_bits": float(global_stats["huffman_bits"]),
            "global_huffman_bpw": float(global_stats["huffman_bits"]) / denominator,
            "global_payload_bytes": float(global_stats["payload_bytes"]),
            "global_codebook_bytes": float(global_stats["codebook_bytes"]),
            "global_header_bytes": float(global_stats["header_bytes"]),
            "global_storage_bytes": float(global_storage),
            "global_metadata_bytes": float(global_meta),
            "global_fixed_state_bytes": float(global_fixed),
            "global_complete_storage_bytes": float(
                global_storage + global_meta + global_fixed
            ),
            "raw_bits": float(raw["raw_bits"]),
            "raw_bpw": float(raw["raw_bits"]) / denominator,
            "raw_payload_bytes": float(raw["payload_bytes"]),
            "raw_codebook_bytes": 0.0,
            "raw_header_bytes": float(raw["header_bytes"]),
            "raw_storage_bytes": float(raw_storage),
            "raw_metadata_bytes": float(raw_meta),
            "raw_fixed_state_bytes": float(raw_fixed),
            "raw_complete_storage_bytes": float(raw_storage + raw_meta + raw_fixed),
            "num_symbols": total_symbols,
        }
    finally:
        for directory, temporary in (
            (perlayer_dir, perlayer_temp),
            (global_dir, global_temp),
            (raw_dir, raw_temp),
        ):
            if temporary:
                shutil.rmtree(directory, ignore_errors=True)


def _read_huffman_artifact_layer(
    directory: Path,
    layer_index: int,
    global_codebook: bool,
) -> list[int]:
    codebook_name = (
        "global_codebook.json"
        if global_codebook else f"layer{layer_index}_codebook.json"
    )
    lengths = {
        int(symbol): int(length)
        for symbol, length in json.loads(
            (directory / codebook_name).read_text(encoding="utf-8")
        ).items()
    }
    header = json.loads(
        (directory / f"layer{layer_index}_header.json").read_text(
            encoding="utf-8"
        )
    )
    return decode_huffman(
        (directory / f"layer{layer_index}_payload.bin").read_bytes(),
        lengths,
        int(header["num_symbols"]),
        header.get("single_symbol"),
    )


def _read_raw_artifact_layer(directory: Path, layer_index: int) -> list[int]:
    header = json.loads(
        (directory / f"layer{layer_index}_header.json").read_text(
            encoding="utf-8"
        )
    )
    num_symbols = int(header["num_symbols"])
    bits_per_symbol = int(header["bits_per_symbol"])
    if num_symbols == 0:
        return []
    packed = np.frombuffer(
        (directory / f"layer{layer_index}_payload.bin").read_bytes(),
        dtype=np.uint8,
    )
    bits = np.unpackbits(packed, bitorder="little")[
        :num_symbols * bits_per_symbol
    ].reshape(num_symbols, bits_per_symbol)
    powers = (1 << np.arange(bits_per_symbol, dtype=np.int64)).reshape(1, -1)
    return (bits.astype(np.int64) * powers).sum(axis=1).tolist()


def load_layered_state(
    artifact_directory: str | Path,
    regime: str = "perlayer",
) -> dict[str, torch.Tensor]:
    """Decode one complete artifact directory into a model state dictionary.

    ``regime`` must match the directory passed by the caller: ``perlayer`` for
    one Huffman codebook per LP layer, ``global`` for the shared Huffman
    codebook, or ``raw`` for tight bit-packing.
    """
    if regime not in {"perlayer", "global", "raw"}:
        raise ValueError("regime must be 'perlayer', 'global', or 'raw'")
    directory = Path(artifact_directory)
    metadata = torch.load(
        directory / "quantizer_metadata.pt",
        map_location="cpu",
        weights_only=False,
    )
    fixed_path = directory / "fixed_fp32_state.pt"
    state = (
        torch.load(fixed_path, map_location="cpu", weights_only=False)
        if fixed_path.is_file() else {}
    )
    layer_indices = sorted({
        int(layer["layer_index"])
        for parameter in metadata["parameters"]
        for layer in parameter["layers"]
    })
    decoded_layers = {}
    for layer_index in layer_indices:
        if regime == "raw":
            decoded_layers[layer_index] = _read_raw_artifact_layer(
                directory, layer_index
            )
        else:
            decoded_layers[layer_index] = _read_huffman_artifact_layer(
                directory,
                layer_index,
                global_codebook=regime == "global",
            )

    for parameter in metadata["parameters"]:
        reconstruction = None
        shape = tuple(parameter["shape"])
        for layer in parameter["layers"]:
            layer_index = int(layer["layer_index"])
            offset = int(layer["offset"])
            count = int(layer["num_symbols"])
            quantized = torch.tensor(
                decoded_layers[layer_index][offset:offset + count],
                dtype=torch.float32,
            ).reshape(shape)
            scale = layer["scale"].float()
            zero_point = layer["zero_point"].float()
            contribution = scale * (quantized - zero_point)
            reconstruction = (
                contribution
                if reconstruction is None else reconstruction + contribution
            )
        if reconstruction is None:
            raise ValueError(f"No decoded layers for {parameter['name']}")
        state[parameter["name"]] = reconstruction
    return state


def saving_percent(layered: float, simulcast: float) -> float:
    return 0.0 if simulcast == 0 else 100.0 * (simulcast - layered) / simulcast
