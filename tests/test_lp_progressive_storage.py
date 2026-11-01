import json

import numpy as np
import torch

from layered_quantization import IntegerFormat
from lp_progressive_storage import (pack_codes, prefix_manifest, read_prefix,
                                    read_stream, unpack_codes, write_stream)
from lp_uniform_calibration import quantize_uniform
from lp_quantizer_families import quantize_family, dequantize_family


def test_five_and_two_bit_pack_roundtrip():
    for bits in (2, 5):
        codes = np.arange(101, dtype=np.uint8) % (1 << bits)
        assert np.array_equal(unpack_codes(pack_codes(codes, bits), len(codes), bits), codes)


def test_raw_entropy_and_prefix_independence(tmp_path):
    weight = torch.tensor([[-0.5, -0.2, 0.1, 0.5, 0.6],
                           [-0.6, -0.3, 0.0, 0.2, 0.7]])
    base_params, base = IntegerFormat("channel", 0).quantize(weight, 5)
    u_params, correction = quantize_uniform(weight-base, "U")
    for regime in ("raw", "entropy"):
        root = tmp_path / regime
        root.mkdir()
        torch.save({"bn.weight": torch.tensor([1.0])}, root / "shared_fp32_state.pt")
        write_stream(root / "streams/B.bin", "B", {"w": base_params}, regime)
        write_stream(root / "streams/BU.bin", "U", {"w": u_params}, regime)
        before = prefix_manifest(tmp_path, "B", regime)
        after = prefix_manifest(tmp_path, "BU", regime)
        assert read_stream(root / "streams/B.bin")[0] == "B"
        assert torch.equal(read_prefix(after)["w"], base+correction)
        (root / "streams/BU.bin").unlink()
        assert torch.equal(read_prefix(before)["w"], base)
        assert torch.equal(read_prefix(after, stop_prefix="B")["w"], base)


def test_entropy_single_symbol_table_roundtrip(tmp_path):
    codes = torch.zeros((1, 512), dtype=torch.uint8)
    params = {"quantized": codes, "scale": torch.ones((1, 1))}
    path = tmp_path / "single.bin"
    stats = write_stream(path, "U", {"w": params}, "entropy")
    assert stats["huffman_tensors"] == 1
    _, decoded = read_stream(path)
    assert torch.equal(decoded["w"]["quantized"], codes)


def test_symmetric_reference_stream_roundtrip(tmp_path):
    values = torch.tensor([[-1.2, -0.4, 0.25, 0.8, 1.1]])
    params, decoded = quantize_family(values, "S")
    path = tmp_path / "s.bin"
    write_stream(path, "S", {"w": params}, "raw")
    family, saved = read_stream(path)
    assert family == "S"
    assert torch.equal(saved["w"]["quantized"], params["quantized"])
    assert torch.equal(dequantize_family(saved["w"], "S"), decoded)


def test_constant_negative_affine_channel_is_serialized(tmp_path):
    values = torch.full((2, 7), -0.75)
    params, decoded = IntegerFormat("channel", 0).quantize(values, 5)
    path = tmp_path / "constant_base.bin"
    stats = write_stream(path, "B", {"w": params}, "raw")
    _, restored = read_stream(path)
    assert stats["metadata_bytes"] == 2 * (4 + 1)
    assert torch.equal(IntegerFormat.dequantize(restored["w"]), decoded)
