import json

import pytest
import torch

from lp_llm.quantize import affine_quantize
from lp_llm.storage import decode_prefix, pack_codes, read_layer, unpack_codes, write_layer


def test_packing_all_widths_and_chunk_boundary(tmp_path):
    for bits in (1, 2, 5, 8, 9, 16):
        codes = (torch.arange((1 << 20) + 13, dtype=torch.int32) % (1 << bits)).reshape(1, -1)
        path = tmp_path / f"{bits}.bin"
        pack_codes(path, codes, bits)
        assert torch.equal(unpack_codes(path, tuple(codes.shape), bits), codes.to(torch.uint8 if bits <= 8 else torch.uint16))
        assert path.stat().st_size == (codes.numel() * bits + 7) // 8


@pytest.mark.parametrize("bits", [5, 16])
def test_layer_metadata_roundtrip(tmp_path, bits):
    source = torch.arange(262, dtype=torch.float32).reshape(2, 131) / 100 - 0.8
    params, decoded = affine_quantize(source, bits, "group", 128)
    record = write_layer(tmp_path, 0, 0, params)
    restored = read_layer(tmp_path, record)
    for key in ("scale", "zero_point", "t_min", "codes"):
        assert torch.equal(restored[key].to(torch.int32) if key == "zero_point" else restored[key],
                           params[key].to(torch.int32) if key == "zero_point" else params[key])
    assert torch.equal(decode_prefix(tmp_path, {"layers": [record]}, 1), decoded)
