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


@pytest.mark.parametrize("bits,granularity", [(5, "channel"), (5, "group"),
                                               (16, "group")])
def test_layer_metadata_roundtrip_and_bytes(tmp_path, bits, granularity):
    source = torch.arange(262, dtype=torch.float32).reshape(2, 131) / 100 - 0.8
    params, decoded = affine_quantize(source, bits, granularity, 128)
    record = write_layer(tmp_path, 0, 0, params)
    restored = read_layer(tmp_path, record)
    assert "t_min" not in params and "t_min" not in restored
    for key in ("scale", "zero_point", "codes"):
        assert torch.equal(restored[key].to(torch.int32) if key == "zero_point" else restored[key],
                           params[key].to(torch.int32) if key == "zero_point" else params[key])
    assert torch.equal(decode_prefix(tmp_path, {"layers": [record]}, 1), decoded)
    assert record["code_bytes"] == (source.numel() * bits + 7) // 8
    assert record["metadata_bytes"] == (tmp_path / record["metadata"]).stat().st_size
    assert record["scale_zero_bytes"] == (restored["scale"].numel() * restored["scale"].element_size()
                                          + restored["zero_point"].numel() * restored["zero_point"].element_size())
    assert record["metadata_bytes"] > record["scale_zero_bytes"]


@pytest.mark.parametrize("granularity", ["channel", "group"])
def test_zero_scale_and_constant_regions_without_fallback_metadata(tmp_path, granularity):
    weight = torch.zeros((2, 133), dtype=torch.float32)
    weight[1, :] = -0.5
    params, encoder_decoded = affine_quantize(weight, 2, granularity, 128)
    assert "t_min" not in params
    assert bool((params["scale"].reshape(2, -1)[0] == 0).all())
    assert bool((params["scale"].reshape(2, -1)[1] > 0).all())
    record = write_layer(tmp_path, 0, 0, params)
    decoded = decode_prefix(tmp_path, {"layers": [record]}, 1)
    assert torch.equal(decoded, encoder_decoded)
    assert torch.equal(decoded, weight)
