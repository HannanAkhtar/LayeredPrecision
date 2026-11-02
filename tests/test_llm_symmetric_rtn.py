import pytest
import torch
from safetensors.torch import load_file

from lp_llm.quantize import affine_decode, affine_quantize, encode_layers, symmetric_rtn_quantize
from lp_llm.storage import decode_prefix, write_layer


def test_exact_hand_computed_ternary_grid():
    x = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=torch.float32)
    params, decoded = symmetric_rtn_quantize(x, 2, "group", 128)
    assert params["family"] == "symmetric_rtn"
    assert torch.equal(params["codes"], torch.tensor([[1, 2, 2, 2, 3]], dtype=torch.uint8))
    assert torch.equal(decoded, torch.tensor([[-2.0, 0.0, 0.0, 0.0, 2.0]]))
    assert torch.equal(affine_decode(params), decoded)
    assert "zero_point" not in params


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_group_tail_zero_and_round_trip(bits, tmp_path):
    torch.manual_seed(45)
    x = torch.randn(3, 259)
    x[0, :] = 0
    params, decoded = symmetric_rtn_quantize(x, bits, "group", 128)
    assert params["scale"].shape == (3, 3)
    assert torch.all(decoded[0] == 0)
    assert torch.equal(decoded, affine_decode(params))
    record = write_layer(tmp_path, 0, 0, params)
    assert record["family"] == "symmetric_rtn"
    assert record["scale_zero_bytes"] == params["scale"].numel() * 4
    metadata = load_file(str(tmp_path / record["metadata"]))
    assert set(metadata) == {"scale"}
    assert torch.equal(decode_prefix(tmp_path, {"layers": [record]}, 1), decoded)


def test_four_prefixes_equal_decoded_contribution_sums(tmp_path):
    x = torch.tensor([[-3.1, -.4, -.2, .1, .6, 2.3]], dtype=torch.float32)
    cumulative = None
    layers = list(encode_layers(x, [2, 2, 2, 2], "group", 128,
                                quantizer_family="symmetric_rtn"))
    for depth, (params, metrics, prefix) in enumerate(layers):
        rec = write_layer(tmp_path, 0, depth, params)
        contribution = affine_decode(params)
        cumulative = contribution if cumulative is None else cumulative + contribution
        assert torch.equal(prefix, cumulative)
        assert metrics["nominal_bits"] is None
        assert rec["family"] == "symmetric_rtn"
    records = [write_layer(tmp_path, 0, n, layer[0]) for n, layer in enumerate(layers)]
    for depth, layer in enumerate(layers, start=1):
        assert torch.equal(decode_prefix(tmp_path, {"layers": records}, depth), layer[2])


def test_symmetric_and_affine_are_distinct_and_1bit_rejected():
    x = torch.tensor([[-5., -.1, 0., .1, 1.]], dtype=torch.float32)
    _, symmetric = symmetric_rtn_quantize(x, 2, "group", 128)
    _, affine = affine_quantize(x, 2, "group", 128)
    assert not torch.equal(symmetric, affine)
    with pytest.raises(ValueError, match="at least 2 bits"):
        symmetric_rtn_quantize(x, 1, "group", 128)
