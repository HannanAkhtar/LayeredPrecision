import pytest
import torch

from layered_quantization import IntegerFormat
from lp_llm.quantize import affine_decode, affine_quantize, encode_layers


def test_channel_reuses_reference_affine_and_fp32():
    weight = torch.tensor([[-0.7, -0.2, 0.3, 0.8], [0.0, 0.1, 0.4, 0.7]], dtype=torch.bfloat16)
    params, decoded = affine_quantize(weight, 2)
    reference, expected = IntegerFormat("channel", 0).quantize(weight.float(), 2)
    assert torch.equal(params["codes"], reference["quantized"])
    assert torch.equal(decoded, expected)
    assert decoded.dtype == torch.float32
    assert torch.equal(affine_decode(params), decoded)


@pytest.mark.parametrize("width", [1, 2, 5, 8, 16])
def test_group_tail_and_constants(width):
    weight = torch.cat([torch.full((1, 133), -0.5), torch.zeros((1, 133))])
    params, decoded = affine_quantize(weight, width, "group", 128)
    assert params["scale"].shape == (2, 2)
    assert torch.equal(affine_decode(params), decoded)
    assert torch.isfinite(decoded).all()
    assert int(params["codes"].to(torch.int32).max()) < 2**width


def test_recursive_prefix_uses_actual_fp32_reconstruction():
    weight = torch.tensor([[-1.4, -0.3, 0.1, 0.7, 1.2]], dtype=torch.float32)
    layers = list(encode_layers(weight, [2, 2, 2, 2], "channel"))
    prefix = None
    for params, metrics, encoded_prefix in layers:
        contribution = affine_decode(params)
        prefix = contribution if prefix is None else prefix + contribution
        assert torch.equal(prefix, encoded_prefix)
        assert metrics["parameter_mse_fp32"] >= 0
    residual = weight - layers[0][2]
    reference, _ = affine_quantize(residual, 2)
    assert torch.equal(layers[1][0]["codes"], reference["codes"])


def test_nonfinite_rejected():
    with pytest.raises(ValueError, match="Nonfinite"):
        affine_quantize(torch.tensor([[float("nan")]]), 2)
