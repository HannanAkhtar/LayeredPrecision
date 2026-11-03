"""Parity with intel/auto-round commit 6c824f741be358983baba87aff83f9c5aa0a01e7.

The reference below is the pinned quant_tensor_rtn_sym arithmetic, with its
reshape/pad helper inlined and an extra copy to expose integer codes. The
upstream function returns only qdq, scale, maxq. Apache-2.0 licensed.
"""

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from lp_llm.quantize import (affine_decode, autoround_rtn_sym_quantize,
                             encode_layers, symmetric_rtn_quantize)
from lp_llm.storage import decode_prefix, read_layer, write_layer


def pinned_autoround_reference(tensor, bits, group_size):
    # Pinned reshape_pad_tensor_by_group_size: padding is row-local and zero.
    orig_shape = tensor.shape
    pad_len = 0
    if tensor.shape[1] >= group_size:
        pad_len = (-tensor.shape[1]) % group_size
        if pad_len:
            tensor = F.pad(tensor, (0, pad_len))
        tensor = tensor.reshape(-1, group_size)
    maxq = int(2.0 ** (bits - 1))
    wmin_tmp = torch.clamp(tensor.min(-1)[0], max=0)
    wmax_tmp = torch.clamp(tensor.max(-1)[0], min=0)
    wmin_abs = -(wmin_tmp * 1.0)
    wmax_abs = wmax_tmp * 1.0
    max_v = (2 * (wmax_abs < wmin_abs).int() - 1) * torch.max(wmax_abs, wmin_abs)
    scale = (max_v / maxq).to(torch.float16)
    scale = torch.where(scale < 0, torch.clamp(scale, max=-1e-5),
                        torch.clamp(scale, min=1e-5))
    scale = scale.unsqueeze(dim=-1)
    int_w = tensor.div(scale).round_().clamp_(-maxq, maxq - 1)
    signed_codes = int_w.clone()
    qdq_result = (int_w.mul_(scale)).to(tensor.dtype)
    # Pinned revert_tensor_by_pad: remove padding after reshaping to rows.
    qdq_result = qdq_result.reshape(orig_shape[0], -1)[:, :orig_shape[1]]
    signed_codes = signed_codes.reshape(orig_shape[0], -1)[:, :orig_shape[1]]
    return qdq_result, scale, signed_codes


@pytest.mark.parametrize("bits", [2, 4, 6, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pinned_reference_parity(bits, dtype):
    torch.manual_seed(2026)
    x = torch.randn(6, 259, dtype=dtype)
    x[0] = 0
    x[1] = 2.5
    x[2] = -2.5
    x[3] = x[3].abs() + 0.1
    x[4] = -x[4].abs() - 0.1
    x[5, 0], x[5, 1] = 50, -100
    qdq, scale, signed = pinned_autoround_reference(x, bits, 128)
    params, actual = autoround_rtn_sym_quantize(x, bits, "group", 128)
    assert params["family"] == "autoround_rtn_sym"
    assert params["scale"].dtype == torch.float16
    assert torch.equal(params["scale"], scale.reshape(6, 3))
    assert torch.equal(params["codes"].int() - (1 << (bits - 1)), signed.int())
    assert torch.equal(actual, qdq)
    assert torch.equal(affine_decode(params).to(dtype), qdq)
    assert torch.all(params["codes"][0] == (1 << (bits - 1)))
    assert torch.all(params["scale"][0] > 0)
    assert torch.all(params["scale"][3] < 0)
    assert torch.all(params["scale"][4] > 0)


def test_full_int2_grid_and_ternary_ablation():
    x = torch.tensor([[-2., -1., 0., 1.]])
    params, decoded = autoround_rtn_sym_quantize(x, 2, "group", 128)
    assert set(params["codes"].flatten().tolist()) == {0, 1, 2, 3}
    assert torch.equal(decoded, x)
    old, _ = symmetric_rtn_quantize(x, 2, "group", 128)
    assert old["family"] == "symmetric_rtn"
    assert set(old["codes"].flatten().tolist()) != {0, 1, 2, 3}


def test_four_prefixes_packed_round_trip(tmp_path):
    torch.manual_seed(74)
    x = torch.randn(3, 259)
    x[0] = 0
    layers = list(encode_layers(x, [2, 2, 2, 2], "group", 128,
                                quantizer_family="autoround_rtn_sym"))
    records = []
    for depth, (params, _, prefix) in enumerate(layers, 1):
        record = write_layer(tmp_path, 0, depth - 1, params)
        records.append(record)
        assert record["family"] == "autoround_rtn_sym"
        assert record["scale_dtype"] == "float16"
        assert record["signed_code_offset"] == 2
        assert record["scale_zero_bytes"] == params["scale"].numel() * 2
        assert record["code_bytes"] == (x.numel() * 2 + 7) // 8
        assert set(load_file(str(tmp_path / record["metadata"]))) == {"scale"}
        stored = read_layer(tmp_path, record)
        assert torch.equal(stored["codes"], params["codes"])
        assert torch.equal(stored["scale"], params["scale"])
        assert torch.equal(affine_decode(stored), affine_decode(params))
        assert torch.equal(decode_prefix(tmp_path, {"layers": records}, depth), prefix)


def test_small_group_and_invalid_one_bit():
    x = torch.tensor([[0., 3., -4.], [0., 0., 0.]])
    qdq, scale, signed = pinned_autoround_reference(x, 2, 128)
    params, actual = autoround_rtn_sym_quantize(x, 2, "group", 128)
    assert torch.equal(actual, qdq)
    assert torch.equal(params["scale"], scale.reshape(2, 1))
    assert torch.equal(params["codes"].int() - 2, signed.int())
    with pytest.raises(ValueError, match="at least 2 bits"):
        autoround_rtn_sym_quantize(x, 1)
