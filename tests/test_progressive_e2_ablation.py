"""Signed AutoRound E2 remains distinct from historical ternary layers."""

import json

import pytest
import torch
from safetensors import safe_open

from lp_llm.progressive_lp import (_read_chunk, encode_layer, encode_residual, identity,
                                    verified_layers)
from lp_llm.progressive_run import materialize
from lp_llm.upstream_reference import direct_qdq
from test_progressive_lp import fixture


@pytest.mark.parametrize("values", [
    torch.zeros(2, 130),
    torch.tensor([[-5.] + [.01] * 129, [7.] + [-0.03] * 129]),
    torch.linspace(-1, 3, 260).reshape(2, 130),
    torch.full((2, 130), -1e-9),
])
def test_signed_e2_matches_pinned_plain_reference(values):
    params, decoded = encode_residual(values, "autoround_signed_rtn")
    upstream, scale = direct_qdq(values.float(), "plain", 128)
    assert torch.equal(decoded, upstream)
    assert torch.equal(params["scale"], scale.float().reshape(2, 2))
    assert params["scale"].dtype == torch.float32
    assert params["codes"].min() >= 0 and params["codes"].max() <= 3


def test_signed_manifest_rejects_mixed_format_and_preserves_ternary(tmp_path):
    source, base = fixture(tmp_path, width=130)
    old_root, new_root = tmp_path / "old", tmp_path / "ablation"
    old = encode_layer("s01", 1, source, base, old_root, rows=2, limit_tensors=7)
    new = encode_layer("s01", 1, source, base, new_root, rows=2, limit_tensors=7,
                       quantizer="autoround_signed_rtn")
    assert old["quantizer"]["family"] == "symmetric_rtn"
    assert new["quantizer"]["family"] == "autoround_rtn_sym"
    assert all(chunk["quantizer"] == "autoround_signed_rtn" for chunk in new["chunks"])
    assert verified_layers(new_root, 1, identity("s01", source, base, "autoround_signed_rtn"),
                           pilot=True) == [new]
    with pytest.raises(ValueError, match="identity or structure"):
        verified_layers(new_root, 1, identity("s01", source, base), pilot=True)
    marker = new_root / "layer1/encoding_complete.json"
    record = json.loads(marker.read_text())
    record["chunks"][0]["quantizer"] = "symmetric_ternary_rtn"
    marker.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="chunk quantizer"):
        verified_layers(new_root, 1, identity("s01", source, base, "autoround_signed_rtn"),
                        pilot=True)
    assert verified_layers(old_root, 1, identity("s01", source, base), pilot=True) == [old]


def test_signed_full_layer_materializes_fp16(tmp_path):
    source, base = fixture(tmp_path, width=128)
    root = tmp_path / "ablation"
    layer = encode_layer("s01", 1, source, base, root, rows=2,
                         quantizer="autoround_signed_rtn")
    assert layer["projection_count"] == 252
    output = tmp_path / "checkpoint"
    report = materialize("s01", 1, source, base, root, output, rows=2,
                         quantizer="autoround_signed_rtn")
    assert report["projection_count"] == 252
    index = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    first = layer["chunks"][0]
    base_index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(base / base_index[first["name"]], framework="pt") as handle:
        base_value = handle.get_tensor(first["name"])
    with safe_open(output / index[first["name"]], framework="pt") as handle:
        materialized = handle.get_tensor(first["name"])
    decoded = _read_chunk(root / "layer1", first)
    assert torch.equal(materialized,
                       (base_value.float() + decoded).to(torch.float16))
    with safe_open(output / index["lm_head.weight"], framework="pt") as handle:
        assert torch.equal(handle.get_tensor("lm_head.weight"),
                           torch.tensor([[.25, -.5]], dtype=torch.bfloat16))
    with pytest.raises(FileExistsError):
        materialize("s01", 1, source, base, root, output, rows=2,
                    quantizer="autoround_signed_rtn")
