import json

import pytest
import torch
from safetensors import safe_open

from lp_llm.progressive_lp import _read_chunk, _slice, encode_layer, encode_residual
from lp_llm.progressive_scaled import (ALPHA, S01_PRODUCER, _external_s01,
                                       encode, evaluate, layers, materialize, reconstruct,
                                       scaled_identity, storage)
from lp_llm import progressive_scaled
from test_progressive_lp import fixture


def test_s01_external_signed_e1_and_fp16_materialization(tmp_path, monkeypatch):
    source, base = fixture(tmp_path, width=128)
    progressive = tmp_path / "progressive_llm"
    old = progressive / "ablations/s01_autoround_signed_rtn"
    monkeypatch.setenv("FOLLOWUP_CODE_SHA", S01_PRODUCER)
    original = encode_layer("s01", 1, source, base, old,
                            quantizer="autoround_signed_rtn", rows=2)
    code = old / "layer1" / original["chunks"][0]["code"]
    code_hash = code.read_bytes()
    monkeypatch.setenv("FOLLOWUP_CODE_SHA", "a" * 40)
    root = progressive / "scaled_signed_v1/s01"
    ident = scaled_identity("s01", source, base)
    external = _external_s01(root, source, base, ident)
    assert external["reference"]["producer_sha"] == S01_PRODUCER
    assert external["reference"]["validating_sha"] == "a" * 40
    assert code.read_bytes() == code_hash
    assert _external_s01(root, source, base, ident) == external
    first = original["chunks"][0]
    base_index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(base / base_index[first["name"]], framework="pt") as handle:
        b = handle.get_tensor(first["name"])
    expected = b.float() + ALPHA * _read_chunk(old / "layer1", first)
    assert torch.equal(reconstruct(b, [external], first["name"], 0, 2), expected)
    output = tmp_path / "scaled-checkpoint"
    report = materialize("s01", 1, source, base, root, output)
    assert report["projection_count"] == 252
    index = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(output / index[first["name"]], framework="pt") as handle:
        assert torch.equal(handle.get_tensor(first["name"]), expected.to(torch.float16))
    with safe_open(output / index["lm_head.weight"], framework="pt") as handle:
        assert torch.equal(handle.get_tensor("lm_head.weight"),
                           torch.tensor([[.25, -.5]], dtype=torch.bfloat16))
    assert storage("s01", 1, source, base, root)["layers"][0]["external"] is True
    with pytest.raises(FileExistsError):
        materialize("s01", 1, source, base, root, output)
    (root / "layer1_reference.json").write_text("{}")
    with pytest.raises(ValueError, match="reference differs"):
        layers("s01", 1, source, base, root, ident)


def test_scaled_evaluation_protocol_and_cleanup(tmp_path, monkeypatch):
    source, base = fixture(tmp_path, width=128)
    progressive = tmp_path / "progressive_llm"
    old = progressive / "ablations/s01_autoround_signed_rtn"
    monkeypatch.setenv("FOLLOWUP_CODE_SHA", S01_PRODUCER)
    encode_layer("s01", 1, source, base, old,
                 quantizer="autoround_signed_rtn", rows=2)
    monkeypatch.setenv("FOLLOWUP_CODE_SHA", "a" * 40)
    monkeypatch.setattr(progressive_scaled, "validate_repaired_base", lambda *_: {})
    monkeypatch.setattr(progressive_scaled, "check_revision", lambda *_: None)
    monkeypatch.setattr(progressive_scaled, "corpus_identity", lambda *_: {
        "token_sha256": progressive_scaled.TOKEN_SHA, "scored_tokens": 298862})
    calls = []
    def fake_eval(_checkout, checkpoint, log):
        calls.append(checkpoint)
        log.write_text("wikitext2: 10.00")
        return {"perplexity": 10.0}
    monkeypatch.setattr(progressive_scaled, "evaluate_matgptq", fake_eval)
    root = progressive / "scaled_signed_v1/s01"
    result = root / "evaluation/depth1"
    first = evaluate("s01", 1, source, base, root, result, tmp_path)
    assert first["corpus"]["scored_tokens"] == 298862
    assert first["storage"]["layers"][0]["external"] is True
    assert not (result / "checkpoint").exists()
    assert evaluate("s01", 1, source, base, root, result, tmp_path) == first
    assert len(calls) == 1


def test_r02_recursive_scaled_residual_and_resume(tmp_path, monkeypatch):
    source, base = fixture(tmp_path, width=128)
    monkeypatch.setattr(progressive_scaled, "validate_repaired_base", lambda *_: {})
    root = tmp_path / "scaled_signed_v1/r02"
    monkeypatch.setenv("FOLLOWUP_CODE_SHA", "b" * 40)
    reports = [encode("r02", depth, source, base, root) for depth in (1, 2, 3)]
    assert all(report["projection_count"] == 252 for report in reports)
    prior = layers("r02", 3, source, base, root, scaled_identity("r02", source, base))
    name = reports[0]["chunks"][0]["name"]
    base_index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(base / base_index[name], framework="pt") as handle:
        b = handle.get_tensor(name)
    expected = b.float()
    for layer in prior:
        expected = expected + ALPHA * _read_chunk(layer["directory"], layer["manifest"]["chunks"][0])
    assert torch.equal(reconstruct(b, prior, name, 0, 2), expected)
    source_index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    for depth in (2, 3):
        precursor = prior[:depth - 1]
        residual = (_slice(source, source_index, name, 0, 2).float() -
                    reconstruct(b, precursor, name, 0, 2))
        _, independently_encoded = encode_residual(residual, "autoround_signed_rtn")
        assert torch.equal(independently_encoded,
            _read_chunk(root / f"layer{depth}", reports[depth - 1]["chunks"][0]))
    assert encode("r02", 3, source, base, root) == reports[-1]
    (root / "layer2.partial").mkdir()
    with pytest.raises(FileExistsError):
        encode("r02", 2, source, base, root)
