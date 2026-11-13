import json
import os
from pathlib import Path

import pytest

from lp_llm import progressive_control_recovery as recovery


def test_w4_partial_requires_process_repair_and_252_projections(tmp_path, monkeypatch):
    source, control, checkout = tmp_path / "source", tmp_path / "control", tmp_path / "checkout"
    source.mkdir(); checkout.mkdir()
    partial = control / "export.partial"
    partial.mkdir(parents=True)
    (partial / "shard.safetensors").write_bytes(b"original weights")
    for name in recovery.TOKEN_FILES:
        (source / name).write_bytes(name.encode())
    ident = {"code_sha": "a" * 40, "source": {"hash": "source"}}
    (control / "run.json").write_text(json.dumps({"identity": ident}))
    (control / "quantization_process.json").write_text(json.dumps({"returncode": 0}))
    (control / "quantization.log").write_text("successful")
    (control / "resolved_config.json").write_text("{}")
    hashes = {str(i): str(i) for i in range(252)}
    repair = {"quantized_projection_byte_identical": True,
              "raw_projection_tensor_sha256": hashes}
    (control / "excluded_repair.json").write_text(json.dumps(repair))
    monkeypatch.setattr(recovery, "control_identity", lambda *_: ident)
    monkeypatch.setattr(recovery, "_files", lambda *_: ident["source"])
    monkeypatch.setattr(recovery, "_resolved_matches", lambda *_: None)
    monkeypatch.setattr(recovery, "validate_control", lambda *_: {
        "projection_tensor_sha256": hashes, "quantized_projection_count": 252})
    output = tmp_path / "audit.json"
    first = recovery.audit(source, control, checkout, output, "b" * 40)
    assert first["status"] == "completed_validated"
    assert first["original_producer_sha"] == "a" * 40
    assert first["code_sha"] == "b" * 40
    assert recovery.audit(source, control, checkout, output, "b" * 40) == first
    monkeypatch.setattr(recovery, "_audit_code_compatible", lambda repo, sha:
                        None if sha == "c" * 40 else pytest.fail("wrong evaluation SHA"))
    monkeypatch.setattr(recovery, "AUDITING_SHA", "b" * 40)
    accepted = recovery._verified_audit(source, control, checkout, output,
                                        "c" * 40, tmp_path)
    assert accepted["code_sha"] == "b" * 40
    assert accepted["original_producer_sha"] == "a" * 40
    seal = output.with_name("audit.json.sha256.json")
    good_seal = seal.read_bytes()
    seal.write_text(json.dumps({"code_sha": "c" * 40, "audit_sha256": recovery.sha256(output)}))
    with pytest.raises(ValueError, match="seal differs"):
        recovery._verified_audit(source, control, checkout, output, "c" * 40, tmp_path)
    seal.write_bytes(good_seal)
    assert (control / "export.partial/shard.safetensors").read_bytes() == b"original weights"
    assert not (control / "export").exists()
    repair["raw_projection_tensor_sha256"] = {"wrong": "hash"}
    (control / "excluded_repair.json").write_text(json.dumps(repair))
    with pytest.raises(ValueError, match="projection or exclusion"):
        recovery.audit(source, control, checkout, output, "b" * 40)


def test_view_uses_original_tokenizer_and_preserves_audited_weights(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    for name in recovery.TOKEN_FILES:
        (source / name).write_bytes(("original " + name).encode())
    partial = tmp_path / "control/export.partial"
    checkpoint = partial / "qwen3-w4"
    checkpoint.mkdir(parents=True)
    for name in ("config.json", "model.safetensors.index.json", "shard.safetensors"):
        (checkpoint / name).write_bytes(name.encode())
    (checkpoint / "tokenizer.json").write_bytes(b"exported tokenizer differs")
    (checkpoint / "special_tokens_map.json").write_bytes(b"exported extra tokenizer file")
    report = {"checkpoint_validation": {"checkpoint": {
        "resolved_checkpoint_path": str(checkpoint)}},
        "original_partial_export": str(partial),
        "snapshot": recovery._snapshot_tree(partial),
        "tokenizer_files": recovery._tokenizers(source)}
    original_shard_hash = recovery.sha256(checkpoint / "shard.safetensors")
    if os.name == "nt":
        real_samefile = os.path.samefile
        monkeypatch.setattr(recovery.os, "link", lambda left, right: Path(right).write_bytes(Path(left).read_bytes()))
        monkeypatch.setattr(recovery.os.path, "samefile", lambda left, right:
            True if Path(left).parent.name == "view" and Path(left).name in
            ("config.json", "model.safetensors.index.json", "shard.safetensors")
            else real_samefile(left, right))
    view = tmp_path / "view"
    manifest = recovery._view(view, source, report)
    assert (view / "tokenizer.json").read_bytes() == (source / "tokenizer.json").read_bytes()
    assert not (view / "special_tokens_map.json").exists()
    assert manifest["exported_tokenizer_differences"]["tokenizer.json"]["differs_from_source"]
    assert manifest["exported_tokenizer_differences"]["special_tokens_map.json"]["differs_from_source"]
    assert recovery.sha256(checkpoint / "shard.safetensors") == original_shard_hash
    assert recovery._view(view, source, report) == manifest
    (checkpoint / "shard.safetensors").write_bytes(b"changed weight shard")
    with pytest.raises(ValueError, match="differs from CPU audit"):
        recovery._view(view, source, report)


def test_recovery_rejects_bad_corpus_before_matgptq(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "check_revision", lambda *_: None)
    monkeypatch.setattr(recovery, "_verified_audit", lambda *_: {"original_producer_sha": "a" * 40,
        "original_partial_export": "partial", "checkpoint_validation": {}})
    monkeypatch.setattr(recovery, "_view", lambda *_: {"verified": True})
    monkeypatch.setattr(recovery, "corpus_identity", lambda *_: {
        "token_sha256": "wrong", "scored_tokens": 298862})
    monkeypatch.setattr(recovery, "evaluate_matgptq", lambda *_:
                        pytest.fail("MatGPTQ must not run on a mismatched corpus"))
    audit = tmp_path / "audit.json"
    audit.write_text("{}")
    with pytest.raises(ValueError, match="corpus identity differs"):
        recovery.evaluate(tmp_path, tmp_path, tmp_path, tmp_path, audit,
                          tmp_path / "result", "b" * 40, tmp_path)
