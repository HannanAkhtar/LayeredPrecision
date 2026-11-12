import json

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
    assert (control / "export.partial/shard.safetensors").read_bytes() == b"original weights"
    assert not (control / "export").exists()
    repair["raw_projection_tensor_sha256"] = {"wrong": "hash"}
    (control / "excluded_repair.json").write_text(json.dumps(repair))
    with pytest.raises(ValueError, match="projection or exclusion"):
        recovery.audit(source, control, checkout, output, "b" * 40)
