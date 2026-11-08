import hashlib
import json

import pytest

from lp_llm.r03_probe_gate import verify


def _report(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "model.safetensors.index.json").write_text("{}")
    (source / "model.safetensors").write_bytes(b"original-shard")
    (source / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"weight": "model.safetensors"}}))
    parent = tmp_path / "r03_feasibility"
    parent.mkdir()
    path = parent / "layer_probe-1.json"
    sha = lambda name: hashlib.sha256((source / name).read_bytes()).hexdigest()
    record = {"status": "representative_layer_validated", "code_sha": "a" * 40,
        "source_layer": 0, "projection_count": 7,
        "backend": "TORCH", "quantization_config": {"bits": 2,
            "group_size": 128, "sym": True, "desc_act": False,
            "act_group_aware": True, "damp_percent": 0.05,
            "damp_auto_increment": 0.01, "static_groups": False,
            "true_sequential": True},
        "largest_projection": "model.layers.0.mlp.down_proj",
        "calibration_sample_count": 128, "calibration_token_sha256": "b" * 64,
        "source_config_sha256": sha("config.json"),
        "source_index_sha256": sha("model.safetensors.index.json"),
        "source_shard_sha256": {"model.safetensors": sha("model.safetensors")},
        "gpu_peak_reserved_bytes": 10, "gpu_total_bytes": 100,
        "process_peak_rss_kib": 10, "host_allocation_bytes": 100_000}
    path.write_text(json.dumps(record))
    return source, path, record


def test_probe_gate_requires_same_source_code_and_headroom(tmp_path):
    source, path, record = _report(tmp_path)
    assert verify(path, source, "a" * 40) == record
    with pytest.raises(ValueError, match="identity or coverage"):
        verify(path, source, "c" * 40)
    record["gpu_peak_reserved_bytes"] = 95
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="headroom"):
        verify(path, source, "a" * 40)


def test_probe_gate_rejects_moved_report_and_modified_source(tmp_path):
    source, path, record = _report(tmp_path)
    (source / "config.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="identity or coverage"):
        verify(path, source, "a" * 40)
    outside = tmp_path / "r03" / "layer_probe.json"
    outside.parent.mkdir()
    outside.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="outside the full result"):
        verify(outside, source, "a" * 40)


def test_probe_gate_rejects_changed_quantization_settings(tmp_path):
    source, path, record = _report(tmp_path)
    record["quantization_config"]["sym"] = False
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="identity or coverage"):
        verify(path, source, "a" * 40)


def test_probe_gate_rejects_modified_source_shard(tmp_path):
    source, path, _ = _report(tmp_path)
    (source / "model.safetensors").write_bytes(b"modified-shard")
    with pytest.raises(ValueError, match="identity or coverage"):
        verify(path, source, "a" * 40)
