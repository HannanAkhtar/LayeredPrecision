import json

from lp_llm.progressive_matrix_inventory import inspect, inventory


def test_inventory_classifies_partial_without_promoting_it(tmp_path):
    root = tmp_path / "control"
    nested = root / "export.partial/qwen3-w4"
    nested.mkdir(parents=True)
    (nested / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"weight": "shard.safetensors"}}))
    (root / "quantization_process.json").write_text(json.dumps({"returncode": 0}))
    candidate = inspect("s01_w4", root, checkpoint=nested)
    assert candidate["state"] == "incomplete"
    assert candidate["missing_indexed_shards"] == ["shard.safetensors"]
    (nested / "shard.safetensors").write_bytes(b"weights")
    candidate = inspect("s01_w4", root, checkpoint=nested)
    assert candidate["state"] == "recoverable_candidate"
    assert candidate["indexed_tensor_count"] == 1
    assert candidate["physical_file_bytes"] > 0
    (root / "full.json").write_text(json.dumps({"status": "completed_validated",
                                                  "evaluation": {"perplexity": 10.0}}))
    (nested / "model.safetensors.index.json").unlink()
    assert inspect("s01_w4", root, checkpoint=nested)["state"] == "incomplete"


def test_inventory_reports_unknown_quota_and_missing_artifacts(tmp_path):
    source, results = tmp_path / "source", tmp_path / "results"
    source.mkdir(); results.mkdir()
    report = inventory(source, results)
    assert report["quota_status"] == "unknown"
    assert report["quota_free_bytes"] is None
    assert len(report["artifacts"]) == 16
    assert all(item["state"] == "missing" for item in report["artifacts"] if item["name"] != "original")
