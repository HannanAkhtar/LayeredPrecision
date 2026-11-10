import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from lp_llm import progressive_stage_a_recovery as recovery


def test_stage_a_audit_preserves_partial_and_original_provenance(tmp_path, monkeypatch):
    source, root, checkout = tmp_path / "source", tmp_path / "results", tmp_path / "auto-round"
    source.mkdir(); checkout.mkdir()
    monkeypatch.setenv("SLURM_JOB_ID", "60000")
    layer = root / "progressive_llm/s01/layer1"
    layer.mkdir(parents=True)
    (layer / "encoding_complete.json").write_text("{}")
    lp_result = root / "progressive_llm/evaluation/s01_lp_2p2"
    lp_result.mkdir(parents=True)
    ident = {"source": {"hash": "source"}}
    (lp_result / "full.json").write_text(json.dumps({
        "status": "completed_validated", "identity": ident, "code_sha": recovery.ORIGINAL_SHA,
        "depth": 1, "corpus": {"token_sha256": recovery.TOKEN_SHA, "scored_tokens": 298862},
        "evaluation": {"perplexity": 13.38}, "materialization": {"projection_count": 252},
        "evaluation_log_sha256": "hash"}))
    (lp_result / "evaluation.log").write_text("completed")
    resources = root / "progressive_llm/job_resources"
    resources.mkdir(parents=True)
    (resources / "s01_lp_2p2-55385.json").write_text(json.dumps({
        "job_id": "55385", "host_allocation_bytes": 12 * 1024**3,
        "host_memory_10pct_headroom_passed": False,
        "slurm_cgroup_peak_bytes": 15_439_818_752}))
    w4 = root / "progressive_llm/controls/s01_w4"
    (w4 / "export.partial").mkdir(parents=True)
    (w4 / "export.partial/shard.safetensors").write_bytes(b"checkpoint")
    (w4 / "run.json").write_text(json.dumps({"identity": {**ident,
        "code_sha": recovery.ORIGINAL_SHA}}))
    (w4 / "quantization_process.json").write_text(json.dumps({"returncode": 0}))
    (w4 / "worker_config.json").write_text(json.dumps({"branch": "s01", "bits": 4}))
    (w4 / "quantization.log").write_text("successful")
    (w4 / "resolved_config.json").write_text("{}")
    hashes = {str(i): "digest" for i in range(252)}
    repair = {"quantized_projection_byte_identical": True,
              "fp16_load_roundtrip_explains_all_excluded_changes": True,
              "restored_excluded_names": [f"excluded_{i}" for i in range(12)],
              "raw_projection_tensor_sha256": hashes}
    (w4 / "excluded_repair.json").write_text(json.dumps(repair))
    monkeypatch.setattr(recovery, "verify_report", lambda *_: {
        "source_identity": ident["source"], "slurm_job_id": "55356"})
    monkeypatch.setattr(recovery.subprocess, "run", lambda *_a, **_k:
                        SimpleNamespace(returncode=0))
    monkeypatch.setattr(recovery.subprocess, "check_output", lambda command, **_:
                        f"{command[5]}|FAILED|1:0\n")
    monkeypatch.setattr(recovery, "identity", lambda *_: ident)
    monkeypatch.setattr(recovery, "verified_layers", lambda *_: [{"code_sha": recovery.ORIGINAL_SHA}])
    monkeypatch.setattr(recovery, "control_identity", lambda *_: {**ident, "code_sha": "a" * 40})
    monkeypatch.setattr(recovery, "_resolved_matches", lambda *_: None)
    monkeypatch.setattr(recovery, "sha256", lambda *_: "hash")
    monkeypatch.setattr(recovery, "validate_control", lambda *_: {
        "projection_tensor_sha256": hashes, "quantized_projection_count": 252})
    output = root / "progressive_llm/recovery/new/audit.json"
    report = recovery.audit(source, root, "a" * 40, tmp_path, checkout, output)
    assert report["status"] == "stage_a_science_validated_resource_uncertified"
    assert report["lp"]["original_job_id"] == "55385"
    assert report["w4"]["original_job_id"] == "55386"
    assert report["recovery_validation_job_id"] == "60000"
    assert report["resource_certification"] == "not_satisfied_by_original_12_gib_jobs"
    assert (w4 / "export.partial").is_dir()
    assert recovery.audit(source, root, "a" * 40, tmp_path, checkout, output) == report
    repair["raw_projection_tensor_sha256"] = {"wrong": "digest"}
    (w4 / "excluded_repair.json").write_text(json.dumps(repair))
    with pytest.raises(ValueError, match="repair provenance"):
        recovery.audit(source, root, "a" * 40, tmp_path, checkout, output)


def test_recovered_w4_evaluation_is_separate_and_idempotent(tmp_path, monkeypatch):
    checkpoint = tmp_path / "export.partial"
    checkpoint.mkdir()
    record = {"status": "stage_a_science_validated_resource_uncertified",
              "code_sha": "a" * 40,
              "original_code_sha": recovery.ORIGINAL_SHA,
              "lp": {"original_job_id": recovery.LP_JOB},
              "resource_certification": "not_satisfied_by_original_12_gib_jobs",
              "w4": {"original_job_id": recovery.W4_JOB,
                     "partial_export": str(checkpoint),
                     "checkpoint_validation": {"quantized_projection_count": 252}}}
    monkeypatch.setattr(recovery, "check_revision", lambda *_: None)
    monkeypatch.setattr(recovery, "corpus_identity", lambda *_: {
        "token_sha256": recovery.TOKEN_SHA, "scored_tokens": 298862})
    calls = []
    def evaluate(_checkout, path, log, **_):
        calls.append(path)
        log.write_text("PPL 13.0")
        return {"perplexity": 13.0}
    monkeypatch.setattr(recovery, "evaluate_matgptq", evaluate)
    monkeypatch.setattr(recovery, "sha256", lambda *_: "loghash")
    output = tmp_path / "recovery"
    first = recovery.evaluate_w4(record, tmp_path, output, "60001")
    assert first["original_quantization_job_id"] == recovery.W4_JOB
    assert first["recovery_evaluation_job_id"] == "60001"
    assert checkpoint.is_dir() and len(calls) == 1
    assert recovery.evaluate_w4(record, tmp_path, output, "60002") == first
    assert len(calls) == 1
    resource_dir = tmp_path / "progressive_llm/job_resources"
    resource_dir.mkdir(parents=True)
    (resource_dir / "s01_w4_recovery-60001.json").write_text(json.dumps({
        "job_id": "60001", "host_allocation_bytes": 15 * 1024**3,
        "host_memory_within_allocation": True, "exit_code_before_resource_check": 0,
        "slurm_cgroup_peak_bytes": 1, "process_peak_rss_bytes": 1}))
    actual = tmp_path / "progressive_llm/recovery" / ("a" * 40) / "s01_w4"
    actual.parent.mkdir(parents=True)
    # The admission reader uses the deployed recovery path, not an original result path.
    import shutil
    shutil.copytree(output, actual)
    monkeypatch.setattr(recovery.subprocess, "check_output", lambda *_a, **_k:
                        "60001|COMPLETED|0:0\n")
    checked = recovery.verify_recovery(record, tmp_path)
    assert checked["stage_b_admission"] == "blocked_pending_lp_full_run_memory_certification"
    high = {"job_id": "60001", "host_allocation_bytes": 15 * 1024**3,
            "host_memory_within_allocation": False, "exit_code_before_resource_check": 0,
            "slurm_cgroup_peak_bytes": 15 * 1024**3 + 1,
            "process_peak_rss_bytes": 1}
    (resource_dir / "s01_w4_recovery-60001.json").write_text(json.dumps(high))
    with pytest.raises(ValueError, match="resource evidence differs"):
        recovery.verify_recovery(record, tmp_path)


def test_evaluate_action_uses_cpu_audit_without_running_full_audit(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "60001")
    monkeypatch.setattr(recovery, "audit", lambda *_: pytest.fail("full CPU audit on GPU"))
    monkeypatch.setattr(recovery, "validate_control", lambda *_:
                        pytest.fail("full tensor comparison on GPU"))
    record = {"status": "stage_a_science_validated_resource_uncertified"}
    monkeypatch.setattr(recovery, "load_audited_inputs", lambda *_: record)
    monkeypatch.setattr(recovery, "evaluate_w4", lambda *_a, **_k:
                        {"status": "completed_validated"})
    monkeypatch.setattr(sys, "argv", ["recovery", "evaluate", "--source", str(tmp_path),
        "--result-root", str(tmp_path), "--code-sha", "a" * 40,
        "--code-repo", str(tmp_path), "--autoround", str(tmp_path),
        "--output", str(tmp_path / "audit.json"), "--matgptq", str(tmp_path)])
    recovery.main()


def test_gpu_hash_gate_rejects_changed_checkpoint_without_tensor_decoding(tmp_path, monkeypatch):
    sha = "a" * 40
    source = tmp_path / "source"
    source.mkdir()
    result = tmp_path / "results"
    w4 = result / "progressive_llm/controls/s01_w4"
    partial = w4 / "export.partial"
    partial.mkdir(parents=True)
    shard = partial / "model.safetensors"
    shard.write_bytes(b"original packed bytes")
    for name in ("run.json", "quantization_process.json", "worker_config.json",
                 "resolved_config.json", "excluded_repair.json", "quantization.log"):
        (w4 / name).write_text(name)
    layer = result / "progressive_llm/s01/layer1"
    layer.mkdir(parents=True)
    (layer / "chunk.bin").write_bytes(b"encoded")
    evaluation = result / "progressive_llm/evaluation/s01_lp_2p2"
    evaluation.mkdir(parents=True)
    (evaluation / "full.json").write_text("{}")
    resource = result / f"progressive_llm/job_resources/s01_lp_2p2-{recovery.LP_JOB}.json"
    resource.parent.mkdir(parents=True)
    resource.write_text("{}")
    audit_path = result / "progressive_llm/recovery" / sha / "stage_a_scientific_audit.json"
    audit_path.parent.mkdir(parents=True)
    names = ("run.json", "quantization_process.json", "worker_config.json",
             "resolved_config.json", "excluded_repair.json", "quantization.log")
    record = {"status": "stage_a_science_validated_resource_uncertified",
        "code_sha": sha, "original_code_sha": recovery.ORIGINAL_SHA,
        "prerequisite_job_id": "55356", "recovery_validation_job_id": "60000",
        "resource_certification": "not_satisfied_by_original_12_gib_jobs",
        "lp": {"original_job_id": recovery.LP_JOB,
               "original_slurm": {"job_id": recovery.LP_JOB},
               "layer_snapshot": recovery._snapshot_tree(layer),
               "evaluation_snapshot": recovery._snapshot_tree(evaluation),
               "original_resource_report_sha256": recovery.sha256(resource)},
        "w4": {"original_job_id": recovery.W4_JOB,
               "original_quantization_code_sha": recovery.ORIGINAL_SHA,
               "original_slurm": {"job_id": recovery.W4_JOB},
               "original_identity": {"source": {"hash": "source"},
                                     "code_sha": recovery.ORIGINAL_SHA,
                                     "upstream_revision": recovery.PR_COMMIT},
               "partial_export": str(partial), "process": {"returncode": 0},
               "repair": {"quantized_projection_byte_identical": True},
               "checkpoint_validation": {"quantized_projection_count": 252,
                   "checkpoint": {"resolved_checkpoint_path": str(partial),
                                  "shard_sha256": {shard.name: recovery.sha256(shard)}}},
               "provenance_sha256": {name: recovery.sha256(w4 / name) for name in names},
               "export_snapshot": recovery._snapshot_tree(partial)}}
    audit_path.write_text(json.dumps(record))
    audit_path.with_name(audit_path.name + ".sha256.json").write_text(json.dumps({
        "code_sha": sha, "recovery_validation_job_id": "60000",
        "audit_sha256": recovery.sha256(audit_path)}))
    monkeypatch.setattr(recovery, "verify_report", lambda *_:
                        {"slurm_job_id": "55356", "source_identity": {"hash": "source"}})
    monkeypatch.setattr(recovery, "_failed_slurm_job", lambda job: {"job_id": job})
    monkeypatch.setattr(recovery.subprocess, "check_output", lambda *_a, **_k:
                        "60000|COMPLETED|0:0\n")
    monkeypatch.setattr(recovery.subprocess, "run", lambda *_a, **_k:
                        SimpleNamespace(returncode=0))
    monkeypatch.setattr(recovery, "check_revision", lambda *_: None)
    monkeypatch.setattr(recovery, "_source_hashes", lambda *_: None)
    monkeypatch.setattr(recovery, "_resolved_matches", lambda *_: None)
    monkeypatch.setattr(recovery, "validate_control", lambda *_:
                        pytest.fail("tensor decoding on GPU"))
    assert recovery.load_audited_inputs(source, result, sha, tmp_path, tmp_path,
                                       audit_path) == record
    shard.write_bytes(b"tampered checkpoint")
    with pytest.raises(ValueError, match="checkpoint changed"):
        recovery.load_audited_inputs(source, result, sha, tmp_path, tmp_path, audit_path)
    shard.write_bytes(b"original packed bytes")
    (w4 / "run.json").write_text("changed provenance")
    with pytest.raises(ValueError, match="provenance changed"):
        recovery.load_audited_inputs(source, result, sha, tmp_path, tmp_path, audit_path)
    (w4 / "run.json").write_text("run.json")
    changed = dict(record, code_sha="b" * 40)
    audit_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="audit integrity differs"):
        recovery.load_audited_inputs(source, result, sha, tmp_path, tmp_path, audit_path)


def test_streaming_source_hashes_reject_modified_shard(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("config.json", "model.safetensors.index.json", "model-1.safetensors"):
        (source / name).write_bytes(name.encode())
    monkeypatch.setattr("lp_llm.rrq_pr2308_compare._weight_map",
                        lambda *_: {"weight": "model-1.safetensors"})
    expected = {"tensor_names": ["weight"],
        "config_sha256": recovery.sha256(source / "config.json"),
        "index_sha256": recovery.sha256(source / "model.safetensors.index.json"),
        "shard_sha256": {"model-1.safetensors": recovery.sha256(
            source / "model-1.safetensors")}}
    recovery._source_hashes(source, expected)
    (source / "model-1.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source shard changed"):
        recovery._source_hashes(source, expected)
