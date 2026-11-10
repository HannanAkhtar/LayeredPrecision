import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lp_llm import progressive_stage_a_recovery as recovery


def test_stage_a_audit_preserves_partial_and_original_provenance(tmp_path, monkeypatch):
    source, root, checkout = tmp_path / "source", tmp_path / "results", tmp_path / "auto-round"
    source.mkdir(); checkout.mkdir()
    monkeypatch.setenv("SLURM_JOB_ID", "60000")
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
