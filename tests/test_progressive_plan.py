import json
import sys
from pathlib import Path

import pytest

from lp_llm.progressive_plan import admit_submission, main, plan, submit


def test_matrix_has_twelve_points_and_scientific_dependencies(tmp_path):
    report = plan(tmp_path, "a" * 40, slots=2)
    assert report["status"] == "dry_run_not_submitted"
    assert report["qos_submit_jobs_count"] == 12
    assert len(report["points"]) == 12
    lookup = {item["id"]: item for item in report["points"]}
    assert lookup["s01_lp_2p2p2"]["predecessor"] == "s01_lp_2p2"
    assert lookup["s01_lp_2p2p2p2"]["predecessor"] == "s01_lp_2p2p2"
    assert lookup["r02_lp_2p2p2"]["predecessor"] == "r02_lp_2p2"
    assert {item["lane"] for item in report["points"]} == {0, 1}
    assert len({item["output"] for item in report["points"]}) == 12
    assert len({item["id"] for item in report["points"] if item["kind"] == "control"}) == 6
    three = plan(tmp_path, "a" * 40, slots=3)
    assert {item["lane"] for item in three["points"]} == {0, 1, 2}
    static = json.loads((Path(__file__).resolve().parents[1] /
                         "configs/progressive_llm/matrix.json").read_text())
    assert [item["id"] for item in static["new_points"]] == [item["id"] for item in report["points"]]
    assert [item["predecessor"] for item in static["new_points"]] == [
        item["predecessor"] for item in report["points"]]


def test_submit_uses_afterok_lane_dependencies_and_resumes(tmp_path, monkeypatch):
    report = plan(tmp_path, "a" * 40, slots=2)
    commands = []

    def fake_sbatch(command, *, text):
        commands.append(command)
        return str(1000 + len(commands))

    monkeypatch.setenv("LP_MATRIX_SUBMIT", "1")
    monkeypatch.setenv("LP_MATRIX_PILOTS_REVIEWED", "1")
    monkeypatch.setattr("lp_llm.progressive_plan.subprocess.check_output", fake_sbatch)
    manifest = tmp_path / "jobs.json"
    completed = submit(report, tmp_path / "run_point.sbatch", manifest)
    assert completed["status"] == "submitted" and len(completed["job_ids"]) == 12
    assert len(commands) == 12
    assert "--dependency=afterok:1001" in commands[1]
    assert "--dependency=afterok:1004" in commands[4]
    assert submit(report, tmp_path / "run_point.sbatch", manifest) == completed
    assert len(commands) == 12
    assert json.loads(manifest.read_text())["job_ids"] == completed["job_ids"]
    three = plan(tmp_path, "a" * 40, slots=3)
    other = submit(three, tmp_path / "run_point.sbatch", tmp_path / "three.json")
    assert other["max_concurrent_gpu_jobs"] == 3
    assert len(other["job_ids"]) == 12
    assert {point["lane"] for point in other["points"]} == {0, 1, 2}
    assert "--dependency=afterok:1013" in commands[13]
    assert "--dependency=afterok:1019" in commands[19]


def test_submit_requires_explicit_review(tmp_path, monkeypatch):
    monkeypatch.delenv("LP_MATRIX_SUBMIT", raising=False)
    monkeypatch.delenv("LP_MATRIX_PILOTS_REVIEWED", raising=False)
    with pytest.raises(ValueError, match="required"):
        submit(plan(tmp_path, "a" * 40), tmp_path / "job.sbatch", tmp_path / "jobs.json")


def test_dry_run_cannot_replace_submitted_manifest(tmp_path, monkeypatch):
    output = tmp_path / "jobs.json"
    prior = plan(tmp_path, "a" * 40)
    prior.update(status="submitting", job_ids={"s01_lp_2p2": "123"})
    output.write_text(json.dumps(prior))
    monkeypatch.setattr(sys, "argv", ["progressive_plan", "dry-run",
        "--result-root", str(tmp_path), "--code-sha", "a" * 40,
        "--manifest", str(output)])
    with pytest.raises(FileExistsError, match="cannot overwrite"):
        main()
    assert json.loads(output.read_text()) == prior


def test_submission_quota_override_only_skips_formal_quota(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr("lp_llm.progressive_plan.estimate",
                        lambda *_args, **_kwargs: {"new_matrix_shared_floor_bytes": 100})
    monkeypatch.setattr("lp_llm.progressive_plan.shutil.disk_usage",
                        lambda _: SimpleNamespace(free=200))
    monkeypatch.delenv("LP_MATRIX_SHARED_QUOTA_FREE_GIB", raising=False)
    monkeypatch.delenv("LP_MATRIX_ALLOW_UNKNOWN_QUOTA", raising=False)
    with pytest.raises(ValueError, match="quota or"):
        admit_submission(tmp_path, tmp_path, 3)
    monkeypatch.setenv("LP_MATRIX_ALLOW_UNKNOWN_QUOTA", "1")
    report = admit_submission(tmp_path, tmp_path, 3)
    assert report["quota_status"] == "unknown_operator_authorized"
    assert report["declared_quota_bytes"] is None and report["slots"] == 3
    monkeypatch.setattr("lp_llm.progressive_plan.shutil.disk_usage",
                        lambda _: SimpleNamespace(free=99))
    with pytest.raises(ValueError, match="Actual shared"):
        admit_submission(tmp_path, tmp_path, 3)
