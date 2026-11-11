import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from lp_llm.progressive_plan import (LP_PILOT_PREVIOUS_SHA, accepted_lp_pilot_sha,
                                     admit_submission, main, plan, submit, verify_stage_a)


def test_matrix_has_twelve_points_and_scientific_dependencies(tmp_path):
    report = plan(tmp_path, "a" * 40, slots=14)
    assert report["status"] == "dry_run_not_submitted"
    assert report["qos_submit_jobs_count"] == 12
    assert len(report["points"]) == 12
    lookup = {item["id"]: item for item in report["points"]}
    assert lookup["s01_lp_2p2p2"]["predecessor"] == "s01_lp_2p2"
    assert lookup["s01_lp_2p2p2p2"]["predecessor"] == "s01_lp_2p2p2"
    assert lookup["r02_lp_2p2p2"]["predecessor"] == "r02_lp_2p2"
    assert all("lane" not in item for item in report["points"])
    assert len({item["output"] for item in report["points"]}) == 12
    assert len({item["id"] for item in report["points"] if item["kind"] == "control"}) == 6
    assert plan(tmp_path, "a" * 40, slots=14)["max_concurrent_gpu_jobs"] == 14
    static = json.loads((Path(__file__).resolve().parents[1] /
                         "configs/progressive_llm/matrix.json").read_text())
    assert [item["id"] for item in static["new_points"]] == [item["id"] for item in report["points"]]
    assert [item["predecessor"] for item in static["new_points"]] == [
        item["predecessor"] for item in report["points"]]


def test_submit_uses_only_scientific_dependencies_and_resumes(tmp_path, monkeypatch):
    report = plan(tmp_path, "a" * 40, slots=14)
    commands = []

    def fake_sbatch(command, *, text):
        if command[0] == "squeue":
            return ""
        commands.append(command)
        return str(1000 + len(commands))

    monkeypatch.setenv("LP_MATRIX_SUBMIT", "1")
    monkeypatch.setenv("LP_MATRIX_PILOTS_REVIEWED", "1")
    monkeypatch.setattr("lp_llm.progressive_plan.subprocess.check_output", fake_sbatch)
    manifest = tmp_path / "jobs.json"
    stage_a = {"s01_w4", "s01_lp_2p2"}
    first = submit(report, tmp_path / "run_point.sbatch", manifest, selected=stage_a)
    assert first["status"] == "submitted" and set(first["job_ids"]) == stage_a
    completed = submit(report, tmp_path / "run_point.sbatch", manifest)
    assert completed["status"] == "submitted" and len(completed["job_ids"]) == 12
    assert len(commands) == 12
    assert any("--dependency=afterok:1001" in command for command in commands)
    assert all(not any("--dependency" in part for part in command)
               for command in commands if any("LP_MATRIX_ID=s01_w4" in part or
                   "LP_MATRIX_ID=r02_w4" in part for part in command))
    assert submit(report, tmp_path / "run_point.sbatch", manifest) == completed
    assert len(commands) == 12
    assert json.loads(manifest.read_text())["job_ids"] == completed["job_ids"]


def test_stage_a_gate_requires_successful_jobs_and_scientific_reports(tmp_path, monkeypatch):
    source = {"config_sha256": "a" * 64}
    manifest = tmp_path / "jobs.json"
    manifest.write_text(json.dumps({"code_sha": "a" * 40,
                                    "job_ids": {"s01_w4": "101", "s01_lp_2p2": "102"}}))
    monkeypatch.setattr("lp_llm.progressive_plan.subprocess.check_output",
                        lambda command, **_: "COMPLETED|0:0\n")
    common = {"status": "completed_validated", "identity": {"source": source},
              "corpus": {"token_sha256":
                  "8ffcee84dd4b684ec631cb52845e18c9c181ce6fdb8b26d65a4ffdc3d35fbda3",
                  "scored_tokens": 298862}, "evaluation": {"perplexity": 12.47}}
    for point, job in (("s01_w4", "101"), ("s01_lp_2p2", "102")):
        folder = tmp_path / "progressive_llm" / ("controls" if point == "s01_w4"
            else "evaluation") / point
        folder.mkdir(parents=True)
        full = dict(common)
        if point == "s01_w4":
            check = {"quantized_projection_count": 252,
                "source_exact_excluded_tensor_count": 147,
                "sampled_group_grid_checks": 1, "actual_checkpoint_file_bytes": 1,
                "distortion": {"mse_fp32": 0.01, "normalized_mse_fp32": 0.02}}
            full.update(identity={"source": source, "code_sha": "a" * 40},
                        checkpoint_validation=check)
            (folder / "encoding_complete.json").write_text(json.dumps({
                "status": "completed_validated", "checkpoint_validation": check}))
        else:
            full.update(code_sha="a" * 40, depth=1,
                        materialization={"status": "completed_validated",
                                         "projection_count": 252},
                        storage={"prefixes": [{"total_reconstruction_bytes": 1}]})
            layer_path = tmp_path / "progressive_llm/s01/layer1/encoding_complete.json"
            layer_path.parent.mkdir(parents=True)
            layer_path.write_text(json.dumps({"status": "completed_validated",
                "identity": {"source": source}, "code_sha": "a" * 40,
                "projection_count": 252, "chunk_count": 1, "chunks": [{}],
                "code_bytes": 1, "metadata_bytes": 1,
                "mse_fp32": 0.01, "normalized_mse_fp32": 0.02}))
        (folder / "full.json").write_text(json.dumps(full))
        resource = tmp_path / "progressive_llm/job_resources" / f"{point}-{job}.json"
        resource.parent.mkdir(parents=True, exist_ok=True)
        resource.write_text(json.dumps({"status": "measured", "point": point,
            "job_id": job, "host_memory_within_allocation": True,
            "host_memory_10pct_headroom_passed": False,
            "slurm_cgroup_peak_bytes": 1, "process_peak_rss_bytes": 1,
            "host_allocation_bytes": 15 * 1024**3,
            "exit_code_before_resource_check": 0}))
    assert len(verify_stage_a(tmp_path, manifest, {"source_identity": source}, "a" * 40)) == 2
    monkeypatch.setattr("lp_llm.progressive_plan.subprocess.check_output",
                        lambda command, **_: "FAILED|1:0\n")
    with pytest.raises(ValueError, match="not completed successfully"):
        verify_stage_a(tmp_path, manifest, {"source_identity": source}, "a" * 40)


def test_partial_submission_respects_fourteen_account_jobs(tmp_path, monkeypatch):
    report = plan(tmp_path, "a" * 40)
    submitted = []
    active = [str(i) for i in range(13)]
    def fake(command, *, text):
        if command[0] == "squeue":
            return "\n".join(active)
        submitted.append(command)
        return str(2000 + len(submitted))
    monkeypatch.setenv("LP_MATRIX_SUBMIT", "1")
    monkeypatch.setenv("LP_MATRIX_PILOTS_REVIEWED", "1")
    monkeypatch.setattr("lp_llm.progressive_plan.subprocess.check_output", fake)
    output = tmp_path / "jobs.json"
    partial = submit(report, tmp_path / "job.sbatch", output)
    assert partial["status"] == "partially_submitted"
    assert len(partial["job_ids"]) == 1
    assert len(submitted) == 1
    active.clear()
    complete = submit(report, tmp_path / "job.sbatch", output)
    assert complete["status"] == "submitted"
    assert len(complete["job_ids"]) == 12
    assert len(submitted) == 12


def test_full_account_and_unresolved_intent_fail_closed(tmp_path, monkeypatch):
    report = plan(tmp_path, "a" * 40)
    calls = []
    active = [str(i) for i in range(14)]
    def fake(command, *, text):
        if command[0] == "squeue":
            return "\n".join(active)
        calls.append(command)
        return "3001"
    monkeypatch.setenv("LP_MATRIX_SUBMIT", "1")
    monkeypatch.setenv("LP_MATRIX_PILOTS_REVIEWED", "1")
    monkeypatch.setattr("lp_llm.progressive_plan.subprocess.check_output", fake)
    output = tmp_path / "jobs.json"
    first = submit(report, tmp_path / "job.sbatch", output)
    assert first["status"] == "partially_submitted" and not calls
    active.clear()
    intent = output.with_name(output.name + ".intents") / "s01_lp_2p2.json"
    intent.parent.mkdir(parents=True)
    intent.write_text('{"status":"submission_in_progress"}')
    with pytest.raises(FileExistsError, match="Unresolved Slurm submission"):
        submit(report, tmp_path / "job.sbatch", output)
    assert not calls


def test_prior_lp_pilot_acceptance_is_exact_and_requires_unchanged_code(monkeypatch):
    calls = []
    def unchanged(command, **_kwargs):
        calls.append(command)
        return type("Result", (), {"returncode": 0})()
    monkeypatch.setattr("lp_llm.progressive_plan.subprocess.run", unchanged)
    monkeypatch.setattr("lp_llm.progressive_plan.verify_executed_code", lambda *_: None)
    source = (Path(__file__).resolve().parents[1] / "lp_llm/progressive_lp.py").read_text()
    monkeypatch.setattr("lp_llm.progressive_plan.subprocess.check_output",
                        lambda command, **_: source if command[3] == "show" else "")
    assert accepted_lp_pilot_sha(LP_PILOT_PREVIOUS_SHA, "a" * 40, Path("repo"), Path("stage")) == LP_PILOT_PREVIOUS_SHA
    assert len(calls) == 1 and "progressive_run.py" in " ".join(calls[0])
    with pytest.raises(ValueError, match="validated prior"):
        accepted_lp_pilot_sha("f" * 40, "a" * 40, Path("repo"), Path("stage"))
    monkeypatch.setattr("lp_llm.progressive_plan.subprocess.run",
                        lambda *_a, **_k: type("Result", (), {"returncode": 1})())
    with pytest.raises(ValueError, match="implementation differs"):
        accepted_lp_pilot_sha(LP_PILOT_PREVIOUS_SHA, "a" * 40, Path("repo"), Path("stage"))
    monkeypatch.setattr("lp_llm.progressive_plan.subprocess.run", unchanged)
    monkeypatch.setattr("lp_llm.progressive_plan.subprocess.check_output",
                        lambda command, **_: source + "\ndef changed_numerical_code(): return 1\n"
                        if command[-1].startswith("a" * 40) else source)
    with pytest.raises(ValueError, match="numerical implementation differs"):
        accepted_lp_pilot_sha(LP_PILOT_PREVIOUS_SHA, "a" * 40, Path("repo"), Path("stage"))


def test_cross_commit_acceptance_from_real_git_archive(tmp_path):
    workspace = Path(__file__).resolve().parents[1]
    checkout, staged = tmp_path / "checkout", tmp_path / "stage"
    subprocess.run(["git", "clone", "--quiet", str(workspace), str(checkout)], check=True)
    for file in ("lp_llm/progressive_plan.py", "lp_llm/progressive_control_pilot.py",
                 "lp_llm/progressive_resources.py", "lp_llm/progressive_lp.py",
                 "lp_llm/progressive_prerequisites.py"):
        shutil.copy2(workspace / file, checkout / file)
    shutil.copy2(workspace / "hpc/progressive_llm/run_point.sbatch",
                 checkout / "hpc/progressive_llm/run_point.sbatch")
    (checkout / "archive-integration-marker.txt").write_text("proposed archive\n")
    subprocess.run(["git", "-C", str(checkout), "add", "lp_llm/progressive_plan.py",
                    "lp_llm/progressive_control_pilot.py", "lp_llm/progressive_resources.py",
                    "lp_llm/progressive_lp.py", "lp_llm/progressive_prerequisites.py",
                    "hpc/progressive_llm/run_point.sbatch",
                    "archive-integration-marker.txt"], check=True)
    subprocess.run(["git", "-C", str(checkout), "-c", "user.name=Test", "-c",
                    "user.email=test@example.com", "commit", "-qm", "proposed archive"], check=True)
    proposed = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                                       text=True).strip()
    staged.mkdir()
    archive = subprocess.check_output(["git", "-C", str(checkout), "archive", proposed])
    import tarfile, io
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for member in stream:
            if member.isfile():
                target = staged / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(stream.extractfile(member).read())
    assert not (staged / ".git").exists()
    assert '--code-repo "$LP_CODE"' in (staged / "hpc/progressive_llm/run_point.sbatch").read_text()
    environment = dict(os.environ, PYTHONPATH=str(staged))
    code = ("from pathlib import Path; from lp_llm.progressive_plan import "
            "accepted_lp_pilot_sha; print(accepted_lp_pilot_sha(" +
            repr(LP_PILOT_PREVIOUS_SHA) + ", " + repr(proposed) +
            ", Path(" + repr(str(checkout)) + "), Path.cwd()))")
    result = subprocess.run([sys.executable, "-c", code], cwd=staged, env=environment,
                            text=True, capture_output=True, check=False)
    numerical_changed = subprocess.run(["git", "-C", str(checkout), "diff", "--quiet",
        LP_PILOT_PREVIOUS_SHA, proposed, "--", "lp_llm/progressive_lp.py"],
        check=False).returncode != 0
    if numerical_changed:
        # A new enhancement format must not inherit historical pilot admission.
        assert result.returncode != 0
        assert ("LP numerical implementation differs" in result.stderr or
                "LP implementation differs" in result.stderr)
        return
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == LP_PILOT_PREVIOUS_SHA
    result_root = tmp_path / "results"
    pilots = result_root / "progressive_llm/pilots"
    metrics = {"projection_count": 7, "encoded_elements": 1, "mse_fp32": 0.1,
               "normalized_mse_fp32": 0.2, "code_bytes": 1, "metadata_bytes": 1}
    for branch, ppl in (("s01", 12.47), ("r02", 32.22)):
        root = pilots / branch
        (root / "base_parity").mkdir(parents=True)
        (root / "report.json").write_text(json.dumps({"status": "pilot_validated",
            "branch": branch, "code_sha": LP_PILOT_PREVIOUS_SHA,
            "identity": {"branch": branch}, "projection_names": list(range(7)),
            "base_checkpoint_source_exact_exclusions": True,
            "base_zero_enhancement_exact": True, "process_peak_rss_kib": 1,
            "host_allocation_bytes": 12 * 1024**3, "artifact_bytes": 1,
            "first": metrics, "second": metrics}))
        (root / "base_parity/base_parity.json").write_text(json.dumps({
            "status": "base_parity_validated", "identity": {"branch": branch},
            "code_sha": LP_PILOT_PREVIOUS_SHA,
            "corpus": {"token_sha256": "8ffcee84dd4b684ec631cb52845e18c9c181ce6fdb8b26d65a4ffdc3d35fbda3",
                       "scored_tokens": 298862}, "evaluation": {"perplexity": ppl}}))
    from lp_llm.quantizer_followups import PR_COMMIT
    recipes = {f"{branch}_w{bits}": {"branch": branch, "bits": bits,
        "group_size": 128, "sym": branch == "s01",
        "algorithm_class": "SignRoundV2Config" if branch == "s01" else "OptimizedRTNConfig",
        "quantizer_class": "SignRoundV2Quantizer" if branch == "s01" else "OptimizedRTNQuantizer",
        "iterations": 200 if branch == "s01" else 0,
        "enable_neuqi": branch == "r02",
        "selected_quant_function": "opt_rtn_int_asym" if branch == "r02" else None,
        "function_name": "quant_tensor_opt_rtn_asym" if branch == "r02" else None,
        "neuqi_grid": {"coarse": 256, "fine": 64} if branch == "r02" else None,
        "calibration": {"dataset": "NeelNanda/pile-10k", "nsamples": 128,
                        "seqlen": 2048, "batch_size": 8, "seed": 42}}
        for branch in ("s01", "r02") for bits in (4, 6, 8)}
    (pilots / f"control_support_{proposed}.json").write_text(json.dumps({
        "status": "control_dispatch_verified", "code_sha": proposed,
        "upstream_revision": PR_COMMIT, "upstream_checkout": str(checkout),
        "recipes": recipes}))
    parts = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
             "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    for branch in ("s01", "r02"):
        for bits in (4, 6, 8):
            (pilots / f"{branch}_w{bits}.json").write_text(json.dumps({
                "status": "control_layer_pilot_validated", "branch": branch,
                "bits": bits, "code_sha": proposed, "source": {"test": True},
                "recipe": recipes[f"{branch}_w{bits}"],
                "reference_module": str(checkout / "auto_round/__init__.py"),
                "projection_count": 7,
                "traversal_metadata_status": "available",
                "traversed_transformer_blocks": [f"model.layers.{i}" for i in range(36)],
                "traversed_transformer_block_count": 36,
                "quantized_projection_names": sorted(f"model.layers.0.{part}.weight" for part in parts),
                "quantized_projection_count": 7,
                "unselected_projection_source_exact_count": 245,
                "projections": {f"model.layers.0.{part}.weight": {"changed_elements": 1,
                    "elements": 2, "sampled_group_levels": 2}
                                for part in parts},
                "gpu_peak_reserved_bytes": 1, "process_peak_rss_kib": 1,
                "slurm_cgroup_peak_bytes": 1,
                "slurm_cgroup_peak_source": "cgroup_v2_memory.peak",
                "after_quantization_cgroup_peak_bytes": 1,
                "after_quantization_process_peak_rss_kib": 1,
                "host_peak_used_for_headroom_bytes": 1024,
                "host_memory_headroom_bytes": 14 * 1024**3 - 1024,
                "host_memory_10pct_headroom_passed": True,
                "host_allocation_bytes": 14 * 1024**3, "gpu_total_bytes": 1000}))
    absent_path = pilots / "s01_w4.json"
    absent = json.loads(absent_path.read_text())
    absent.update(traversal_metadata_status="unavailable",
                  traversed_transformer_blocks=None,
                  traversed_transformer_block_count=None)
    absent_path.write_text(json.dumps(absent))
    for branch in ("s01", "r02"):
        for bits in (4, 6, 8):
            (pilots / f"{branch}_w{bits}.json").unlink()
    preflight = '''from pathlib import Path
from unittest.mock import patch
from lp_llm import progressive_plan as p
import sys
root, repo, sha = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
layer = {"code_sha": p.LP_PILOT_PREVIOUS_SHA, "projection_count": 7,
         "encoded_elements": 1, "mse_fp32": 0.1, "normalized_mse_fp32": 0.2,
         "code_bytes": 1, "metadata_bytes": 1}
with patch.object(p, "validate_repaired_base", return_value={}), \\
     patch.object(p, "identity", side_effect=lambda branch, *_: {"branch": branch}), \\
     patch.object(p, "verified_layers", return_value=[layer, layer]), \\
     patch.object(p, "_files", return_value={"test": True}):
    report = p.verify_pilots(root, root, sha, repo)
assert report["s01"]["cross_commit_acceptance"]
assert report["r02"]["cross_commit_acceptance"]
print("ARCHIVED_PREFLIGHT_VALIDATED")
'''
    script = staged / "preflight_fixture.py"
    script.write_text(preflight)
    preflight_result = subprocess.run([sys.executable, str(script), str(result_root),
        str(checkout), proposed], cwd=staged, env=environment, text=True,
        capture_output=True, check=False)
    assert preflight_result.returncode == 0, preflight_result.stderr
    assert "ARCHIVED_PREFLIGHT_VALIDATED" in preflight_result.stdout
    assert (pilots / f"lp_cross_commit_acceptance_{proposed}.json").is_file()
    lp_file = staged / "lp_llm/progressive_lp.py"
    lp_file.write_bytes(lp_file.read_bytes() + b"\n# tampered staged code\n")
    rejected = subprocess.run([sys.executable, "-c", code], cwd=staged, env=environment,
                              text=True, capture_output=True, check=False)
    assert rejected.returncode != 0 and "Executed code differs" in rejected.stderr


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
