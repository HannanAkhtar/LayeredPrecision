import json
import sys
from pathlib import Path

import pytest

from lp_llm import progressive_prerequisites as gate
from lp_llm.progressive_lp import BRANCHES, _files
from lp_llm.quantizer_followups import PR_COMMIT


SHA = "a" * 40
OLD_SHA = "b650efb085b00b801f0f42543485c84ae9fbbbc0"


def fixture_tree(tmp_path):
    source = tmp_path / "source"
    result = tmp_path / "results"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "model.safetensors.index.json").write_text("{}")
    (source / "model-1.safetensors").write_bytes(b"source")
    for branch, repair in BRANCHES.items():
        parent = result / "quantizer_followups" / repair
        (parent / "export").mkdir(parents=True)
        (parent / "export/model-1.safetensors").write_bytes(branch.encode())
        (parent / "encoding_complete.json").write_text("{}")
        pilot = result / "progressive_llm/pilots" / branch
        pilot.mkdir(parents=True)
        (pilot / "report.json").write_text("{}")
    pilots = result / "progressive_llm/pilots"
    recipes = {f"{branch}_w{bits}": {"branch": branch, "bits": bits,
        "group_size": 128, "sym": branch == "s01",
        "algorithm_class": "SignRoundV2Config" if branch == "s01" else "OptimizedRTNConfig",
        "quantizer_class": "SignRoundV2Quantizer" if branch == "s01" else "OptimizedRTNQuantizer",
        "iterations": 200 if branch == "s01" else 0, "enable_neuqi": branch == "r02",
        "calibration": {"dataset": "NeelNanda/pile-10k", "nsamples": 128,
                        "seqlen": 2048, "batch_size": 8, "seed": 42},
        "selected_quant_function": "opt_rtn_int_asym" if branch == "r02" else None,
        "function_name": "quant_tensor_opt_rtn_asym" if branch == "r02" else None,
        "neuqi_grid": {"coarse": 256, "fine": 64} if branch == "r02" else None}
        for branch in BRANCHES for bits in (4, 6, 8)}
    support = {"status": "control_dispatch_verified", "code_sha": SHA,
               "upstream_revision": PR_COMMIT, "recipes": recipes}
    support_path = pilots / f"control_support_{SHA}.json"
    support_path.write_text(json.dumps(support))
    names = [f"model.layers.{i}.{part}.weight" for i in range(36) for part in
             ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
              "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]
    ident = {"config_sha256": "1" * 64, "index_sha256": "2" * 64,
             "shard_sha256": {"model-1.safetensors": "3" * 64},
             "tensor_names": sorted(names)}
    checked = {branch: {"accepted_pilot_code_sha": OLD_SHA,
        "cross_commit_acceptance": True,
        "source_base_identity": {"source": ident, "repaired_base": ident,
                                 "eligible_projection_count": 252},
        "base_validation": {"quantized_projection_count": 252,
                            "quantized_projection_names": names,
                            "unchanged_tensor_count": 1, "checked_w2_row_groups": 1,
                            "config_sha256": ident["config_sha256"],
                            "index_sha256": ident["index_sha256"],
                            "shard_sha256": ident["shard_sha256"]},
        "base_parity_perplexity": ppl,
        "report": str(pilots / branch / "report.json"),
        "base_parity_report": str(pilots / branch / "base_parity/base_parity.json")}
        for branch, ppl in (("s01", 12.47), ("r02", 32.22))}
    checked["control_support"] = str(support_path)
    acceptance = pilots / f"lp_cross_commit_acceptance_{SHA}.json"
    acceptance.write_text(json.dumps({"status": "prior_lp_pilots_validated",
        "current_code_sha": SHA, "prior_code_sha": OLD_SHA,
        "lp_numerical_ast_verified_unchanged": True,
        "accepted": {branch: checked[branch] for branch in BRANCHES}}))
    checked["cross_commit_acceptance_report"] = str(acceptance)
    return source, result, checked


def test_cpu_audit_report_and_lightweight_resume(tmp_path, monkeypatch):
    source, result, checked = fixture_tree(tmp_path)
    monkeypatch.setattr("lp_llm.progressive_plan.verify_pilots", lambda *_: checked)
    monkeypatch.setattr("lp_llm.progressive_plan.verify_executed_code", lambda *_: None)
    report = gate.verify_full(source, result, SHA, tmp_path, job_id="123")
    assert report["status"] == "prerequisites_validated"
    assert report["source_identity"] == checked["s01"]["source_base_identity"]["source"]
    assert report["bases"]["r02"]["base_validation"]["quantized_projection_count"] == 252
    monkeypatch.setattr("lp_llm.progressive_lp.sha256", lambda *_: pytest.fail("rehash"))
    monkeypatch.setattr("lp_llm.quantizer_followups.validate_checkpoint",
                        lambda *_: pytest.fail("tensor scan"))
    assert gate.verify_report(source, result, SHA, tmp_path) == report
    monkeypatch.setenv("LP_PREREQUISITE_REPORT", str(gate.report_path(result, SHA)))
    monkeypatch.setenv("FOLLOWUP_CODE_SHA", SHA)
    monkeypatch.setenv("RESULT_ROOT", str(result))
    monkeypatch.setenv("LP_SOURCE", str(source))
    monkeypatch.setenv("LP_CODE", str(tmp_path))
    gate._REPORT_CACHE.clear()
    assert _files(source) == report["source_identity"]
    assert _files(result / "quantizer_followups/s01_fp16_load_repair/export") == \
        report["bases"]["s01"]["source_base_identity"]["repaired_base"]
    (source / "model-1.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="metadata changed|inputs changed"):
        gate.verify_report(source, result, SHA, tmp_path)
    with pytest.raises(ValueError, match="inputs changed"):
        _files(source)


def test_failed_cpu_audit_never_writes_success(tmp_path, monkeypatch):
    source, result, _ = fixture_tree(tmp_path)
    def fail(*_):
        raise ValueError("bad checkpoint")
    monkeypatch.setattr("lp_llm.progressive_plan.verify_pilots", fail)
    with pytest.raises(ValueError, match="bad checkpoint"):
        gate.verify_full(source, result, SHA, tmp_path, job_id="124")
    assert not gate.report_path(result, SHA).exists()
    diagnostic = gate.report_path(result, SHA).with_name(f"prerequisites_{SHA}.failure.124.json")
    assert json.loads(diagnostic.read_text())["status"] == "prerequisites_failed"


def test_reverification_preserves_prior_success_on_failure(tmp_path, monkeypatch):
    source, result, checked = fixture_tree(tmp_path)
    monkeypatch.setattr("lp_llm.progressive_plan.verify_pilots", lambda *_: checked)
    first = gate.verify_full(source, result, SHA, tmp_path, job_id="130")
    monkeypatch.setattr("lp_llm.progressive_plan.verify_pilots",
                        lambda *_: (_ for _ in ()).throw(ValueError("fresh audit failed")))
    with pytest.raises(ValueError, match="fresh audit failed"):
        gate.verify_full(source, result, SHA, tmp_path, job_id="131")
    assert json.loads(gate.report_path(result, SHA).read_text()) == first
    monkeypatch.setattr("lp_llm.progressive_plan.verify_pilots", lambda *_: checked)
    second = gate.verify_full(source, result, SHA, tmp_path, job_id="132")
    assert second["slurm_job_id"] == "132"
    backup = gate.report_path(result, SHA).with_name(f"prerequisites_{SHA}.superseded.132.json")
    assert json.loads(backup.read_text()) == first


def test_mismatched_report_and_missing_artifact_fail_closed(tmp_path, monkeypatch):
    source, result, checked = fixture_tree(tmp_path)
    monkeypatch.setattr("lp_llm.progressive_plan.verify_pilots", lambda *_: checked)
    monkeypatch.setattr("lp_llm.progressive_plan.verify_executed_code", lambda *_: None)
    with pytest.raises(FileNotFoundError):
        gate.verify_report(source, result, SHA, tmp_path)
    gate.verify_full(source, result, SHA, tmp_path, job_id="125")
    path = gate.report_path(result, SHA)
    saved = json.loads(path.read_text())
    saved["code_sha"] = "b" * 40
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="schema"):
        gate.verify_report(source, result, SHA, tmp_path)
    path.write_text(json.dumps({**saved, "code_sha": SHA}))
    (result / "quantizer_followups/r02_fp16_load_repair/export/model-1.safetensors").unlink()
    with pytest.raises(ValueError, match="metadata changed"):
        gate.verify_report(source, result, SHA, tmp_path)


def test_gpu_script_uses_report_gate_and_keeps_output_validation():
    root = Path(__file__).resolve().parents[1]
    script = (root / "hpc/progressive_llm/run_point.sbatch").read_text()
    assert "lp_llm.progressive_prerequisites check" in script
    assert "lp_llm.progressive_plan verify-prerequisites" not in script
    assert "lp_llm.progressive_run evaluate" in script
    assert "lp_llm.progressive_control" in script
    cpu = (root / "hpc/progressive_llm/verify_prerequisites.sbatch").read_text()
    assert "#SBATCH --partition=hpc" in cpu
    assert "#SBATCH --gres" not in cpu
    assert "flock -n 9" in cpu


def test_submission_without_validated_report_is_blocked(tmp_path, monkeypatch):
    from lp_llm.progressive_plan import main
    source, result, _ = fixture_tree(tmp_path)
    monkeypatch.setattr(sys, "argv", ["progressive_plan", "submit",
        "--source", str(source), "--result-root", str(result),
        "--code-sha", SHA, "--code-repo", str(tmp_path),
        "--manifest", str(tmp_path / "matrix.json")])
    with pytest.raises(FileNotFoundError):
        main()
    assert not (tmp_path / "matrix.json").exists()
