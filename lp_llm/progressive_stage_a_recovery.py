"""Read-only scientific audit of the two failed Stage A Slurm jobs."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from .progressive_control import _resolved_matches, control_identity, validate_control
from .progressive_lp import identity, verified_layers
from .progressive_prerequisites import verify_report
from .quantizer_followups import TOKEN_SHA256
from .rrq_reference_eval import corpus_identity, run as evaluate_matgptq, check_revision
from .rrq_w2 import MATGPTQ_COMMIT
from .storage import atomic_json, sha256


ORIGINAL_SHA = "0c6557e79569176a42b0b861787d21fc98bd62c6"
LP_JOB = "55385"
W4_JOB = "55386"
TOKEN_SHA = "8ffcee84dd4b684ec631cb52845e18c9c181ce6fdb8b26d65a4ffdc3d35fbda3"
LP_NUMERICAL_PATHS = ("lp_llm/progressive_lp.py", "lp_llm/progressive_run.py",
                      "lp_llm/quantize.py", "lp_llm/storage.py",
                      "lp_llm/progressive_control_worker.py",
                      "lp_llm/rrq_reference_eval.py", "lp_llm/rrq_pr2308_compare.py",
                      "lp_llm/evaluate.py")


def _failed_slurm_job(job_id: str) -> dict:
    rows = subprocess.check_output(["sacct", "-n", "-P", "-X", "-j", job_id,
        "-o", "JobIDRaw,State,ExitCode"], text=True).strip().splitlines()
    if len(rows) != 1:
        raise ValueError(f"Original Stage A Slurm job record unavailable: {job_id}")
    fields = rows[0].split("|")
    if len(fields) != 3 or fields[0] != job_id or fields[1] != "FAILED" or fields[2] == "0:0":
        raise ValueError(f"Original Stage A Slurm failure differs: {job_id}: {rows}")
    return {"job_id": job_id, "state": fields[1], "exit_code": fields[2]}


def audit(source: Path, result_root: Path, code_sha: str, code_repo: Path,
          checkout: Path, output: Path) -> dict:
    previous = json.loads(output.read_text()) if output.exists() else None
    validation_job_id = (previous.get("recovery_validation_job_id") if previous else
                         os.environ.get("SLURM_JOB_ID", ""))
    if not str(validation_job_id).isdigit():
        raise ValueError("Stage A recovery audit requires a Slurm validation job ID")
    prerequisite = verify_report(source, result_root, code_sha, code_repo)
    lp_slurm = _failed_slurm_job(LP_JOB)
    w4_slurm = _failed_slurm_job(W4_JOB)
    if subprocess.run(["git", "-C", str(code_repo), "diff", "--quiet", ORIGINAL_SHA,
                       code_sha, "--", *LP_NUMERICAL_PATHS], check=False).returncode:
        raise ValueError("LP numerical implementation changed since original Stage A job")
    lp_root = result_root / "progressive_llm/s01"
    lp_result = result_root / "progressive_llm/evaluation/s01_lp_2p2"
    full_path = lp_result / "full.json"
    full = json.loads(full_path.read_text())
    base = result_root / "quantizer_followups/s01_fp16_load_repair/export"
    ident = identity("s01", source, base)
    if (full.get("status") != "completed_validated" or full.get("identity") != ident or
            full.get("code_sha") != ORIGINAL_SHA or full.get("depth") != 1 or
            full.get("corpus", {}).get("token_sha256") != TOKEN_SHA or
            full.get("corpus", {}).get("scored_tokens") != 298862 or
            round(full.get("evaluation", {}).get("perplexity", -1), 2) != 13.38 or
            full.get("materialization", {}).get("projection_count") != 252 or
            sha256(lp_result / "evaluation.log") != full.get("evaluation_log_sha256") or
            (lp_result / "checkpoint").exists() or
            (lp_result / "checkpoint.partial").exists()):
        raise ValueError("Original LP scientific output differs")
    with patch.dict(os.environ, {"FOLLOWUP_CODE_SHA": ORIGINAL_SHA}):
        layers = verified_layers(lp_root, 1, ident)
    if len(layers) != 1 or layers[0].get("code_sha") != ORIGINAL_SHA:
        raise ValueError("Original LP encoded layer differs")
    lp_resource_path = result_root / f"progressive_llm/job_resources/s01_lp_2p2-{LP_JOB}.json"
    lp_resource = json.loads(lp_resource_path.read_text())
    if (lp_resource.get("job_id") != LP_JOB or
            lp_resource.get("host_allocation_bytes") != 12 * 1024**3 or
            lp_resource.get("host_memory_10pct_headroom_passed") is not False or
            lp_resource.get("slurm_cgroup_peak_bytes") != 15_439_818_752):
        raise ValueError("Original LP resource failure evidence differs")
    w4_root = result_root / "progressive_llm/controls/s01_w4"
    partial = w4_root / "export.partial"
    if (w4_root / "encoding_complete.json").exists() or not partial.is_dir():
        raise ValueError("Expected original unmarked W4 partial export")
    current_ident = control_identity("s01", 4, source, checkout)
    if current_ident.get("code_sha") != code_sha:
        raise ValueError("Recovery audit is not running under its pinned code SHA")
    original_ident = {**current_ident, "code_sha": ORIGINAL_SHA}
    run = json.loads((w4_root / "run.json").read_text())
    process = json.loads((w4_root / "quantization_process.json").read_text())
    config = json.loads((w4_root / "worker_config.json").read_text())
    repair = json.loads((w4_root / "excluded_repair.json").read_text())
    if (run.get("identity") != original_ident or
            original_ident["source"] != prerequisite["source_identity"] or
            process.get("returncode") != 0 or
            config != {"branch": "s01", "bits": 4} or
            not (w4_root / "quantization.log").is_file() or
            repair.get("quantized_projection_byte_identical") is not True or
            repair.get("fp16_load_roundtrip_explains_all_excluded_changes") is not True or
            len(repair.get("restored_excluded_names", [])) != 12 or
            len(repair.get("raw_projection_tensor_sha256", {})) != 252):
        raise ValueError("Original W4 process, source, or repair provenance differs")
    _resolved_matches(w4_root / "resolved_config.json", "s01", 4, checkout)
    validated = validate_control(source, partial, 4)
    if validated["projection_tensor_sha256"] != repair["raw_projection_tensor_sha256"]:
        raise ValueError("Repaired W4 projections differ from original quantized bytes")
    report = {"status": "stage_a_science_validated_resource_uncertified",
              "code_sha": code_sha, "original_code_sha": ORIGINAL_SHA,
              "recovery_validation_job_id": str(validation_job_id),
              "prerequisite_job_id": prerequisite["slurm_job_id"],
              "lp": {"original_job_id": LP_JOB, "original_job_failed_resource_gate": True,
                     "original_slurm": lp_slurm,
                     "original_allocation_bytes": 12 * 1024**3,
                     "original_cgroup_peak_bytes": lp_resource["slurm_cgroup_peak_bytes"],
                     "original_process_peak_rss_bytes": lp_resource.get("process_peak_rss_bytes"),
                     "original_headroom_fraction": (12 * 1024**3 -
                         max(lp_resource["slurm_cgroup_peak_bytes"],
                             lp_resource.get("process_peak_rss_bytes", 0))) / (12 * 1024**3),
                     "measured_peak_below_15_gib": max(
                         lp_resource["slurm_cgroup_peak_bytes"],
                         lp_resource.get("process_peak_rss_bytes", 0)) <= 15 * 1024**3,
                     "original_resource_report_sha256": sha256(lp_resource_path),
                     "full_json_sha256": sha256(full_path),
                     "evaluation_log_sha256": full["evaluation_log_sha256"],
                     "perplexity": full["evaluation"]["perplexity"],
                     "encoded_layer": layers[0]},
              "w4": {"original_job_id": W4_JOB, "original_quantization_code_sha": ORIGINAL_SHA,
                     "original_slurm": w4_slurm,
                     "partial_export": str(partial), "process": process,
                     "quantization_log_sha256": sha256(w4_root / "quantization.log"),
                     "repair": repair, "checkpoint_validation": validated},
              "resource_certification": "not_satisfied_by_original_12_gib_jobs"}
    if output.exists():
        if previous != report:
            raise ValueError("Existing Stage A recovery audit differs")
    else:
        atomic_json(output, report)
    return report


def evaluate_w4(audit_record: dict, matgptq: Path, output: Path,
                recovery_job_id: str, *, python: str = sys.executable) -> dict:
    """Evaluate the validated original partial export without moving its shards."""
    if (audit_record.get("status") != "stage_a_science_validated_resource_uncertified" or
            not recovery_job_id.isdigit()):
        raise ValueError("Validated Stage A recovery and Slurm job ID required")
    check_revision(matgptq, MATGPTQ_COMMIT)
    checkpoint = Path(audit_record["w4"]["partial_export"])
    full_path = output / "full.json"
    if full_path.exists():
        full = json.loads(full_path.read_text())
        if (full.get("status") != "completed_validated" or
                full.get("code_sha") != audit_record["code_sha"] or
                full.get("original_quantization_job_id") != W4_JOB or
                full.get("original_quantization_code_sha") != ORIGINAL_SHA or
                full.get("checkpoint_validation") != audit_record["w4"]["checkpoint_validation"] or
                full.get("corpus", {}).get("token_sha256") != TOKEN_SHA or
                full.get("corpus", {}).get("scored_tokens") != 298862 or
                sha256(output / "evaluation.log") != full.get("evaluation_log_sha256")):
            raise ValueError("Existing recovered W4 evaluation differs")
        return full
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Incomplete recovered W4 evaluation requires inspection")
    output.mkdir(parents=True, exist_ok=True)
    corpus = corpus_identity(checkpoint)
    if corpus.get("token_sha256") != TOKEN_SHA256 or corpus.get("scored_tokens") != 298862:
        raise ValueError("Recovered W4 evaluation corpus identity differs")
    measurement = evaluate_matgptq(matgptq, checkpoint, output / "evaluation.log", python=python)
    full = {"status": "completed_validated", "code_sha": audit_record["code_sha"],
            "original_quantization_code_sha": ORIGINAL_SHA,
            "original_quantization_job_id": W4_JOB,
            "recovery_evaluation_job_id": recovery_job_id,
            "checkpoint_validation": audit_record["w4"]["checkpoint_validation"],
            "corpus": corpus, "evaluation": measurement,
            "evaluation_log_sha256": sha256(output / "evaluation.log")}
    atomic_json(full_path, full)
    return full


def verify_recovery(audit_record: dict, result_root: Path) -> dict:
    """Report scientific recovery while refusing to certify the old LP memory run."""
    if (audit_record.get("status") != "stage_a_science_validated_resource_uncertified" or
            audit_record.get("original_code_sha") != ORIGINAL_SHA or
            audit_record.get("lp", {}).get("original_job_id") != LP_JOB or
            audit_record.get("w4", {}).get("original_job_id") != W4_JOB or
            audit_record.get("w4", {}).get("checkpoint_validation", {}).get(
                "quantized_projection_count") != 252 or
            audit_record.get("resource_certification") !=
                "not_satisfied_by_original_12_gib_jobs"):
        raise ValueError("Stage A scientific audit provenance differs")
    root = result_root / "progressive_llm/recovery" / audit_record["code_sha"] / "s01_w4"
    full = json.loads((root / "full.json").read_text())
    job_id = str(full.get("recovery_evaluation_job_id", ""))
    slurm = subprocess.check_output(["sacct", "-n", "-P", "-X", "-j", job_id,
        "-o", "JobIDRaw,State,ExitCode"], text=True).strip().splitlines()
    resource = json.loads((result_root / "progressive_llm/job_resources" /
                           f"s01_w4_recovery-{job_id}.json").read_text())
    if (full.get("status") != "completed_validated" or
            full.get("code_sha") != audit_record["code_sha"] or
            full.get("original_quantization_job_id") != W4_JOB or
            full.get("checkpoint_validation") != audit_record["w4"]["checkpoint_validation"] or
            full.get("corpus", {}).get("token_sha256") != TOKEN_SHA or
            full.get("corpus", {}).get("scored_tokens") != 298862 or
            not math.isfinite(full.get("evaluation", {}).get("perplexity", float("nan"))) or
            sha256(root / "evaluation.log") != full.get("evaluation_log_sha256") or
            slurm != [f"{job_id}|COMPLETED|0:0"] or
            resource.get("job_id") != job_id or
            resource.get("host_allocation_bytes") != 15 * 1024**3 or
            resource.get("host_memory_within_allocation") is not True or
            resource.get("slurm_cgroup_peak_bytes", 0) <= 0 or
            resource.get("process_peak_rss_bytes", 0) <= 0 or
            max(resource["slurm_cgroup_peak_bytes"],
                resource["process_peak_rss_bytes"]) > 15 * 1024**3 or
            resource.get("exit_code_before_resource_check") != 0):
        raise ValueError("Recovered W4 evaluation, Slurm or resource evidence differs")
    return {"status": "stage_a_science_validated_resource_uncertified",
            "lp_original_job_id": LP_JOB, "lp_original_resource_headroom_passed": False,
            "w4_original_quantization_job_id": W4_JOB,
            "w4_recovery_evaluation_job_id": job_id,
            "w4_perplexity": full["evaluation"]["perplexity"],
            "stage_b_admission": "blocked_pending_lp_full_run_memory_certification"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("audit", "evaluate", "verify"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--code-repo", type=Path, required=True)
    parser.add_argument("--autoround", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matgptq", type=Path)
    args = parser.parse_args()
    if args.action == "verify":
        verify_report(args.source, args.result_root, args.code_sha, args.code_repo)
        record = json.loads(args.output.read_text())
        if record.get("code_sha") != args.code_sha:
            raise ValueError("Stage A recovery audit code SHA differs")
        record = verify_recovery(record, args.result_root)
    else:
        record = audit(args.source, args.result_root, args.code_sha, args.code_repo,
                       args.autoround, args.output)
    if args.action == "evaluate":
        if args.matgptq is None:
            parser.error("evaluate requires --matgptq")
        result = args.result_root / "progressive_llm/recovery" / args.code_sha / "s01_w4"
        record = evaluate_w4(record, args.matgptq, result,
                             os.environ.get("SLURM_JOB_ID", ""))
    print(json.dumps({"status": record["status"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
