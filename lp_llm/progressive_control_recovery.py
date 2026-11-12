"""Validate a historical S01 W4 partial export and evaluate it without requantization."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch

from .progressive_control import _resolved_matches, control_identity, validate_control
from .progressive_lp import _files
from .progressive_stage_a_recovery import _snapshot_tree
from .quantizer_followups import PR_COMMIT
from .rrq_reference_eval import check_revision, corpus_identity, run as evaluate_matgptq
from .storage import atomic_json, sha256


TOKEN_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
TOKEN_SHA = "8ffcee84dd4b684ec631cb52845e18c9c181ce6fdb8b26d65a4ffdc3d35fbda3"


def _tokenizers(source: Path) -> dict:
    result = {}
    for name in TOKEN_FILES:
        path = source / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Original tokenizer file is unavailable: {name}")
        result[name] = {"sha256": sha256(path), "size": path.stat().st_size}
    return result


def audit(source: Path, control: Path, checkout: Path, output: Path,
          code_sha: str) -> dict:
    """Full CPU tensor validation, preserving export.partial and its producer SHA."""
    partial = control / "export.partial"
    if (not partial.is_dir() or (control / "encoding_complete.json").exists() or
            (control / "export").exists()):
        raise ValueError("Expected original unmarked partial W4 export")
    run = json.loads((control / "run.json").read_text())
    original = run.get("identity", {})
    producer = original.get("code_sha")
    if not isinstance(producer, str) or len(producer) != 40:
        raise ValueError("Original W4 producing SHA unavailable")
    with patch.dict(os.environ, {"FOLLOWUP_CODE_SHA": producer}):
        expected = control_identity("s01", 4, source, checkout)
    if original != expected or original.get("source") != _files(source):
        raise ValueError("Original W4 recipe or source identity differs")
    process = json.loads((control / "quantization_process.json").read_text())
    if process.get("returncode") != 0 or not (control / "quantization.log").is_file():
        raise ValueError("Original W4 quantization did not succeed")
    _resolved_matches(control / "resolved_config.json", "s01", 4, checkout)
    repair = json.loads((control / "excluded_repair.json").read_text())
    checked = validate_control(source, partial, 4)
    if (repair.get("quantized_projection_byte_identical") is not True or
            repair.get("raw_projection_tensor_sha256") != checked.get("projection_tensor_sha256") or
            checked.get("quantized_projection_count") != 252):
        raise ValueError("W4 projection or exclusion repair integrity differs")
    report = {"status": "completed_validated", "code_sha": code_sha,
              "original_producer_sha": producer, "original_identity": original,
              "original_partial_export": str(partial.resolve()),
              "original_process": process, "original_repair": repair,
              "checkpoint_validation": checked, "snapshot": _snapshot_tree(partial),
              "tokenizer_files": _tokenizers(source),
              "source_identity": _files(source),
              "provenance_sha256": {name: sha256(control / name) for name in
                ("run.json", "quantization_process.json", "quantization.log",
                 "resolved_config.json", "excluded_repair.json")},
              "original_resource_certification": "not_assessed_by_this_audit"}
    if output.exists():
        if json.loads(output.read_text()) != report:
            raise ValueError("Existing W4 recovery audit differs")
    else:
        atomic_json(output, report)
    seal = output.with_name(output.name + ".sha256.json")
    expected_seal = {"code_sha": code_sha, "audit_sha256": sha256(output)}
    if seal.exists():
        if json.loads(seal.read_text()) != expected_seal:
            raise ValueError("Existing W4 recovery audit seal differs")
    else:
        atomic_json(seal, expected_seal)
    return report


def _verified_audit(source: Path, control: Path, checkout: Path,
                    audit_path: Path, code_sha: str) -> dict:
    report = json.loads(audit_path.read_text())
    seal = json.loads(audit_path.with_name(audit_path.name + ".sha256.json").read_text())
    if seal != {"code_sha": code_sha, "audit_sha256": sha256(audit_path)}:
        raise ValueError("W4 recovery audit seal differs")
    partial = control / "export.partial"
    if (report.get("status") != "completed_validated" or report.get("code_sha") != code_sha or
            report.get("original_partial_export") != str(partial.resolve()) or
            report.get("source_identity") != _files(source) or
            report.get("tokenizer_files") != _tokenizers(source) or
            _snapshot_tree(partial) != report.get("snapshot")):
        raise ValueError("W4 recovery audit inputs changed")
    for name, digest in report["provenance_sha256"].items():
        if sha256(control / name) != digest:
            raise ValueError(f"W4 original provenance changed: {name}")
    producer = report["original_producer_sha"]
    with patch.dict(os.environ, {"FOLLOWUP_CODE_SHA": producer}):
        if report["original_identity"] != control_identity("s01", 4, source, checkout):
            raise ValueError("W4 producing identity changed")
    _resolved_matches(control / "resolved_config.json", "s01", 4, checkout)
    return report


def _view(view: Path, source: Path, report: dict) -> dict:
    checkpoint = Path(report["checkpoint_validation"]["checkpoint"]["resolved_checkpoint_path"])
    partial = Path(report["original_partial_export"])
    files = {path.relative_to(checkpoint).as_posix(): path for path in checkpoint.rglob("*")
             if path.is_file() and not path.is_symlink()}
    if not {"config.json", "model.safetensors.index.json"} <= set(files):
        raise ValueError("W4 checkpoint view lacks config or index")
    expected = {"checkpoint": str(checkpoint), "source": str(source.resolve()),
                "model_files": {name: {"size": path.stat().st_size, "sha256": sha256(path)}
                                for name, path in files.items() if name not in TOKEN_FILES},
                "tokenizer_files": report["tokenizer_files"]}
    for name, path in files.items():
        if report["snapshot"]["files"].get(path.relative_to(partial).as_posix()) != {
                "size": path.stat().st_size, "sha256": sha256(path)}:
            raise ValueError("W4 view file differs from CPU audit")
        if name in TOKEN_FILES and sha256(path) != report["tokenizer_files"][name]["sha256"]:
            raise ValueError("W4 exported tokenizer differs from original")
    if not view.exists():
        temporary = view.with_name(view.name + ".partial")
        if temporary.exists():
            raise FileExistsError("Interrupted W4 evaluation view requires inspection")
        temporary.mkdir(parents=True)
        try:
            for name, digest in expected["model_files"].items():
                target = temporary / name
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(files[name], target)
            for name in TOKEN_FILES:
                shutil.copyfile(source / name, temporary / name)
            atomic_json(temporary / "view.json", expected)
            os.replace(temporary, view)
        except Exception:
            shutil.rmtree(temporary)
            raise
    if view.is_symlink() or json.loads((view / "view.json").read_text()) != expected:
        raise ValueError("W4 evaluation view manifest differs")
    entries = list(view.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ValueError("W4 evaluation view contains a symbolic link")
    actual = {path.relative_to(view).as_posix() for path in entries if path.is_file()}
    if actual != set(expected["model_files"]) | set(TOKEN_FILES) | {"view.json"}:
        raise ValueError("W4 evaluation view inventory differs")
    for name, digest in expected["model_files"].items():
        target = view / name
        if (target.is_symlink() or not os.path.samefile(target, files[name]) or
                sha256(target) != digest["sha256"]):
            raise ValueError("W4 evaluation view model changed")
    for name in TOKEN_FILES:
        if sha256(view / name) != report["tokenizer_files"][name]["sha256"]:
            raise ValueError("W4 evaluation view tokenizer changed")
    return expected


def evaluate(source: Path, control: Path, checkout: Path, matgptq: Path,
             audit_path: Path, result: Path, code_sha: str) -> dict:
    check_revision(matgptq)
    report = _verified_audit(source, control, checkout, audit_path, code_sha)
    view = result.parent / "w4_view"
    if result.exists() and not (result / "full.json").is_file():
        raise FileExistsError("Incomplete W4 recovery evaluation requires inspection")
    view_identity = _view(view, source, report)
    if (result / "full.json").is_file():
        saved = json.loads((result / "full.json").read_text())
        if (saved.get("code_sha") != code_sha or saved.get("view") != view_identity or
                sha256(result / "evaluation.log") != saved.get("evaluation_log_sha256")):
            raise ValueError("Existing W4 recovery evaluation differs")
        return saved
    result.mkdir(parents=True)
    corpus = corpus_identity(view)
    if corpus.get("token_sha256") != TOKEN_SHA or corpus.get("scored_tokens") != 298862:
        raise ValueError("W4 recovery corpus identity differs")
    measured = evaluate_matgptq(matgptq, view, result / "evaluation.log")
    if _view(view, source, report) != view_identity:
        raise ValueError("W4 evaluation view changed during evaluation")
    full = {"status": "completed_validated", "code_sha": code_sha,
            "audit_sha256": sha256(audit_path), "view": view_identity,
            "original_producer_sha": report["original_producer_sha"],
            "original_partial_export": report["original_partial_export"],
            "checkpoint_validation": report["checkpoint_validation"],
            "corpus": corpus, "evaluation": measured,
            "evaluation_log_sha256": sha256(result / "evaluation.log")}
    atomic_json(result / "full.json", full)
    return full


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("audit", "evaluate"))
    for key in ("source", "control", "checkout", "audit", "matgptq", "result"):
        p.add_argument(f"--{key}", type=Path)
    p.add_argument("--code-sha", required=True)
    a = p.parse_args()
    if a.action == "audit":
        value = audit(a.source, a.control, a.checkout, a.audit, a.code_sha)
    else:
        value = evaluate(a.source, a.control, a.checkout, a.matgptq,
                         a.audit, a.result, a.code_sha)
    print(json.dumps({"status": value["status"]}, indent=2))


if __name__ == "__main__":
    main()
