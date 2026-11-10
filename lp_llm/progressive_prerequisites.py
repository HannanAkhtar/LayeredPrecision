"""Full CPU prerequisite audit and metadata-based freshness checks."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from .storage import atomic_json

VERSION = 1


def _peak_rss_kib() -> int | None:
    try:
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except ImportError:  # Local Windows tests; AUS CPU jobs use Linux.
        return None


def _cgroup_peak(job_id: str) -> tuple[int | None, str | None]:
    try:
        from .progressive_control_pilot import slurm_cgroup_peak
        return slurm_cgroup_peak(job_id)
    except (RuntimeError, OSError, ValueError):
        return None, None


def report_path(result_root: Path, code_sha: str) -> Path:
    return result_root / "progressive_llm/prerequisites" / f"prerequisites_{code_sha}.json"


def _inventory(roots: dict[str, Path]) -> dict[str, dict]:
    result = {}
    for label, root in roots.items():
        if not root.is_dir() or root.is_symlink():
            raise ValueError(f"Prerequisite directory missing or linked: {root}")
        files = {}
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Prerequisite symlink is not allowed: {path}")
            if path.is_file():
                stat = path.stat()
                files[path.relative_to(root).as_posix()] = {
                    "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                    "ctime_ns": stat.st_ctime_ns, "inode": stat.st_ino,
                    "device": stat.st_dev}
        if not files:
            raise ValueError(f"Prerequisite directory has no files: {root}")
        result[label] = {"path": str(root.resolve()), "files": files}
    return result


def _roots(source: Path, result_root: Path, code_sha: str) -> dict[str, Path]:
    from .progressive_lp import BRANCHES
    roots = {"source": source}
    for branch, repair in BRANCHES.items():
        roots[f"{branch}_base"] = result_root / "quantizer_followups" / repair
        roots[f"{branch}_pilot"] = result_root / "progressive_llm/pilots" / branch
    roots["control_support"] = result_root / "progressive_llm/pilots"
    return roots


def _watched_inventory(source: Path, result_root: Path, code_sha: str) -> dict:
    roots = _roots(source, result_root, code_sha)
    # The top-level pilot directory also contains run-specific diagnostics.
    # Watch only the pinned support file there, not unrelated later reports.
    support = roots.pop("control_support") / f"control_support_{code_sha}.json"
    if not support.is_file() or support.is_symlink():
        raise ValueError("Pinned control support report is missing")
    inventory = _inventory(roots)
    stat = support.stat()
    inventory["control_support"] = {"path": str(support.resolve()), "files": {
        support.name: {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                       "ctime_ns": stat.st_ctime_ns, "inode": stat.st_ino,
                       "device": stat.st_dev}}}
    acceptance = result_root / "progressive_llm/pilots" / f"lp_cross_commit_acceptance_{code_sha}.json"
    if acceptance.is_file():
        if acceptance.is_symlink():
            raise ValueError("Cross-commit acceptance link is not allowed")
        stat = acceptance.stat()
        inventory["cross_commit_acceptance"] = {"path": str(acceptance.resolve()),
            "files": {acceptance.name: {"size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns,
                "inode": stat.st_ino, "device": stat.st_dev}}}
    return inventory


def verify_report(source: Path, result_root: Path, code_sha: str,
                  code_repo: Path, *, staged_root: Path | None = None) -> dict:
    """No tensor reads or checkpoint hashes; reject any watched metadata change."""
    from .progressive_plan import verify_executed_code
    from .rrq_pr2308_compare import _eligible
    if not re.fullmatch(r"[0-9a-f]{40}", code_sha):
        raise ValueError("Invalid prerequisite code SHA")
    path = report_path(result_root, code_sha)
    if path.is_symlink():
        raise ValueError("Prerequisite report link is not allowed")
    record = json.loads(path.read_text())
    expected_recipes = {f"{branch}_w{bits}" for branch in ("s01", "r02")
                        for bits in (4, 6, 8)}
    if (record.get("status") != "prerequisites_validated" or
            record.get("verification_version") != VERSION or
            record.get("code_sha") != code_sha or
            not str(record.get("slurm_job_id", "")).isdigit() or
            not record.get("completed_at_utc") or
            not record.get("started_at_utc") or
            record.get("elapsed_seconds", -1) < 0 or
            record.get("pinned_autoround_revision") !=
            "7985c1b0f4eef65fb717511668fda7e347e584c3" or
            set(record.get("bases", {})) != {"s01", "r02"} or
            record.get("source_identity") != record["bases"]["s01"].get("source_base_identity", {}).get("source") or
            record.get("source_identity") != record["bases"]["r02"].get("source_base_identity", {}).get("source") or
            any(record["bases"][branch].get("accepted_pilot_code_sha") !=
                "b650efb085b00b801f0f42543485c84ae9fbbbc0" or
                record["bases"][branch].get("cross_commit_acceptance") is not True or
                record["bases"][branch].get("source_base_identity", {}).get(
                    "eligible_projection_count") != 252 or
                round(record["bases"][branch].get("base_parity_perplexity", -1), 2) !=
                    {"s01": 12.47, "r02": 32.22}[branch]
                for branch in ("s01", "r02")) or
            record.get("control_support", {}).get("status") != "control_dispatch_verified" or
            record.get("control_support", {}).get("code_sha") != code_sha or
            record.get("control_support", {}).get("upstream_revision") !=
            record.get("pinned_autoround_revision") or
            record.get("calibration_corpus_identity") !=
            {"dataset": "NeelNanda/pile-10k", "nsamples": 128,
             "seqlen": 2048, "batch_size": 8, "seed": 42} or
            set(record.get("control_support", {}).get("recipes", {})) !=
            expected_recipes):
        raise ValueError("Prerequisite report schema, identity or provenance differs")
    for branch in ("s01", "r02"):
        ident = record["bases"][branch]["source_base_identity"]
        if ident["source"] != record["source_identity"] or \
                ident["repaired_base"].get("tensor_names") != record["source_identity"].get("tensor_names"):
            raise ValueError("Source/base tensor inventories differ")
        validation = record["bases"][branch]["base_validation"]
        if (validation.get("quantized_projection_count") != 252 or
                len(validation.get("quantized_projection_names", [])) != 252 or
                validation.get("unchanged_tensor_count", 0) <= 0 or
                validation.get("checked_w2_row_groups", 0) <= 0 or
                validation.get("config_sha256") != ident["repaired_base"]["config_sha256"] or
                validation.get("index_sha256") != ident["repaired_base"]["index_sha256"] or
                validation.get("shard_sha256") != ident["repaired_base"]["shard_sha256"]):
            raise ValueError("Repaired checkpoint coverage audit missing")
        for files in (record["source_identity"], ident["repaired_base"]):
            if (not isinstance(files.get("config_sha256"), str) or
                    not isinstance(files.get("index_sha256"), str) or
                    not files.get("shard_sha256") or
                    not files.get("tensor_names")):
                raise ValueError("Prerequisite checkpoint hashes or inventory missing")
            if any(not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in
                   [files["config_sha256"], files["index_sha256"],
                    *files["shard_sha256"].values()]):
                raise ValueError("Prerequisite checkpoint hash malformed")
            if (len([name for name in files["tensor_names"] if _eligible(name)]) != 252 or
                    files["tensor_names"] != sorted(set(files["tensor_names"]))):
                raise ValueError("Prerequisite projection inventory differs")
        if not record["bases"][branch].get("base_validation"):
            raise ValueError("Repaired checkpoint validation missing")
        if (record["bases"][branch].get("report") != str(result_root /
                "progressive_llm/pilots" / branch / "report.json") or
                record["bases"][branch].get("base_parity_report") != str(result_root /
                "progressive_llm/pilots" / branch / "base_parity/base_parity.json")):
            raise ValueError("Historical pilot paths differ")
    for branch in ("s01", "r02"):
        for bits in (4, 6, 8):
            recipe = record["control_support"]["recipes"][f"{branch}_w{bits}"]
            if (recipe.get("branch") != branch or recipe.get("bits") != bits or
                    recipe.get("group_size") != 128 or
                    recipe.get("sym") is not (branch == "s01") or
                    recipe.get("algorithm_class") != ("SignRoundV2Config" if branch == "s01"
                                                       else "OptimizedRTNConfig") or
                    recipe.get("quantizer_class") != ("SignRoundV2Quantizer" if branch == "s01"
                                                       else "OptimizedRTNQuantizer") or
                    recipe.get("iterations") != (200 if branch == "s01" else 0) or
                    recipe.get("enable_neuqi") is not (branch == "r02") or
                    recipe.get("calibration") != {"dataset": "NeelNanda/pile-10k",
                        "nsamples": 128, "seqlen": 2048, "batch_size": 8, "seed": 42}):
                raise ValueError("Pinned control recipe differs")
            if branch == "r02" and (recipe.get("selected_quant_function") !=
                    "opt_rtn_int_asym" or recipe.get("function_name") !=
                    "quant_tensor_opt_rtn_asym" or recipe.get("neuqi_grid") !=
                    {"coarse": 256, "fine": 64}):
                raise ValueError("Pinned NeUQI recipe differs")
    acceptance_path = result_root / "progressive_llm/pilots" / f"lp_cross_commit_acceptance_{code_sha}.json"
    if record.get("cross_commit_acceptance_report") != str(acceptance_path):
        raise ValueError("Historical LP cross-commit acceptance path differs")
    acceptance = json.loads(acceptance_path.read_text())
    if (acceptance.get("status") != "prior_lp_pilots_validated" or
            acceptance.get("current_code_sha") != code_sha or
            acceptance.get("prior_code_sha") != "b650efb085b00b801f0f42543485c84ae9fbbbc0" or
            acceptance.get("lp_numerical_ast_verified_unchanged") is not True or
            set(acceptance.get("accepted", {})) != {"s01", "r02"} or
            any(acceptance["accepted"][branch] != record["bases"][branch]
                for branch in ("s01", "r02"))):
        raise ValueError("Historical LP cross-commit acceptance differs")
    if record.get("inventory") != _watched_inventory(source, result_root, code_sha):
        raise ValueError("Prerequisite file metadata changed; run full CPU verification")
    verify_executed_code(code_repo, code_sha,
                         staged_root or Path(__file__).resolve().parents[1])
    return record


_REPORT_CACHE: dict[tuple[str, str], tuple[dict, tuple[int, int, int]]] = {}


def _report_stamp(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


def verified_input_identity(directory: Path, *, branch: str | None = None) -> dict | None:
    """Return a CPU-audited source/base identity only for an exact watched path."""
    path = os.environ.get("LP_PREREQUISITE_REPORT")
    if not path:
        return None
    reported_path = Path(path)
    if reported_path.is_symlink():
        raise ValueError("Prerequisite report link is not allowed")
    report = reported_path.resolve()
    code_sha = os.environ.get("FOLLOWUP_CODE_SHA", "")
    expected = Path(os.environ["RESULT_ROOT"]) / "progressive_llm/prerequisites" / f"prerequisites_{code_sha}.json"
    if report != expected.resolve():
        raise ValueError("Prerequisite report path differs from pinned code SHA")
    source = Path(os.environ["LP_SOURCE"])
    result_root = Path(os.environ["RESULT_ROOT"])
    key = (str(report), code_sha)
    if key not in _REPORT_CACHE or _REPORT_CACHE[key][1] != _report_stamp(report):
        checked = verify_report(source, result_root, code_sha,
                                Path(os.environ["LP_CODE"]))
        _REPORT_CACHE[key] = checked, _report_stamp(report)
    record = _REPORT_CACHE[key][0]
    if record["inventory"] != _watched_inventory(source, result_root, code_sha):
        raise ValueError("Prerequisite inputs changed during experiment")
    if directory.is_symlink():
        raise ValueError("Prerequisite input directory link is not allowed")
    absolute = os.path.normcase(os.path.abspath(directory))
    if absolute == os.path.normcase(os.path.abspath(source)):
        return record["source_identity"]
    from .progressive_lp import BRANCHES
    for name, repair in BRANCHES.items():
        base = result_root / "quantizer_followups" / repair / "export"
        if absolute == os.path.normcase(os.path.abspath(base)):
            if branch is not None and branch != name:
                raise ValueError("Prerequisite base branch differs")
            return record["bases"][name]["base_validation"] if branch else record[
                "bases"][name]["source_base_identity"]["repaired_base"]
    return None


def verify_full(source: Path, result_root: Path, code_sha: str,
                code_repo: Path, *, job_id: str) -> dict:
    from .progressive_plan import verify_pilots
    from .progressive_lp import BRANCHES
    from .quantizer_followups import PR_COMMIT

    if not job_id.isdigit():
        raise ValueError("Full verification requires a Slurm job ID")
    path = report_path(result_root, code_sha)
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        print("[PREREQUISITE VERIFICATION STARTED]", flush=True)
        checked = verify_pilots(source, result_root, code_sha, code_repo)
        support = json.loads(Path(checked["control_support"]).read_text())
        source_identity = checked["s01"]["source_base_identity"]["source"]
        if checked["r02"]["source_base_identity"]["source"] != source_identity:
            raise ValueError("S01/R02 source identities differ")
        cgroup_peak, cgroup_source = _cgroup_peak(job_id)
        record = {"status": "prerequisites_validated", "verification_version": VERSION,
            "code_sha": code_sha, "slurm_job_id": job_id,
            "started_at_utc": started_at,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - started,
            "process_peak_rss_kib": _peak_rss_kib(),
            "slurm_cgroup_peak_bytes": cgroup_peak,
            "slurm_cgroup_peak_source": cgroup_source,
            "host_allocation_bytes": int(os.environ.get("SLURM_MEM_PER_NODE", "0")) * 1024**2,
            "source_identity": source_identity,
            "bases": {branch: checked[branch] for branch in BRANCHES},
            "cross_commit_acceptance_report": checked.get("cross_commit_acceptance_report"),
            "control_support": support,
            "calibration_corpus_identity": support["recipes"]["s01_w4"]["calibration"],
            "pinned_autoround_revision": PR_COMMIT,
            "inventory": _watched_inventory(source, result_root, code_sha)}
        if path.exists():
            prior = json.loads(path.read_text())
            if prior.get("source_identity") != record["source_identity"] or any(
                    prior.get("bases", {}).get(branch, {}).get("source_base_identity") !=
                    record["bases"][branch]["source_base_identity"] for branch in BRANCHES):
                raise ValueError("Existing prerequisite report identities differ; preserve and investigate")
            backup = path.with_name(f"{path.stem}.superseded.{job_id}.json")
            if backup.exists():
                raise FileExistsError(f"Prior prerequisite backup exists: {backup}")
            shutil.copy2(path, backup)
        atomic_json(path, record)
        print(f"[PREREQUISITES COMPLETE] {path}", flush=True)
        return record
    except Exception as error:
        failure = path.with_name(f"prerequisites_{code_sha}.failure.{job_id}.json")
        if not failure.exists():
            failure.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(failure, {"status": "prerequisites_failed", "code_sha": code_sha,
                "job_id": job_id, "error_type": type(error).__name__, "error": str(error),
                "elapsed_seconds": time.monotonic() - started,
                "process_peak_rss_kib": _peak_rss_kib()})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("full", "check"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--code-repo", type=Path, required=True)
    parser.add_argument("--job-id", default=os.environ.get("SLURM_JOB_ID", ""))
    args = parser.parse_args()
    if args.action == "full":
        verify_full(args.source, args.result_root, args.code_sha,
                    args.code_repo, job_id=args.job_id)
    else:
        value = verify_report(args.source, args.result_root, args.code_sha, args.code_repo)
        print(f"[PREREQUISITES REPORT VALIDATED] {report_path(args.result_root, args.code_sha)} "
              f"job={value['slurm_job_id']}")


if __name__ == "__main__":
    main()
