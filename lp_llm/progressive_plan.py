"""Dry-run first, explicitly gated Slurm plan for twelve LP-LLM operating points."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from .progressive_lp import BRANCHES, _files, identity, verified_layers
from .progressive_resources import estimate
from .quantizer_followups import PR_COMMIT
from .storage import atomic_json


def points(root: Path, slots: int) -> list[dict]:
    if slots not in (2, 3):
        raise ValueError("Concurrency cap must be 2 or 3")
    result = []
    for branch in BRANCHES:
        predecessor = None
        for depth in (1, 2, 3):
            # Explicit IDs avoid ambiguity in the published experiment table.
            name = {1: f"{branch}_lp_2p2", 2: f"{branch}_lp_2p2p2",
                    3: f"{branch}_lp_2p2p2p2"}[depth]
            result.append({"id": name, "kind": "lp", "branch": branch,
                           "depth": depth, "predecessor": predecessor,
                           "output": str(root / "progressive_llm/evaluation" / name),
                           "stage": "encode_validate_evaluate"})
            predecessor = name
    for branch in BRANCHES:
        for bits in (4, 6, 8):
            name = f"{branch}_w{bits}"
            result.append({"id": name, "kind": "control", "branch": branch,
                           "bits": bits, "predecessor": None,
                           "output": str(root / "progressive_llm/controls" / name),
                           "stage": "encode_validate_evaluate"})
    for point in result:
        point["lane"] = (0 if point["branch"] == "s01" else 1) if slots == 2 else (
            2 if point["kind"] == "control" else 0 if point["branch"] == "s01" else 1)
    assert len(result) == 12 and len({p["id"] for p in result}) == 12
    return result


def verify_pilots(source: Path, result_root: Path, code_sha: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", code_sha):
        raise ValueError("Invalid reviewed code SHA")
    reports = {}
    for branch, repair in BRANCHES.items():
        base = result_root / "quantizer_followups" / repair / "export"
        root = result_root / "progressive_llm/pilots" / branch
        record = json.loads((root / "report.json").read_text())
        ident = identity(branch, source, base)
        if (record.get("status") != "pilot_validated" or record.get("branch") != branch or
                record.get("code_sha") != code_sha or record.get("identity") != ident or
                len(record.get("projection_names", [])) != 7 or
                record.get("base_checkpoint_source_exact_exclusions") is not True or
                record.get("base_zero_enhancement_exact") is not True or
                record.get("process_peak_rss_kib", 0) * 1024 >=
                record.get("host_allocation_bytes", 0) * 0.9 or
                record.get("host_allocation_bytes", 0) <= 0):
            raise ValueError(f"Unreviewed or invalid {branch} full-size layer pilot")
        verified_layers(root, 2, ident, pilot=True)
        parity_path = root / "base_parity/base_parity.json"
        parity = json.loads(parity_path.read_text())
        if (parity.get("status") != "base_parity_validated" or
                (root / "base_parity/checkpoint").exists() or
                (root / "base_parity/checkpoint.partial").exists() or
                parity.get("identity") != ident or parity.get("code_sha") != code_sha or
                parity.get("corpus", {}).get("token_sha256") !=
                "8ffcee84dd4b684ec631cb52845e18c9c181ce6fdb8b26d65a4ffdc3d35fbda3" or
                parity.get("corpus", {}).get("scored_tokens") != 298862 or
                round(parity.get("evaluation", {}).get("perplexity", -1), 2) !=
                {"s01": 12.47, "r02": 32.22}[branch]):
            raise ValueError(f"Independent {branch} W2 perplexity parity is unverified")
        reports[branch] = {"process_peak_rss_kib": record["process_peak_rss_kib"],
                           "artifact_bytes": record["artifact_bytes"],
                           "report": str(root / "report.json"),
                           "base_parity_report": str(parity_path),
                           "base_parity_perplexity": parity["evaluation"]["perplexity"]}
    support = json.loads((result_root / "progressive_llm/pilots/control_support.json").read_text())
    if (support.get("status") != "control_dispatch_verified" or
            support.get("code_sha") != code_sha or
            support.get("upstream_revision") != PR_COMMIT or
            set(support.get("recipes", {})) !=
            {f"{branch}_w{bits}" for branch in BRANCHES for bits in (4, 6, 8)}):
        raise ValueError("Six pinned monolithic-control dispatches are not verified")
    reports["control_support"] = str(result_root / "progressive_llm/pilots/control_support.json")
    source_files = _files(source)
    for branch in BRANCHES:
        for bits in (4, 6, 8):
            name = f"{branch}_w{bits}"
            path = result_root / "progressive_llm/pilots" / f"{name}.json"
            control = json.loads(path.read_text())
            if (control.get("status") != "control_layer_pilot_validated" or
                    control.get("branch") != branch or control.get("bits") != bits or
                    control.get("code_sha") != code_sha or control.get("source") != source_files or
                    control.get("recipe") != support["recipes"][name] or
                    not Path(control.get("reference_module", "")).resolve().is_relative_to(
                        Path(support.get("upstream_checkout", "/nonexistent")).resolve()) or
                    control.get("projection_count") != 7 or
                    control.get("gpu_peak_reserved_bytes", 0) <= 0 or
                    control.get("process_peak_rss_kib", 0) <= 0 or
                    control.get("process_peak_rss_kib", 0) * 1024 >=
                    control.get("host_allocation_bytes", 0) * 0.9 or
                    control.get("gpu_peak_reserved_bytes", 0) >=
                    control.get("gpu_total_bytes", 0) * 0.9 or
                    control.get("host_allocation_bytes", 0) <= 0 or
                    control.get("gpu_total_bytes", 0) <= 0):
                raise ValueError(f"Unreviewed or invalid {name} control layer pilot")
            reports[name] = {"process_peak_rss_kib": control["process_peak_rss_kib"],
                             "gpu_peak_reserved_bytes": control["gpu_peak_reserved_bytes"],
                             "report": str(path)}
    return reports


def plan(root: Path, code_sha: str, *, slots: int = 2) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", code_sha):
        raise ValueError("Invalid pinned code SHA")
    return {"status": "dry_run_not_submitted", "code_sha": code_sha,
            "max_concurrent_gpu_jobs": slots, "qos_submit_jobs_count": 12,
            "point_count": 12, "points": points(root, slots),
            "dependency_rule": "afterok scientific predecessor plus prior job in same GPU lane"}


def submit(manifest: dict, script: Path, output: Path) -> dict:
    if (os.environ.get("LP_MATRIX_SUBMIT") != "1" or
            os.environ.get("LP_MATRIX_PILOTS_REVIEWED") != "1"):
        raise ValueError("LP_MATRIX_SUBMIT=1 and LP_MATRIX_PILOTS_REVIEWED=1 are required")
    submitted = dict(manifest)
    submitted["status"] = "submitting"
    job_ids = {}
    lane_last = {}
    if output.exists():
        previous = json.loads(output.read_text())
        if previous.get("code_sha") != manifest["code_sha"] or previous.get("points") != manifest["points"]:
            raise ValueError("Existing matrix manifest conflicts with plan")
        job_ids = previous.get("job_ids", {})
        if not isinstance(job_ids, dict) or any(not re.fullmatch(r"[0-9]+", str(v)) for v in job_ids.values()):
            raise ValueError("Existing matrix job IDs are invalid")
    submitted["job_ids"] = job_ids
    for point in manifest["points"]:
        predecessor = point["predecessor"]
        dependencies = {job_ids[predecessor]} if predecessor else set()
        lane = point["lane"]
        if lane in lane_last:
            dependencies.add(lane_last[lane])
        if point["id"] not in job_ids:
            command = ["sbatch", "--parsable",
                "--export=ALL,LP_MATRIX_ID=" + point["id"] +
                ",FOLLOWUP_CODE_SHA=" + manifest["code_sha"] +
                ",LP_MATRIX_ALLOW=1,LP_MATRIX_SLOTS=" + str(manifest["max_concurrent_gpu_jobs"])]
            if dependencies:
                command.append("--dependency=afterok:" + ":".join(sorted(dependencies)))
            command.append(str(script))
            response = subprocess.check_output(command, text=True).strip()
            job_id = response.split(";")[0]
            if not re.fullmatch(r"[0-9]+", job_id):
                raise ValueError(f"Unexpected sbatch job ID: {response}")
            job_ids[point["id"]] = job_id
            submitted["job_ids"] = job_ids
            atomic_json(output, submitted)
        lane_last[lane] = job_ids[point["id"]]
    submitted["status"] = "submitted"
    atomic_json(output, submitted)
    return submitted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("dry-run", "verify-pilots", "submit"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--slots", type=int, choices=(2, 3), default=2)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--script", type=Path,
                        default=Path("hpc/progressive_llm/run_point.sbatch"))
    args = parser.parse_args()
    if args.action == "verify-pilots":
        if args.source is None:
            parser.error("verify-pilots requires --source")
        print(json.dumps(verify_pilots(args.source, args.result_root, args.code_sha), indent=2))
        return
    manifest = plan(args.result_root, args.code_sha, slots=args.slots)
    if args.action == "submit":
        if args.source is None or args.manifest is None:
            parser.error("submit requires --source and --manifest")
        verify_pilots(args.source, args.result_root, args.code_sha)
        try:
            quota = float(os.environ["LP_MATRIX_SHARED_QUOTA_FREE_GIB"])
        except (KeyError, ValueError) as exc:
            raise ValueError("Site-confirmed LP_MATRIX_SHARED_QUOTA_FREE_GIB is required") from exc
        from .r03_resources import GIB
        if quota * GIB < estimate(args.source, slots=args.slots)["new_matrix_shared_floor_bytes"]:
            raise ValueError("Declared remaining shared quota is below matrix floor")
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        if actual != args.code_sha:
            raise ValueError("Submission code SHA differs from current checkout")
        prior = json.loads(args.manifest.read_text()) if args.manifest.is_file() else {}
        remaining = 12 - len(prior.get("job_ids", {}))
        existing_jobs = subprocess.check_output(["squeue", "-h", "-u",
            os.environ["USER"], "-o", "%i"], text=True).splitlines()
        if len(existing_jobs) + remaining > 15:
            raise ValueError("QoS GrpSubmitJobs=15 would be exceeded by this submission")
        manifest = submit(manifest, args.script, args.manifest)
    elif args.manifest:
        if args.manifest.exists():
            previous = json.loads(args.manifest.read_text())
            if previous.get("job_ids") or previous.get("status") in ("submitting", "submitted"):
                raise FileExistsError("Dry run cannot overwrite a submitted job manifest")
        atomic_json(args.manifest, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
