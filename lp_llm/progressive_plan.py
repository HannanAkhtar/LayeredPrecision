"""Dry-run first, explicitly gated Slurm plan for twelve LP-LLM operating points."""

from __future__ import annotations

import argparse
import io
import tarfile
from unittest.mock import patch
import json
import os
import math
import re
import shutil
import subprocess
from pathlib import Path

from .progressive_lp import BRANCHES, _files, identity, validate_repaired_base, verified_layers
from .progressive_resources import estimate, quota_status
from .quantizer_followups import PR_COMMIT
from .storage import atomic_json


LP_PILOT_PREVIOUS_SHA = "b650efb085b00b801f0f42543485c84ae9fbbbc0"
LP_UNCHANGED_PATHS = ("lp_llm/progressive_lp.py", "lp_llm/progressive_run.py",
                      "lp_llm/progressive_pilot.py", "lp_llm/progressive_base_parity.py",
                      "lp_llm/quantize.py", "lp_llm/storage.py")


def verify_executed_code(code_repo: Path, code_sha: str, staged_root: Path) -> None:
    code_repo, staged_root = code_repo.resolve(), staged_root.resolve()
    if not (code_repo / ".git").exists() or not staged_root.is_dir():
        raise ValueError("Explicit trusted Git checkout and staged code directory are required")
    kind = subprocess.check_output(["git", "-C", str(code_repo), "cat-file", "-t", code_sha],
                                   text=True).strip()
    if kind != "commit":
        raise ValueError("Requested code SHA is not a commit in the trusted checkout")
    archive = subprocess.check_output(["git", "-C", str(code_repo), "archive",
                                       "--format=tar", code_sha, "lp_llm"])
    tracked = set()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as stream:
        for member in stream:
            if not member.isfile() or not member.name.endswith(".py"):
                continue
            tracked.add(member.name)
            expected = stream.extractfile(member).read()
            path = staged_root / member.name
            if not path.is_file() or path.read_bytes() != expected:
                raise ValueError(f"Executed code differs from requested commit: {member.name}")
    actual = {path.relative_to(staged_root).as_posix() for path in
              (staged_root / "lp_llm").rglob("*.py")}
    if not tracked or actual != tracked:
        raise ValueError("Staged Python module inventory differs from requested commit")


def accepted_lp_pilot_sha(record_sha: str, code_sha: str,
                          code_repo: Path, staged_root: Path) -> str:
    verify_executed_code(code_repo, code_sha, staged_root)
    if record_sha == code_sha:
        return code_sha
    if record_sha != LP_PILOT_PREVIOUS_SHA:
        raise ValueError("LP pilot code SHA is not the validated prior commit")
    command = ["git", "-C", str(code_repo.resolve()), "diff", "--quiet",
               LP_PILOT_PREVIOUS_SHA, code_sha, "--", *LP_UNCHANGED_PATHS]
    if subprocess.run(command, check=False).returncode != 0:
        raise ValueError("LP implementation differs from the validated pilot commit")
    return LP_PILOT_PREVIOUS_SHA


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


def verify_pilots(source: Path, result_root: Path, code_sha: str, code_repo: Path) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", code_sha):
        raise ValueError("Invalid reviewed code SHA")
    reports = {}
    for branch, repair in BRANCHES.items():
        base = result_root / "quantizer_followups" / repair / "export"
        root = result_root / "progressive_llm/pilots" / branch
        record = json.loads((root / "report.json").read_text())
        validate_repaired_base(branch, source, base)
        ident = identity(branch, source, base)
        accepted_sha = accepted_lp_pilot_sha(record.get("code_sha"), code_sha,
                                            code_repo, Path(__file__).resolve().parents[1])
        if (record.get("status") != "pilot_validated" or record.get("branch") != branch or
                record.get("code_sha") != accepted_sha or record.get("identity") != ident or
                len(record.get("projection_names", [])) != 7 or
                record.get("base_checkpoint_source_exact_exclusions") is not True or
                record.get("base_zero_enhancement_exact") is not True or
                record.get("process_peak_rss_kib", 0) * 1024 >=
                record.get("host_allocation_bytes", 0) * 0.9 or
                record.get("host_allocation_bytes", 0) <= 0):
            raise ValueError(f"Unreviewed or invalid {branch} full-size layer pilot")
        with patch.dict(os.environ, {"FOLLOWUP_CODE_SHA": accepted_sha}):
            layers = verified_layers(root, 2, ident, pilot=True)
        if (any(layer.get("code_sha") != accepted_sha or
                layer.get("encoded_elements", 0) <= 0 or
                not math.isfinite(layer.get("mse_fp32", float("nan"))) or
                not math.isfinite(layer.get("normalized_mse_fp32", float("nan")))
                for layer in layers) or
                any(any(record.get(key, {}).get(field) != layers[index][field]
                        for field in ("projection_count", "encoded_elements", "mse_fp32",
                                      "normalized_mse_fp32", "code_bytes", "metadata_bytes"))
                    for index, key in enumerate(("first", "second")))):
            raise ValueError("Prior LP pilot numerical report differs from verified layers")
        parity_path = root / "base_parity/base_parity.json"
        parity = json.loads(parity_path.read_text())
        if (parity.get("status") != "base_parity_validated" or
                (root / "base_parity/checkpoint").exists() or
                (root / "base_parity/checkpoint.partial").exists() or
                parity.get("identity") != ident or parity.get("code_sha") != accepted_sha or
                parity.get("corpus", {}).get("token_sha256") !=
                "8ffcee84dd4b684ec631cb52845e18c9c181ce6fdb8b26d65a4ffdc3d35fbda3" or
                parity.get("corpus", {}).get("scored_tokens") != 298862 or
                round(parity.get("evaluation", {}).get("perplexity", -1), 2) !=
                {"s01": 12.47, "r02": 32.22}[branch]):
            raise ValueError(f"Independent {branch} W2 perplexity parity is unverified")
        reports[branch] = {"accepted_pilot_code_sha": accepted_sha,
                           "cross_commit_acceptance": accepted_sha != code_sha,
                           "source_base_identity": ident,
                           "process_peak_rss_kib": record["process_peak_rss_kib"],
                           "artifact_bytes": record["artifact_bytes"],
                           "report": str(root / "report.json"),
                           "base_parity_report": str(parity_path),
                           "base_parity_perplexity": parity["evaluation"]["perplexity"]}
    support_path = result_root / "progressive_llm/pilots" / f"control_support_{code_sha}.json"
    support = json.loads(support_path.read_text())
    if (support.get("status") != "control_dispatch_verified" or
            support.get("code_sha") != code_sha or
            support.get("upstream_revision") != PR_COMMIT or
            set(support.get("recipes", {})) !=
            {f"{branch}_w{bits}" for branch in BRANCHES for bits in (4, 6, 8)}):
        raise ValueError("Six pinned monolithic-control dispatches are not verified")
    reports["control_support"] = str(support_path)
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
                    not isinstance(control.get("traversed_transformer_blocks"), list) or
                    "model.layers.0" not in control["traversed_transformer_blocks"] or
                    len(control["traversed_transformer_blocks"]) !=
                    len(set(control["traversed_transformer_blocks"])) or
                    any(not re.fullmatch(r"model\.layers\.(?:[0-9]|[12][0-9]|3[0-5])", block)
                        for block in control["traversed_transformer_blocks"]) or
                    control.get("traversed_transformer_block_count") !=
                    len(control["traversed_transformer_blocks"]) or
                    control.get("quantized_projection_count") != 7 or
                    control.get("quantized_projection_names") !=
                    sorted(control.get("projections", {})) or
                    control.get("unselected_projection_source_exact_count") != 245 or
                    set(control.get("projections", {})) != {
                        f"model.layers.0.{part}.weight" for part in
                        ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                         "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
                    } or
                    any(item.get("changed_elements", 0) <= 0 or
                        item.get("elements", 0) < item.get("changed_elements", 0) or
                        not 1 <= item.get("sampled_group_levels", 0) <= 1 << bits
                        for item in control.get("projections", {}).values()) or
                    control.get("gpu_peak_reserved_bytes", 0) <= 0 or
                    control.get("process_peak_rss_kib", 0) <= 0 or
                    max(control.get("process_peak_rss_kib", 0) * 1024,
                        control.get("slurm_cgroup_peak_bytes", 0)) >=
                    control.get("host_allocation_bytes", 0) * 0.9 or
                    control.get("slurm_cgroup_peak_bytes", 0) <= 0 or
                    control.get("slurm_cgroup_peak_source") not in (
                        "cgroup_v2_memory.peak", "cgroup_v1_memory.max_usage_in_bytes") or
                    control.get("after_quantization_cgroup_peak_bytes", 0) <= 0 or
                    control.get("after_quantization_cgroup_peak_bytes", 0) >
                    control.get("slurm_cgroup_peak_bytes", 0) or
                    control.get("after_quantization_process_peak_rss_kib", 0) <= 0 or
                    control.get("after_quantization_process_peak_rss_kib", 0) >
                    control.get("process_peak_rss_kib", 0) or
                    control.get("host_allocation_bytes", 0) < 14 * 1024**3 or
                    control.get("host_peak_used_for_headroom_bytes") != max(
                        control.get("process_peak_rss_kib", 0) * 1024,
                        control.get("slurm_cgroup_peak_bytes", 0)) or
                    control.get("host_memory_headroom_bytes") != (
                        control.get("host_allocation_bytes", 0) -
                        control.get("host_peak_used_for_headroom_bytes", 0)) or
                    control.get("host_memory_10pct_headroom_passed") is not True or
                    control.get("gpu_peak_reserved_bytes", 0) >=
                    control.get("gpu_total_bytes", 0) * 0.9 or
                    control.get("host_allocation_bytes", 0) <= 0 or
                    control.get("gpu_total_bytes", 0) <= 0):
                raise ValueError(f"Unreviewed or invalid {name} control layer pilot")
            reports[name] = {"process_peak_rss_kib": control["process_peak_rss_kib"],
                             "gpu_peak_reserved_bytes": control["gpu_peak_reserved_bytes"],
                             "report": str(path)}
    if any(reports[branch]["cross_commit_acceptance"] for branch in BRANCHES):
        acceptance = {"status": "prior_lp_pilots_validated", "current_code_sha": code_sha,
            "prior_code_sha": LP_PILOT_PREVIOUS_SHA,
            "lp_implementation_paths_verified_unchanged": list(LP_UNCHANGED_PATHS),
            "accepted": {branch: reports[branch] for branch in BRANCHES}}
        path = result_root / "progressive_llm/pilots" / f"lp_cross_commit_acceptance_{code_sha}.json"
        if path.is_file() and json.loads(path.read_text()) != acceptance:
            raise ValueError("Existing LP cross-commit acceptance report differs")
        atomic_json(path, acceptance)
        reports["cross_commit_acceptance_report"] = str(path)
    return reports


def plan(root: Path, code_sha: str, *, slots: int = 2) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", code_sha):
        raise ValueError("Invalid pinned code SHA")
    return {"status": "dry_run_not_submitted", "code_sha": code_sha,
            "quota_status": quota_status(),
            "max_concurrent_gpu_jobs": slots, "qos_submit_jobs_count": 12,
            "point_count": 12, "points": points(root, slots),
            "dependency_rule": "afterok scientific LP predecessor; afterany scheduling-only lane predecessor"}


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
        scientific = {job_ids[predecessor]} if predecessor else set()
        scheduling = set()
        lane = point["lane"]
        if lane in lane_last:
            if lane_last[lane] not in scientific:
                scheduling.add(lane_last[lane])
        if point["id"] not in job_ids:
            command = ["sbatch", "--parsable",
                "--export=ALL,LP_MATRIX_ID=" + point["id"] +
                ",FOLLOWUP_CODE_SHA=" + manifest["code_sha"] +
                ",LP_MATRIX_ALLOW=1,LP_MATRIX_SLOTS=" + str(manifest["max_concurrent_gpu_jobs"])]
            dependencies = []
            if scientific:
                dependencies.append("afterok:" + ":".join(sorted(scientific)))
            if scheduling:
                dependencies.append("afterany:" + ":".join(sorted(scheduling)))
            if dependencies:
                command.append("--dependency=" + ",".join(dependencies))
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


def admit_submission(source: Path, result_root: Path, slots: int) -> dict:
    from .r03_resources import GIB
    floor = estimate(source, slots=slots)["new_matrix_shared_floor_bytes"]
    free = shutil.disk_usage(result_root).free
    if free < floor:
        raise ValueError("Actual shared filesystem free space is below matrix floor")
    quota = os.environ.get("LP_MATRIX_SHARED_QUOTA_FREE_GIB")
    if quota:
        try:
            quota_bytes = float(quota) * GIB
        except ValueError as exc:
            raise ValueError("Invalid declared shared quota") from exc
        if not math.isfinite(quota_bytes) or quota_bytes < floor:
            raise ValueError("Declared remaining shared quota is below matrix floor")
    elif os.environ.get("LP_MATRIX_ALLOW_UNKNOWN_QUOTA") != "1":
        raise ValueError("Site-confirmed quota or LP_MATRIX_ALLOW_UNKNOWN_QUOTA=1 is required")
    return {"quota_status": "site_confirmed" if quota else "unknown_operator_authorized",
            "declared_quota_bytes": quota_bytes if quota else None,
            "actual_filesystem_free_bytes": free, "matrix_floor_bytes": floor,
            "slots": slots}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("dry-run", "verify-pilots", "submit"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--code-repo", type=Path,
                        help="Trusted original Git checkout, required for pilot verification")
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--slots", type=int, choices=(2, 3), default=2)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--script", type=Path,
                        default=Path("hpc/progressive_llm/run_point.sbatch"))
    args = parser.parse_args()
    if args.action == "verify-pilots":
        if args.source is None or args.code_repo is None:
            parser.error("verify-pilots requires --source and --code-repo")
        print(json.dumps(verify_pilots(args.source, args.result_root, args.code_sha,
                                       args.code_repo), indent=2))
        return
    manifest = plan(args.result_root, args.code_sha, slots=args.slots)
    if args.action == "submit":
        if args.source is None or args.manifest is None or args.code_repo is None:
            parser.error("submit requires --source, --manifest and --code-repo")
        verify_pilots(args.source, args.result_root, args.code_sha, args.code_repo)
        manifest["admission"] = admit_submission(args.source, args.result_root, args.slots)
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
