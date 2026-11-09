"""Read-only AUS capacity floor for two LP branches and six QDQ controls."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

from .r03_resources import GIB, _headers
from .progressive_lp import ELIGIBLE_PARAMETERS, TOTAL_PARAMETERS


def quota_status() -> str:
    if os.environ.get("LP_MATRIX_ALLOW_UNKNOWN_QUOTA") == "1" and not os.environ.get("LP_MATRIX_SHARED_QUOTA_FREE_GIB"):
        return "unknown_operator_authorized"
    return "site_confirmed" if os.environ.get("LP_MATRIX_SHARED_QUOTA_FREE_GIB") else "unknown_unapproved"


def estimate(source: Path, *, slots: int = 2) -> dict:
    if slots not in (2, 3):
        raise ValueError("LP matrix GPU cap must be two or three")
    config, tensors = _headers(source)
    if (config.get("model_type") != "qwen3" or config.get("num_hidden_layers") != 36 or
            config.get("hidden_size") != 4096):
        raise ValueError("Not the pinned Qwen3-8B architecture")
    from .rrq_pr2308_compare import _eligible
    names = [name for name in tensors if _eligible(name)]
    if len(names) != 252:
        raise ValueError("Expected 252 source projections")
    projected = sum(math.prod(tensors[name]["shape"]) for name in names)
    total_elements = sum(math.prod(entry["shape"]) for entry in tensors.values())
    groups = sum(tensors[name]["shape"][0] * math.ceil(tensors[name]["shape"][1] / 128)
                 for name in names)
    dense = sum(math.prod(entry["shape"]) * 2 for entry in tensors.values())
    residual_code = (projected * 2 + 7) // 8
    residual_scales = groups * 4
    # 5% accommodates per-chunk Safetensors headers and JSON manifests; this is modeled.
    one_residual_floor = math.ceil((residual_code + residual_scales) * 1.05)
    controls = 6 * math.ceil(dense * 1.02)
    matrix = (2 * 3 * one_residual_floor + controls +
              slots * math.ceil(dense * 1.02) + 10 * GIB)
    point = one_residual_floor + math.ceil(dense * 1.05) + 5 * GIB
    return {"source_dense_payload_bytes": dense, "source_total_elements": total_elements,
            "projected_elements": projected,
            "projection_group_count": groups, "one_e2_code_bytes": residual_code,
            "one_e2_fp32_scale_bytes": residual_scales,
            "one_e2_modeled_floor_bytes": one_residual_floor,
            "six_monolithic_qdq_modeled_floor_bytes": controls,
            "concurrent_temporary_eval_checkpoint_floor_bytes": slots * math.ceil(dense * 1.02),
            "new_matrix_shared_floor_bytes": matrix,
            "one_point_remaining_shared_floor_bytes": point,
            "one_layer_pilot_shared_floor_bytes": max(5 * GIB,
                                                      math.ceil(dense * 1.05) + 5 * GIB),
            "note": "source and two repaired bases already exist; estimates exclude their existing bytes"}


def assess(source: Path, shared: Path, scratch: Path, *, mode: str = "matrix",
           slots: int = 2, quota_gib: float | None = None,
           gpu_free_bytes: int | None = None, host_limit_bytes: int | None = None,
           strict_source: bool = True) -> dict:
    sizes = estimate(source, slots=slots)
    if strict_source and (sizes["source_total_elements"] != TOTAL_PARAMETERS or
                          sizes["projected_elements"] != ELIGIBLE_PARAMETERS):
        raise ValueError("Source parameter inventory differs from pinned Qwen3-8B-Base")
    if mode not in ("pilot", "control_pilot", "matrix", "point") or not shared.is_dir() or not scratch.is_dir():
        raise ValueError("Invalid mode or missing real shared/scratch directory")
    floor = (sizes["one_layer_pilot_shared_floor_bytes"] if mode in ("pilot", "control_pilot") else
             sizes["one_point_remaining_shared_floor_bytes"] if mode == "point" else
             sizes["new_matrix_shared_floor_bytes"])
    if quota_gib is None:
        raw = os.environ.get("LP_MATRIX_SHARED_QUOTA_FREE_GIB", "")
        if raw:
            try:
                quota_gib = float(raw)
            except ValueError as exc:
                raise ValueError("Invalid declared shared quota") from exc
    quota_bytes = int(quota_gib * GIB) if quota_gib is not None and math.isfinite(quota_gib) and quota_gib > 0 else None
    if gpu_free_bytes is None:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free",
            "--format=csv,noheader,nounits"], text=True).strip().splitlines()
        if len(output) != 1:
            raise ValueError("Expected one Slurm-visible A10G")
        gpu_free_bytes = int(output[0]) * 1024**2
    if host_limit_bytes is None:
        host_limit_bytes = int(os.environ.get("SLURM_MEM_PER_NODE", "0")) * 1024**2
    shared_free = shutil.disk_usage(shared).free
    scratch_free = shutil.disk_usage(scratch).free
    allow_unknown = (os.environ.get("LP_MATRIX_ALLOW_UNKNOWN_QUOTA") == "1" and
                     not os.environ.get("LP_MATRIX_SHARED_QUOTA_FREE_GIB") and quota_bytes is None)
    checks = {"shared_filesystem_floor": shared_free >= floor,
              "site_confirmed_shared_quota": allow_unknown or (quota_bytes is not None and quota_bytes >= floor),
              "local_scratch_2_gib": scratch_free >= 2 * GIB,
              "host_allocation_12_gib": host_limit_bytes >= 12 * GIB,
              "a10g_free_20_gib": gpu_free_bytes >= 20 * GIB}
    if mode == "control_pilot":
        checks["control_host_allocation_14_gib"] = host_limit_bytes >= 14 * GIB
    return {"status": "capacity_floor_passed" if all(checks.values()) else "capacity_floor_failed",
            "mode": mode, "slots": slots, "sizes": sizes,
            "shared_path": str(shared), "shared_free_bytes": shared_free,
            "scratch_path": str(scratch), "scratch_free_bytes": scratch_free,
            "quota_free_bytes_declared": quota_bytes, "shared_floor_bytes": floor,
            "quota_status": "unknown_operator_authorized" if allow_unknown else
                            "site_confirmed" if quota_bytes is not None else "unknown_unapproved",
            "gpu_free_bytes": gpu_free_bytes, "host_limit_bytes": host_limit_bytes,
            "checks": checks,
            "scope": "read-only capacity gate; no measured full-model peak or verified site quota"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--mode", choices=("pilot", "control_pilot", "matrix", "point"), default="matrix")
    parser.add_argument("--slots", type=int, choices=(2, 3), default=2)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = assess(args.source, args.shared, args.scratch, mode=args.mode, slots=args.slots)
    print(json.dumps(report, indent=2))
    if args.strict and report["status"] != "capacity_floor_passed":
        raise SystemExit("LP matrix capacity floor failed; no quantization started")


if __name__ == "__main__":
    main()
