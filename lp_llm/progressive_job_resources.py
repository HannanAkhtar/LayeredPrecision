"""Job-level memory evidence for full LP and AutoRound matrix points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .progressive_control_pilot import slurm_cgroup_peak
from .storage import atomic_json


def record(output: Path, time_dir: Path, job_id: str, allocation_mib: int,
           exit_code: int, point: str) -> dict:
    if output.exists() or allocation_mib <= 0 or not job_id.isdigit():
        raise ValueError("Resource record already exists or Slurm allocation is invalid")
    stages = {}
    for path in sorted(time_dir.glob("*.rss_kib")):
        stages[path.stem] = int(path.read_text().strip())
    peak, source = slurm_cgroup_peak(job_id)
    allocation = allocation_mib * 1024**2
    process_peak = max(stages.values(), default=0) * 1024
    used = max(process_peak, peak)
    result = {"status": "measured", "point": point, "job_id": job_id,
              "exit_code_before_resource_check": exit_code,
              "stage_process_peak_rss_kib": stages,
              "process_peak_rss_bytes": process_peak,
              "slurm_cgroup_peak_bytes": peak, "slurm_cgroup_peak_source": source,
              "host_allocation_bytes": allocation,
              "host_memory_headroom_bytes": allocation - used,
              "host_memory_10pct_headroom_passed": used * 10 < allocation * 9}
    atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-dir", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--allocation-mib", type=int, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--point", required=True)
    args = parser.parse_args()
    result = record(args.output, args.time_dir, args.job_id,
                    args.allocation_mib, args.exit_code, args.point)
    print(json.dumps(result, indent=2))
    if not result["host_memory_10pct_headroom_passed"]:
        raise SystemExit("Full-model job lacks 10% host-memory headroom")


if __name__ == "__main__":
    main()
