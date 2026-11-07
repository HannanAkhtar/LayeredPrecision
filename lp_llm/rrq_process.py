"""Stream upstream logs and sample GPU memory during a child process."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path


def host_rss_mb(pid: int) -> int | None:
    """Sample Linux RSS of a child and its current descendants, when available."""
    root = Path("/proc")
    if not root.is_dir():
        return None
    pending, seen, total = [pid], set(), 0
    while pending:
        child = pending.pop()
        if child in seen:
            continue
        seen.add(child)
        try:
            for line in (root / str(child) / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1])
                    break
            children = root / str(child) / "task" / str(child) / "children"
            pending.extend(int(value) for value in children.read_text().split())
        except (OSError, ValueError):
            continue
    return total // 1024


def gpu_used_mb() -> int | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=5)
        return max(int(line.strip()) for line in output.splitlines() if line.strip())
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return None


def run_logged(command: list[str], log: Path, *, cwd: Path | None = None,
               env: dict[str, str] | None = None) -> dict:
    log.parent.mkdir(parents=True, exist_ok=True)
    begun = time.monotonic()
    peak = None
    host_peak = None
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=cwd, env=env, text=True, stdout=handle,
                                   stderr=subprocess.STDOUT)
        while process.poll() is None:
            used = gpu_used_mb()
            if used is not None:
                peak = max(used, peak or 0)
            host_used = host_rss_mb(process.pid)
            if host_used is not None:
                host_peak = max(host_used, host_peak or 0)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        return {"returncode": process.returncode, "runtime_seconds": time.monotonic() - begun,
                "peak_gpu_memory_used_mb": peak,
                "gpu_measurement": "nvidia-smi total used memory on busiest GPU; includes other processes",
                "sampled_peak_process_tree_rss_mb": host_peak,
                "host_measurement": "sum of parent and live child VmRSS sampled every five seconds; misses short peaks"}
