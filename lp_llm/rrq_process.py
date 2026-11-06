"""Stream upstream logs and sample GPU memory during a child process."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path


def gpu_used_mb() -> int | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=5)
        return max(int(line.strip()) for line in output.splitlines() if line.strip())
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return None


def run_logged(command: list[str], log: Path, *, cwd: Path | None = None) -> dict:
    log.parent.mkdir(parents=True, exist_ok=True)
    begun = time.monotonic()
    peak = None
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=cwd, text=True, stdout=handle,
                                   stderr=subprocess.STDOUT)
        while process.poll() is None:
            used = gpu_used_mb()
            if used is not None:
                peak = max(used, peak or 0)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        return {"returncode": process.returncode, "runtime_seconds": time.monotonic() - begun,
                "peak_gpu_memory_used_mb": peak,
                "gpu_measurement": "nvidia-smi total used memory on busiest GPU; includes other processes"}
