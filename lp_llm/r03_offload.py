"""Job-owned shared-storage lease for GPTQModel's temporary disk offload."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path


GIB = 1 << 30
DEFAULT_ROOT = Path("/shared/makhtar/lp_llm/tmp/r03_offload")


def quota_free_bytes() -> int:
    raw = os.environ.get("R03_SHARED_QUOTA_FREE_GIB", "")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("Set R03_SHARED_QUOTA_FREE_GIB from the site's available-quota report") from exc
    if not 0 < value < 1_000_000:
        raise ValueError("R03_SHARED_QUOTA_FREE_GIB must be a positive finite GiB value")
    return int(value * GIB)


def offload_root(path: Path | None = None, *, allowed_root: Path = Path("/shared/makhtar/lp_llm")) -> Path:
    root = path or Path(os.environ.get("R03_OFFLOAD_ROOT", str(DEFAULT_ROOT)))
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError(f"R03 offload root must be an existing real directory: {root}")
    resolved = root.resolve()
    if not resolved.is_relative_to(allowed_root.resolve()):
        raise ValueError(f"R03 offload root must remain under {allowed_root}")
    return resolved


class OffloadLease:
    def __init__(self, root: Path, directory: Path, token: str):
        self.root, self.directory, self.token = root, directory, token

    @classmethod
    def create(cls, root: Path, *, minimum_free_bytes: int = 40 * GIB) -> "OffloadLease":
        if root.is_symlink():
            raise ValueError("Offload root cannot be a symlink")
        root = root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("Offload root is not a real directory")
        if shutil.disk_usage(root).free < minimum_free_bytes or quota_free_bytes() < minimum_free_bytes:
            raise ValueError("Shared offload capacity or declared free quota is insufficient")
        job = os.environ.get("SLURM_JOB_ID", "manual")
        if not job.isalnum():
            raise ValueError("Unsafe Slurm job identifier")
        directory = Path(tempfile.mkdtemp(prefix=f"r03-{job}-", dir=root))
        token = uuid.uuid4().hex
        (directory / ".owner.json").write_text(json.dumps({"token": token,
            "job": job, "root": str(root), "pid": os.getpid()}))
        return cls(root, directory, token)

    def cleanup(self) -> None:
        if (self.directory.is_symlink() or self.directory.parent.resolve() != self.root or
                not self.directory.is_dir()):
            raise ValueError("Offload cleanup target is no longer the job-owned directory")
        marker = json.loads((self.directory / ".owner.json").read_text())
        if marker != {"token": self.token, "job": os.environ.get("SLURM_JOB_ID", "manual"),
                      "root": str(self.root), "pid": os.getpid()}:
            raise ValueError("Offload ownership marker changed; refusing cleanup")
        shutil.rmtree(self.directory)

    def __enter__(self) -> "OffloadLease":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()
