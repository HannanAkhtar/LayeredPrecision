"""Lightweight process and numerical diagnostics."""

from __future__ import annotations

import os
import time


def peak_rss_bytes() -> int | None:
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value * (1024 if os.name != "darwin" else 1))
    except ImportError:
        return None


def started() -> float:
    return time.monotonic()


def elapsed(begin: float) -> float:
    return time.monotonic() - begin
