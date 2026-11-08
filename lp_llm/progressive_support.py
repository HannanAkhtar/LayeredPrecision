"""Read-only runtime check of the six pinned AutoRound control recipes."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from .quantizer_followups import PR_COMMIT
from .rrq_reference_eval import check_revision
from .rrq_runtime import source_environment
from .storage import atomic_json


def run(checkout: Path, output: Path, code_sha: str, *, python: str = sys.executable) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", code_sha):
        raise ValueError("Invalid pinned code SHA")
    check_revision(checkout, PR_COMMIT)
    if output.exists():
        existing = json.loads(output.read_text())
        if (existing.get("status") != "control_dispatch_verified" or
                existing.get("code_sha") != code_sha or
                existing.get("upstream_revision") != PR_COMMIT or
                existing.get("upstream_checkout") != str(checkout.resolve())):
            raise ValueError("Existing control support audit differs")
        return existing
    env = source_environment(checkout)
    process = subprocess.run([python, "-m", "lp_llm.progressive_control_worker", "--support-only"],
                             env=env, text=True, capture_output=True, check=False)
    if process.returncode:
        raise RuntimeError(f"Pinned AutoRound control support failed: {process.stderr[-2000:]}")
    recipes = json.loads(process.stdout)
    expected = {f"{branch}_w{bits}" for branch in ("s01", "r02") for bits in (4, 6, 8)}
    if (set(recipes) != expected or any(record.get("bits") not in (4, 6, 8) or
            record.get("group_size") != 128 for record in recipes.values())):
        raise ValueError("Control recipe inventory or group size differs")
    report = {"status": "control_dispatch_verified", "code_sha": code_sha,
              "upstream_revision": PR_COMMIT, "upstream_checkout": str(checkout.resolve()),
              "recipes": recipes,
              "scope": "runtime dispatch and configuration only; no full model quantization or memory proof"}
    atomic_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.checkout, args.output, args.code_sha), indent=2))


if __name__ == "__main__":
    main()
