"""Bounded CPU diagnosis of previously sampled fake-QDQ grid misses."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import torch
from safetensors import safe_open

from .progressive_storage_feasibility import (
    GROUP, _bits, _candidate_scales, _checkpoint, _weight_map, inventory, recover_group,
)
from .storage import atomic_json, sha256

PINNED_AUTOROUND = "7985c1b0f4eef65fb717511668fda7e347e584c3"
PINNED_SAMPLE_SHA = "5b8c3dee6e612027e98f3f0768a47ae6b5be6370"
LIMITS = (256, 1024, 4096)


def _interval_candidates(values: torch.Tensor, bits: int, branch: str,
                         *, limit: int = 4096) -> list[float]:
    """Heuristic rounding-boundary candidates; not a complete constraint solver."""
    if values.dtype != torch.float16:
        return []
    unique = torch.unique(values.float())
    selected = unique[torch.linspace(0, len(unique) - 1, min(len(unique), 8)).long()]
    max_code = (1 << (bits - 1)) if branch == "s01" else (1 << bits) - 1
    candidates = set()
    for v in selected.tolist():
        if v == 0:
            continue
        half = torch.tensor(v, dtype=torch.float16)
        lower = torch.nextafter(half, torch.tensor(float("-inf"), dtype=torch.float16))
        upper = torch.nextafter(half, torch.tensor(float("inf"), dtype=torch.float16))
        bounds = ((float(lower) + v) / 2, v, (v + float(upper)) / 2)
        for magnitude in range(1, max_code + 1):
            for code in (magnitude, -magnitude):
                for bound in bounds:
                    guess = torch.tensor(bound / code, dtype=torch.float32)
                    for scale in (guess, torch.nextafter(guess, torch.tensor(float("-inf"))),
                                  torch.nextafter(guess, torch.tensor(float("inf")))):
                        if torch.isfinite(scale) and scale != 0 and (branch == "s01" or scale > 0):
                            candidates.add(float(scale))
    ordered = sorted(candidates, key=lambda s: (abs(s), s))
    if len(ordered) <= limit:
        return ordered
    return [ordered[int(i * len(ordered) / limit)] for i in range(limit)]


def known_grid_oracle() -> list[dict]:
    results = []
    for branch in ("s01", "r02"):
        for bits in (2, 4, 6, 8):
            for scale in (0.03125, 0.333251953125, 1.0):
                for dtype in (torch.float16, torch.bfloat16):
                    # Quantizer output is FP16 before a possible BF16 export cast.
                    q = torch.tensor([-2, -1, 0, 1] if branch == "s01" else [0, 1, 2, 3],
                                     dtype=torch.float16)
                    zero = 0 if branch == "s01" else 2
                    stored = ((q - zero) * torch.tensor(scale, dtype=torch.float16)).to(torch.float16).to(dtype)
                    legacy = recover_group(stored, bits, branch, "fp16", search_limit=256)
                    pinned = recover_group(stored, bits, branch, "fp16", search_limit=256,
                                           pre_export_fp16=True)
                    results.append({"branch": branch, "bits": bits, "scale": scale,
                                    "stored_dtype": str(dtype), "legacy": legacy["outcome"],
                                    "pinned": pinned["outcome"]})
    return results


def _selected(sample: dict, model: str) -> list[dict]:
    info = sample["models"][model]
    group_file = Path(info["group_file"])
    if not group_file.resolve().is_relative_to(Path(sample["report_directory"]).resolve()):
        raise ValueError("Sample group file escapes original report directory")
    if sha256(group_file) != info["group_file_sha256"]:
        raise ValueError(f"Original sampled groups changed: {model}")
    failed, controls = [], []
    with group_file.open() as handle:
        for line in handle:
            entry = json.loads(line)
            if entry.get("model_id") != model or entry.get("checkpoint_record_sha256") != \
                    info["inventory"]["full_sha256"]:
                raise ValueError("Sample group provenance differs")
            if any(entry["candidates"][fmt]["outcome"] == "exact_constructive"
                   for fmt in ("fp16", "fp32")):
                if len(controls) < 1:
                    controls.append(entry)
            elif len(failed) < 2:
                failed.append(entry)
    if not failed or not controls:
        raise ValueError(f"Missing failure/control groups in {model}")
    return failed + controls


def _group(item: dict, entry: dict) -> torch.Tensor:
    name, row, group = entry["tensor"], entry["row"], entry["group"]
    info = item["tensors"][name]
    if info["shape"] != entry["shape"] or row < 0 or row >= info["shape"][0] or \
            group < 0 or group >= math.ceil(info["shape"][1] / GROUP):
        raise ValueError("Sample coordinate differs from checkpoint inventory")
    start = group * GROUP
    with safe_open(Path(item["checkpoint"]) / info["shard"], framework="pt", device="cpu") as handle:
        return handle.get_slice(name)[row:row + 1, start:min(start + GROUP, info["shape"][1])].reshape(-1)


def run(source: Path, root: Path, sample_path: Path, checkout: Path,
        output: Path, code_sha: str) -> dict:
    if output.exists() or len(code_sha) != 40:
        raise FileExistsError("Existing diagnostic output or invalid code SHA")
    revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                                       text=True).strip()
    if revision != PINNED_AUTOROUND:
        raise ValueError("AutoRound checkout differs from pinned quantizer revision")
    sample = json.loads(sample_path.read_text())
    if (sample.get("mode") != "sample" or sample.get("status") != "completed" or
            sample.get("code_sha") != sample_path.parent.name or
            sample.get("code_sha") != PINNED_SAMPLE_SHA):
        raise ValueError("Validated original sample report required")
    sample["report_directory"] = str(sample_path.parent.resolve())
    result = {"status": "completed", "code_sha": code_sha, "sample_sha256": sha256(sample_path),
              "upstream_revision": revision, "oracle": known_grid_oracle(), "models": {},
              "limitations": "Interval candidates are heuristic; misses remain inconclusive"}
    for branch in ("s01", "r02"):
        for bits in (4, 6, 8):
            model = f"{branch}_w{bits}"
            current = _checkpoint(root, branch, bits)
            previous = sample["models"][model]["inventory"]
            if (current["full_sha256"] != previous["full_sha256"] or
                    current["checkpoint"] != previous["checkpoint"]):
                raise ValueError(f"Checkpoint differs from original sample: {model}")
            item = inventory(source, root, branch, bits)
            entries = []
            for original in _selected(sample, model):
                values = _group(item, original)
                checks = {}
                for fmt in ("fp16", "fp32"):
                    checks[fmt] = {}
                    for limit in LIMITS:
                        for decoder, cast in (("legacy", False), ("pinned_fp16_qdq", True)):
                            started = time.perf_counter()
                            found = recover_group(values, bits, branch, fmt,
                                                  search_limit=limit, pre_export_fp16=cast)
                            checks[fmt][f"{decoder}_{limit}"] = {
                                "outcome": found["outcome"], "scale": found.get("scale"),
                                "zero_point": found.get("zero_point"),
                                "candidates_tested": found.get("candidates_tested", 0),
                                "cpu_seconds": time.perf_counter() - started}
                    interval = _interval_candidates(values, bits, branch)
                    if not interval:
                        checks[fmt]["interval_guided"] = {"outcome": "unknown",
                            "reason": "FP16 rounding interval prototype does not model BF16 stored values",
                            "candidates_tested": 0, "cpu_seconds": 0.0}
                    else:
                        started = time.perf_counter()
                        found = recover_group(values, bits, branch, fmt, search_limit=len(interval),
                                              candidate_scales=interval, pre_export_fp16=True)
                        checks[fmt]["interval_guided"] = {"outcome": found["outcome"],
                            "candidates_tested": found.get("candidates_tested", 0),
                            "cpu_seconds": time.perf_counter() - started,
                            "method": "FP16 rounding-boundary heuristic, not completeness proof"}
                entries.append({"coordinate": {key: original[key] for key in
                                ("tensor", "shape", "row", "group", "valid_elements")},
                                "original_outcomes": original["candidates"], "checks": checks})
            result["models"][model] = entries
    atomic_json(output, result)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ("source", "result-root", "sample", "checkout", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--code-sha", required=True)
    a = p.parse_args()
    result = run(a.source, a.result_root, a.sample, a.checkout, a.output, a.code_sha)
    print(json.dumps({"output": str(a.output), "models": list(result["models"])}))


if __name__ == "__main__":
    main()
