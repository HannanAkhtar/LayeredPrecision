"""Actual pinned AutoRound first-layer quantization, without full-model export."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path


def verify_unselected_projections(model, read_original, projections, *, layers: int = 36) -> int:
    import torch
    checked = 0
    for layer in range(1, layers):
        for path in projections:
            name = f"model.layers.{layer}.{path}.weight"
            value = model.get_submodule(f"model.layers.{layer}.{path}").weight.detach()
            if value.device.type == "meta":
                raise ValueError(f"Unselected pilot weight cannot be audited on meta device: {name}")
            original = read_original(name)
            if (value.shape != original.shape or not bool(torch.isfinite(value).all()) or
                    not torch.equal(value.cpu(), original.to(value.dtype))):
                raise ValueError(f"Unselected transformer projection changed: {name}")
            checked += 1
            del value, original
    return checked


def traversed_blocks(quant_block_list, *, layers: int = 36) -> list[str]:
    """Record AutoRound traversal; weight comparison determines quantization coverage."""
    if not isinstance(quant_block_list, (list, tuple)):
        raise ValueError("AutoRound block traversal is unavailable")
    names = []
    for group in quant_block_list:
        if not isinstance(group, (list, tuple)):
            raise ValueError("Malformed AutoRound block traversal")
        for name in group:
            if not isinstance(name, str) or not re.fullmatch(r"model\.layers\.[0-9]+", name):
                raise ValueError("Malformed AutoRound block traversal")
            names.append(name)
    if not names or "model.layers.0" not in names or len(names) != len(set(names)) or any(
            int(name.rsplit(".", 1)[1]) >= layers for name in names):
        raise ValueError("Unexpected AutoRound block traversal")
    return names


def slurm_cgroup_peak_bytes(job_id: str, *, proc: Path = Path("/proc/self/cgroup"),
                            root: Path = Path("/sys/fs/cgroup")) -> int:
    """Read the job's cgroup peak, including subprocesses and filesystem cache."""
    if not job_id or not proc.is_file():
        raise RuntimeError("Slurm cgroup memory peak is unavailable")
    for line in proc.read_text().splitlines():
        _, controllers, relative = line.split(":", 2)
        if not re.search(rf"(?:job_|job-){re.escape(job_id)}(?:/|$)", relative):
            continue
        if controllers == "":
            counter = root / relative.lstrip("/") / "memory.peak"
        elif "memory" in controllers.split(","):
            counter = root / "memory" / relative.lstrip("/") / "memory.max_usage_in_bytes"
        else:
            continue
        if counter.is_file():
            return int(counter.read_text().strip())
    raise RuntimeError("Slurm job cgroup memory peak is unavailable")


def run(branch: str, bits: int, source: Path, report: Path, code_sha: str) -> dict:
    import resource
    import torch
    import auto_round
    from auto_round import AutoRound

    from .progressive_control_worker import resolved_recipe
    from .progressive_lp import _files
    from .rrq_pr2308_compare import _get, _weight_map
    from .storage import atomic_json
    from .progressive_resources import quota_status

    if os.environ.get("LP_CONTROL_PILOT") != "1" or report.exists() or len(code_sha) != 40:
        raise ValueError("Control pilot requires a new report, guard and pinned code SHA")
    if branch == "r02":
        os.environ.update(AR_NEUQI_COARSE="256", AR_NEUQI_FINE="64", AR_NEUQI_BACKEND="auto")
    algorithm, recipe = resolved_recipe(branch, bits)
    started = time.monotonic()
    engine = AutoRound(model=str(source), scheme={"bits": bits, "group_size": 128,
        "sym": branch == "s01", "data_type": "int", "act_bits": 16},
        alg_configs=algorithm, format="fake", dataset="NeelNanda/pile-10k",
        nsamples=128, seqlen=2048, batch_size=8, seed=42, model_dtype="float16",
        scale_dtype="fp16", device_map="cuda:0", low_gpu_mem_usage=True,
        to_quant_block_names=r"^model\.layers\.0$")
    configured = [item for item in engine._alg_configs if isinstance(item, type(algorithm))]
    if (len(configured) != 1 or configured[0].bits != bits or
            engine.to_quant_block_names != r"^model\.layers\.0$"):
        raise ValueError("First-layer control selection or algorithm was replaced")
    model, _ = engine.quantize()
    blocks = traversed_blocks(getattr(engine, "quant_block_list", None))
    source_map = _weight_map(source)
    projections = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                   "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    measured = {}
    for path in projections:
        name = f"model.layers.0.{path}.weight"
        value = model.get_submodule(f"model.layers.0.{path}").weight.detach()
        if value.device.type == "meta":
            raise ValueError(f"Pilot projection remained on meta device: {name}")
        value = value.cpu()
        original = _get(source, source_map, name)
        if value.shape != original.shape or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Invalid control pilot projection: {name}")
        changed = int((value != original.to(value.dtype)).sum())
        if not changed:
            raise ValueError(f"Control pilot projection did not change: {name}")
        sample = value[0, :128]
        if torch.unique(sample).numel() > 1 << bits:
            raise ValueError(f"Control pilot W{bits} grid check failed: {name}")
        measured[name] = {"changed_elements": changed, "elements": value.numel(),
                          "sampled_group_levels": int(torch.unique(sample).numel())}
        del value, original
    unselected = verify_unselected_projections(
        model, lambda name: _get(source, source_map, name), projections)
    host_allocation = int(os.environ.get("SLURM_MEM_PER_NODE", "0")) * 1024**2
    cgroup_peak = slurm_cgroup_peak_bytes(os.environ.get("SLURM_JOB_ID", ""))
    process_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report_value = {"status": "control_layer_pilot_validated", "branch": branch, "bits": bits,
                    "quota_status": quota_status(),
                    "code_sha": code_sha, "reference_module": str(Path(auto_round.__file__).resolve()),
                    "source": _files(source),
                    "recipe": recipe, "source_layer": 0, "projection_count": len(measured),
                    "traversed_transformer_blocks": blocks,
                    "traversed_transformer_block_count": len(blocks),
                    "quantized_projection_names": sorted(measured),
                    "quantized_projection_count": len(measured),
                    "unselected_projection_source_exact_count": unselected,
                    "projections": measured, "runtime_seconds": time.monotonic() - started,
                    "process_peak_rss_kib": process_peak,
                    "slurm_cgroup_peak_bytes": cgroup_peak,
                    "slurm_cgroup_peak_source": "memory.peak or memory.max_usage_in_bytes",
                    "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                    "host_allocation_bytes": host_allocation,
                    "exported_checkpoint_bytes": 0,
                    "scope": "first real Qwen3 layer only; no full-model export"}
    if (host_allocation < 18 * 1024**3 or
            max(process_peak * 1024, cgroup_peak) >= 0.9 * host_allocation or
            report_value["gpu_peak_reserved_bytes"] >= 0.9 * report_value["gpu_total_bytes"]):
        raise RuntimeError("Control pilot lacks 10% resource headroom")
    atomic_json(report, report_value)
    return report_value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=("s01", "r02"), required=True)
    parser.add_argument("--bits", type=int, choices=(4, 6, 8), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.branch, args.bits, args.source,
                         args.report, args.code_sha), indent=2))


if __name__ == "__main__":
    main()
