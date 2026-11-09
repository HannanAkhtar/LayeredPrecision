"""Actual pinned AutoRound first-layer quantization, without full-model export."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import json
import os
import re
import time
from pathlib import Path


@contextmanager
def source_view(source: Path, mapping: dict[str, str], name: str):
    """Keep one Safetensors shard mapped while comparing bounded row slices."""
    from safetensors import safe_open
    with safe_open(source / mapping[name], framework="pt", device="cpu") as handle:
        yield handle.get_slice(name)


def _shape(view) -> tuple[int, ...]:
    return tuple(view.get_shape() if hasattr(view, "get_shape") else view.shape)


def _compare_rows(value, original, name: str, *, rows: int = 64) -> int:
    import torch
    if value.device.type == "meta":
        raise ValueError(f"Pilot projection cannot be audited on meta device: {name}")
    if _shape(original) != tuple(value.shape):
        raise ValueError(f"Pilot projection shape differs from source: {name}")
    changed = 0
    for first in range(0, value.shape[0], rows):
        stop = min(first + rows, value.shape[0])
        loaded = value[first:stop].detach().cpu()
        source_chunk = original[first:stop].to(loaded.dtype)
        if not bool(torch.isfinite(loaded).all()):
            raise ValueError(f"Nonfinite pilot projection: {name}")
        changed += int(torch.count_nonzero(loaded != source_chunk))
        del loaded, source_chunk
    return changed


def verify_unselected_projections(model, read_original, projections, *, layers: int = 36) -> int:
    checked = 0
    for layer in range(1, layers):
        for path in projections:
            name = f"model.layers.{layer}.{path}.weight"
            value = model.get_submodule(f"model.layers.{layer}.{path}").weight
            with read_original(name) as original:
                if _compare_rows(value, original, name):
                    raise ValueError(f"Unselected transformer projection changed: {name}")
            checked += 1
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


def slurm_cgroup_peak(job_id: str, *, proc: Path = Path("/proc/self/cgroup"),
                      root: Path = Path("/sys/fs/cgroup")) -> tuple[int, str]:
    """Read the job's cgroup peak, including subprocesses and filesystem cache."""
    if not job_id or not proc.is_file():
        raise RuntimeError("Slurm cgroup memory peak is unavailable")
    for line in proc.read_text().splitlines():
        _, controllers, relative = line.split(":", 2)
        if not re.search(rf"(?:job_|job-){re.escape(job_id)}(?:/|$)", relative):
            continue
        if controllers == "":
            counter = root / relative.lstrip("/") / "memory.peak"
            source = "cgroup_v2_memory.peak"
        elif "memory" in controllers.split(","):
            counter = root / "memory" / relative.lstrip("/") / "memory.max_usage_in_bytes"
            source = "cgroup_v1_memory.max_usage_in_bytes"
        else:
            continue
        if counter.is_file():
            return int(counter.read_text().strip()), source
    raise RuntimeError("Slurm job cgroup memory peak is unavailable")


def host_memory_assessment(process_peak_kib: int, cgroup_peak_bytes: int,
                           allocation_bytes: int) -> dict:
    if min(process_peak_kib, cgroup_peak_bytes, allocation_bytes) <= 0:
        raise ValueError("Missing host memory measurement or allocation")
    peak = max(process_peak_kib * 1024, cgroup_peak_bytes)
    return {"host_peak_used_for_headroom_bytes": peak,
            "host_memory_headroom_bytes": allocation_bytes - peak,
            "host_memory_headroom_fraction": (allocation_bytes - peak) / allocation_bytes,
            "host_memory_10pct_headroom_passed": peak * 10 < allocation_bytes * 9}


def run(branch: str, bits: int, source: Path, report: Path, code_sha: str) -> dict:
    import resource
    import torch
    import auto_round
    from auto_round import AutoRound
    from auto_round.utils.device import memory_monitor

    from .progressive_control_worker import resolved_recipe
    from .progressive_lp import _files
    from .rrq_pr2308_compare import _weight_map
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
    model, quantization_info = engine.quantize()
    autoround_peak_ram_gib = float(memory_monitor.peak_ram)
    quantization_process_peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    quantization_cgroup_peak_bytes, _ = slurm_cgroup_peak(os.environ.get("SLURM_JOB_ID", ""))
    blocks = traversed_blocks(getattr(engine, "quant_block_list", None))
    # The returned model remains live; release calibration and engine state before validation.
    del engine, quantization_info
    gc.collect()
    source_map = _weight_map(source)
    projections = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                   "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    measured = {}
    for path in projections:
        name = f"model.layers.0.{path}.weight"
        value = model.get_submodule(f"model.layers.0.{path}").weight
        with source_view(source, source_map, name) as original:
            changed = _compare_rows(value, original, name)
        if not changed:
            raise ValueError(f"Control pilot projection did not change: {name}")
        sample = value[0, :128].detach().cpu()
        levels = int(torch.unique(sample).numel())
        if levels > 1 << bits:
            raise ValueError(f"Control pilot W{bits} grid check failed: {name}")
        measured[name] = {"changed_elements": changed, "elements": value.numel(),
                          "sampled_group_levels": levels}
    unselected = verify_unselected_projections(
        model, lambda name: source_view(source, source_map, name), projections)
    source_identity = _files(source)
    host_allocation = int(os.environ.get("SLURM_MEM_PER_NODE", "0")) * 1024**2
    cgroup_peak, cgroup_source = slurm_cgroup_peak(os.environ.get("SLURM_JOB_ID", ""))
    process_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    memory = host_memory_assessment(process_peak, cgroup_peak, host_allocation)
    report_value = {"status": "control_layer_pilot_validated", "branch": branch, "bits": bits,
                    "quota_status": quota_status(),
                    "code_sha": code_sha, "reference_module": str(Path(auto_round.__file__).resolve()),
                    "source": source_identity,
                    "recipe": recipe, "source_layer": 0, "projection_count": len(measured),
                    "traversed_transformer_blocks": blocks,
                    "traversed_transformer_block_count": len(blocks),
                    "quantized_projection_names": sorted(measured),
                    "quantized_projection_count": len(measured),
                    "unselected_projection_source_exact_count": unselected,
                    "projections": measured, "runtime_seconds": time.monotonic() - started,
                    "process_peak_rss_kib": process_peak,
                    "after_quantization_process_peak_rss_kib": quantization_process_peak_kib,
                    "after_quantization_cgroup_peak_bytes": quantization_cgroup_peak_bytes,
                    "slurm_cgroup_peak_bytes": cgroup_peak,
                    "slurm_cgroup_peak_source": cgroup_source,
                    "autoround_peak_ram": {"value": autoround_peak_ram_gib, "unit": "GiB",
                        "meaning": "sampled sum of process-tree RSS; upstream log labels GB; not used for resource gate"},
                    **memory,
                    "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                    "host_allocation_bytes": host_allocation,
                    "exported_checkpoint_bytes": 0,
                    "scope": "first real Qwen3 layer only; no full-model export"}
    if (host_allocation < 14 * 1024**3 or
            not memory["host_memory_10pct_headroom_passed"] or
            report_value["gpu_peak_reserved_bytes"] >= 0.9 * report_value["gpu_total_bytes"]):
        raise RuntimeError(f"Control pilot lacks 10% resource headroom: "
                           f"allocation={host_allocation}, process_rss_peak={process_peak * 1024}, "
                           f"cgroup_peak={cgroup_peak}, headroom={memory['host_memory_headroom_bytes']}")
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
