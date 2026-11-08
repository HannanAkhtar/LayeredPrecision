"""Actual pinned AutoRound first-layer quantization, without full-model export."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def run(branch: str, bits: int, source: Path, report: Path, code_sha: str) -> dict:
    import resource
    import torch
    import auto_round
    from auto_round import AutoRound

    from .progressive_control_worker import resolved_recipe
    from .progressive_lp import _files
    from .rrq_pr2308_compare import _get, _weight_map
    from .storage import atomic_json

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
        to_quant_block_names=["model.layers.0"])
    configured = [item for item in engine._alg_configs if isinstance(item, type(algorithm))]
    if (len(configured) != 1 or configured[0].bits != bits or
            engine.to_quant_block_names != ["model.layers.0"]):
        raise ValueError("First-layer control selection or algorithm was replaced")
    model, _ = engine.quantize()
    selected_blocks = getattr(engine, "quant_block_list", None)
    if (selected_blocks is None or len(selected_blocks) != 1 or
            len(selected_blocks[0]) != 1 or selected_blocks[0][0] != "model.layers.0"):
        raise ValueError("Control pilot selected more than one decoder layer")
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
    report_value = {"status": "control_layer_pilot_validated", "branch": branch, "bits": bits,
                    "code_sha": code_sha, "reference_module": str(Path(auto_round.__file__).resolve()),
                    "source": _files(source),
                    "recipe": recipe, "source_layer": 0, "projection_count": len(measured),
                    "projections": measured, "runtime_seconds": time.monotonic() - started,
                    "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                    "host_allocation_bytes": int(os.environ.get("SLURM_MEM_PER_NODE", "0")) * 1024**2,
                    "exported_checkpoint_bytes": 0,
                    "scope": "first real Qwen3 layer only; no full-model export"}
    if (report_value["host_allocation_bytes"] and
            report_value["process_peak_rss_kib"] * 1024 >= 0.9 * report_value["host_allocation_bytes"] or
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
