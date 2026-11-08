"""Read-only capacity assessment for the pinned Qwen3-8B R03 workflow.

Reads Safetensors headers, filesystem capacity and nvidia-smi. It never loads
weights, downloads data, creates result directories or starts quantization.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import subprocess
from pathlib import Path


GIB = 1 << 30
PROJECTIONS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
               "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def _existing(path: Path) -> Path:
    path = path.resolve()
    while not path.exists():
        if path == path.parent:
            raise ValueError(f"No existing filesystem ancestor: {path}")
        path = path.parent
    return path


def _headers(source: Path) -> tuple[dict, dict]:
    config = json.loads((source / "config.json").read_text())
    index = json.loads((source / "model.safetensors.index.json").read_text())
    mapping = index.get("weight_map")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Source Safetensors index has no tensor map")
    tensors = {}
    for filename in sorted(set(mapping.values())):
        if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".safetensors"):
            raise ValueError(f"Unsafe source shard name: {filename}")
        path = source / filename
        with path.open("rb") as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError(f"Truncated Safetensors shard: {filename}")
            length = struct.unpack("<Q", raw)[0]
            if length > 100_000_000 or length + 8 > path.stat().st_size:
                raise ValueError(f"Invalid Safetensors header: {filename}")
            header = json.loads(stream.read(length))
        header.pop("__metadata__", None)
        if set(header) != {name for name, shard in mapping.items() if shard == filename}:
            raise ValueError(f"Source shard tensor inventory differs: {filename}")
        payload_bytes = path.stat().st_size - 8 - length
        for name, entry in header.items():
            start, end = entry["data_offsets"]
            if not 0 <= start < end <= payload_bytes:
                raise ValueError(f"Invalid source tensor offsets: {name}")
        tensors.update(header)
    if set(tensors) != set(mapping):
        raise ValueError("Source tensor inventory differs from index")
    if {path.name for path in source.glob("*.safetensors")} != set(mapping.values()):
        raise ValueError("Source has missing or unindexed Safetensors shards")
    return config, tensors


def estimate(source: Path) -> dict:
    config, tensors = _headers(source)
    if (config.get("model_type") != "qwen3" or config.get("num_hidden_layers") != 36 or
            config.get("hidden_size") != 4096 or config.get("intermediate_size") != 12288 or
            config.get("vocab_size") != 151936):
        raise ValueError("Source is not the expected Qwen3-8B architecture")
    expected = {f"model.layers.{layer}.{projection}.weight"
                for layer in range(36) for projection in PROJECTIONS}
    if not expected <= set(tensors):
        raise ValueError(f"Missing {len(expected - set(tensors))} expected R03 projections")
    code_bytes = scale_bytes = zero_bytes = group_index_bytes = projected_elements = 0
    dense_bytes = excluded_bytes = 0
    for name, entry in tensors.items():
        shape, dtype = entry["shape"], entry["dtype"]
        if dtype not in ("F16", "BF16"):
            raise ValueError(f"Unsupported source dtype: {name}: {dtype}")
        elements = math.prod(shape)
        dense_bytes += elements * 2
        if name not in expected:
            excluded_bytes += elements * 2
            continue
        if len(shape) != 2:
            raise ValueError(f"Projection is not a matrix: {name}")
        out_features, in_features = shape
        if in_features % 128 or in_features % 16 or out_features % 16:
            raise ValueError(f"Unsupported packed projection dimensions: {name}: {shape}")
        projected_elements += elements
        groups = in_features // 128
        code_bytes += in_features // 16 * out_features * 4
        scale_bytes += groups * out_features * 2
        zero_bytes += groups * (out_features // 16) * 4
        group_index_bytes += in_features * 4
    packed_estimate = excluded_bytes + code_bytes + scale_bytes + zero_bytes + group_index_bytes
    source_shard_bytes = sum(path.stat().st_size for path in source.glob("*.safetensors"))
    return {"projection_count": len(expected), "projected_elements": projected_elements,
            "source_tensor_count": len(tensors),
            "source_shard_bytes": source_shard_bytes, "dense_tensor_bytes": dense_bytes,
            "excluded_dense_bytes": excluded_bytes, "packed_code_bytes": code_bytes,
            "packed_scale_bytes": scale_bytes, "packed_zero_bytes": zero_bytes,
            "packed_group_index_bytes": group_index_bytes,
            "packed_tensor_estimate_bytes": packed_estimate,
            "calibration_activation_single_copy_bytes": 128 * 2048 * 4096 * 2,
            "largest_fp32_hessian_single_copy_bytes": 12288 ** 2 * 4}


def _gpu() -> dict:
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            raise ValueError(f"Unexpected nvidia-smi row: {line}")
        rows.append({"index": fields[0], "uuid": fields[1], "name": fields[2],
                     "total_bytes": int(fields[3]) * 1024**2,
                     "free_bytes": int(fields[4]) * 1024**2})
    if not rows:
        raise ValueError("nvidia-smi reported no GPUs")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0]
    selected = next((row for row in rows if visible and visible in (row["index"], row["uuid"])), None)
    if selected is None and len(rows) == 1:
        selected = rows[0]
    if selected is None:
        raise ValueError("Cannot identify the Slurm-allocated GPU from CUDA_VISIBLE_DEVICES")
    return selected


def _host_memory() -> dict:
    result = {}
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                result["node_mem_available_bytes"] = int(line.split()[1]) * 1024
                break
    limit, current = Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")
    if limit.is_file() and current.is_file():
        raw = limit.read_text().strip()
        if raw != "max":
            result["cgroup_headroom_bytes"] = max(0, int(raw) - int(current.read_text().strip()))
    return result


def assess(source: Path, result_root: Path, scratch: Path, *, gpu: dict | None = None,
           memory_limit_bytes: int | None = None) -> dict:
    sizes = estimate(source)
    scratch_base, result_base = _existing(scratch), _existing(result_root)
    scratch_free = shutil.disk_usage(scratch_base).free
    result_free = shutil.disk_usage(result_base).free
    same_filesystem = scratch_base.stat().st_dev == result_base.stat().st_dev
    # Capacity floors include a duplicate offload copy and room for checkpoint
    # serialization. They are conservative gates, not predicted peak usage.
    scratch_floor = max(40 * GIB, 2 * sizes["dense_tensor_bytes"] +
                        sizes["packed_tensor_estimate_bytes"] + 4 * GIB)
    result_floor = sizes["packed_tensor_estimate_bytes"] + sizes["dense_tensor_bytes"] + 4 * GIB
    # The current QDQ materializer allocates all 252 FP16 projection Linear
    # weights on CPU before save_pretrained. This alone exceeds 12 GiB.
    host_floor = max(20 * GIB, 2 * sizes["projected_elements"] + 4 * GIB)
    gpu = _gpu() if gpu is None else gpu
    memory_limit_bytes = memory_limit_bytes if memory_limit_bytes is not None else int(
        os.environ.get("SLURM_MEM_PER_NODE", "0")) * 1024**2
    checks = {"gpu_is_a10g": "A10G" in gpu["name"],
              "gpu_free_at_least_20_gib": gpu["free_bytes"] >= 20 * GIB,
              "host_allocation_for_dense_qdq": memory_limit_bytes >= host_floor,
              "scratch_and_output_capacity": (scratch_free >= scratch_floor + result_floor
                    if same_filesystem else scratch_free >= scratch_floor and result_free >= result_floor)}
    return {"status": "capacity_floor_passed" if all(checks.values()) else "capacity_floor_failed",
            "scope": "read-only capacity floors; not a peak-memory prediction or full-run approval",
            "source": str(source.resolve()), "sizes": sizes, "gpu": gpu,
            "host_allocation_bytes": memory_limit_bytes, "host_floor_bytes": host_floor,
            "host_observed": _host_memory(),
            "scratch": {"path": str(scratch_base), "free_bytes": scratch_free,
                        "floor_bytes": scratch_floor},
            "output": {"path": str(result_base), "free_bytes": result_free,
                       "floor_bytes": result_floor},
            "same_filesystem": same_filesystem, "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = assess(args.source, args.result_root, args.scratch)
    print(json.dumps(report, indent=2))
    if args.strict and report["status"] != "capacity_floor_passed":
        raise SystemExit("R03 resource capacity floor failed; no quantization started")


if __name__ == "__main__":
    main()
