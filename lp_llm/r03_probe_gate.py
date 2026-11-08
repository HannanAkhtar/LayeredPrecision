"""Read-only attestation check before a full R03 allocation can start."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .storage import sha256


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_shard_hashes(source: Path) -> dict[str, str]:
    mapping = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Source Safetensors index has no tensor map")
    shards = set(mapping.values())
    if (not all(isinstance(name, str) and Path(name).name == name and
                name.endswith(".safetensors") for name in shards) or
            {path.name for path in source.glob("*.safetensors")} != shards):
        raise ValueError("Source Safetensors shard inventory differs")
    return {name: sha256(source / name) for name in sorted(shards)}


def verify(report_path: Path, source: Path, code_sha: str) -> dict:
    if report_path.parent.name != "r03_feasibility":
        raise ValueError("R03 probe report must be outside the full result directory")
    record = json.loads(report_path.read_text())
    expected = {"status": "representative_layer_validated", "code_sha": code_sha,
                "source_layer": 0, "projection_count": 7,
                "backend": "TORCH",
                "quantization_config": {"bits": 2, "group_size": 128, "sym": True,
                    "desc_act": False, "act_group_aware": True, "damp_percent": 0.05,
                    "damp_auto_increment": 0.01, "static_groups": False,
                    "true_sequential": True},
                "largest_projection": "model.layers.0.mlp.down_proj",
                "calibration_sample_count": 128,
                "source_config_sha256": _sha(source / "config.json"),
                "source_index_sha256": _sha(source / "model.safetensors.index.json"),
                "source_shard_sha256": source_shard_hashes(source)}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Representative-layer report identity or coverage differs")
    token_hash = record.get("calibration_token_sha256", "")
    if len(token_hash) != 64 or any(c not in "0123456789abcdef" for c in token_hash):
        raise ValueError("Representative calibration fingerprint missing")
    for peak, limit in (("gpu_peak_reserved_bytes", "gpu_total_bytes"),
                        ("process_peak_rss_kib", "host_allocation_bytes")):
        used = record.get(peak)
        capacity = record.get(limit)
        if peak.endswith("rss_kib") and isinstance(used, int):
            used *= 1024
        if not isinstance(used, int) or not isinstance(capacity, int) or not 0 < used < capacity * 0.9:
            raise ValueError(f"Representative resource headroom is insufficient: {peak}")
    if "cgroup_peak_bytes" in record and record["cgroup_peak_bytes"] >= record["host_allocation_bytes"] * 0.9:
        raise ValueError("Representative cgroup memory headroom is insufficient")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    args = parser.parse_args()
    value = verify(args.report, args.source, args.code_sha)
    print(json.dumps({"status": "r03_probe_identity_verified", "report": str(args.report),
                      "sampled_peak_shared_temporary_bytes": value.get("sampled_peak_shared_temporary_bytes")}, indent=2))


if __name__ == "__main__":
    main()
