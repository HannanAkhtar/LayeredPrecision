"""Deterministic Qwen3 checkpoint tensor-coverage inventory for Phase 1."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .model_adapter import SelectionPolicy, expected_qwen3, inventory
from .storage import atomic_json, sha256


def audit(source: Path) -> dict:
    source = Path(source)
    config, mapping = inventory(source)
    shapes = expected_qwen3(config)
    policy = SelectionPolicy()
    quantized, excluded = [], []
    by_projection, by_block = {}, {}
    for name in sorted(mapping):
        numel = 1
        for dimension in shapes[name]:
            numel *= dimension
        item = {"name": name, "shape": list(shapes[name]), "parameters": numel,
                "source_dtype": "bfloat16", "shard": mapping[name]}
        if policy.eligible(name):
            quantized.append(item)
            match = re.fullmatch(r"model\.layers\.(\d+)\.(?:self_attn|mlp)\.([a-z_]+)\.weight", name)
            block, projection = match.groups()
            by_projection[projection] = by_projection.get(projection, 0) + numel
            by_block[block] = by_block.get(block, 0) + numel
        else:
            excluded.append(item)
    quantized_count = sum(item["parameters"] for item in quantized)
    excluded_count = sum(item["parameters"] for item in excluded)
    return {"model_id": "Qwen/Qwen3-8B-Base", "revision": "49e3418fbbbca6ecbdf9608b4d22e5a407081db4",
            "source": str(source.resolve()),
            "config_sha256": sha256(source / "config.json"),
            "index_sha256": sha256(source / "model.safetensors.index.json"),
            "source_shard_sha256": {shard: sha256(source / shard) for shard in sorted(set(mapping.values()))},
            "quantized_tensors": quantized, "excluded_tensors": excluded,
            "parameters_by_projection": dict(sorted(by_projection.items())),
            "parameters_by_block": dict(sorted(by_block.items(), key=lambda item: int(item[0]))),
            "quantized_parameters": quantized_count, "excluded_parameters": excluded_count,
            "total_parameters": quantized_count + excluded_count,
            "matches_established_qwen3_8b_coverage": (quantized_count == 6_945_767_424
                                                    and excluded_count == 1_244_967_936)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = audit(args.source)
    atomic_json(args.output, value)
    print(json.dumps({key: value[key] for key in
                      ("quantized_parameters", "excluded_parameters", "matches_established_qwen3_8b_coverage")},
                     indent=2))


if __name__ == "__main__":
    main()
