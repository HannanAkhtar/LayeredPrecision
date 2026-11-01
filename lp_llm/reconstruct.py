"""Reconstruct one LP prefix as a sharded, Transformers-loadable checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
import torch

from . import FORMAT_VERSION
from .diagnostics import elapsed, peak_rss_bytes, started
from .storage import atomic_json, atomic_safetensors, decode_prefix, sha256


def reconstruct(artifact: Path, output: Path, depth: int) -> dict:
    artifact, output = Path(artifact), Path(output)
    if artifact.resolve() == output.resolve():
        raise ValueError("Output must differ from LP artifact")
    complete = json.loads((artifact / "COMPLETE.json").read_text())
    if complete["manifest_sha256"] != sha256(artifact / "manifest.json"):
        raise ValueError("LP manifest integrity check failed")
    manifest = json.loads((artifact / "manifest.json").read_text())
    if manifest["format"] != FORMAT_VERSION or depth < 1 or depth > len(manifest["bits"]):
        raise ValueError("Unsupported format or prefix depth")
    output.mkdir(parents=True, exist_ok=True)
    identity = {"manifest_sha256": complete["manifest_sha256"], "depth": depth}
    run = output / "reconstruction.json"
    if run.exists():
        if json.loads(run.read_text())["identity"] != identity:
            raise ValueError("Output belongs to another reconstruction")
    elif any(output.iterdir()):
        raise ValueError("Nonempty reconstruction directory has no identity")
    else:
        atomic_json(run, {"identity": identity})
    for sidecar in manifest["sidecars"]:
        shutil.copyfile(artifact / sidecar, output / Path(sidecar).name)
    mapping = {}
    begun = started()
    for record in manifest["tensors"]:
        name = record["name"]
        filename = f"model-{record['tensor_id'] + 1:05d}-of-{len(manifest['tensors']):05d}.safetensors"
        target = output / filename
        marker = output / f".done-{record['tensor_id']:04d}.json"
        if marker.exists() and target.exists():
            saved = json.loads(marker.read_text())
            if saved.get("sha256") == sha256(target) and saved.get("name") == name:
                mapping[name] = filename
                continue
        if record["eligible"]:
            tensor = decode_prefix(artifact, record, depth).to(dtype=torch.bfloat16)
        else:
            shared = artifact / record["shared"]
            if sha256(shared) != record["shared_sha256"]:
                raise ValueError(f"Corrupt shared tensor: {name}")
            with safe_open(shared, framework="pt", device="cpu") as handle:
                tensor = handle.get_tensor(name)
        atomic_safetensors(target, {name: tensor})
        atomic_json(marker, {"name": name, "sha256": sha256(target)})
        mapping[name] = filename
        del tensor
        print(f"[{len(mapping)}/{len(manifest['tensors'])}] {name}", flush=True)
    index = {"metadata": {"total_size": sum(r["numel"] * 2 for r in manifest["tensors"])},
             "weight_map": mapping}
    atomic_json(output / "model.safetensors.index.json", index)
    atomic_json(output / "COMPLETE.json", {"identity": identity,
                "index_sha256": sha256(output / "model.safetensors.index.json"),
                "seconds": elapsed(begun), "peak_rss_bytes": peak_rss_bytes()})
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth", type=int, required=True)
    args = parser.parse_args()
    reconstruct(args.artifact, args.output, args.depth)


if __name__ == "__main__":
    main()
