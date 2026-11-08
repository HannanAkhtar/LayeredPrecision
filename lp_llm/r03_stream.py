"""Bounded, independently checked dense QDQ export from a packed R03 model."""

from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path

import torch
from safetensors.torch import save_file

from .rrq_pr2308_compare import _eligible, _weight_map
from .storage import atomic_json, sha256


def _decoded_projection(module, name: str) -> torch.Tensor:
    from .quantizer_followups_worker import decode_gptq_w2_packed

    independent = decode_gptq_w2_packed(module)
    official = module.dequantize_weight().T.detach().to(device="cpu", dtype=torch.float16).contiguous()
    if not torch.equal(independent, official):
        raise ValueError(f"Independent packed decode differs from official Torch decoder: {name}")
    return official


def stream_gptq_qdq(packed_model, source: Path, target: Path, *, expected_count: int = 252) -> dict:
    """Write one source-named tensor per shard; leave partial output on failure."""
    from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear

    partial = target.with_name(target.name + ".partial")
    if target.exists() or partial.exists():
        raise FileExistsError(f"Existing QDQ output or partial export requires inspection: {target}")
    source_names = sorted(_weight_map(source))
    expected = {name for name in source_names if _eligible(name)}
    modules = {name: module for name, module in packed_model.model.named_modules()
               if isinstance(module, TorchQuantLinear)}
    if len(expected) != expected_count or {name + ".weight" for name in modules} != expected:
        raise ValueError("Packed projection inventory differs from source")
    parameters = dict(packed_model.model.named_parameters())
    buffers = dict(packed_model.model.named_buffers())
    partial.mkdir(parents=True)
    config = copy.deepcopy(packed_model.model.config)
    if hasattr(config, "quantization_config"):
        del config.quantization_config
    config.torch_dtype = torch.float16
    config.save_pretrained(partial)
    packed_model.tokenizer.save_pretrained(partial)
    generation = getattr(packed_model.model, "generation_config", None)
    if generation is not None:
        generation.save_pretrained(partial)
    mapping, parity_hashes, shard_hashes = {}, {}, {}
    max_tensor_bytes = total_bytes = 0
    for index, name in enumerate(source_names, 1):
        if name in expected:
            module = modules[name[:-7]]
            value = _decoded_projection(module, name)
            parity_hashes[name] = hashlib.sha256(value.numpy().tobytes()).hexdigest()
        else:
            original = parameters.get(name, buffers.get(name))
            if original is None or original.device.type == "meta":
                raise ValueError(f"Excluded source tensor is unavailable after packed reload: {name}")
            value = original.detach().to(device="cpu").contiguous()
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite QDQ tensor: {name}")
        filename = f"model-{index:05d}-of-{len(source_names):05d}.safetensors"
        save_file({name: value}, partial / filename, metadata={"format": "pt"})
        mapping[name] = filename
        shard_hashes[filename] = sha256(partial / filename)
        tensor_bytes = value.numel() * value.element_size()
        total_bytes += tensor_bytes
        max_tensor_bytes = max(max_tensor_bytes, tensor_bytes)
        del value
    atomic_json(partial / "model.safetensors.index.json", {
        "metadata": {"total_size": total_bytes}, "weight_map": mapping})
    report = {"format": "dense_fp16_qdq_streamed", "tensor_count": len(mapping),
              "quantized_projection_count": len(parity_hashes),
              "projection_sha256": parity_hashes, "shard_sha256": shard_hashes,
              "max_single_tensor_bytes": max_tensor_bytes,
              "total_tensor_bytes": total_bytes}
    atomic_json(partial / "stream_integrity.json", report)
    os.replace(partial, target)
    return report
