"""One-layer Qwen3-8B resource probe; never writes the full R03 result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
from pathlib import Path


def retain_first_layer(model) -> None:
    """Limit both GPTQModel shell and turtle to one real 8B decoder layer."""
    import torch

    for branch in (model.model, model.turtle_model):
        if branch is None:
            raise RuntimeError("Representative probe requires GPTQModel's offloaded turtle")
        layers = branch.model.layers
        if len(layers) != 36:
            raise RuntimeError(f"Expected 36 Qwen3 layers before probe, got {len(layers)}")
        branch.model.layers = torch.nn.ModuleList([layers[0]])
        branch.config.num_hidden_layers = 1
        if len(branch.model.layers) != 1:
            raise RuntimeError("Probe layer limit failed")


def _tree_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except FileNotFoundError:
            continue  # GPTQModel can replace an offload file during sampling.
    return total


class DiskPeak:
    def __init__(self, path: Path):
        self.path, self.peak = path, 0
        self.error = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(2):
            try:
                self.peak = max(self.peak, _tree_bytes(self.path))
            except OSError as exc:
                self.error = exc
                return

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join()
        if self.error is not None:
            raise RuntimeError("Temporary-storage monitor failed") from self.error
        self.peak = max(self.peak, _tree_bytes(self.path))


def _guard(torch) -> None:
    total = torch.cuda.get_device_properties(0).total_memory
    if torch.cuda.memory_reserved(0) >= total * 0.9:
        raise RuntimeError("Probe reached 90% GPU reserved-memory abort threshold")
    current, limit = Path("/sys/fs/cgroup/memory.current"), Path("/sys/fs/cgroup/memory.max")
    if current.is_file() and limit.is_file() and limit.read_text().strip() != "max":
        if int(current.read_text()) >= int(limit.read_text()) * 0.9:
            raise RuntimeError("Probe reached 90% cgroup host-memory abort threshold")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    args = parser.parse_args()
    if (args.report.parent.name != "r03_feasibility" or args.report.exists() or
            (args.report.parent / (args.report.name + ".tmp")).exists()):
        raise RuntimeError("Probe report must be a new file under r03_feasibility")
    if len(args.code_sha) != 40 or any(c not in "0123456789abcdef" for c in args.code_sha):
        raise RuntimeError("Invalid pinned probe code SHA")
    if os.environ.get("R03_LAYER_PROBE") != "1":
        raise RuntimeError("R03_LAYER_PROBE=1 is required for the separate one-layer test")
    os.environ["GPTQ_TORCH_TRITON_DEQUANT"] = "0"
    import resource
    import torch
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
    from lp_llm.quantizer_followups_worker import (decode_gptq_w2_packed,
        make_r03_quantize_config, prepare_r03_calibration, require_loaded_gptq_quantizer)
    from lp_llm.r03_offload import OffloadLease, offload_root
    from lp_llm.r03_probe_gate import source_shard_hashes

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Probe requires exactly one visible CUDA GPU")
    tokenizer, calibration, fingerprint = prepare_r03_calibration(args.source)
    qcfg = make_r03_quantize_config()
    if qcfg.offload_to_disk is not True:
        raise RuntimeError("Pinned GPTQModel disk offloading is unavailable")
    _guard(torch)
    with OffloadLease.create(offload_root()) as lease, DiskPeak(lease.directory) as disk_peak:
        qcfg.offload_to_disk_path = str(lease.directory / "weights")
        model = GPTQModel.load(str(args.source), qcfg, device="cuda:0", dtype=torch.float16)
        require_loaded_gptq_quantizer(model, qcfg)
        if not qcfg.offload_to_disk or qcfg.offload_to_disk_path != str(lease.directory / "weights"):
            raise RuntimeError("GPTQModel changed the pinned shared offload configuration")
        retain_first_layer(model)
        _guard(torch)
        model.quantize(calibration, batch_size=1, backend=BACKEND.TORCH,
                       calibration_sort=None, calibration_data_min_length=2048)
        names = {name for name, module in model.model.named_modules()
                 if isinstance(module, TorchQuantLinear)}
        if len(names) != 7 or "model.layers.0.mlp.down_proj" not in names:
            raise RuntimeError("Probe did not pack exactly the seven first-layer projections")
        _guard(torch)
        packed_path = lease.directory / "probe_packed"
        model.save(str(packed_path))
        del model
        torch.cuda.empty_cache()
        reloaded = GPTQModel.load(str(packed_path), backend=BACKEND.TORCH)
        modules = {name: module for name, module in reloaded.model.named_modules()
                   if isinstance(module, TorchQuantLinear)}
        if set(modules) != names:
            raise RuntimeError("Probe packed reload changed projection coverage")
        for name, module in modules.items():
            independent = decode_gptq_w2_packed(module)
            official = module.dequantize_weight().T.detach().to(device="cpu", dtype=torch.float16)
            if not torch.equal(independent, official):
                raise RuntimeError(f"Probe packed decoder parity failed: {name}")
            del independent, official
        _guard(torch)
        result = {"status": "representative_layer_validated", "source_layer": 0,
                  "code_sha": args.code_sha,
                  "backend": "TORCH",
                  "quantization_config": {key: getattr(qcfg, key) for key in (
                      "bits", "group_size", "sym", "desc_act", "act_group_aware",
                      "damp_percent", "damp_auto_increment", "static_groups",
                      "true_sequential")},
                  "source_config_sha256": hashlib.sha256((args.source / "config.json").read_bytes()).hexdigest(),
                  "source_index_sha256": hashlib.sha256((args.source / "model.safetensors.index.json").read_bytes()).hexdigest(),
                  "source_shard_sha256": source_shard_hashes(args.source),
                  "projection_count": len(modules), "largest_projection": "model.layers.0.mlp.down_proj",
                  "calibration_sample_count": len(calibration), "calibration_token_sha256": fingerprint,
                  "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                  "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
                  "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                  "host_allocation_bytes": int(os.environ.get("SLURM_MEM_PER_NODE", "0")) * 1024**2,
                  "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "shared_temporary_bytes_at_finish": _tree_bytes(lease.directory),
                  "packed_checkpoint_bytes": _tree_bytes(packed_path),
                  "offload_cleanup": "job-owned temporary directory removed on normal exit"}
        cgroup_peak = Path("/sys/fs/cgroup/memory.peak")
        if cgroup_peak.is_file():
            result["cgroup_peak_bytes"] = int(cgroup_peak.read_text().strip())
    result["sampled_peak_shared_temporary_bytes"] = disk_peak.peak
    from lp_llm.storage import atomic_json
    atomic_json(args.report, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
