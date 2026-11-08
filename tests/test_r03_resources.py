"""The R03 capacity check reads headers and never loads Qwen3-8B weights."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from lp_llm.r03_resources import GIB, PROJECTIONS, assess, estimate


def source_fixture(root: Path, *, omit: str | None = None) -> Path:
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"model_type": "qwen3",
        "num_hidden_layers": 36, "hidden_size": 4096,
        "intermediate_size": 12288, "vocab_size": 151936}))
    tensors = {f"model.layers.{layer}.{projection}.weight":
               torch.zeros(16, 128, dtype=torch.bfloat16)
               for layer in range(36) for projection in PROJECTIONS}
    tensors["lm_head.weight"] = torch.zeros(16, 128, dtype=torch.bfloat16)
    if omit:
        del tensors[omit]
    save_file(tensors, root / "model.safetensors")
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "model.safetensors" for name in tensors}}))
    return root


def test_r03_capacity_counts_codes_scales_zeros_and_dense_exclusions(tmp_path, monkeypatch):
    source = source_fixture(tmp_path / "source")
    sizes = estimate(source)
    assert sizes["projection_count"] == 252
    assert sizes["projected_elements"] == 252 * 16 * 128
    assert sizes["packed_code_bytes"] == 252 * 8 * 16 * 4
    assert sizes["packed_scale_bytes"] == 252 * 16 * 2
    assert sizes["packed_zero_bytes"] == 252 * 4
    assert sizes["packed_group_index_bytes"] == 252 * 128 * 4
    assert sizes["excluded_dense_bytes"] == 16 * 128 * 2
    assert sizes["packed_tensor_estimate_bytes"] == sum(sizes[key] for key in (
        "packed_code_bytes", "packed_scale_bytes", "packed_zero_bytes",
        "packed_group_index_bytes", "excluded_dense_bytes"))
    monkeypatch.setattr("lp_llm.r03_resources.shutil.disk_usage",
                        lambda _: SimpleNamespace(free=100 * GIB))
    gpu = {"name": "A10G", "free_bytes": 22 * GIB, "total_bytes": 23 * GIB}
    result = assess(source, tmp_path, tmp_path, tmp_path, gpu=gpu,
                    memory_limit_bytes=12 * GIB, quota_bytes=100 * GIB)
    assert result["status"] == "capacity_floor_passed"
    assert all(result["checks"].values())


def test_r03_capacity_rejects_missing_projection_and_low_space(tmp_path, monkeypatch):
    missing = "model.layers.35.mlp.down_proj.weight"
    with pytest.raises(ValueError, match="Missing 1 expected"):
        estimate(source_fixture(tmp_path / "source", omit=missing))
    source = source_fixture(tmp_path / "complete")
    monkeypatch.setattr("lp_llm.r03_resources.shutil.disk_usage",
                        lambda _: SimpleNamespace(free=16 * GIB))
    gpu = {"name": "A10G", "free_bytes": 19 * GIB, "total_bytes": 23 * GIB}
    result = assess(source, tmp_path, tmp_path, tmp_path, gpu=gpu,
                    memory_limit_bytes=8 * GIB, quota_bytes=16 * GIB)
    assert result["status"] == "capacity_floor_failed"
    assert not result["checks"]["filesystem_capacity"]
    assert not result["checks"]["gpu_free_at_least_20_gib"]
    assert not result["checks"]["host_allocation_for_streaming"]
    assert not result["checks"]["declared_shared_quota"]


def test_r03_read_only_preflight_reports_missing_quota(tmp_path, monkeypatch):
    source = source_fixture(tmp_path / "source")
    monkeypatch.delenv("R03_SHARED_QUOTA_FREE_GIB", raising=False)
    monkeypatch.setattr("lp_llm.r03_resources.shutil.disk_usage",
                        lambda _: SimpleNamespace(free=100 * GIB))
    result = assess(source, tmp_path, tmp_path, tmp_path,
                    gpu={"name": "A10G", "free_bytes": 22 * GIB},
                    memory_limit_bytes=12 * GIB)
    assert not result["checks"]["declared_shared_quota"]
    assert result["declared_shared_quota_free_bytes"] is None
    assert "R03_SHARED_QUOTA_FREE_GIB" in result["quota_note"]


def test_r03_full_job_requires_explicit_feasibility_gate():
    script = (Path(__file__).resolve().parents[1] /
              "hpc/quantizer_followups/run_experiment.sbatch").read_text()
    assert 'R03_FULL_FEASIBILITY_OK:-0' in script
    assert 'R03_PROBE_REPORT:?' in script
    assert 'lp_llm.r03_resources' in script
    assert '--scratch "$CODE_STAGE"' in script and '--strict' in script
    assert '--offload-root "${R03_OFFLOAD_ROOT:-$LP_ROOT/tmp/r03_offload}"' in script
