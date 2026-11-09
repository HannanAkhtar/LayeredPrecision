import json
from types import SimpleNamespace

import torch
import pytest
from safetensors.torch import save_file

from lp_llm.progressive_resources import GIB, assess, estimate
from lp_llm.progressive_aggregate import plots, write


def _source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    names = [f"model.layers.{layer}.{part}.weight" for layer in range(36)
             for part in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                          "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]
    tensors = {name: torch.zeros(16, 128, dtype=torch.bfloat16) for name in names}
    tensors["lm_head.weight"] = torch.zeros(16, 128, dtype=torch.bfloat16)
    save_file(tensors, source / "model.safetensors")
    (source / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "model.safetensors" for name in tensors}}))
    (source / "config.json").write_text(json.dumps({"model_type": "qwen3",
        "num_hidden_layers": 36, "hidden_size": 4096}))
    return source


def test_matrix_estimate_counts_codes_scales_controls_and_concurrent_qdq(tmp_path):
    source = _source(tmp_path)
    two = estimate(source, slots=2)
    three = estimate(source, slots=3)
    assert two["projected_elements"] == 252 * 16 * 128
    assert two["projection_group_count"] == 252 * 16
    assert two["one_e2_code_bytes"] == 252 * 16 * 128 // 4
    assert two["one_e2_fp32_scale_bytes"] == 252 * 16 * 4
    assert three["new_matrix_shared_floor_bytes"] > two["new_matrix_shared_floor_bytes"]
    assert two["new_matrix_shared_floor_bytes"] > two["one_point_remaining_shared_floor_bytes"]


def test_matrix_preflight_fails_unknown_quota_even_when_df_reports_space(tmp_path, monkeypatch):
    source = _source(tmp_path)
    with pytest.raises(ValueError, match="parameter inventory"):
        assess(source, tmp_path, tmp_path, mode="pilot", quota_gib=20,
               gpu_free_bytes=22 * GIB, host_limit_bytes=12 * GIB)
    monkeypatch.delenv("LP_MATRIX_SHARED_QUOTA_FREE_GIB", raising=False)
    monkeypatch.setattr("lp_llm.progressive_resources.shutil.disk_usage",
                        lambda _: SimpleNamespace(free=1000 * GIB))
    report = assess(source, tmp_path, tmp_path, mode="pilot", quota_gib=None,
                    gpu_free_bytes=22 * GIB, host_limit_bytes=12 * GIB,
                    strict_source=False)
    assert report["status"] == "capacity_floor_failed"
    assert not report["checks"]["site_confirmed_shared_quota"]
    passing = assess(source, tmp_path, tmp_path, mode="pilot", quota_gib=20,
                     gpu_free_bytes=22 * GIB, host_limit_bytes=12 * GIB,
                     strict_source=False)
    assert passing["status"] == "capacity_floor_passed"
    point = assess(source, tmp_path, tmp_path, mode="point", quota_gib=20,
                   gpu_free_bytes=22 * GIB, host_limit_bytes=12 * GIB,
                   strict_source=False)
    assert point["status"] == "capacity_floor_passed"
    assert point["shared_floor_bytes"] < assess(
        source, tmp_path, tmp_path, mode="matrix", quota_gib=20,
        gpu_free_bytes=22 * GIB, host_limit_bytes=12 * GIB,
        strict_source=False)["shared_floor_bytes"]


def test_unknown_quota_override_keeps_every_other_capacity_gate(tmp_path, monkeypatch):
    source = _source(tmp_path)
    monkeypatch.delenv("LP_MATRIX_SHARED_QUOTA_FREE_GIB", raising=False)
    monkeypatch.setenv("LP_MATRIX_ALLOW_UNKNOWN_QUOTA", "1")
    monkeypatch.setattr("lp_llm.progressive_resources.shutil.disk_usage",
                        lambda _: SimpleNamespace(free=1000 * GIB))
    kwargs = dict(mode="pilot", quota_gib=None, gpu_free_bytes=22 * GIB,
                  host_limit_bytes=12 * GIB, strict_source=False)
    accepted = assess(source, tmp_path, tmp_path, **kwargs)
    assert accepted["status"] == "capacity_floor_passed"
    assert accepted["quota_status"] == "unknown_operator_authorized"
    assert accepted["quota_free_bytes_declared"] is None
    assert assess(source, tmp_path, tmp_path, **dict(kwargs, gpu_free_bytes=1))["status"] == "capacity_floor_failed"
    assert assess(source, tmp_path, tmp_path, **dict(kwargs, host_limit_bytes=1))["status"] == "capacity_floor_failed"
    monkeypatch.setattr("lp_llm.progressive_resources.shutil.disk_usage",
                        lambda _: SimpleNamespace(free=1))
    low_disk = assess(source, tmp_path, tmp_path, **kwargs)
    assert not low_disk["checks"]["shared_filesystem_floor"]
    assert not low_disk["checks"]["local_scratch_2_gib"]


def test_control_pilot_requires_14_gib_without_raising_lp_or_matrix_floor(tmp_path, monkeypatch):
    source = _source(tmp_path)
    monkeypatch.setenv("LP_MATRIX_ALLOW_UNKNOWN_QUOTA", "1")
    monkeypatch.delenv("LP_MATRIX_SHARED_QUOTA_FREE_GIB", raising=False)
    monkeypatch.setattr("lp_llm.progressive_resources.shutil.disk_usage",
                        lambda _: SimpleNamespace(free=1000 * GIB))
    kwargs = dict(quota_gib=None, gpu_free_bytes=22 * GIB, strict_source=False)
    short = assess(source, tmp_path, tmp_path, mode="control_pilot",
                   host_limit_bytes=12 * GIB, **kwargs)
    assert not short["checks"]["control_host_allocation_14_gib"]
    assert short["status"] == "capacity_floor_failed"
    assert assess(source, tmp_path, tmp_path, mode="pilot",
                  host_limit_bytes=12 * GIB, **kwargs)["status"] == "capacity_floor_passed"
    ready = assess(source, tmp_path, tmp_path, mode="control_pilot",
                   host_limit_bytes=14 * GIB, **kwargs)
    assert ready["status"] == "capacity_floor_passed"


def test_aggregate_reports_all_pending_points_without_invented_measurements(tmp_path):
    source = _source(tmp_path)
    report = write(source, tmp_path, tmp_path / "aggregate")
    assert len(report["rows"]) == 17  # five established references plus twelve new points
    pending = [row for row in report["rows"] if row["kind"] in ("lp", "control")]
    assert len(pending) == 12 and all(row["perplexity"] is None for row in pending)
    assert all(row["status"] == "pending" for row in pending)
    assert all(row["storage_basis"].startswith("modeled") for row in pending
               if row["kind"] == "control")
    assert (tmp_path / "aggregate/comparison.csv").is_file()
    import matplotlib
    matplotlib.use("Agg")
    plots(report, tmp_path / "aggregate")
    assert all((tmp_path / "aggregate" / name).is_file() for name in (
        "perplexity_vs_total_bpp.png", "lp_vs_monolithic.png", "multirate_storage.png"))
    attempted = tmp_path / "progressive_llm/controls/s01_w4"
    attempted.mkdir(parents=True)
    (attempted / "quantization_process.json").write_text('{"returncode":0}')
    refreshed = write(source, tmp_path, tmp_path / "aggregate")
    assert next(row for row in refreshed["rows"] if row["id"] == "s01_w4")["status"] == "attempted"
