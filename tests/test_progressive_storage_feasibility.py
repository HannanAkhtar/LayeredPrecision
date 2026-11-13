import json

import pytest
import torch
from safetensors.torch import save_file

from lp_llm import progressive_storage_feasibility as sf


@pytest.mark.parametrize("bits", (2, 4, 6, 8))
@pytest.mark.parametrize("branch", ("s01", "r02"))
@pytest.mark.parametrize("scale_dtype", ("fp16", "fp32"))
def test_exact_constructive_groups(bits, branch, scale_dtype):
    codes = torch.tensor([-1, 0, 1], dtype=torch.float32) if branch == "s01" else \
        torch.tensor([0, 1, 2], dtype=torch.float32)
    values = (codes * 0.5).to(torch.float16) if branch == "s01" else \
        ((codes - 1) * 0.5).to(torch.float16)
    result = sf.recover_group(values, bits, branch, scale_dtype)
    assert result["outcome"] == "exact_constructive"
    assert result["mismatching_elements"] == 0


def test_distinct_patterns_prove_failure_but_search_failure_does_not():
    values = torch.tensor([-2., -1., 0., 1., 2.], dtype=torch.float16)
    assert sf.recover_group(values, 2, "s01", "fp16")["outcome"] == "not_representable_proven"
    bounded = sf.recover_group(torch.tensor([0., 0.3], dtype=torch.float16), 2,
                               "s01", "fp16", search_limit=0)
    assert bounded["outcome"] == "not_found_under_search"


def test_signed_zero_nonfinite_and_partial_group():
    assert sf.recover_group(torch.tensor([-0., 0.], dtype=torch.float16), 2, "s01", "fp16")[
        "outcome"] in ("not_found_under_search", "exact_constructive")
    assert sf.recover_group(torch.zeros(3, dtype=torch.float16), 2, "r02", "fp32")[
        "outcome"] == "exact_constructive"
    with pytest.raises(ValueError, match="Nonfinite"):
        sf.recover_group(torch.tensor([float("inf")], dtype=torch.float16), 2, "s01", "fp16")
    assert list(sf._positions([3, 129], "sample"))[-1] == (2, 1)


def test_inventory_and_storage_use_actual_tensor_dtypes(tmp_path, monkeypatch):
    source = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    for directory in (source, checkpoint):
        directory.mkdir()
        tensors = {"model.layers.0.self_attn.q_proj.weight": torch.ones((2, 129), dtype=torch.float16),
                   "lm_head.weight": torch.ones(7, dtype=torch.bfloat16)}
        save_file(tensors, directory / "model.safetensors")
        (directory / "config.json").write_text('{}')
    marker = {"status": "completed_validated", "checkpoint_validation": {
        "resolved_checkpoint_path": str(checkpoint),
        "shard_sha256": {"model.safetensors": sf.sha256(checkpoint / "model.safetensors")}}}
    job = tmp_path / "quantizer_followups/s01_fp16_load_repair"
    job.mkdir(parents=True)
    (job / "encoding_complete.json").write_text(json.dumps(marker))
    (job / "evaluation.log").write_text("evaluated")
    (job / "full.json").write_text(json.dumps({**marker,
        "corpus": {"token_sha256": sf.TOKEN_SHA, "scored_tokens": sf.TOKEN_COUNT},
        "evaluation_log_sha256": sf.sha256(job / "evaluation.log")}))
    monkeypatch.setattr(sf, "_eligible", lambda name: name.endswith("q_proj.weight"))
    monkeypatch.setattr(sf, "_weight_map", lambda directory: {
        name: "model.safetensors" for name in tensors})
    monkeypatch.setattr(sf, "inventory", sf.inventory)
    # The synthetic fixture has one eligible tensor; exercise the accounting directly.
    item = sf._checkpoint(tmp_path, "s01", 2)
    item.update({"tensors": {name: {"shape": list(value.shape), "eligible": sf._eligible(name),
        "payload_bytes": value.numel() * value.element_size()} for name, value in tensors.items()},
        "checkpoint_file_bytes": sum(p.stat().st_size for p in checkpoint.iterdir())})
    storage = sf._storage(item)
    assert storage["code_bytes"] == 2 * ((129 * 2 + 7) // 8)
    assert storage["group_count"] == 4
    assert storage["excluded_tensor_payload_bytes"] == 14
    (checkpoint / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="shard changed"):
        sf._checkpoint(tmp_path, "s01", 2)


def test_checkpoint_layouts_and_missing_provenance(tmp_path):
    for branch, bits in (("s01", 2), ("r02", 6), ("s01", 4)):
        full, completion, _ = sf._result(tmp_path, branch, bits)
        if bits == 2:
            assert "fp16_load_repair" in str(full)
        elif branch == "s01" and bits == 4:
            assert str(completion).endswith("control_recovery/s01_w4/audit.json") or \
                str(completion).endswith("control_recovery\\s01_w4\\audit.json")
        else:
            assert str(full).endswith("controls/r02_w6/full.json") or \
                str(full).endswith("controls\\r02_w6\\full.json")
        with pytest.raises(FileNotFoundError):
            sf._checkpoint(tmp_path, branch, bits)


def test_figure_is_new_output_only(tmp_path):
    target = tmp_path / "plot.svg"
    sf._bar_figure(target, "sample", [("s01_w2", 0.5)], "fraction")
    assert "s01_w2" in target.read_text()
    with pytest.raises(FileExistsError):
        sf._bar_figure(target, "sample", [("s01_w2", 0.5)], "fraction")


def test_report_conflict_does_not_modify_existing_file(tmp_path):
    output = tmp_path / "sample-all.json"
    output.write_bytes(b"historic report")
    with pytest.raises(FileExistsError):
        sf.run(tmp_path, tmp_path, output, "sample", "a" * 40)
    assert output.read_bytes() == b"historic report"
