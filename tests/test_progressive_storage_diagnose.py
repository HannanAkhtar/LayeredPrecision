import json

import pytest
import torch
from safetensors.torch import save_file

from lp_llm import progressive_storage_diagnose as diag
from lp_llm import progressive_storage_feasibility as audit


def test_known_grid_oracle_covers_pinned_families_and_bits():
    rows = diag.known_grid_oracle()
    assert {(r["branch"], r["bits"]) for r in rows} == {
        (b, n) for b in ("s01", "r02") for n in (2, 4, 6, 8)}
    assert all(r["pinned"] in ("exact_constructive", "not_found_under_search") for r in rows)


def test_fp16_intermediate_can_change_bf16_export_bits():
    q = torch.tensor([1.0], dtype=torch.float32)
    scale = torch.tensor(0.003700540168210864, dtype=torch.float32)
    direct = (q * scale).to(torch.bfloat16)
    pinned = (q * scale).to(torch.float16).to(torch.bfloat16)
    assert not torch.equal(direct.view(torch.int16), pinned.view(torch.int16))


def test_interval_candidates_and_inconclusive_search():
    values = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float16)
    candidates = diag._interval_candidates(values, 2, "s01")
    assert candidates
    result = audit.recover_group(values, 2, "s01", "fp32", search_limit=len(candidates),
                                 candidate_scales=candidates, pre_export_fp16=True)
    assert result["outcome"] == "exact_constructive"
    assert diag._interval_candidates(values.to(torch.bfloat16), 2, "s01") == []
    missed = audit.recover_group(values, 2, "s01", "fp32", search_limit=0,
                                 candidate_scales=candidates)
    assert missed["outcome"] == "not_found_under_search"


def test_format_specific_full_certificates(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    name = "model.layers.0.self_attn.q_proj.weight"
    save_file({name: torch.tensor([[0.0, 0.5]], dtype=torch.float16)},
              checkpoint / "model.safetensors")
    item = {"id": "s01_w2", "branch": "s01", "bits": 2,
            "checkpoint": str(checkpoint), "full_sha256": "a" * 64,
            "eligible_elements": 2, "checkpoint_file_bytes": 100,
            "tensors": {name: {"eligible": True, "shape": [1, 2],
                               "shard": "model.safetensors", "payload_bytes": 4}}}
    def fake(values, bits, branch, fmt):
        return {"outcome": "exact_constructive" if fmt == "fp16" else
                "not_found_under_search", "max_abs_error": 0.0}
    monkeypatch.setattr(audit, "recover_group", fake)
    result = audit.inspect(item, "full", tmp_path / "groups.jsonl")
    assert result["exact_certificates"] == {"fp16": True, "fp32": False, "mixed": True}
    assert json.loads((tmp_path / "groups.jsonl").read_text())["row"] == 0
