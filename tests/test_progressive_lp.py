import json
import sys
import types

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from lp_llm.progressive_lp import (encode_layer, encode_residual, identity,
                                    reconstruct_block, verified_layers)
from lp_llm.progressive_run import materialize
from lp_llm import progressive_pilot


def fixture(tmp_path, *, width=130):
    source, base = tmp_path / "source", tmp_path / "base"
    source.mkdir()
    base.mkdir()
    names = [f"model.layers.{layer}.{kind}.weight" for layer in range(36)
             for kind in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                          "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]
    original = {name: torch.linspace(-0.5, 0.7, width).repeat(2, 1).to(torch.bfloat16)
                for name in names}
    original["lm_head.weight"] = torch.tensor([[0.25, -0.5]], dtype=torch.bfloat16)
    recovered = {name: (value.to(torch.float16) * 0.75 if name != "lm_head.weight"
                        else value.clone()) for name, value in original.items()}
    for directory, tensors in ((source, original), (base, recovered)):
        save_file(tensors, directory / "model.safetensors", metadata={"format": "pt"})
        (directory / "config.json").write_text(json.dumps({"model_type": "qwen3",
            "num_hidden_layers": 36, "hidden_size": 4096,
            "intermediate_size": 12288, "vocab_size": 151936}))
        (directory / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {name: "model.safetensors" for name in tensors}}))
    return source, base


@pytest.mark.parametrize("values", [
    torch.zeros(2, 130),
    torch.full((2, 130), -1e-9),
    torch.full((2, 130), 0.125),
    torch.tensor([[-5.0] + [0.01] * 129, [7.0] + [-0.03] * 129]),
])
def test_signed_e2_roundtrip_edge_cases(values):
    params, decoded = encode_residual(values)
    assert torch.isfinite(decoded).all()
    assert params["scale"].dtype == torch.float32
    assert params["scale"].shape == (2, 2)  # 128 plus partial group
    assert not ((params["codes"] < 1) | (params["codes"] > 3)).any()
    if torch.count_nonzero(values) == 0 or torch.all(values == values[0, 0]):
        assert torch.equal(decoded, values)


def test_pilot_layers_preserve_base_and_refuse_partial_restart(tmp_path):
    source, base = fixture(tmp_path)
    root = tmp_path / "pilot"
    ident = identity("s01", source, base)
    first = encode_layer("s01", 1, source, base, root, rows=1, limit_tensors=7)
    second = encode_layer("s01", 2, source, base, root, rows=1, limit_tensors=7)
    assert first["projection_count"] == second["projection_count"] == 7
    assert first["status"] == second["status"] == "pilot_validated"
    assert len(verified_layers(root, 2, ident, pilot=True)) == 2
    original_base = torch.tensor([[1.0, -0.5]], dtype=torch.float16)
    assert torch.equal(reconstruct_block(original_base, []), original_base)
    assert reconstruct_block(original_base, []).dtype == torch.float16
    other = tmp_path / "interrupted"
    other.mkdir()
    (other / "layer1.partial").mkdir()
    with pytest.raises(FileExistsError, match="Interrupted"):
        encode_layer("s01", 1, source, base, other)


def test_full_tiny_hierarchy_materializes_source_exact_exclusions(tmp_path):
    source, base = fixture(tmp_path, width=128)
    root = tmp_path / "hierarchy"
    first = encode_layer("r02", 1, source, base, root, rows=2)
    assert first["projection_count"] == 252
    assert first["encoded_elements"] == 252 * 2 * 128
    output = tmp_path / "prefix1"
    report = materialize("r02", 1, source, base, root, output, rows=2)
    assert report["projection_count"] == 252
    index = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(output / index["lm_head.weight"], framework="pt") as handle:
        assert torch.equal(handle.get_tensor("lm_head.weight"),
                           torch.tensor([[0.25, -0.5]], dtype=torch.bfloat16))
    with pytest.raises(FileExistsError):
        materialize("r02", 1, source, base, root, output, rows=2)
    second = encode_layer("r02", 2, source, base, root, rows=2)
    third = encode_layer("r02", 3, source, base, root, rows=2)
    assert second["projection_count"] == third["projection_count"] == 252
    assert len(third["layerwise"]) == 36
    final = tmp_path / "prefix3"
    materialize("r02", 3, source, base, root, final, rows=2)
    name = first["chunks"][0]["name"]
    base_map = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(base / base_map[name], framework="pt") as handle:
        base_value = handle.get_tensor(name)
    entries = [(root / f"layer{depth}", layer["chunks"][0]) for depth, layer in
               enumerate((first, second, third), 1)]
    expected = reconstruct_block(base_value, entries).to(torch.float16)
    final_map = json.loads((final / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(final / final_map[name], framework="pt") as handle:
        assert torch.equal(handle.get_tensor(name), expected)
    corrupt = root / "layer1" / first["chunks"][0]["code"]
    corrupt.write_bytes(b"bad")
    with pytest.raises(ValueError, match="hash differs"):
        verified_layers(root, 1, identity("r02", source, base))


def test_zero_enhancement_checkpoint_is_exact_repaired_base(tmp_path):
    source, base = fixture(tmp_path, width=128)
    output = tmp_path / "zero"
    report = materialize("s01", 0, source, base, tmp_path / "empty", output)
    assert report["checkpoint_dtype"] == "base_exact"
    base_map = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    output_map = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    for name in base_map:
        with safe_open(base / base_map[name], framework="pt") as source_handle:
            original = source_handle.get_tensor(name)
        with safe_open(output / output_map[name], framework="pt") as output_handle:
            reconstructed = output_handle.get_tensor(name)
        assert reconstructed.dtype == original.dtype
        assert torch.equal(reconstructed, original)


def test_materialization_preserves_interrupted_partial(tmp_path, monkeypatch):
    from lp_llm import progressive_run

    source, base = fixture(tmp_path, width=128)
    output = tmp_path / "interrupted"
    original_save = progressive_run.save_file
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated shard interruption")
        return original_save(*args, **kwargs)

    monkeypatch.setattr(progressive_run, "save_file", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        materialize("s01", 0, source, base, tmp_path / "empty", output)
    assert not output.exists()
    assert (tmp_path / "interrupted.partial").is_dir()
    with pytest.raises(FileExistsError):
        materialize("s01", 0, source, base, tmp_path / "empty", output)


def test_incomplete_row_coverage_cannot_be_marked_valid(tmp_path):
    source, base = fixture(tmp_path, width=128)
    root = tmp_path / "pilot"
    encode_layer("s01", 1, source, base, root, rows=1, limit_tensors=7)
    marker = root / "layer1/encoding_complete.json"
    report = json.loads(marker.read_text())
    omitted = report["chunks"].pop()
    report["chunk_count"] -= 1
    marker.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="row coverage"):
        verified_layers(root, 1, identity("s01", source, base), pilot=True)
    assert (root / "layer1" / omitted["code"]).is_file()


def test_two_stage_pilot_report_and_idempotent_resume(tmp_path, monkeypatch):
    source, base = fixture(tmp_path, width=128)
    monkeypatch.setitem(sys.modules, "resource", types.SimpleNamespace(
        RUSAGE_SELF=0, getrusage=lambda _: types.SimpleNamespace(ru_maxrss=1000)))
    monkeypatch.setattr(progressive_pilot, "validate_repaired_base", lambda *_: {})
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "12288")
    root = tmp_path / "pilot"
    report_path = root / "report.json"
    report = progressive_pilot.run("s01", source, base, root, report_path, "a" * 40)
    assert report["first"]["projection_count"] == 7
    assert report["second"]["projection_count"] == 7
    assert report["artifact_bytes"] > 0
    assert progressive_pilot.run("s01", source, base, root, report_path, "a" * 40) == report
