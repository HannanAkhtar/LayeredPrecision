import json
import sys
import types
import weakref
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from lp_llm.r03_stream import stream_gptq_qdq


class FakeConfig:
    def __init__(self):
        self.quantization_config = {"bits": 2}

    def save_pretrained(self, target):
        (Path(target) / "config.json").write_text(json.dumps({"model_type": "qwen3"}))


class FakeQuantLinear(torch.nn.Module):
    def __init__(self, wrong=False):
        super().__init__()
        self.bits, self.group_size, self.sym, self.pack_dtype_bits = 2, 128, True, 32
        self.in_features, self.out_features = 128, 16
        self.register_buffer("qweight", torch.zeros(8, 16, dtype=torch.int32))
        self.register_buffer("qzeros", torch.zeros(1, 1, dtype=torch.int32))
        self.register_buffer("scales", torch.ones(1, 16, dtype=torch.float16))
        self.register_buffer("g_idx", torch.zeros(128, dtype=torch.int32))
        self.wrong = wrong

    def dequantize_weight(self):
        return torch.full((128, 16), int(self.wrong), dtype=torch.float16)


def fixture(tmp_path, monkeypatch, *, wrong=False):
    fake = types.ModuleType("gptqmodel.nn_modules.qlinear.torch")
    fake.TorchQuantLinear = FakeQuantLinear
    for name in ("gptqmodel", "gptqmodel.nn_modules", "gptqmodel.nn_modules.qlinear"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, fake.__name__, fake)

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = torch.nn.Module()
            self.self_attn.q_proj = FakeQuantLinear(wrong)
            self.self_attn.k_proj = FakeQuantLinear()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([Block()])
            self.model.norm = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float16))
            self.config = FakeConfig()

    source = tmp_path / "source"
    source.mkdir()
    names = {f"model.layers.0.self_attn.{kind}.weight":
             torch.ones(16, 128, dtype=torch.bfloat16) for kind in ("q_proj", "k_proj")}
    names["model.norm"] = torch.ones(1, dtype=torch.float16)
    save_file(names, source / "model.safetensors")
    (source / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "model.safetensors" for name in names}}))
    packed = types.SimpleNamespace(model=Model(), tokenizer=types.SimpleNamespace(
        save_pretrained=lambda target: (Path(target) / "tokenizer_config.json").write_text("{}")))
    return source, packed


def test_r03_stream_one_tensor_per_shard_and_exact_parity(tmp_path, monkeypatch):
    source, packed = fixture(tmp_path, monkeypatch)
    output = tmp_path / "qdq"
    result = stream_gptq_qdq(packed, source, output, expected_count=2)
    assert result["format"] == "dense_fp16_qdq_streamed"
    assert result["tensor_count"] == 3 and result["quantized_projection_count"] == 2
    assert result["max_single_tensor_bytes"] == 128 * 16 * 2
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert len(set(index["weight_map"].values())) == 3
    for name, shard in index["weight_map"].items():
        with safe_open(output / shard, framework="pt") as handle:
            assert handle.keys() == [name]
            if name.endswith(".weight"):
                assert torch.equal(handle.get_tensor(name), torch.zeros(16, 128, dtype=torch.float16))
    assert not (output / "model.safetensors").exists()
    with pytest.raises(FileExistsError):
        stream_gptq_qdq(packed, source, output, expected_count=2)


def test_r03_stream_preserves_partial_on_error_and_refuses_restart(tmp_path, monkeypatch):
    source, packed = fixture(tmp_path, monkeypatch, wrong=True)
    output = tmp_path / "qdq"
    with pytest.raises(ValueError, match="Independent packed decode differs"):
        stream_gptq_qdq(packed, source, output, expected_count=2)
    assert not output.exists()
    assert (tmp_path / "qdq.partial").is_dir()
    with pytest.raises(FileExistsError, match="partial export"):
        stream_gptq_qdq(packed, source, output, expected_count=2)


def test_r03_stream_releases_previous_dense_projection_before_next_decode(tmp_path, monkeypatch):
    from lp_llm import r03_stream

    source, packed = fixture(tmp_path, monkeypatch)
    original = r03_stream._decoded_projection
    previous = []

    def observe(module, name):
        assert all(ref() is None for ref in previous), "Prior dense projection is still live"
        value = original(module, name)
        previous.append(weakref.ref(value))
        return value

    monkeypatch.setattr(r03_stream, "_decoded_projection", observe)
    stream_gptq_qdq(packed, source, tmp_path / "qdq", expected_count=2)
    assert all(ref() is None for ref in previous)


def test_r03_stream_tiny_hf_checkpoint_reloads(tmp_path, monkeypatch):
    from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

    fake = types.ModuleType("gptqmodel.nn_modules.qlinear.torch")
    fake.TorchQuantLinear = FakeQuantLinear
    for name in ("gptqmodel", "gptqmodel.nn_modules", "gptqmodel.nn_modules.qlinear"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=256, hidden_size=128,
        intermediate_size=256, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=64,
        tie_word_embeddings=False)).eval().half()
    source = tmp_path / "source"
    model.save_pretrained(source)
    for path in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                 "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
        parent_name, field = path.rsplit(".", 1)
        parent = model.model.layers[0].get_submodule(parent_name)
        linear = getattr(parent, field)
        replacement = FakeQuantLinear()
        replacement.in_features, replacement.out_features = linear.in_features, linear.out_features
        groups = linear.in_features // 128
        replacement.qweight = torch.zeros(linear.in_features // 16, linear.out_features, dtype=torch.int32)
        replacement.qzeros = torch.zeros(groups, linear.out_features // 16, dtype=torch.int32)
        replacement.scales = torch.ones(groups, linear.out_features, dtype=torch.float16)
        replacement.g_idx = torch.arange(linear.in_features, dtype=torch.int32) // 128
        replacement.dequantize_weight = lambda module=replacement: torch.zeros(
            module.in_features, module.out_features, dtype=torch.float16)
        setattr(parent, field, replacement)
    packed = types.SimpleNamespace(model=model, tokenizer=types.SimpleNamespace(
        save_pretrained=lambda target: (Path(target) / "tokenizer_config.json").write_text("{}")))
    output = tmp_path / "qdq"
    report = stream_gptq_qdq(packed, source, output, expected_count=7)
    assert report["quantized_projection_count"] == 7
    reloaded = AutoModelForCausalLM.from_pretrained(output, torch_dtype=torch.float16)
    assert torch.count_nonzero(reloaded.model.layers[0].mlp.down_proj.weight) == 0
    logits = reloaded(torch.tensor([[1, 2, 3]])).logits
    assert torch.isfinite(logits).all()
