import json
import sys
import types

import pytest
import torch
from safetensors.torch import load_file, save_file

from lp_llm.encode import encode
from lp_llm.model_adapter import expected_qwen3
from lp_llm.reconstruct import reconstruct
from lp_llm.rtn_audit import audit
from lp_llm.rtn_diagnostic_eval import blocks, score_blocks, evaluate as diagnostic_evaluate
from lp_llm.rtn_logits import logit_stats
from lp_llm.rtn_reproduction import MODES, quantize_base
from lp_llm.storage import decode_prefix, read_layer
from lp_llm.upstream_reference import direct_qdq
from lp_llm.run_rtn_reproduction import numerical, prepare, report, full, _paths
from lp_llm.aggregate_rtn_reproduction import aggregate, markdown


CONFIG = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
          "num_hidden_layers": 1, "hidden_size": 16, "intermediate_size": 24,
          "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
          "vocab_size": 32, "tie_word_embeddings": False, "torch_dtype": "bfloat16"}


def _checkpoint(path):
    path.mkdir()
    (path / "config.json").write_text(json.dumps(CONFIG))
    tensors = {}
    for name, shape in expected_qwen3(CONFIG).items():
        tensors[name] = (torch.randn(shape) * 0.02).bfloat16()
    tensors["model.layers.0.self_attn.q_proj.weight"][0] = 0
    save_file(tensors, str(path / "model.safetensors"))
    (path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {name: "model.safetensors" for name in tensors}}))
    return tensors


@pytest.mark.parametrize("family", list(MODES))
def test_two_bit_artifact_matches_upstream_and_checkpoint(tmp_path, family):
    torch.manual_seed(711)
    source = tmp_path / "source"
    originals = _checkpoint(source)
    coverage = audit(source)
    assert coverage["quantized_parameters"] + coverage["excluded_parameters"] == coverage["total_parameters"]
    assert len(coverage["parameters_by_projection"]) == 7
    assert len(coverage["parameters_by_block"]) == 1
    settings = {"experiment_id": family, "method": "lp", "bits": [2],
                "granularity": "group", "group_size": 128,
                "quantizer_family": family, "compute_device": "cpu"}
    artifact = tmp_path / family
    manifest = encode(source, artifact, settings)
    checkpoint = tmp_path / f"{family}-decoded"
    index = reconstruct(artifact, checkpoint, 1)
    algorithm, dtype = MODES[family]
    for record in manifest["tensors"]:
        name = record["name"]
        saved = load_file(str(checkpoint / index["weight_map"][name]))[name]
        if not record["eligible"]:
            assert torch.equal(saved, originals[name])
            continue
        upstream, scale = direct_qdq(originals[name].to(dtype),
                                     "plain" if algorithm == "plain" else "optimized")
        in_memory_params, in_memory = quantize_base(originals[name], family)
        stored = read_layer(artifact, record["layers"][0])
        decoded_fp32 = decode_prefix(artifact, record, 1)
        assert torch.equal(in_memory, upstream)
        assert torch.equal(in_memory_params["codes"], stored["codes"])
        assert stored["scale"].dtype == torch.float16
        assert torch.equal(stored["scale"], scale.reshape_as(stored["scale"]))
        assert torch.equal(decoded_fp32.to(dtype), upstream)
        assert torch.equal(saved, upstream.to(torch.bfloat16))
        before_cast = decoded_fp32 - upstream.float()
        after_cast = decoded_fp32.to(dtype).to(torch.bfloat16) - saved
        assert record["metrics"][0]["qdq_decode_max_abs_before_cast"] == pytest.approx(
            float(before_cast.abs().max()))
        assert torch.equal(after_cast, torch.zeros_like(after_cast))
        assert record["layers"][0]["qdq_dtype"] == str(dtype).removeprefix("torch.")
    assert manifest["identity"]["quantizer_spec"]["quantization_input_dtype"] == str(dtype).removeprefix("torch.")
    assert manifest["bits"] == [2]
    report = json.loads((artifact / "storage_report.json").read_text())["prefixes"][0]
    assert report["total_physical_bytes"] == sum(report[key] for key in (
        "packed_code_bytes", "scale_zero_metadata_bytes", "other_decoder_framing_bytes",
        "shared_unquantized_bytes"))


def test_nonoverlap_token_weighting_and_logit_metrics():
    tokens = torch.arange(22).reshape(1, 22)
    selected = list(blocks(tokens, length=8, limit=2))
    assert [x.tolist()[0] for x in selected] == [list(range(8)), list(range(8, 16))]
    class FakeModel:
        def __call__(self, input_ids, labels):
            return type("Output", (), {"loss": torch.tensor(float(input_ids[0, 0] + 1))})()
    score = score_blocks(FakeModel(), selected, "cpu")
    assert score["scored_tokens"] == 14
    assert score["negative_log_likelihood"] == 7 * (1 + 9)
    assert score["mean_cross_entropy"] == 5
    assert score["perplexity"] == pytest.approx(torch.exp(torch.tensor(5.)).item())
    reference = torch.randn(1, 8, 32)
    same = logit_stats(reference, reference, torch.arange(8).reshape(1, 8))
    assert same["max_abs_logit_error"] == 0
    assert same["logit_rmse"] == 0
    assert same["predicted_token_agreement"] == 1


def test_diagnostic_evaluator_small_model_and_identity(tmp_path, monkeypatch):
    import lp_llm.rtn_diagnostic_eval as diagnostic_module
    monkeypatch.setattr(diagnostic_module, "_identity", lambda checkpoint, task: {"checkpoint": str(checkpoint), "task": task})
    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert kwargs["add_special_tokens"] is False
            return {"input_ids": torch.arange(22).reshape(1, 22)}
    class Model:
        def eval(self):
            return self
        def __call__(self, input_ids, labels):
            return type("Output", (), {"loss": torch.tensor(float(input_ids[0, 0] + 1))})()
    datasets = types.ModuleType("datasets")
    datasets.load_dataset = lambda *args, **kwargs: {"text": ["a", "b"]}
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = type("AutoTokenizer", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: Tokenizer())})
    transformers.AutoModelForCausalLM = type("AutoModelForCausalLM", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: Model())})
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    output = tmp_path / "diagnostic.json"
    value = diagnostic_evaluate(tmp_path / "checkpoint", output, block_length=8, block_limit=2, device="cpu")
    assert value["status"] == "complete"
    assert value["scored_tokens"] == 14
    assert value["selected_input_tokens"] == 16
    assert value["blocks"][1]["mean_cross_entropy"] == 9
    assert diagnostic_evaluate(tmp_path / "checkpoint", output, block_length=8, block_limit=2, device="cpu") == value


def test_phase1_rejects_hierarchy_and_wrong_group(tmp_path):
    source = tmp_path / "source"
    _checkpoint(source)
    with pytest.raises(ValueError, match="bits=\\[2\\]"):
        encode(source, tmp_path / "bad", {"method": "lp", "bits": [2, 2, 2, 2],
                                          "granularity": "group", "group_size": 128,
                                          "quantizer_family": "autoround_plain_fp16"})


def test_restart_safe_numerical_prepare_and_aggregation(tmp_path):
    source = tmp_path / "source"
    _checkpoint(source)
    result_root = tmp_path / "results"
    paths = _paths(result_root, "b1_plain_fp16")
    paths["root"].mkdir(parents=True)
    settings = {"experiment_id": "b1_plain_fp16", "method": "lp", "bits": [2],
                "granularity": "group", "group_size": 128,
                "quantizer_family": "autoround_plain_fp16", "compute_device": "cpu",
                "model_id": "Qwen/Qwen3-8B-Base",
                "revision": "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"}
    first = numerical(settings, "fixed-config-hash", paths["numerical"])
    assert first["scale_exact"] and first["codes_exact"] and first["qdq_exact"]
    assert numerical(settings, "fixed-config-hash", paths["numerical"]) == first
    with pytest.raises(ValueError, match="different identity"):
        numerical(settings, "changed-config-hash", paths["numerical"])
    prepare(source, settings, "fixed-config-hash", paths)
    initial = (paths["artifact"] / "COMPLETE.json").read_bytes()
    prepare(source, settings, "fixed-config-hash", paths)
    assert (paths["artifact"] / "COMPLETE.json").read_bytes() == initial
    value = report(settings, "fixed-config-hash", paths)
    assert value["validation_status"]["numerical"]
    assert value["physical_storage"]["packed_code_bytes"] > 0
    assert value["qdq_decoder_difference"]["qdq_decode_max_abs_after_bf16_cast"] == 0
    comparison = aggregate(result_root)
    assert comparison["rows"][0]["full_perplexity"] == 9730.84365
    assert comparison["rows"][1]["experiment_id"] == "b1_plain_fp16"
    assert "b0_lp_plain_fp32" in markdown(comparison)
    evaluator_package = tmp_path / "eval_code" / "lp_llm"
    evaluator_package.mkdir(parents=True)
    (evaluator_package / "__init__.py").write_text("")
    evaluator = evaluator_package / "evaluate_matgptq.py"
    evaluator.write_text(
        "import argparse, json\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser()\n"
        "p.add_argument('--checkpoint'); p.add_argument('--output')\n"
        "p.add_argument('--sequence-length'); p.add_argument('--dtype')\n"
        "a=p.parse_args()\n"
        "Path(a.output).write_text(json.dumps({'perplexity': 17.0}))\n")
    assert full(paths, evaluator)["perplexity"] == 17.0
    assert full(paths, evaluator)["perplexity"] == 17.0
    assert paths["full"].with_name(paths["full"].name + ".validated.json").exists()
    paths["full"].write_text(json.dumps({"perplexity": 99.0}))
    with pytest.raises(ValueError, match="identity/content"):
        full(paths, evaluator)


def test_small_model_logit_pipeline_with_saved_reload(tmp_path, monkeypatch):
    from lp_llm.rtn_logits import compare
    source = tmp_path / "source"
    _checkpoint(source)
    artifact = tmp_path / "artifact"
    encode(source, artifact, {"method": "lp", "bits": [2], "granularity": "group",
                              "group_size": 128, "quantizer_family": "autoround_plain_fp16",
                              "compute_device": "cpu"})
    checkpoint = tmp_path / "checkpoint"
    reconstruct(artifact, checkpoint, 1)

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(32, 16)
            self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
            self.model.layers[0].self_attn = torch.nn.Module()
            self.model.layers[0].self_attn.q_proj = torch.nn.Linear(16, 16, bias=False)
            self.lm_head = torch.nn.Linear(16, 32, bias=False)
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            model = cls()
            index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
            for name, parameter in model.named_parameters():
                parameter.data = load_file(str(path / index[name]))[name]
            return model
        def forward(self, input_ids):
            hidden = self.model.embed_tokens(input_ids)
            hidden = self.model.layers[0].self_attn.q_proj(hidden)
            return type("Output", (), {"logits": self.lm_head(hidden)})()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = TinyModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    value = compare(source, artifact, checkpoint, torch.tensor([[1, 2, 3, 4]]), "cpu")
    assert value["lp_in_memory"]["max_abs_logit_error"] == 0
    assert value["saved_vs_lp_in_memory"]["max_abs_logit_error"] == 0
    assert value["saved_reloaded"]["predicted_token_agreement"] == 1


def test_tiny_qwen3_logit_workflow_when_transformers_supports_it(tmp_path):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3Config"):
        pytest.skip("Qwen3 requires Transformers >= 4.51")
    from lp_llm.rtn_logits import compare
    config = transformers.Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=64, tie_word_embeddings=False)
    model = transformers.Qwen3ForCausalLM(config).to(torch.bfloat16)
    source = tmp_path / "source"
    model.save_pretrained(source, safe_serialization=True, max_shard_size="10KB")
    artifact = tmp_path / "artifact"
    encode(source, artifact, {"method": "lp", "bits": [2], "granularity": "group",
                              "group_size": 128, "quantizer_family": "autoround_plain_fp16",
                              "compute_device": "cpu"})
    checkpoint = tmp_path / "checkpoint"
    reconstruct(artifact, checkpoint, 1)
    result = compare(source, artifact, checkpoint, torch.tensor([[1, 2, 3, 4]]), "cpu")
    assert result["saved_vs_lp_in_memory"]["max_abs_logit_error"] == 0
