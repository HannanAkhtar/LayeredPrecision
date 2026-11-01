import json
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from lp_llm.evaluate import _identity, sliding_nll, wikitext, zeroshot


class RecordingModel:
    def __init__(self):
        self.calls = []

    def __call__(self, input_ids, labels):
        loss = (input_ids.float().sum() + labels[labels != -100].float().sum()) / 37
        self.calls.append((input_ids.clone(), labels.clone(), float(loss)))
        return SimpleNamespace(loss=loss)


def baseline_sliding_nll(model, tokens, context, stride):
    total_tokens = tokens.size(1)
    nll_sum = 0.0
    scored_tokens = 0
    valid_per_window = []
    for i in range(0, total_tokens, stride):
        end = min(i + stride, total_tokens)
        begin = max(end - context, 0)
        input_ids = tokens[:, begin:end].to("cpu")
        labels = input_ids.clone()
        target_length = end - i
        labels[:, :-target_length] = -100
        valid = int((labels[:, 1:] != -100).sum().item())
        valid_per_window.append(valid)
        if valid > 0:
            with torch.inference_mode():
                loss = model(input_ids, labels=labels).loss
            nll_sum += loss.item() * valid
            scored_tokens += valid
    return nll_sum, scored_tokens, valid_per_window


@pytest.mark.parametrize("length,context,stride", [(10, 6, 2), (13, 7, 3),
                                                    (1, 6, 2), (20, 5, 5)])
def test_sliding_window_matches_original_protocol(length, context, stride):
    tokens = torch.arange(1, length + 1).reshape(1, length)
    reference, actual = RecordingModel(), RecordingModel()
    expected_nll, expected_count, valid_per_window = baseline_sliding_nll(
        reference, tokens, context, stride)
    nll, count = sliding_nll(actual, tokens, context, stride, "cpu")
    assert (nll, count) == (expected_nll, expected_count)
    assert len(reference.calls) == len(actual.calls)
    for expected, observed in zip(reference.calls, actual.calls):
        assert torch.equal(expected[0], observed[0])
        assert torch.equal(expected[1], observed[1])
        assert expected[2] == observed[2]
    assert sum(valid_per_window) == count
    if length == 10:
        assert [call[0].tolist()[0] for call in actual.calls] == [
            [1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7, 8], [5, 6, 7, 8, 9, 10]]
        assert valid_per_window == [1, 2, 2, 2, 2]


def make_hf_checkpoint(path):
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (path / "tokenizer.json").write_text('{"version":"1"}')
    save_file({"model.norm.weight": torch.ones(4, dtype=torch.bfloat16)},
              str(path / "model-00001.safetensors"))
    (path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"model.norm.weight": "model-00001.safetensors"}}))


def test_original_hf_checkpoint_identity_and_strict_resume(tmp_path, monkeypatch):
    source, output = tmp_path / "hf", tmp_path / "result.json"
    make_hf_checkpoint(source)
    identity = _identity(source, "wikitext-2-raw-v1")
    assert identity["kind"] == "huggingface_safetensors"
    assert identity["shard_sha256"]

    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert text == "a\n\nb" and kwargs["add_special_tokens"] is False
            return {"input_ids": torch.arange(5).reshape(1, 5)}

    class Model:
        def eval(self):
            pass

    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **k: Model()))
    fake_datasets = SimpleNamespace(load_dataset=lambda *a, **k: {"text": ["a", "b"]})
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setattr("lp_llm.evaluate.sliding_nll", lambda *a: (10.0, 4))
    result = wikitext(source, output)
    assert result["identity"]["kind"] == "huggingface_safetensors"
    assert result["corpus_tokens"] == 5 and result["scored_tokens"] == 4
    assert wikitext(source, output) == result
    (source / "tokenizer.json").write_text('{"version":"2"}')
    with pytest.raises(ValueError, match="different identity"):
        wikitext(source, output)


def test_harness_metadata_is_preserved(tmp_path, monkeypatch):
    source, output = tmp_path / "hf", tmp_path / "harness.json"
    make_hf_checkpoint(source)
    raw = {"results": {"arc_easy": {"acc,none": 0.8, "acc_stderr,none": 0.01,
                                     "acc_norm,none": 0.79, "acc_norm_stderr,none": 0.02}},
           "versions": {"arc_easy": 1}, "configs": {"arc_easy": {"num_fewshot": 0}},
           "n-shot": {"arc_easy": 0}, "higher_is_better": {"arc_easy": {"acc": True}}}
    calls = []
    monkeypatch.setitem(sys.modules, "lm_eval", SimpleNamespace(
        simple_evaluate=lambda **kwargs: calls.append(kwargs) or raw))
    monkeypatch.setitem(sys.modules, "lm_eval.utils", SimpleNamespace(
        handle_non_serializable=str))
    value = zeroshot(source, output)
    assert value["harness"] == raw
    assert value["results"]["arc_easy"]["acc_stderr,none"] == 0.01
    assert value["harness"]["versions"]["arc_easy"] == 1
    assert value["task_versions"]["arc_easy"] == 1
    assert value["settings"]["num_fewshot"] == 0
    assert calls[0]["batch_size"] == 1
    assert calls[0]["log_samples"] is False
    assert zeroshot(source, output) == value and len(calls) == 1
