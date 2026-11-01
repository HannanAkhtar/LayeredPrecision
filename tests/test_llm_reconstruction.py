import json

import pytest
import torch
import weakref
from safetensors.torch import load_file, save_file

from lp_llm.encode import encode
from lp_llm.evaluate import _identity
from lp_llm.model_adapter import SelectionPolicy, expected_qwen3, inventory
from lp_llm.reconstruct import reconstruct
from lp_llm.storage import decode_prefix, read_layer, storage_report
from lp_llm.storage import atomic_json
from lp_llm.quantize import affine_quantize, encode_layers
from lp_llm.validate_stage import encoding as validate_encoding
from lp_llm.validate_stage import reconstruction as validate_reconstruction
from lp_llm.validate_stage import evaluation as validate_evaluation
from lp_llm.evaluate import _package_version


CONFIG = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
          "num_hidden_layers": 1, "hidden_size": 16, "intermediate_size": 24,
          "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
          "vocab_size": 32, "tie_word_embeddings": False, "torch_dtype": "bfloat16"}


def make_checkpoint(path):
    path.mkdir()
    (path / "config.json").write_text(json.dumps(CONFIG))
    names = expected_qwen3(CONFIG)
    mapping = {}
    tensors = {}
    for i, (name, shape) in enumerate(sorted(names.items())):
        tensor = (torch.randn(shape) * 0.02).bfloat16()
        if "norm" in name:
            tensor.fill_(1)
        tensors[name] = tensor
        shard = f"model-{i // 7:05d}.safetensors"
        mapping[name] = shard
    for shard in set(mapping.values()):
        save_file({name: tensors[name] for name in mapping if mapping[name] == shard}, str(path / shard))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    return tensors


@pytest.mark.parametrize("granularity", ["channel", "group"])
def test_incremental_encode_resume_and_reconstruct(tmp_path, granularity):
    source, artifact = tmp_path / "source", tmp_path / "artifact"
    originals = make_checkpoint(source)
    _, mapping = inventory(source)
    assert set(mapping) == set(originals)
    settings = {"bits": [2, 2, 2, 2], "granularity": granularity, "group_size": 7}
    assert encode(source, artifact, settings, stop_after=5) is None
    marker = artifact / "markers/0000.json"
    first = marker.stat().st_mtime_ns
    manifest = encode(source, artifact, settings)
    assert marker.stat().st_mtime_ns == first
    manifest_hash = (artifact / "COMPLETE.json").read_text()
    assert encode(source, artifact, settings) == manifest
    assert (artifact / "COMPLETE.json").read_text() == manifest_hash
    assert manifest["eligible_parameters"] > 0 and manifest["excluded_parameters"] > 0
    assert len(manifest["prefix_diagnostics"]) == 4
    assert [row["nominal_bits"] for row in manifest["prefix_diagnostics"]] == [2, 4, 6, 8]
    assert sum(r["numel"] for r in manifest["tensors"]) == manifest["eligible_parameters"] + manifest["excluded_parameters"]
    assert storage_report(artifact, manifest) == json.loads((artifact / "storage_report.json").read_text())
    for row in storage_report(artifact, manifest)["prefixes"]:
        assert row["total_physical_bytes"] == sum(row[key] for key in (
            "packed_code_bytes", "scale_zero_metadata_bytes", "other_decoder_framing_bytes",
            "shared_unquantized_bytes"))
        required = ["manifest.json", "COMPLETE.json", *manifest["sidecars"]]
        required += [r["shared"] for r in manifest["tensors"] if not r["eligible"]]
        selected = [layer for r in manifest["tensors"] if r["eligible"]
                    for layer in r["layers"][:row["depth"]]]
        required += [path for layer in selected for path in (layer["code"], layer["metadata"])]
        assert row["total_physical_bytes"] == sum((artifact / path).stat().st_size for path in required)
        assert row["scale_zero_metadata_bytes"] == sum(
            read_layer(artifact, layer)[key].numel() * read_layer(artifact, layer)[key].element_size()
            for layer in selected for key in ("scale", "zero_point"))
    assert sum(1 for r in manifest["tensors"] if r["eligible"]) == 7
    for record in manifest["tensors"]:
        if not record["eligible"]:
            continue
        original = originals[record["name"]].float()
        for depth, (params, _, encoder_prefix) in enumerate(
                encode_layers(original, settings["bits"], granularity, settings["group_size"]), 1):
            stored = read_layer(artifact, record["layers"][depth - 1])
            assert torch.equal(stored["codes"], params["codes"])
            assert torch.equal(stored["scale"], params["scale"])
            assert torch.equal(stored["zero_point"], params["zero_point"])
            assert torch.equal(decode_prefix(artifact, record, depth), encoder_prefix)
    for depth in range(1, 5):
        out = tmp_path / f"prefix-{depth}"
        index = reconstruct(artifact, out, depth)
        assert _identity(out, "test")["kind"] == "lp_reconstructed"
        assert set(index["weight_map"]) == set(originals)
        for record in manifest["tensors"]:
            name = record["name"]
            actual = load_file(str(out / index["weight_map"][name]))[name]
            if not record["eligible"]:
                assert torch.equal(actual, originals[name])
            else:
                assert actual.dtype == torch.bfloat16
                assert torch.equal(actual, decode_prefix(artifact, record, depth).to(torch.bfloat16))
    for shard in source.glob("*.safetensors"):
        shard.unlink()
    assert reconstruct(artifact, tmp_path / "independent", 2)


def test_lp_checkpoint_identity_rejects_bad_completion_marker(tmp_path):
    source = tmp_path / "source"
    make_checkpoint(source)
    artifact = tmp_path / "artifact"
    encode(source, artifact, {"bits": [2], "granularity": "channel"})
    out = tmp_path / "decoded"
    reconstruct(artifact, out, 1)
    marker = next(out.glob(".done-*.json"))
    data = json.loads(marker.read_text())
    data["sha256"] = "0" * 64
    marker.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="integrity check failed"):
        _identity(out, "test")


def test_index_detects_missing_or_unexpected_keys(tmp_path):
    source = tmp_path / "source"
    make_checkpoint(source)
    index = source / "model.safetensors.index.json"
    value = json.loads(index.read_text())
    value["weight_map"].pop("lm_head.weight")
    index.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Checkpoint key mismatch"):
        inventory(source)


@pytest.mark.parametrize("granularity", ["channel", "group"])
def test_monolithic_is_independent_quantization(tmp_path, granularity):
    source = tmp_path / "source"
    originals = make_checkpoint(source)
    artifact = tmp_path / "mono"
    manifest = encode(source, artifact, {"method": "monolithic", "bits": [4],
                                         "granularity": granularity, "group_size": 7})
    out = tmp_path / "decoded"
    index = reconstruct(artifact, out, 1)
    for record in manifest["tensors"]:
        if record["eligible"]:
            name = record["name"]
            expected = affine_quantize(originals[name].float(), 4, granularity, 7)[1].bfloat16()
            actual = load_file(str(out / index["weight_map"][name]))[name]
            assert torch.equal(actual, expected)


def test_source_tensors_are_released_before_next_read(tmp_path, monkeypatch):
    import lp_llm.encode as encoder
    source = tmp_path / "source"
    make_checkpoint(source)
    actual_read = encoder.read_tensor
    previous = None
    count = 0

    def bounded_read(*args):
        nonlocal previous, count
        assert previous is None or previous() is None
        tensor = actual_read(*args)
        previous = weakref.ref(tensor)
        count += 1
        return tensor

    monkeypatch.setattr(encoder, "read_tensor", bounded_read)
    encode(source, tmp_path / "artifact", {"bits": [2], "granularity": "channel"})
    assert count == len(expected_qwen3(CONFIG))


@pytest.mark.parametrize("granularity", ["channel", "group"])
def test_monolithic_batch_stage_validation(tmp_path, granularity):
    source, artifact = tmp_path / "source", tmp_path / "artifact"
    make_checkpoint(source)
    checkpoint, result = artifact / "checkpoint", artifact / "wikitext.json"
    assert not validate_encoding(artifact, source, 2, granularity)
    encode(source, artifact, {"method": "monolithic", "bits": [2],
                              "granularity": granularity, "group_size": 128})
    assert validate_encoding(artifact, source, 2, granularity)
    assert not validate_reconstruction(artifact, checkpoint)
    reconstruct(artifact, checkpoint, 1)
    assert validate_reconstruction(artifact, checkpoint)
    assert not validate_evaluation(checkpoint, result)
    identity = _identity(checkpoint, "wikitext-2-raw-v1") | {
        "context": 2048, "stride": 512, "datasets": _package_version("datasets"),
        "join": "\\n\\n", "add_special_tokens": False}
    atomic_json(result, {"identity": identity, "status": "complete", "corpus_tokens": 10,
                         "scored_tokens": 9, "perplexity": 2.0})
    assert validate_evaluation(checkpoint, result)
    atomic_json(result, {"identity": identity, "status": "incomplete"})
    with pytest.raises(ValueError, match="different identity"):
        validate_evaluation(checkpoint, result)
