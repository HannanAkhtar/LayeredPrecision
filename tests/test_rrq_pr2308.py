"""CPU integrity tests for the pinned RRQ PR experiment path."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

from lp_llm.rrq_pr2308 import (PR_COMMIT, config_for, experiment_identity,
                                pr03_command, run_pr03, source_audit)
from lp_llm.rrq_pr2308_compare import (compare_checkpoints, materialize_qdq,
                                        same_hf_contents, unpack_int2, validate_packed_base,
                                        validate_residual, run_pr04)
from lp_llm.rrq_pr2308_parity import run_parity
from lp_llm.storage import atomic_json


NAME = "model.layers.0.self_attn.q_proj"


def _checkpoint(directory: Path, tensors: dict, *, packed=False, residual=False):
    directory.mkdir(parents=True)
    config = {"model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36,
              "vocab_size": 151936, "intermediate_size": 12288,
              "num_attention_heads": 32, "num_key_value_heads": 8}
    if packed:
        config["quantization_config"] = {"bits": 2, "group_size": 128, "sym": True}
    if residual:
        config["quantization_config"] = {"packing_format": "auto_round:rrq",
                                         "total_planes": 2, "group_size": 128, "sym": True}
    (directory / "config.json").write_text(json.dumps(config))
    save_file(tensors, directory / "model.safetensors")
    atomic_json(directory / "model.safetensors.index.json",
                {"metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
                 "weight_map": {n: "model.safetensors" for n in tensors}})
    return directory


def _fixture(tmp_path):
    original = torch.arange(16 * 128, dtype=torch.float32).reshape(16, 128).sin().to(torch.bfloat16)
    embeddings = torch.randn(8, 16).to(torch.bfloat16)
    source = _checkpoint(tmp_path / "original", {f"{NAME}.weight": original,
                                                  "model.embed_tokens.weight": embeddings})
    codes = (torch.arange(128) % 4).to(torch.int32)
    shifts = torch.arange(16, dtype=torch.int32) * 2
    words = ((codes.reshape(8, 16) << shifts).sum(1)).reshape(8, 1).expand(8, 16).contiguous().to(torch.int32)
    zeros = torch.tensor([[int(sum(1 << int(s) for s in shifts))]], dtype=torch.int32)
    scales = torch.full((1, 16), .25, dtype=torch.float16)
    base = _checkpoint(tmp_path / "base", {f"{NAME}.qweight": words,
                                            f"{NAME}.scales": scales,
                                            f"{NAME}.qzeros": zeros,
                                            "model.embed_tokens.weight": embeddings.clone()}, packed=True)
    residual = _checkpoint(tmp_path / "residual", {f"{NAME}.qweight_1": words.clone(),
                                                    f"{NAME}.scales_1": scales.clone(),
                                                    f"{NAME}.qzeros_1": zeros.clone()}, residual=True)
    return source, base, residual, original, embeddings


def test_config_and_command_are_explicit():
    config = config_for("pr03")
    assert config["iterations"] == 0 and config["residual_planes"] == 1
    assert config["calibration"]["nsamples"] == 128
    cmd = pr03_command(Path("/model"), Path("/out"), Path("/runtime"), "python")
    script = cmd[2]
    assert script.count(".quantize_and_save(") == 1
    assert ".quantize()" not in script
    for keyword in ("iters=0", "num_residual_planes=1", "nsamples=128", "batch_size=8",
                    "model_dtype=\"float16\"", "low_gpu_mem_usage=True"):
        assert keyword in script


def test_checkpoint_content_identity_ignores_runtime_only_fields():
    left = {"checkpoint": "/model", "config_sha256": "a", "index_sha256": "b",
            "shard_sha256": {"x.safetensors": "c"}, "sidecar_sha256": {"tokenizer.json": "d"},
            "torch": "old"}
    right = {**left, "torch": "new", "transformers": "new"}
    assert same_hf_contents(left, right)
    right["shard_sha256"] = {"x.safetensors": "changed"}
    assert not same_hf_contents(left, right)


def test_aggregate_marks_attempted_incomplete_and_keeps_old_results(tmp_path):
    from lp_llm.rrq_pr2308_aggregate import collect
    root = tmp_path / "results"
    old = root / "rrq_w2_reproduction/j05/full.json"
    old.parent.mkdir(parents=True)
    old.write_text('{"perplexity": 65.62}')
    attempted = root / "rrq_pr2308/pr03"
    attempted.mkdir(parents=True)
    (attempted / "run.json").write_text('{"status": "running"}')
    rows = {row["id"]: row for row in collect(root)["experiments"]}
    assert rows["pr03"]["status"] == "running"
    assert rows["pr04"]["status"] == "not_started"
    assert old.read_text() == '{"perplexity": 65.62}'


def test_unpack_int2_and_qdq_nonquant_preservation(tmp_path):
    source, base, residual, _, embeddings = _fixture(tmp_path)
    info = validate_packed_base(base, source)
    assert info["quantized_module_count"] == 1
    assert validate_residual(residual, info)["tensor_count"] == 3
    with __import__("safetensors").safe_open(base / "model.safetensors", framework="pt", device="cpu") as handle:
        decoded, codes = unpack_int2(handle.get_tensor(f"{NAME}.qweight"),
                                     handle.get_tensor(f"{NAME}.scales"),
                                     handle.get_tensor(f"{NAME}.qzeros"),
                                     in_features=128, out_features=16)
    assert torch.equal(codes[0], (torch.arange(128) % 4).to(torch.int8))
    assert torch.equal(decoded[0], ((torch.arange(128) % 4) - 2) * .25)
    destination = tmp_path / "qdq"
    result = materialize_qdq(base, source, destination)
    assert result["nonquant_tensors_copied_exactly"] == 1
    from lp_llm.rrq_pr2308_compare import _get, _weight_map
    mapping = _weight_map(destination)
    assert torch.equal(_get(destination, mapping, "model.embed_tokens.weight"), embeddings)
    assert torch.equal(_get(destination, mapping, f"{NAME}.weight"), decoded.to(torch.float16))
    with pytest.raises(ValueError, match="overwrite"):
        materialize_qdq(base, source, destination)


def test_corruption_rejected(tmp_path):
    source, base, residual, *_ = _fixture(tmp_path)
    (residual / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="shards"):
        validate_residual(residual, validate_packed_base(base, source))
    with __import__("safetensors").safe_open(base / "model.safetensors", framework="pt", device="cpu") as handle:
        tensors = {n: handle.get_tensor(n).clone() for n in handle.keys()}
    tensors[f"{NAME}.scales"][0, 0] = float("nan")
    save_file(tensors, base / "model.safetensors")
    with pytest.raises(ValueError, match="Nonfinite"):
        validate_packed_base(base, source)


def test_interrupted_qdq_shard_recovery_is_exact(tmp_path):
    from lp_llm.storage import sha256
    source, base, _, *_ = _fixture(tmp_path)
    destination = tmp_path / "qdq"
    original = materialize_qdq(base, source, destination)
    shards = sorted(destination.glob("*.safetensors"))
    preserved = sha256(shards[0])
    shards[1].unlink()
    (destination / "model.safetensors.index.json").unlink()
    recovered = materialize_qdq(base, source, destination, resume=True)
    assert recovered == original
    assert sha256(shards[0]) == preserved
    with __import__("safetensors").safe_open(shards[0], framework="pt", device="cpu") as handle:
        wrong = {name: handle.get_tensor(name).clone() + 1 for name in handle.keys()}
    save_file(wrong, shards[0])
    with pytest.raises(ValueError, match="differs from expected decode"):
        materialize_qdq(base, source, destination, resume=True)


def test_resume_uses_exact_marker_and_never_requantizes(tmp_path):
    source, base, residual, *_ = _fixture(tmp_path)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    result = tmp_path / "results/rrq_pr2308/pr03"
    from lp_llm.evaluate import _identity
    atomic_json(tmp_path / "results/rrq_w2_reproduction/j05/run.json",
                {"revision": config_for("pr03")["model_revision"],
                 "source_checkpoint": _identity(source, "rrq-source")})
    (result / "export").mkdir(parents=True)
    base.rename(result / "export/base")
    residual.rename(result / "export/residual")
    config = config_for("pr03")
    identity = experiment_identity(config, source, checkout)
    base_info = validate_packed_base(result / "export/base", source)
    residual_info = validate_residual(result / "export/residual", base_info)
    atomic_json(result / "run.json", {"identity": identity, "status": "running"})
    marker = {"status": "completed_validated", "identity": identity,
              "checkpoint": base_info, "residual": residual_info}
    atomic_json(result / "quantization_complete.json", marker)
    with patch("lp_llm.rrq_pr2308.source_audit", return_value={}), \
         patch("lp_llm.rrq_pr2308.run_logged", side_effect=AssertionError("quantizer launched")):
        assert run_pr03(config, source, checkout, result, python="python") == marker
        wrong = dict(marker)
        wrong["identity"] = {"wrong": True}
        atomic_json(result / "quantization_complete.json", wrong)
        with pytest.raises(ValueError, match="identity"):
            run_pr03(config, source, checkout, result, python="python")


def test_unmarked_export_is_not_overwritten(tmp_path):
    source, base, residual, *_ = _fixture(tmp_path)
    result = tmp_path / "results/rrq_pr2308/pr03"
    from lp_llm.evaluate import _identity
    atomic_json(tmp_path / "results/rrq_w2_reproduction/j05/run.json",
                {"revision": config_for("pr03")["model_revision"],
                 "source_checkpoint": _identity(source, "rrq-source")})
    (result / "export").mkdir(parents=True)
    (result / "export" / "partial").write_text("incomplete")
    config = config_for("pr03")
    identity = experiment_identity(config, source, tmp_path)
    atomic_json(result / "run.json", {"identity": identity, "status": "running"})
    with patch("lp_llm.rrq_pr2308.source_audit", return_value={}), \
         patch("lp_llm.rrq_pr2308.run_logged", side_effect=AssertionError("quantizer launched")):
        with pytest.raises(ValueError, match="Unmarked"):
            run_pr03(config, source, tmp_path, result, python="python")


def test_compare_complete_qdq_and_nonquant_coverage(tmp_path):
    source, base, _, *_ = _fixture(tmp_path)
    qdq = tmp_path / "qdq"
    materialize_qdq(base, source, qdq)
    report = compare_checkpoints(source, qdq, qdq)
    assert report["quantized_tensor_count"] == 1
    assert report["overall_weighted_mse_j05_vs_pr"] == 0
    assert report["nonquant_pr_exact_count"] == 1


def test_pinned_source_audit_and_cpu_parity_when_available(tmp_path):
    checkout = Path(__file__).resolve().parents[1] / "outputs" / "rrq_pr2308_source"
    if not checkout.is_dir():
        pytest.skip("Pinned source checkout is optional for ordinary repository tests")
    assert source_audit(checkout)["revision"] == PR_COMMIT
    parity = run_parity(checkout, tmp_path)
    assert all(case["weighted_codes_exact"] and case["weighted_scales_exact"]
               for case in parity["cases"])
    assert all(case["weighted_qdq_exact_fraction"] == 1 for case in parity["cases"])
    from lp_llm.rrq_pr2308_parity import _extract
    from lp_llm.rrq_pr2308_compare import _get, _weight_map
    source, base, _, *_ = _fixture(tmp_path / "packing")
    mapping = _weight_map(base)
    qweight = _get(base, mapping, f"{NAME}.qweight")
    scales = _get(base, mapping, f"{NAME}.scales")
    qzeros = _get(base, mapping, f"{NAME}.qzeros")
    namespace = {"torch": torch, "nn": torch.nn,
                 "get_packing_device": lambda *_: torch.device("cpu"),
                 "transformers": __import__("types").SimpleNamespace(
                     pytorch_utils=__import__("types").SimpleNamespace(Conv1D=type("Conv1D", (), {})))}
    official_dequant = _extract(checkout / "auto_round_extension/torch/qlinear_torch_zp.py",
                                "_dequantize", namespace, class_name="QuantLinear")
    official_pack = _extract(checkout / "auto_round_extension/torch/qlinear_torch_zp.py",
                             "pack_248_bits", namespace, class_name="QuantLinear")
    from types import SimpleNamespace
    official = official_dequant(SimpleNamespace(qweight=qweight, scales=scales, qzeros=qzeros,
                                                bits=2, group_size=128, maxq=3, infeatures=128,
                                                wf=(torch.arange(16, dtype=torch.int32) * 2).unsqueeze(0),
                                                dequant_dtype=torch.int8, use_generic_bit_packing=False))
    decoded, _ = unpack_int2(qweight, scales, qzeros, in_features=128, out_features=16)
    assert torch.equal(official.T, decoded.to(official.dtype))
    # Official scalar-zero pack path needs at least 32 output columns.
    linear = torch.nn.Linear(128, 32, bias=False)
    linear.weight.data.copy_(torch.cat([decoded, decoded], dim=0))
    packed = SimpleNamespace(bits=2, maxq=3, group_size=128, bias=None,
                             scales=torch.empty(1, 32, dtype=torch.float16), qweight=None, qzeros=None)
    official_pack(packed, linear, torch.cat([scales, scales], dim=1).T.contiguous(), 2, device="cpu")
    assert torch.equal(packed.qweight, torch.cat([qweight, qweight], dim=1))
    assert torch.equal(packed.qzeros, torch.cat([qzeros, qzeros], dim=1))


def test_eval_failure_resumes_without_rematerialization(tmp_path):
    from lp_llm.evaluate import _identity
    from lp_llm.rrq_autoround import validate_output
    from lp_llm.rrq_pr2308 import J05_RELATIVE, RESULT_SUBDIR, TOKEN_SHA256
    from lp_llm.rrq_w2 import AUTOROUND_COMMIT, load_config

    source, base, residual, *_ = _fixture(tmp_path)
    result_root = tmp_path / "results"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    pr03 = result_root / RESULT_SUBDIR / "pr03"
    (pr03 / "export").mkdir(parents=True)
    base.rename(pr03 / "export/base")
    residual.rename(pr03 / "export/residual")
    packed = validate_packed_base(pr03 / "export/base", source)
    atomic_json(pr03 / "quantization_complete.json",
                {"status": "completed_validated",
                 "identity": experiment_identity(config_for("pr03"), source, checkout),
                 "checkpoint": packed})
    j05 = result_root / J05_RELATIVE
    materialize_qdq(pr03 / "export/base", source, j05)
    j05_info = validate_output(j05, source)
    atomic_json(j05.parents[1] / "encoding_complete.json",
                {"status": "complete", "identity": {
                    "config_sha256": load_config("j05")["config_sha256"],
                    "autoround_commit": AUTOROUND_COMMIT,
                    "source": _identity(source, "rrq-original")},
                 "checkpoint": j05_info})
    atomic_json(j05.parents[1] / "full.json",
                {"status": "complete", "identity": {
                    "matgptq_commit": config_for("pr04")["matgptq_commit"]},
                 "score": {"token_sha256": TOKEN_SHA256, "perplexity": 65.62},
                 "inference_dtype": "float16",
                 "checkpoint_identity": _identity(j05, "rrq-result")})
    corpus = {"token_sha256": TOKEN_SHA256, "scored_tokens": 2047}
    with patch("lp_llm.rrq_pr2308.source_audit", return_value={}), \
         patch("lp_llm.rrq_pr2308_compare.check_revision"), \
         patch("lp_llm.rrq_pr2308_compare.corpus_identity",
               return_value={"token_sha256": "wrong", "scored_tokens": 2047}), \
         patch("lp_llm.rrq_pr2308_compare.evaluate_matgptq",
               side_effect=AssertionError("evaluator launched on wrong tokens")):
        with pytest.raises(ValueError, match="tokenizer fingerprint"):
            run_pr04(config_for("pr04"), source, checkout, tmp_path / "MatGPTQ",
                     result_root, python="python")
    with patch("lp_llm.rrq_pr2308.source_audit", return_value={}), \
         patch("lp_llm.rrq_pr2308_compare.check_revision"), \
         patch("lp_llm.rrq_pr2308_compare.corpus_identity", return_value=corpus), \
         patch("lp_llm.rrq_pr2308_compare.evaluate_matgptq", side_effect=RuntimeError("evaluator failed")):
        with pytest.raises(RuntimeError, match="evaluator failed"):
            run_pr04(config_for("pr04"), source, checkout, tmp_path / "MatGPTQ",
                     result_root, python="python")
    from lp_llm.rrq_pr2308_aggregate import collect
    assert json.loads((result_root / RESULT_SUBDIR / "pr04/failure.json").read_text())["stage"] == "evaluation"
    assert collect(result_root)["experiments"][3]["status"] == "failed"
    qdq = result_root / RESULT_SUBDIR / "pr04/qdq_checkpoint"
    assert qdq.is_dir()
    with patch("lp_llm.rrq_pr2308.source_audit", return_value={}), \
         patch("lp_llm.rrq_pr2308_compare.check_revision"), \
         patch("lp_llm.rrq_pr2308_compare.corpus_identity", return_value=corpus), \
         patch("lp_llm.rrq_pr2308_compare.materialize_qdq", side_effect=AssertionError("rematerialized")), \
         patch("lp_llm.rrq_pr2308_compare.evaluate_matgptq", return_value={"perplexity": 65.62}) as evaluator:
        completed = run_pr04(config_for("pr04"), source, checkout, tmp_path / "MatGPTQ",
                             result_root, python="python")
    assert completed["status"] == "completed_validated"
    evaluator.assert_called_once()
    with patch("lp_llm.rrq_pr2308.source_audit", return_value={}), \
         patch("lp_llm.rrq_pr2308_compare.check_revision"), \
         patch("lp_llm.rrq_pr2308_compare.evaluate_matgptq", side_effect=AssertionError("reevaluated")):
        assert run_pr04(config_for("pr04"), source, checkout, tmp_path / "MatGPTQ",
                        result_root, python="python") == completed
