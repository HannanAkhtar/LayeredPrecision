import json
import math
import os
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from lp_llm.encode import encode
from lp_llm.model_adapter import expected_qwen3
from lp_llm.reconstruct import reconstruct
from lp_llm.rrq_aggregate import aggregate as report
from lp_llm.rrq_autoround import (command as autoround_command, compare_coverage, configured_modules,
                                  verify_upstream_recipe,
                                  validate_output, run as autoround_run)
from lp_llm.rrq_hybrid import build, selected_names
from lp_llm.rrq_reference_eval import (aggregate, blocks, check_revision, command,
                                       local_score, parse_perplexity)
from lp_llm.rrq_process import run_logged
from lp_llm.rrq_runtime import probe, source_environment
from lp_llm.rrq_overlay import planned_packages, install as install_overlay
from lp_llm.rrq_w2 import (AUTOROUND_COMMIT, AUTOROUND_HISTORICAL_COMMIT,
                           MODES, completion_matches, load_config)


def test_all_explicit_mappings_and_upstream_commands(tmp_path):
    assert len(MODES) == 14
    for job, (kind, mode, group) in MODES.items():
        config = load_config(job)
        assert (config["kind"], config["mode"], config["group_size"]) == (kind, mode, group)
        assert config["bits"] == 2 and config["symmetric"] is True
        if kind == "autoround":
            args = autoround_command(config, Path("/model"), Path("/out"))
            assert args[0] == sys.executable and args[1] == "-c"
            function = "run_rtn" if mode == "plain" else "run_opt_rtn"
            assert args[2] == f"from auto_round.cli.main import {function}; {function}()"
            assert args[args.index("--scheme") + 1] == ("W2A16" if group == 128 else f"W2A16G{group}")
            assert args[args.index("--format") + 1] == "fake"
            assert args[args.index("--iters") + 1] == "0"
            assert args[args.index("--nsamples") + 1] == ("1" if mode == "plain" else "128")
            assert args[args.index("--model_dtype") + 1] == "float16"
            assert config["quantizer_configuration"]["configured_nsamples"] == (1 if mode == "plain" else 128)
            assert config["quantizer_configuration"]["observed_calibration_examples_consumed"] is None
    args = command(Path("/upstream"), Path("/checkpoint"))
    assert "--quant_weights_path" not in args
    assert "--use_fast_tokenizer" not in args
    assert args[args.index("--sequence_length") + 1] == "2048"
    assert load_config("j08")["availability"] == "blocked"


def test_source_pythonpath_and_actual_import_resolution(tmp_path):
    checkout = tmp_path / "auto-round"
    overlay = tmp_path / "extra_deps"
    module = checkout / "auto_round"
    module.mkdir(parents=True)
    (module / "__init__.py").write_text("__version__ = 'test-source'\n")
    overlay.mkdir()
    env = source_environment(checkout, overlay, base={"PYTHONPATH": "existing"})
    assert env["PYTHONPATH"].split(os.pathsep) == [str(checkout.resolve()),
                                                     str(overlay.resolve()), "existing"]
    runtime = probe(sys.executable, ["auto_round"], env=source_environment(checkout, overlay))
    assert Path(runtime["modules"]["auto_round"]["file"]).resolve().is_relative_to(checkout)
    assert runtime["modules"]["auto_round"]["version"] == "test-source"


def test_source_based_autoround_help_invocation(tmp_path):
    checkout = tmp_path / "upstream"
    cli = checkout / "auto_round/cli"
    cli.mkdir(parents=True)
    (cli.parent / "__init__.py").write_text("")
    (cli / "__init__.py").write_text("")
    (cli / "main.py").write_text("import sys\n"
                                "def run_rtn(): print('plain', sys.argv[1:])\n"
                                "def run_opt_rtn(): print('optimized', sys.argv[1:])\n")
    for mode, expected in (("plain", "plain"), ("optimized", "optimized")):
        args = autoround_command(load_config("j04" if mode == "plain" else "j05"),
                                 Path("/model"), Path("/out"))
        result = run_logged(args[:3] + ["--help"], tmp_path / f"{mode}.log",
                            env=source_environment(checkout))
        assert result["returncode"] == 0
        assert expected in (tmp_path / f"{mode}.log").read_text()


def test_overlay_plans_only_missing_supplemental_packages(monkeypatch):
    import lp_llm.rrq_overlay as overlay
    present = {"filelock", "requests", "typing_extensions", "charset_normalizer",
               "idna", "urllib3", "certifi"}
    monkeypatch.setattr(overlay, "available", lambda module: module in present)
    pins = planned_packages()
    assert "pydantic==2.12.5" in pins and "pydantic-core==2.41.5" in pins
    assert "py-cpuinfo==9.0.0" in pins
    assert not any(pin.startswith(("torch", "transformers", "numpy", "requests", "filelock")) for pin in pins)


def test_overlay_install_is_targeted_atomic_and_idempotent(tmp_path, monkeypatch):
    import lp_llm.rrq_overlay as overlay
    commands = []
    core_version = ["2.41.5"]
    monkeypatch.setattr(overlay, "planned_packages", lambda: ["pydantic==2.12.5", "pydantic-core==2.41.5"])
    def fake_run(command, **kwargs):
        commands.append(command)
        return type("Done", (), {"returncode": 0, "stderr": ""})()
    monkeypatch.setattr(overlay.subprocess, "run", fake_run)
    def fake_probe(_python, modules, *, env):
        root = Path(env["PYTHONPATH"].split(os.pathsep)[0])
        return {"modules": {name: {"file": str(root / name / "__init__.py"),
                                    "version": "2.12.5" if name == "pydantic" else
                                    core_version[0] if name == "pydantic_core" else "unknown"}
                            for name in modules}}
    monkeypatch.setattr(overlay, "probe", fake_probe)
    target = tmp_path / "extra_deps"
    assert install_overlay(target)["status"] == "complete"
    assert (target / "OVERLAY_COMPLETE.json").exists()
    assert "--target" in commands[0] and "--no-deps" in commands[0]
    assert "--only-binary=:all:" in commands[0]
    assert install_overlay(target)["status"] == "already_complete"
    assert len(commands) == 1
    core_version[0] = "2.41.4"
    with pytest.raises(ValueError, match="did not resolve"):
        install_overlay(target)
    core_version[0] = "2.41.5"
    (target / "OVERLAY_COMPLETE.json").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        install_overlay(target)


def test_reference_blocks_loss_weighting_and_printed_result():
    tokens = torch.arange(22).reshape(1, 22)
    assert [b.tolist()[0] for b in blocks(tokens, 8)] == [list(range(8)), list(range(8, 16))]
    score = aggregate([1.0, 3.0], 8)
    assert score["scored_tokens"] == 14 and score["cross_entropy"] == 2.0
    assert score["perplexity"] == pytest.approx(math.exp(2))
    assert parse_perplexity("Perplexities\nwikitext2: 7.01\n") == 7.01

    class FakeModel:
        def __call__(self, inputs):
            logits = torch.zeros((*inputs.shape, 32))
            logits[:, :, 0] = inputs.float() / 10
            return type("Output", (), {"logits": logits})()

    actual = local_score(FakeModel(), blocks(tokens, 8), "cpu")
    losses = []
    for block in blocks(tokens, 8):
        logits = FakeModel()(block).logits
        losses.append(float(torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, 32), block[:, 1:].reshape(-1))))
    expected = aggregate(losses, 8)
    assert actual["scored_tokens"] == expected["scored_tokens"]
    assert actual["cross_entropy"] == pytest.approx(expected["cross_entropy"])
    assert actual["perplexity"] == pytest.approx(expected["perplexity"])


def test_revision_and_resume_reject_wrong_identity(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "bad\n")
    with pytest.raises(ValueError, match="expected"):
        check_revision(tmp_path)
    path = tmp_path / "complete.json"
    path.write_text(json.dumps({"status": "complete", "identity": {"x": 1}}))
    assert completion_matches(path, {"x": 1})
    with pytest.raises(ValueError, match="another identity"):
        completion_matches(path, {"x": 2})


def test_official_output_requires_valid_qdq_and_can_index_single_shard(tmp_path):
    with pytest.raises(ValueError, match="did not produce"):
        validate_output(tmp_path)
    (tmp_path / "config.json").write_text('{"model_type":"qwen3"}')
    save_file({"x.weight": torch.ones(2, 2)}, str(tmp_path / "model.safetensors"))
    value = validate_output(tmp_path)
    assert value["tensor_names"] == ["x.weight"]
    assert (tmp_path / "model.safetensors.index.json").exists()


def test_official_coverage_uses_actual_weight_differences(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    original = {"a.weight": torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
                "b.weight": torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16)}
    save_file(original, str(source / "model.safetensors"))
    save_file({"a.weight": original["a.weight"].half(),
               "b.weight": torch.tensor([[3.0, 5.0]], dtype=torch.float16)},
              str(output / "model.safetensors"))
    index = {"weight_map": {name: "model.safetensors" for name in original}}
    for path in (source, output):
        (path / "model.safetensors.index.json").write_text(json.dumps(index))
    coverage = compare_coverage(source, output)
    assert coverage["changed_tensor_names"] == ["b.weight"]
    assert coverage["unchanged_tensor_names"] == ["a.weight"]


def test_upstream_module_metadata_is_distinct_from_weight_change(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {
        "layer_config": {"a": {"bits": 2}, "b": {"bits": 16}}}}))
    selection = configured_modules(tmp_path)
    assert selection["selected_module_names"] == ["a"]
    assert selection["excluded_module_names"] == ["b"]
    (tmp_path / "config.json").write_text("{}")
    assert configured_modules(tmp_path)["selected_module_names"] is None


def test_current_and_preprint_candidate_are_distinct_pins():
    assert AUTOROUND_COMMIT != AUTOROUND_HISTORICAL_COMMIT
    assert AUTOROUND_HISTORICAL_COMMIT == "24bfce00bb3bd053498232f0e8395118fe10a906"


def test_upstream_recipe_dispatch_and_defaults_are_checked(tmp_path):
    cli = tmp_path / "auto_round/cli/main.py"
    cli.parent.mkdir(parents=True)
    cli.write_text("RECIPES = {'rtn': {'iters': 0, 'nsamples': 1, 'seqlen': 2048, "
                   "'disable_opt_rtn': True}, 'opt_rtn': {'iters': 0, 'nsamples': 128, "
                   "'seqlen': 2048, 'disable_opt_rtn': False}}\n"
                   "def run_rtn(): start('rtn')\n"
                   "def run_opt_rtn(): start('opt_rtn')\n")
    assert verify_upstream_recipe(tmp_path, load_config("j04"))["recipe_name"] == "rtn"
    assert verify_upstream_recipe(tmp_path, load_config("j05"))["recipe_name"] == "opt_rtn"
    cli.write_text(cli.read_text().replace("'nsamples': 128", "'nsamples': 64"))
    with pytest.raises(ValueError, match="differs"):
        verify_upstream_recipe(tmp_path, load_config("j05"))


def test_official_runner_refuses_to_replace_partial_checkpoint(tmp_path, monkeypatch):
    import lp_llm.rrq_autoround as upstream
    monkeypatch.setattr(upstream, "check_revision", lambda *args: None)
    output = tmp_path / "output"
    output.mkdir()
    (output / "partial").write_text("interrupted")
    with pytest.raises(ValueError, match="overwrite"):
        autoround_run(load_config("j04"), tmp_path / "source", output,
                      tmp_path / "upstream", tmp_path / "log")


def test_missing_autoround_dependency_fails_before_quantization(tmp_path, monkeypatch):
    import lp_llm.rrq_autoround as upstream
    monkeypatch.setattr(upstream, "check_revision", lambda *args: None)
    monkeypatch.setattr(upstream, "verify_upstream_recipe", lambda *args: {"recipe_name": "rtn"})
    monkeypatch.setattr(upstream, "probe", lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("No module named pydantic")))
    monkeypatch.setattr(upstream, "run_logged", lambda *args, **kwargs: pytest.fail("quantization launched"))
    with pytest.raises(RuntimeError, match="pydantic"):
        autoround_run(load_config("j04"), tmp_path / "source", tmp_path / "output",
                      tmp_path / "upstream", tmp_path / "log")


def test_incomplete_overlay_fails_before_autoround_import(tmp_path, monkeypatch):
    import lp_llm.rrq_autoround as upstream
    overlay = tmp_path / "extra_deps"
    overlay.mkdir()
    monkeypatch.setattr(upstream, "check_revision", lambda *args: None)
    monkeypatch.setattr(upstream, "verify_upstream_recipe", lambda *args: {"recipe_name": "rtn"})
    monkeypatch.setattr(upstream, "probe", lambda *args, **kwargs: pytest.fail("import probe started"))
    with pytest.raises(ValueError, match="Incomplete"):
        autoround_run(load_config("j04"), tmp_path / "source", tmp_path / "output",
                      tmp_path / "upstream", tmp_path / "log", overlay=overlay)


def test_unavailable_recipe_fails_before_any_model_access(tmp_path):
    from lp_llm.rrq_runner import run
    with pytest.raises(RuntimeError, match="J08|j08"):
        run("j08", tmp_path / "missing", tmp_path, tmp_path, tmp_path)


def test_upstream_process_log_and_exit_status(tmp_path):
    log = tmp_path / "upstream.log"
    success = run_logged([sys.executable, "-c", "print('finished')"], log)
    assert success["returncode"] == 0 and "finished" in log.read_text()
    failure = run_logged([sys.executable, "-c", "raise SystemExit(7)"], log)
    assert failure["returncode"] == 7


def test_missing_official_dataset_fails_clearly(tmp_path, monkeypatch):
    import types
    import transformers
    from lp_llm.rrq_reference_eval import corpus_identity
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("dataset cache absent")
    stub = types.ModuleType("datasets")
    stub.load_dataset = unavailable
    monkeypatch.setitem(sys.modules, "datasets", stub)
    with pytest.raises(RuntimeError, match="WikiText-2 raw test dataset is unavailable"):
        corpus_identity(tmp_path)


def _tiny_source(path):
    config = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
              "num_hidden_layers": 3, "hidden_size": 8, "intermediate_size": 16,
              "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 8,
              "vocab_size": 16, "tie_word_embeddings": False, "torch_dtype": "bfloat16"}
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config))
    tensors = {name: torch.randn(shape).bfloat16() for name, shape in expected_qwen3(config).items()}
    save_file(tensors, str(path / "model.safetensors"))
    (path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {name: "model.safetensors" for name in tensors}}))
    return tensors


@pytest.mark.parametrize("mode,expected", [
    ("q_proj,k_proj", {"q_proj", "k_proj"}), ("v_proj,o_proj", {"v_proj", "o_proj"}),
    ("gate_proj,up_proj", {"gate_proj", "up_proj"}), ("down_proj", {"down_proj"})])
def test_projection_selection(mode, expected):
    manifest = {"tensors": [{"name": f"model.layers.{i}.self_attn.{p}.weight", "eligible": True}
                            for i in range(3) for p in ("q_proj", "k_proj", "v_proj", "o_proj")] +
                           [{"name": f"model.layers.{i}.mlp.{p}.weight", "eligible": True}
                            for i in range(3) for p in ("gate_proj", "up_proj", "down_proj")]}
    names = selected_names(manifest, mode)
    assert {name.rsplit(".", 2)[-2] for name in names} == expected
    assert len(names) == len(expected) * 3
    assert {int(name.split(".")[2]) for name in selected_names(manifest, "first_third")} == {0}
    assert {int(name.split(".")[2]) for name in selected_names(manifest, "final_third")} == {2}


def test_hybrid_restores_exact_original_and_resumes(tmp_path):
    torch.manual_seed(11)
    source = tmp_path / "source"
    originals = _tiny_source(source)
    artifact = tmp_path / "artifact"
    manifest = encode(source, artifact, {"experiment_id": "b3_opt_fp16", "method": "lp",
                                         "bits": [2], "granularity": "group", "group_size": 128,
                                         "quantizer_family": "autoround_opt_fp16", "compute_device": "cpu"})
    base = tmp_path / "base"
    reconstruct(artifact, base, 1, checkpoint_dtype="float16")
    hybrid = tmp_path / "hybrid"
    summary = build(source, artifact, base, hybrid, "q_proj,k_proj")
    assert build(source, artifact, base, hybrid, "q_proj,k_proj") == summary
    selected = selected_names(manifest, "q_proj,k_proj")
    index = json.loads((hybrid / "model.safetensors.index.json").read_text())["weight_map"]
    base_index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    for name in index:
        actual = load_file(str(hybrid / index[name]))[name]
        expected = originals[name] if name in selected else load_file(str(base / base_index[name]))[name]
        assert torch.equal(actual, expected)
    restored = sum(originals[name].numel() for name in selected)
    assert summary["restored_parameters"] == restored
    assert summary["remaining_2bit_parameters"] == manifest["eligible_parameters"] - restored
    assert summary["estimated_logical_compact_bytes"] == (
        json.loads((artifact / "storage_report.json").read_text())["prefixes"][0]["total_physical_bytes"]
        - summary["removed_B3_stream_bytes"] + summary["restored_safetensors_bytes"]
        + summary["compact_manifest_bytes"])
    assert summary["independently_deployable_compressed_bytes"] is None
    assert summary["actual_qdq_checkpoint_safetensors_bytes"] > 0
    damaged = hybrid / index[next(iter(selected))]
    damaged.write_bytes(b"bad")
    with pytest.raises(ValueError, match="corrupt hybrid"):
        build(source, artifact, base, hybrid, "q_proj,k_proj")


def test_hybrid_recovers_only_exact_orphaned_shard(tmp_path, monkeypatch):
    import lp_llm.rrq_hybrid as hybrid_module
    torch.manual_seed(18)
    source = tmp_path / "source"
    originals = _tiny_source(source)
    artifact = tmp_path / "artifact"
    manifest = encode(source, artifact, {"experiment_id": "b3_opt_fp16", "method": "lp",
                                         "bits": [2], "granularity": "group", "group_size": 128,
                                         "quantizer_family": "autoround_opt_fp16", "compute_device": "cpu"})
    base = tmp_path / "base"
    reconstruct(artifact, base, 1, checkpoint_dtype="float16")
    output = tmp_path / "hybrid"
    selected = sorted(selected_names(manifest, "q_proj,k_proj"))[0]
    filename = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"][selected]
    marker = output / (filename + ".done.json")
    actual_atomic = hybrid_module.atomic_json
    with monkeypatch.context() as patch:
        def interrupt(path, value):
            if path == marker:
                raise RuntimeError("interrupted before marker")
            return actual_atomic(path, value)
        patch.setattr(hybrid_module, "atomic_json", interrupt)
        with pytest.raises(RuntimeError, match="interrupted"):
            build(source, artifact, base, output, "q_proj,k_proj")
    shard = output / filename
    assert shard.exists() and not marker.exists()
    digest = shard.read_bytes()
    build(source, artifact, base, output, "q_proj,k_proj")
    assert marker.exists() and shard.read_bytes() == digest
    assert torch.equal(load_file(str(shard))[selected], originals[selected])
    marker.unlink()
    shard.write_bytes(b"bad")
    with pytest.raises(Exception, match="header|Orphaned|safetensor"):
        build(source, artifact, base, output, "q_proj,k_proj")
    assert shard.read_bytes() == b"bad" and not marker.exists()


@pytest.mark.parametrize("first,second", [("screen", "full"), ("full", "screen")])
@pytest.mark.parametrize("job_id", ["j01", "j04"])
def test_full_jobs_have_one_canonical_stage_and_resume(tmp_path, monkeypatch, first, second, job_id):
    import lp_llm.rrq_runner as runner
    counter = {"eval": 0}
    monkeypatch.setattr(runner, "identity", lambda config, source, checkpoint=None, **kw: {
        "id": config["id"], "checkpoint": str(checkpoint) if checkpoint else None,
        "variant": kw.get("autoround_variant")})
    monkeypatch.setattr(runner, "check_revision", lambda *args: None)
    monkeypatch.setattr(runner, "_checkpoint", lambda *args: (tmp_path / "checkpoint", {}))
    monkeypatch.setattr(runner, "_identity", lambda *args: {"stable": True})
    monkeypatch.setattr(runner, "_checkpoint_dtypes", lambda *args: {"F16": 1})
    monkeypatch.setattr(runner, "_git_revision", lambda: "test")
    monkeypatch.setattr(runner, "corpus_identity", lambda *args: {"scored_tokens": 2047,
                                                                   "token_sha256": "abc"})
    monkeypatch.setattr(runner, "local_evaluate", lambda *args: {"scored_tokens": 2047,
                                                                   "token_sha256": "abc",
                                                                   "perplexity": 7.0,
                                                                   "cross_entropy": 1.94})
    def reference(*args):
        counter["eval"] += 1
        return {"perplexity": 7.0, "cross_entropy": 1.94}
    monkeypatch.setattr(runner, "run_reference", reference)
    one = runner.run(job_id, tmp_path, tmp_path, tmp_path, tmp_path, stage=first)
    two = runner.run(job_id, tmp_path, tmp_path, tmp_path, tmp_path, stage=second)
    assert one == two and one["identity"]["stage"] == "full"
    assert counter["eval"] == 1
    job = tmp_path / "rrq_w2_reproduction" / job_id
    assert (job / "full.json").exists() and not (job / "diagnostic.json").exists()
    if job_id == "j04":
        historical = runner.run(job_id, tmp_path, tmp_path, tmp_path, tmp_path,
                                stage="screen", autoround_variant="preprint_candidate")
        assert historical["identity"]["variant"] == "preprint_candidate"
        assert historical["configuration"]["quantizer_configuration"]["revision"] == AUTOROUND_HISTORICAL_COMMIT
        assert (job / "preprint_candidate" / "full.json").exists()
        assert counter["eval"] == 2
        assert runner.run(job_id, tmp_path, tmp_path, tmp_path, tmp_path, stage="screen") == one


def test_hybrid_keeps_diagnostic_and_full_identities_separate(tmp_path, monkeypatch):
    import lp_llm.rrq_runner as runner
    import lp_llm.rtn_diagnostic_eval as diagnostic
    monkeypatch.setattr(runner, "identity", lambda config, source, checkpoint=None, **kw: {
        "id": config["id"], "checkpoint": str(checkpoint) if checkpoint else None})
    monkeypatch.setattr(runner, "check_revision", lambda *args: None)
    monkeypatch.setattr(runner, "_checkpoint", lambda *args: (tmp_path / "checkpoint", {}))
    monkeypatch.setattr(runner, "_identity", lambda *args: {"stable": True})
    monkeypatch.setattr(runner, "_checkpoint_dtypes", lambda *args: {"F16": 1})
    monkeypatch.setattr(runner, "_git_revision", lambda: "test")
    monkeypatch.setattr(diagnostic, "evaluate", lambda *args: {
        "perplexity": 8.0, "mean_cross_entropy": 2.0, "scored_tokens": 16376,
        "selected_token_sha256": "diag"})
    monkeypatch.setattr(runner, "corpus_identity", lambda *args: {"scored_tokens": 2047})
    monkeypatch.setattr(runner, "run_reference", lambda *args: {
        "perplexity": 9.0, "cross_entropy": 2.2})
    screen = runner.run("j09", tmp_path, tmp_path, tmp_path, tmp_path, stage="screen")
    full = runner.run("j09", tmp_path, tmp_path, tmp_path, tmp_path, stage="full")
    assert screen["identity"]["stage"] == "screen"
    assert full["identity"]["stage"] == "full"
    assert (tmp_path / "rrq_w2_reproduction/j09/diagnostic.json").exists()
    assert (tmp_path / "rrq_w2_reproduction/j09/full.json").exists()


def test_aggregate_reports_only_measured_values(tmp_path):
    rows = report(tmp_path)["rows"]
    assert len(rows) == 14
    assert rows[0]["full_ppl"] is None
    assert rows[0]["bits"] is None and rows[0]["group"] is None
    assert next(row for row in rows if row["id"] == "j09")["bits"] == "mixed (2/BF16)"
    assert next(row for row in rows if row["id"] == "j08")["status"] == "unavailable"


def test_aggregate_includes_historical_reference_without_relabeling_current(tmp_path):
    path = tmp_path / "rrq_w2_reproduction/j04/preprint_candidate/full.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"status": "complete", "identity": {
        "autoround_variant": "preprint_candidate"},
        "configuration": {"config_sha256": load_config("j04")["config_sha256"]},
        "construction": {}, "score": {"perplexity": 10.0}}))
    rows = report(tmp_path)["rows"]
    assert len(rows) == 15
    current = next(row for row in rows if row["id"] == "j04")
    historical = next(row for row in rows if row["id"] == "j04/preprint_candidate")
    assert current["status"] == "pending" and historical["full_ppl"] == 10.0


def test_slurm_scripts_are_explicit_and_do_not_submit_on_import():
    root = Path(__file__).resolve().parents[1] / "hpc/rrq_w2"
    batch = (root / "run_experiment.sbatch").read_text()
    helper = (root / "submit_selected.sh").read_text()
    preflight = (root / "preflight.sh").read_text()
    setup = (root / "setup_reference_env.sh").read_text()
    assert "set -euo pipefail" in batch and "#SBATCH --mem=12G" in batch
    assert "--account=acc-makhtar" in batch and "--partition=gpu" in batch
    assert "lp_llm.rrq_runner" in batch and "EXPERIMENT_ID" in batch
    assert "j08)" in batch and "exit 1" in batch
    assert "sbatch" in helper and "RRQ_CONCURRENCY" in helper
    assert "sbatch" not in batch
    assert 'RRQ_SETUP_AUTOROUND:-1' in setup and 'RRQ_SETUP_HISTORICAL:-0' in setup
    assert 'j04|j05|j06|j07)' in preflight
    assert 'RRQ_AUTOROUND_MEM:-12G' in helper
    assert "conda create" not in setup and "pip install" not in setup
    assert 'RRQ_PYTHON:-$LP_ROOT/envs/lp_llm/bin/python' in batch
    assert "rrq_matgptq" not in batch and "auto-round-rtn" not in preflight
