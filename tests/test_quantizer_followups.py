from __future__ import annotations

import ast
import json
import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from lp_llm.quantizer_followups import (_grid, _check_all_w2_groups, _read_marker, aggregate, load_config,
                                         validate_checkpoint, validate_resolved,
                                         verify_original_checkpoint, PR_COMMIT, GPTQ_COMMIT)
from lp_llm.rrq_pr2308 import PR_COMMIT as EXISTING_PR_COMMIT
from lp_llm.storage import atomic_json
from lp_llm.rrq_reference_eval import command as matgptq_command


ROOT = Path(__file__).resolve().parents[1]
AUTO = ROOT / "outputs/rrq_pr2308_source/auto_round"
GPTQ = ROOT / "outputs/gptqmodel_v570_source/gptqmodel"
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def test_pinned_configs_and_shared_reference():
    configs = [load_config(id_) for id_ in ("s01", "r01", "r02", "r03")]
    assert PR_COMMIT == EXISTING_PR_COMMIT
    assert all(c["autoround_commit"] == PR_COMMIT for c in configs[:3])
    assert configs[3]["gptqmodel_commit"] == GPTQ_COMMIT
    assert configs[1]["neuqi_coarse"] == configs[2]["neuqi_coarse"] == 256
    assert configs[1]["neuqi_fine"] == configs[2]["neuqi_fine"] == 64
    assert configs[1]["neuqi_backend"] == configs[2]["neuqi_backend"] == "auto"
    assert configs[1]["symmetric"] and not configs[2]["symmetric"]
    assert configs[1]["enable_neuqi"] and configs[2]["enable_neuqi"]
    cmd = matgptq_command(Path("/reference"), Path("/checkpoint"), "python")
    assert cmd[2:] == ["--model_name_or_path", str(Path("/checkpoint")),
                       "--sequence_length", "2048", "--eval_datasets", "wikitext2",
                       "--eval_batch_size", "1", "--dtype", "float16"]
    setup = (ROOT / "hpc/quantizer_followups/setup_gptq_env.sh").read_text()
    runner = (ROOT / "hpc/quantizer_followups/run_experiment.sbatch").read_text()
    extras = (ROOT / "hpc/quantizer_followups/gptq_v570_extras.txt").read_text().splitlines()
    assert "gptq_followups_v570_light" in setup and "gptq_followups_v570_light" in runner
    assert "-m venv --system-site-packages" in setup
    assert "--copy --prefix \"$GPTQ_ENV\" --clone \"$LP_ENV\"" in setup
    assert setup.count("--no-deps") >= 2
    assert not any(entry.startswith(("torch=", "transformers=")) for entry in extras)


def test_gptq_lightweight_dependency_selection_and_marker_gate(tmp_path, monkeypatch):
    import importlib.util
    path = ROOT / "hpc/quantizer_followups/gptq_env_check.py"
    spec = importlib.util.spec_from_file_location("gptq_env_check_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text('[project]\ndependencies = ["torch>=2.8.0", "transformers>=4.57.1", "logbar>=0.2.1", "numpy==2.2.6", "device-smi>=0.5.3"]\n')
    pins = tmp_path / "pins.txt"
    pins.write_text("logbar==0.2.1\nnumpy==2.2.6\ndevice-smi==0.5.3\n")
    versions = {"logbar": "0.2.1", "numpy": "1.26.0", "device-smi": "0.5.2"}
    monkeypatch.setattr(module.metadata, "version", lambda name: versions[name])
    assert module.missing_extras(source, pins) == ["numpy==2.2.6", "device-smi==0.5.3"]
    versions["numpy"] = "2.2.6"
    assert module.missing_extras(source, pins) == ["device-smi==0.5.3"]
    versions["device-smi"] = "0.5.3"
    assert module.missing_extras(source, pins) == []
    pins.write_text("logbar==0.2.1\nnumpy==2.2.6\ndevice-smi==0.5.2\n")
    versions["device-smi"] = "0.5.2"
    with pytest.raises(RuntimeError, match="violates"):
        module.missing_extras(source, pins)
    setup = (ROOT / "hpc/quantizer_followups/setup_gptq_env.sh").read_text()
    assert '[[ -f "$GPTQ_ENV/quantizer_followups_ready.json"' in setup
    assert "[GPTQ ENV RESUME]" in setup and "dist.version == \"5.7.0\"" in setup
    assert '"$CHECK" verify' in setup and '"$CHECK" initialize' in setup


def test_device_smi_pin_matches_pinned_gptq_package_metadata():
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    from packaging.requirements import Requirement
    dependencies = tomllib.loads((GPTQ.parent / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    requirement = next(Requirement(line) for line in dependencies if Requirement(line).name == "device-smi")
    pins = (ROOT / "hpc/quantizer_followups/gptq_v570_extras.txt").read_text().splitlines()
    pin = next(Requirement(line) for line in pins if Requirement(line).name == "device-smi")
    assert "0.5.3" in requirement.specifier
    assert "0.5.3" in pin.specifier
    assert "0.5.2" not in requirement.specifier


def test_effective_upstream_dispatch_contract():
    registry = (AUTO / "algorithms/registry.py").read_text()
    v2 = (AUTO / "algorithms/quantization/sign_roundv2/quantizer.py").read_text()
    dispatch = (AUTO / "data_type/utils.py").read_text()
    neuqi = (AUTO / "data_type/neuqi.py").read_text()
    assert "def resolve_pipeline_member(config:" in registry
    assert "@register_quantizer(SignRoundV2Config)\nclass SignRoundV2Quantizer" in v2
    assert 'f"{rtn_data_type}_sym_neuqi"' in dispatch
    assert 'if not enable_neuqi and not sym:' in dispatch
    assert '@register_dtype("opt_rtn_int_asym")' in neuqi
    assert '@register_dtype("opt_rtn_int_sym_neuqi")' in neuqi
    assert "class Qwen3QModel" in (GPTQ / "models/definitions/qwen3.py").read_text()
    source = (GPTQ / "quantization/config.py").read_text(encoding="utf-8")
    assert "class QuantizeConfig" in source and "true_sequential: bool = field(default=True)" in source
    assert "def dequantize_model(model:" in (GPTQ / "nn_modules/qlinear/torch.py").read_text()
    worker = (ROOT / "lp_llm/quantizer_followups_worker.py").read_text()
    assert "from gptqmodel.quantization import QuantizeConfig" in worker
    assert "backend=BACKEND.TORCH" in worker
    assert "module.dequantize_weight().T" in worker
    assert "bias=module.bias is not None" in worker


def test_gptq_materializer_preserves_bias_inventory_and_exact_decode(tmp_path, monkeypatch):
    from lp_llm.quantizer_followups_worker import materialize_gptq_qdq

    class QuantLinear(torch.nn.Module):
        def __init__(self, bias):
            super().__init__()
            self.in_features, self.out_features = 2, 2
            self.bias = torch.nn.Parameter(torch.tensor([1., -1.])) if bias else None

        def dequantize_weight(self):
            return torch.tensor([[1.25, 2.5], [-3.5, 4.25]])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = torch.nn.Module()
            self.block.no_bias = QuantLinear(False)
            self.block.has_bias = QuantLinear(True)
            self.config = types.SimpleNamespace(quantization_config={"bits": 2})

        def save_pretrained(self, target):
            Path(target).mkdir()
            save_file({name: value.detach().contiguous() for name, value in self.state_dict().items()},
                      Path(target) / "model.safetensors")

    module = types.ModuleType("gptqmodel.nn_modules.qlinear.torch")
    module.TorchQuantLinear = QuantLinear
    for name in ("gptqmodel", "gptqmodel.nn_modules", "gptqmodel.nn_modules.qlinear"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    model = Model()
    packed = types.SimpleNamespace(model=model, tokenizer=types.SimpleNamespace(
        save_pretrained=lambda target: (Path(target) / "tokenizer_config.json").write_text("{}")))
    assert materialize_gptq_qdq(packed, tmp_path / "qdq") == 2
    with __import__("safetensors").safe_open(tmp_path / "qdq/model.safetensors", framework="pt") as handle:
        assert "block.no_bias.bias" not in handle.keys()
        assert torch.equal(handle.get_tensor("block.no_bias.weight"),
                           torch.tensor([[1.25, -3.5], [2.5, 4.25]], dtype=torch.float16))
        assert torch.equal(handle.get_tensor("block.has_bias.bias"),
                           torch.tensor([1., -1.], dtype=torch.float16))
    assert not hasattr(model.config, "quantization_config")
    with pytest.raises(FileExistsError):
        materialize_gptq_qdq(packed, tmp_path / "qdq")


def test_actual_pinned_dispatch_function(monkeypatch):
    source = (AUTO / "data_type/utils.py").read_text()
    function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
                    and node.name == "get_quant_func")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(AUTO / "data_type/utils.py"), "exec"),
         namespace)
    package = types.ModuleType("auto_round")
    package.__path__ = []
    data_type = types.ModuleType("auto_round.data_type")
    data_type.QUANT_FUNC_WITH_DTYPE = {name: (lambda: None) for name in (
        "opt_rtn_int_sym_neuqi", "opt_rtn_int_asym", "rtn_int_asym", "opt_rtn_int_sym")}
    monkeypatch.setitem(sys.modules, "auto_round", package)
    monkeypatch.setitem(sys.modules, "auto_round.data_type", data_type)
    get = namespace["get_quant_func"]
    assert get("int", 2, True, group_size=128, iters=0, enable_neuqi=True)[1] == "opt_rtn_int_sym_neuqi"
    assert get("int", 2, False, group_size=128, iters=0, enable_neuqi=True)[1] == "opt_rtn_int_asym"
    assert get("int", 2, False, group_size=128, iters=0, enable_neuqi=False)[1] == "rtn_int_asym"


def test_two_bit_grid_and_nonfinite_rejection():
    symmetric = torch.tensor([-2., -1., 0., 1.] * 32, dtype=torch.float16) * 0.25
    asymmetric = torch.tensor([-1., 0., 1., 2.] * 32, dtype=torch.float16) * 0.5
    assert _grid(symmetric, True)["code_range"] in ([-2, 1], [-1, 2])
    assert _grid(asymmetric, False)["levels"] == 4
    with pytest.raises(ValueError):
        _grid(torch.arange(5, dtype=torch.float16), False)
    partial = torch.tensor([-.5, -.25, 0., .25, -.5, 0., .5], dtype=torch.float16)[None, :]
    assert _check_all_w2_groups(partial, group_size=4) == 2
    bad = torch.tensor([0., 1., 2., 4.], dtype=torch.float16)[None, :]
    with pytest.raises(ValueError, match="affine"):
        _check_all_w2_groups(bad, group_size=4)


def test_pinned_gptq_symmetric_w2_qdq_arithmetic():
    pinned_source = (GPTQ / "quantization/quantizer.py").read_text(encoding="utf-8")
    tree = ast.parse(pinned_source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
                 node.name in {"_replace_zero_scale", "quantize"}]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "pinned_gptq_quantizer.py", "exec"),
         namespace)
    source = torch.tensor([-1.2, -0.8, -0.1, 0.4, 1.2], dtype=torch.float32)
    scale = torch.tensor(0.5)
    zero = torch.tensor(2.0)
    codes = torch.clamp(torch.round(source / scale) + zero, 0, 3)
    assert set(codes.tolist()) <= {0, 1, 2, 3}
    decoded = namespace["quantize"](source, scale, zero, 3, False)
    assert torch.equal(decoded, scale * (codes - zero))
    # The pinned revision avoids zero scales in find_params, before quantize.
    assert "xmin[tmp] = -1" in pinned_source and "xmax[tmp] = +1" in pinned_source


def _checkpoint(directory: Path, weights: dict[str, torch.Tensor]) -> None:
    directory.mkdir(parents=True)
    save_file(weights, directory / "model.safetensors")
    atomic_json(directory / "config.json", {"model_type": "qwen3", "hidden_size": 4096,
                 "num_hidden_layers": 36, "vocab_size": 151936,
                 "intermediate_size": 12288, "num_attention_heads": 32,
                 "num_key_value_heads": 8})
    atomic_json(directory / "model.safetensors.index.json", {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in weights.values())},
        "weight_map": {name: "model.safetensors" for name in weights}})


def _fixture(tmp_path: Path, asym: bool = False):
    source_weights, output_weights = {}, {}
    original = (torch.arange(128, dtype=torch.float32).remainder(11) / 7 - 0.7).to(torch.bfloat16)[None, :]
    code = torch.tensor([-1., 0., 1., 2.] if asym else [-2., -1., 0., 1.], dtype=torch.float16)
    quantized = code.repeat(32)[None, :] * 0.25
    for layer in range(36):
        for kind in PROJECTIONS:
            prefix = "self_attn" if kind in PROJECTIONS[:4] else "mlp"
            name = f"model.layers.{layer}.{prefix}.{kind}.weight"
            source_weights[name] = original.clone()
            output_weights[name] = quantized.clone()
    source_weights["model.embed_tokens.weight"] = torch.tensor([1.125], dtype=torch.bfloat16)
    output_weights["model.embed_tokens.weight"] = source_weights["model.embed_tokens.weight"].to(torch.float16)
    source = tmp_path / "source"
    export = tmp_path / "export"
    _checkpoint(source, source_weights)
    _checkpoint(export / "qwen3-8b-base-w2g128", output_weights)
    return source, export


@pytest.mark.parametrize("experiment_id,asym", [("r01", False), ("r02", True)])
def test_nested_qdq_validation_and_corruption(tmp_path, experiment_id, asym):
    source, export = _fixture(tmp_path, asym)
    config = load_config(experiment_id)
    record = validate_checkpoint(config, source, export)
    assert record["quantized_projection_count"] == 252
    assert record["unchanged_tensor_count"] == 1
    assert record["scale_and_zero_point_bytes"] == "unavailable_in_fake_qdq"
    assert len(record["shard_sha256"]) == 1
    shard = export / "qwen3-8b-base-w2g128/model.safetensors"
    shard.unlink()
    with pytest.raises(ValueError):
        validate_checkpoint(config, source, export)


def test_resume_identity_and_aggregate(tmp_path):
    marker = tmp_path / "marker.json"
    expected = {"id": "r01"}
    atomic_json(marker, {"status": "completed_validated", "identity": expected})
    assert _read_marker(marker, expected)["status"] == "completed_validated"
    with pytest.raises(ValueError):
        _read_marker(marker, {"id": "r02"})
    root = tmp_path / "quantizer_followups"
    (root / "s01").mkdir(parents=True)
    atomic_json(root / "s01/run.json", {"status": "started"})
    statuses = {row["id"]: row["status"] for row in aggregate(root)["experiments"]}
    assert statuses == {"s01": "incomplete_or_failed", "r01": "not_started",
                        "r02": "not_started", "r03": "not_started"}


def test_resolved_neuqi_identity_rejects_plain_fallback(tmp_path):
    checkout = tmp_path / "source"
    checkout.mkdir()
    imported = checkout / "auto_round/__init__.py"
    imported.parent.mkdir()
    imported.write_text("")
    path = tmp_path / "resolved.json"
    config = load_config("r02")
    record = {"reference_module": str(imported), "resolved_config_class": "OptimizedRTNConfig",
              "resolved_quantizer_class": "OptimizedRTNQuantizer",
              "selected_quant_function": "opt_rtn_int_asym", "neuqi_grid": {"coarse": 256, "fine": 64, "backend": "auto"},
              "resolved_algorithm": {"iters": 0, "bits": 2, "group_size": 128,
                 "sym": False, "enable_neuqi": True, "disable_opt_rtn": False}}
    atomic_json(path, record)
    assert validate_resolved(config, path, checkout) == record
    record["resolved_algorithm"]["iters"] = None  # Actual pinned OptimizedRTNConfig record.
    atomic_json(path, record)
    assert validate_resolved(config, path, checkout) == record
    record["resolved_config_class"] = "SignRoundV2Config"
    atomic_json(path, record)
    with pytest.raises(ValueError, match="resolved configuration"):
        validate_resolved(config, path, checkout)
    record["resolved_config_class"] = "OptimizedRTNConfig"
    record["resolved_algorithm"]["iters"] = 1
    atomic_json(path, record)
    with pytest.raises(ValueError, match="resolved configuration"):
        validate_resolved(config, path, checkout)
    record["resolved_algorithm"]["iters"] = None
    record["selected_quant_function"] = "rtn_int_asym"
    atomic_json(path, record)
    with pytest.raises(ValueError, match="NeUQI"):
        validate_resolved(config, path, checkout)


def test_r03_resolved_identity_requires_pinned_backend_and_materializer(tmp_path):
    checkout = tmp_path / "gptq"
    checkout.mkdir()
    path = tmp_path / "resolved.json"
    record = {"reference_module": str(checkout / "gptqmodel/__init__.py"),
              "config": {"bits": "2", "group_size": "128", "sym": "True",
                         "desc_act": "False", "act_group_aware": "True",
                         "damp_percent": "0.05", "damp_auto_increment": "0.01",
                         "static_groups": "False", "true_sequential": "True",
                         "format": "gptq"},
              "backend": "TORCH", "model_dtype": "float16", "calibration_sample_count": 128,
              "materializer": "pinned TorchQuantLinear.dequantize_weight, preserving bias inventory",
              "packed_to_qdq_exact_projection_parity": True,
              "packed_to_qdq_projection_count": 252}
    atomic_json(path, record)
    assert validate_resolved(load_config("r03"), path, checkout) == record
    for key, bad in (("backend", "AUTO"), ("calibration_sample_count", 0),
                     ("materializer", "unverified")):
        changed = dict(record, **{key: bad})
        atomic_json(path, changed)
        with pytest.raises(ValueError, match="GPTQ settings"):
            validate_resolved(load_config("r03"), path, checkout)


def test_source_must_match_existing_j05_identity(tmp_path):
    result = tmp_path / "qwen3_8b_base/quantizer_followups/r01"
    prior = tmp_path / "qwen3_8b_base/rrq_w2_reproduction/j05/run.json"
    current = {key: {"x": "a"} for key in ("shard_sha256", "sidecar_sha256")}
    current.update({"checkpoint": "original", "config_sha256": "cfg", "index_sha256": "idx"})
    atomic_json(prior, {"source_checkpoint": current})
    verify_original_checkpoint(current, result)
    changed = dict(current, index_sha256="other")
    with pytest.raises(ValueError, match="differs"):
        verify_original_checkpoint(changed, result)


def test_gptq_subprocess_gets_separate_conda_prefix(tmp_path):
    from lp_llm.quantizer_followups import reference_environment
    checkout = tmp_path / "reference"
    checkout.mkdir()
    selected = tmp_path / "gptq_env/bin/python"
    selected.parent.mkdir(parents=True)
    selected.write_text("")
    env = reference_environment(checkout, None, str(selected))
    assert Path(env["CONDA_PREFIX"]) == selected.parent.parent
    assert env["PATH"].split(__import__("os").pathsep)[0] == str(selected.parent)
    assert env["PYTHONPATH"].split(__import__("os").pathsep)[0] == str(checkout.resolve())


def test_read_only_excluded_audit_quantifies_head_and_other_tensors(tmp_path):
    from lp_llm.quantizer_followups_audit import audit
    source = tmp_path / "source"
    export = tmp_path / "export/nested"
    original = {"lm_head.weight": torch.tensor([[1., 2., 3., 4.]], dtype=torch.bfloat16),
                "model.embed_tokens.weight": torch.tensor([[1., 2., 3., 4.]], dtype=torch.bfloat16),
                "model.norm.weight": torch.tensor([1., 1.], dtype=torch.bfloat16),
                "model.layers.0.self_attn.q_proj.weight": torch.ones(1, 128, dtype=torch.bfloat16)}
    changed = {name: tensor.to(torch.float16) for name, tensor in original.items()}
    changed["lm_head.weight"] = torch.tensor([[1., 2.5, 3., 4.]], dtype=torch.float16)
    changed["model.layers.0.self_attn.q_proj.weight"] = torch.zeros(1, 128, dtype=torch.float16)
    _checkpoint(source, original)
    _checkpoint(export, changed)
    report = audit(source, export.parent)
    assert report["changed_excluded_names"] == ["lm_head.weight"]
    assert report["lm_head"]["changed"] == 1
    assert report["lm_head"]["changed_fraction"] == 0.25
    assert report["lm_head"]["max_abs_error_after_cast"] == 0.5
    assert report["lm_head"]["rmse_after_cast"] == 0.25
    assert report["lm_head"]["fp16_roundtrip_mismatches"] == 1
    assert report["head_vs_embeddings"]["source"]["different_elements"] == 0
    assert report["head_vs_embeddings"]["export"]["different_elements"] == 1


def test_s01_audit_detects_bf16_fp16_underflow_without_changing_export(tmp_path):
    from lp_llm.quantizer_followups_audit import audit
    source, export = tmp_path / "source", tmp_path / "export"
    tiny = torch.tensor([[2.9802322387695312e-8, 1.0]], dtype=torch.bfloat16)
    original = {"lm_head.weight": tiny,
                "model.embed_tokens.weight": tiny.clone(),
                "model.layers.0.self_attn.q_proj.weight": torch.ones(1, 128, dtype=torch.bfloat16)}
    output = dict(original)
    output["lm_head.weight"] = tiny.to(torch.float16).to(torch.bfloat16)
    _checkpoint(source, original)
    _checkpoint(export, output)
    report = audit(source, export)
    assert report["lm_head"]["changed"] == 1
    assert report["lm_head"]["fp16_roundtrip_mismatches"] == 0
    assert report["lm_head"]["source_nonzero_to_export_zero"] == 1
    assert report["lm_head"]["first_mismatches"][0]["export"] == 0.0


def test_s01_repair_copies_only_excluded_bytes_and_resumes(tmp_path, monkeypatch):
    import lp_llm.quantizer_followups_repair as module
    source = tmp_path / "source"
    raw_result = tmp_path / "quantizer_followups/s01"
    raw = raw_result / "export"
    repaired_result = tmp_path / "quantizer_followups/s01_fp16_load_repair"
    names = [f"model.layers.{i}.self_attn.q_proj.weight" for i in range(252)]
    original = {name: torch.tensor([[float(i + 1)]], dtype=torch.bfloat16)
                for i, name in enumerate(names)}
    original["lm_head.weight"] = torch.tensor([[2.9802322387695312e-8]], dtype=torch.bfloat16)
    exported = dict(original)
    exported["lm_head.weight"] = torch.zeros(1, 1, dtype=torch.bfloat16)
    _checkpoint(source, original)
    _checkpoint(raw, exported)
    pinned = {"id": "s01", "source": {}}
    atomic_json(raw_result / "run.json", {"identity": pinned})
    atomic_json(raw_result / "quantization_process.json", {"returncode": 0})
    (raw_result / "quantization.log").write_text("success")
    (raw_result / "resolved_config.json").write_text("{}")
    monkeypatch.setattr(module, "check_revision", lambda *args: None)
    monkeypatch.setattr(module, "audit_source", lambda *args: {})
    monkeypatch.setattr(module, "identity", lambda *args: pinned)
    monkeypatch.setattr(module, "verify_original_checkpoint", lambda *args: None)
    monkeypatch.setattr(module, "validate_resolved", lambda *args: {})
    monkeypatch.setattr(module, "audit", lambda *args: {
        "quantized_projection_names": names,
        "excluded_tensors": [{"name": "lm_head.weight", "changed": 1,
                              "source_dtype": "torch.bfloat16", "export_dtype": "torch.bfloat16",
                              "fp16_roundtrip_mismatches": 0}]})

    def validate(config, input_dir, output_dir):
        mapping = json.loads((output_dir / "model.safetensors.index.json").read_text())["weight_map"]
        with __import__("safetensors").safe_open(output_dir / mapping["lm_head.weight"], framework="pt") as handle:
            assert torch.equal(handle.get_tensor("lm_head.weight"), original["lm_head.weight"])
        return {"resolved_checkpoint_path": str(output_dir.resolve()), "quantized_projection_count": 252}

    monkeypatch.setattr(module, "validate_checkpoint", validate)
    first = module.repair(source, raw_result, repaired_result, tmp_path / "upstream",
                          tmp_path / "matgptq", "encode")
    second = module.repair(source, raw_result, repaired_result, tmp_path / "upstream",
                           tmp_path / "matgptq", "encode")
    assert first == second
    with __import__("safetensors").safe_open(raw / "model.safetensors", framework="pt") as handle:
        assert torch.equal(handle.get_tensor("lm_head.weight"), exported["lm_head.weight"])
    assert len(first["projection_tensor_sha256"]) == 252
    assert first["restored_excluded_tensors"] == ["lm_head.weight"]


def test_s01_repair_rejects_unexplained_excluded_change(tmp_path, monkeypatch):
    import lp_llm.quantizer_followups_repair as module
    monkeypatch.setattr(module, "check_revision", lambda *args: None)
    monkeypatch.setattr(module, "identity", lambda *args: {"id": "s01", "source": {}})
    monkeypatch.setattr(module, "verify_original_checkpoint", lambda *args: None)
    monkeypatch.setattr(module, "audit_source", lambda *args: {})
    monkeypatch.setattr(module, "validate_resolved", lambda *args: {})
    monkeypatch.setattr(module, "validate_output", lambda *args: {"resolved_checkpoint_path": str(tmp_path)})
    monkeypatch.setattr(module, "audit", lambda *args: {
        "quantized_projection_names": [str(i) for i in range(252)],
        "excluded_tensors": [{"name": "lm_head.weight", "changed": 1,
                              "fp16_roundtrip_mismatches": 1}]})
    raw_result = tmp_path / "s01"
    raw_result.mkdir()
    atomic_json(raw_result / "run.json", {"identity": {"id": "s01", "source": {}}})
    atomic_json(raw_result / "quantization_process.json", {"returncode": 0})
    (raw_result / "quantization.log").write_text("success")
    (raw_result / "resolved_config.json").write_text("{}")
    with pytest.raises(ValueError, match="not fully explained"):
        module.repair(tmp_path, raw_result, tmp_path / "repaired", tmp_path, tmp_path, "encode")
    assert not (tmp_path / "repaired").exists()


@pytest.mark.parametrize("experiment_id", ("r01", "r02"))
def test_recover_successful_export_without_requantization_and_resume_eval(tmp_path, monkeypatch, experiment_id):
    import lp_llm.quantizer_followups as module
    config = load_config(experiment_id)
    result = tmp_path / experiment_id
    export = result / "export/model-w2g128"
    export.mkdir(parents=True)
    (export / "model.safetensors").write_bytes(b"fixture")
    (result / "quantization.log").write_text("upstream complete")
    atomic_json(result / "quantization_process.json", {"returncode": 0})
    atomic_json(result / "resolved_config.json", {"selected_quant_function": "opt_rtn_int_sym_neuqi"})
    identity = {"id": experiment_id, "config_sha256": config["config_sha256"], "source": {}}
    atomic_json(result / "run.json", {"identity": identity})
    monkeypatch.setattr(module, "audit_source", lambda *args: {"revision": PR_COMMIT})
    monkeypatch.setattr(module, "check_revision", lambda *args: None)
    monkeypatch.setattr(module, "identity", lambda *args: identity)
    monkeypatch.setattr(module, "verify_original_checkpoint", lambda *args: None)
    monkeypatch.setattr(module, "validate_resolved", lambda *args: {})
    checkpoint = {"resolved_checkpoint_path": str(export), "shard_sha256": {"model.safetensors": "hash"}}
    monkeypatch.setattr(module, "validate_checkpoint", lambda *args: checkpoint)
    monkeypatch.setattr(module, "corpus_identity", lambda *args: {
        "token_sha256": module.TOKEN_SHA256, "scored_tokens": 298862})
    monkeypatch.setattr(module, "run_logged", lambda *args, **kwargs: pytest.fail("quantizer relaunched"))
    calls = []

    def evaluate(checkout, checkpoint_path, log, *, python):
        calls.append(checkpoint_path)
        log.write_text("wikitext2: 20.00")
        return {"perplexity": 20.0}

    monkeypatch.setattr(module, "evaluate_matgptq", evaluate)
    first = module.run_one(config, tmp_path / "source", result, tmp_path / "upstream",
                           tmp_path / "matgptq", "python")
    assert first["evaluation"]["perplexity"] == 20.0
    assert json.loads((result / "encoding_complete.json").read_text())["recovered_after_successful_export"]
    assert len(calls) == 1
    second = module.run_one(config, tmp_path / "source", result, tmp_path / "upstream",
                            tmp_path / "matgptq", "python")
    assert second == first and len(calls) == 1


def test_unverified_export_never_requantizes(tmp_path, monkeypatch):
    import lp_llm.quantizer_followups as module
    config = load_config("r01")
    result = tmp_path / "r01"
    (result / "export").mkdir(parents=True)
    (result / "export/partial.safetensors").write_bytes(b"partial")
    identity = {"id": "r01", "source": {}}
    atomic_json(result / "run.json", {"identity": identity})
    monkeypatch.setattr(module, "audit_source", lambda *args: {})
    monkeypatch.setattr(module, "check_revision", lambda *args: None)
    monkeypatch.setattr(module, "identity", lambda *args: identity)
    monkeypatch.setattr(module, "verify_original_checkpoint", lambda *args: None)
    monkeypatch.setattr(module, "validate_resolved", lambda *args: {})
    monkeypatch.setattr(module, "run_logged", lambda *args, **kwargs: pytest.fail("quantizer relaunched"))
    with pytest.raises(ValueError, match="Unmarked export"):
        module.run_one(config, tmp_path / "source", result, tmp_path / "upstream",
                       tmp_path / "matgptq", "python")


def test_recovery_only_mode_refuses_missing_export_without_quantization(tmp_path, monkeypatch):
    import lp_llm.quantizer_followups as module
    config = load_config("r01")
    result = tmp_path / "r01"
    result.mkdir()
    ident = {"id": "r01", "source": {}}
    atomic_json(result / "run.json", {"identity": ident})
    monkeypatch.setenv("FOLLOWUP_REUSE_EXPORT_ONLY", "1")
    monkeypatch.setattr(module, "audit_source", lambda *args: {})
    monkeypatch.setattr(module, "check_revision", lambda *args: None)
    monkeypatch.setattr(module, "identity", lambda *args: ident)
    monkeypatch.setattr(module, "verify_original_checkpoint", lambda *args: None)
    monkeypatch.setattr(module, "run_logged", lambda *args, **kwargs: pytest.fail("quantizer relaunched"))
    with pytest.raises(ValueError, match="quantization was not launched"):
        module.run_one(config, tmp_path / "source", result, tmp_path / "upstream",
                       tmp_path / "matgptq", "python")
