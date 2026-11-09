import sys
import types
import json
import runpy
from contextlib import nullcontext
from pathlib import Path

import pytest

from lp_llm.progressive_control_worker import resolved_recipe
from lp_llm import progressive_control
from lp_llm.progressive_control_pilot import (verify_unselected_projections,
    traversed_blocks, slurm_cgroup_peak, host_memory_assessment, _compare_rows,
    source_view)


def test_pinned_autoround_exact_block_selector_normalizes_single_layer():
    find_matching_blocks = runpy.run_path(str(Path(__file__).parent / "fixtures" /
        "autoround_block_selector_7985c1b.py"))["find_matching_blocks"]
    blocks = [[f"model.layers.{number}" for number in range(36)]]
    assert find_matching_blocks(None, blocks, r"^model\.layers\.0$") == [["model.layers.0"]]
    assert find_matching_blocks(None, blocks, ["model.layers.0"]) == ["model.layers.0"]


def test_control_pilot_detects_quantization_outside_selected_block():
    import torch
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList()
    for _ in range(2):
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.self_attn.q_proj = torch.nn.Linear(4, 2, bias=False)
        model.model.layers.append(layer)
    original = model.model.layers[1].self_attn.q_proj.weight.detach().clone()
    assert verify_unselected_projections(model, lambda _: nullcontext(original),
        ("self_attn.q_proj",), layers=2) == 1
    with torch.no_grad():
        model.model.layers[1].self_attn.q_proj.weight[0, 0] += 1
    with pytest.raises(ValueError, match="Unselected transformer projection changed"):
        verify_unselected_projections(model, lambda _: nullcontext(original),
            ("self_attn.q_proj",), layers=2)


def test_all_blocks_traversed_while_only_layer_zero_is_quantized():
    import torch
    blocks = [[f"model.layers.{index}" for index in range(36)]]
    assert traversed_blocks(blocks) == blocks[0]
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList()
    originals = {}
    for index in range(36):
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.self_attn.q_proj = torch.nn.Linear(4, 2, bias=False)
        model.model.layers.append(layer)
        name = f"model.layers.{index}.self_attn.q_proj.weight"
        originals[name] = layer.self_attn.q_proj.weight.detach().clone()
    with torch.no_grad():
        model.model.layers[0].self_attn.q_proj.weight[0, 0] += 1
    assert verify_unselected_projections(model, lambda name: nullcontext(originals[name]),
        ("self_attn.q_proj",)) == 35
    assert not torch.equal(model.model.layers[0].self_attn.q_proj.weight,
                           originals["model.layers.0.self_attn.q_proj.weight"])
    with pytest.raises(ValueError, match="Unexpected AutoRound block traversal"):
        traversed_blocks([["model.layers.36"]])


def test_slurm_cgroup_peak_requires_job_scoped_counter(tmp_path):
    proc = tmp_path / "cgroup"
    root = tmp_path / "sys/fs/cgroup"
    counter = root / "system.slice/slurm/job_123/step_0/memory.peak"
    counter.parent.mkdir(parents=True)
    counter.write_text(str(15 * 1024**3))
    proc.write_text("0::/system.slice/slurm/job_123/step_0\n")
    assert slurm_cgroup_peak("123", proc=proc, root=root) == (
        15 * 1024**3, "cgroup_v2_memory.peak")
    with pytest.raises(RuntimeError, match="unavailable"):
        slurm_cgroup_peak("124", proc=proc, root=root)


def test_bounded_projection_comparison_preserves_exact_counts(tmp_path):
    import torch
    from safetensors.torch import save_file
    class Sliced:
        def __init__(self, tensor):
            self.tensor = tensor
            self.largest_rows = 0
        def get_shape(self):
            return self.tensor.shape
        def __getitem__(self, key):
            self.largest_rows = max(self.largest_rows, key.stop - key.start)
            return self.tensor[key]
    original = torch.zeros(150, 128, dtype=torch.float16)
    value = original.clone()
    value[0, 0] += 1
    value[149, 127] += 1
    view = Sliced(original)
    assert _compare_rows(value, view, "projection.weight") == 2
    assert view.largest_rows <= 64
    assert host_memory_assessment(10 * 1024**2, 12 * 1024**3,
                                  14 * 1024**3)["host_memory_10pct_headroom_passed"]
    assert not host_memory_assessment(10 * 1024**2, 13 * 1024**3,
                                      14 * 1024**3)["host_memory_10pct_headroom_passed"]
    with pytest.raises(ValueError, match="shape differs"):
        _compare_rows(value, Sliced(original[:149]), "projection.weight")
    source = tmp_path / "source"
    source.mkdir()
    save_file({"projection.weight": original.to(torch.bfloat16)},
              source / "model.safetensors")
    with source_view(source, {"projection.weight": "model.safetensors"},
                     "projection.weight") as mapped:
        assert _compare_rows(value, mapped, "projection.weight") == 2
        assert _compare_rows(original, mapped, "projection.weight") == 0


def test_all_six_control_recipes_keep_pinned_dispatch(monkeypatch):
    modules = ["auto_round", "auto_round.algorithms", "auto_round.algorithms.quantization",
        "auto_round.algorithms.quantization.rtn", "auto_round.algorithms.quantization.sign_round",
        "auto_round.data_type"]
    for name in modules:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    class SignRoundV2Config:
        need_calib = True

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class OptimizedRTNConfig:
        need_calib = True

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    config_s = types.ModuleType("auto_round.algorithms.quantization.sign_round.config")
    config_s.SignRoundV2Config = SignRoundV2Config
    config_r = types.ModuleType("auto_round.algorithms.quantization.rtn.config")
    config_r.OptimizedRTNConfig = OptimizedRTNConfig
    registry = types.ModuleType("auto_round.algorithms.registry")
    registry.resolve_pipeline_member = lambda config: types.SimpleNamespace(
        __name__="SignRoundV2Quantizer" if isinstance(config, SignRoundV2Config)
        else "OptimizedRTNQuantizer")
    functions = types.ModuleType("auto_round.data_type.utils")

    def quant_tensor_opt_rtn_asym():
        pass

    def dispatch(dtype, bits, sym, **settings):
        assert dtype == "int" and bits in (4, 6, 8) and sym is False
        assert settings == {"disable_opt_rtn": False, "group_size": 128,
                            "iters": 0, "enable_neuqi": True}
        return quant_tensor_opt_rtn_asym, "opt_rtn_int_asym"

    functions.get_quant_func = dispatch
    for module in (config_s, config_r, registry, functions):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    for bits in (4, 6, 8):
        sign, sign_record = resolved_recipe("s01", bits)
        rtn, rtn_record = resolved_recipe("r02", bits)
        assert sign.bits == rtn.bits == bits
        assert sign_record["quantizer_class"] == "SignRoundV2Quantizer"
        assert sign_record["iterations"] == 200
        assert sign_record["enable_neuqi"] is False
        assert rtn_record["selected_quant_function"] == "opt_rtn_int_asym"
        assert rtn_record["function_name"] == "quant_tensor_opt_rtn_asym"
        assert rtn_record["iterations"] == 0 and rtn_record["enable_neuqi"] is True
        assert rtn_record["neuqi_grid"] == {"coarse": 256, "fine": 64}


def test_control_recipe_rejects_unexpected_dispatch(monkeypatch):
    with pytest.raises(ValueError, match="Unsupported"):
        resolved_recipe("r02", 3)


def test_successful_partial_control_recovers_without_requantization(tmp_path, monkeypatch):
    result = tmp_path / "control"
    result.mkdir()
    (result / "export.partial").mkdir()
    (result / "quantization_process.json").write_text('{"returncode":0}')
    (result / "quantization.log").write_text("completed")
    (result / "resolved_config.json").write_text("{}")
    hashes = {"projection.weight": "abc"}
    (result / "excluded_repair.json").write_text(json.dumps({
        "raw_projection_tensor_sha256": hashes,
        "quantized_projection_byte_identical": True}))
    ident = {"branch": "s01", "bits": 4}
    (result / "run.json").write_text(json.dumps({"identity": ident}))
    monkeypatch.setattr(progressive_control, "control_identity", lambda *_: ident)
    monkeypatch.setattr(progressive_control, "check_revision", lambda *_: None)
    monkeypatch.setattr(progressive_control, "_resolved_matches", lambda *_: None)
    monkeypatch.setattr(progressive_control, "validate_output", lambda *_: {})
    monkeypatch.setattr(progressive_control, "validate_control", lambda *_, **__: {
        "projection_tensor_sha256": hashes})
    monkeypatch.setattr(progressive_control, "run_logged", lambda *_args, **_kwargs:
                        pytest.fail("Quantization subprocess must not run during recovery"))
    record = progressive_control.run("s01", 4, tmp_path, result, tmp_path,
                                     tmp_path, stage="encode")
    assert record["status"] == "completed_validated"
    assert (result / "export").is_dir() and not (result / "export.partial").exists()
    assert progressive_control.run("s01", 4, tmp_path, result, tmp_path,
                                   tmp_path, stage="encode") == record


def test_control_recovery_refuses_wrong_identity_or_incomplete_process(tmp_path, monkeypatch):
    result = tmp_path / "control"
    result.mkdir()
    (result / "export.partial").mkdir()
    (result / "run.json").write_text(json.dumps({"identity": {"branch": "r02"}}))
    monkeypatch.setattr(progressive_control, "control_identity", lambda *_: {"branch": "s01"})
    monkeypatch.setattr(progressive_control, "check_revision", lambda *_: None)
    monkeypatch.setattr(progressive_control, "run_logged", lambda *_args, **_kwargs:
                        pytest.fail("Incomplete recovery must not quantize"))
    with pytest.raises(ValueError, match="identity differs"):
        progressive_control.run("s01", 4, tmp_path, result, tmp_path,
                                tmp_path, stage="encode")
    (result / "run.json").write_text(json.dumps({"identity": {"branch": "s01"}}))
    with pytest.raises(FileExistsError, match="successful process evidence"):
        progressive_control.run("s01", 4, tmp_path, result, tmp_path,
                                tmp_path, stage="encode")
    assert (result / "export.partial").is_dir()


def test_resolved_control_rejects_changed_calibration_or_quantizer(tmp_path):
    checkout = tmp_path / "reference"
    checkout.mkdir()
    path = tmp_path / "resolved.json"
    expected = {"branch": "r02", "bits": 6, "group_size": 128,
        "sym": False, "algorithm_class": "OptimizedRTNConfig",
        "quantizer_class": "OptimizedRTNQuantizer",
        "selected_quant_function": "opt_rtn_int_asym",
        "function_name": "quant_tensor_opt_rtn_asym", "iterations": 0,
        "enable_neuqi": True, "need_calib": True, "data_type": "int",
        "model_dtype": "float16", "scale_dtype": "fp16",
        "calibration": {"dataset": "NeelNanda/pile-10k", "nsamples": 128,
                        "seqlen": 2048, "batch_size": 8, "seed": 42},
        "neuqi_grid": {"coarse": 256, "fine": 64},
        "resolved": {"bits": 6, "group_size": 128, "sym": False,
                     "enable_neuqi": True, "iters": None},
        "reference_module": str(checkout / "auto_round/__init__.py")}
    path.write_text(json.dumps(expected))
    progressive_control._resolved_matches(path, "r02", 6, checkout)
    for key, value in (("iterations", 1), ("quantizer_class", "RTNQuantizer"),
                       ("calibration", {"nsamples": 0}),
                       ("neuqi_grid", {"coarse": 1, "fine": 1})):
        changed = dict(expected, **{key: value})
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="resolved recipe"):
            progressive_control._resolved_matches(path, "r02", 6, checkout)
