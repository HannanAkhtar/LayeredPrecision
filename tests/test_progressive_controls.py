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
from lp_llm import progressive_control_pilot


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
    assert traversed_blocks(None) is None


def test_successful_quantization_with_unavailable_traversal_metadata(tmp_path, monkeypatch):
    import torch
    from types import SimpleNamespace
    from lp_llm import progressive_control_worker, progressive_lp, rrq_pr2308_compare

    paths = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
             "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList()
    originals = {}
    for layer_number in range(36):
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        for path in paths:
            parent = layer.self_attn if path.startswith("self_attn.") else layer.mlp
            setattr(parent, path.rsplit(".", 1)[1], torch.nn.Linear(4, 2, bias=False))
        model.model.layers.append(layer)
        for path in paths:
            name = f"model.layers.{layer_number}.{path}.weight"
            originals[name] = model.get_submodule(
                f"model.layers.{layer_number}.{path}").weight.detach().clone()

    class Algorithm:
        bits = 4

    class AutoRound:
        def __init__(self, **kwargs):
            self._alg_configs = [kwargs["alg_configs"]]
            self.to_quant_block_names = kwargs["to_quant_block_names"]
            self.quant_block_list = None
        def quantize(self):
            with torch.no_grad():
                for path in paths:
                    model.get_submodule(f"model.layers.0.{path}").weight[0, 0] += 1
            return model, {}

    auto_round = types.ModuleType("auto_round")
    auto_round.__file__ = str(tmp_path / "auto_round/__init__.py")
    auto_round.AutoRound = AutoRound
    device = types.ModuleType("auto_round.utils.device")
    device.memory_monitor = SimpleNamespace(peak_ram=14.71, peak_vram={"0": 1.0})
    monkeypatch.setitem(sys.modules, "auto_round", auto_round)
    monkeypatch.setitem(sys.modules, "auto_round.utils", types.ModuleType("auto_round.utils"))
    monkeypatch.setitem(sys.modules, "auto_round.utils.device", device)
    monkeypatch.setitem(sys.modules, "resource", SimpleNamespace(RUSAGE_SELF=0,
        getrusage=lambda *_: SimpleNamespace(ru_maxrss=1000)))
    monkeypatch.setattr(progressive_control_worker, "resolved_recipe",
                        lambda *_: (Algorithm(), {"test_recipe": True}))
    monkeypatch.setattr(rrq_pr2308_compare, "_weight_map", lambda *_: {})
    monkeypatch.setattr(progressive_lp, "_files", lambda *_: {"test_source": True})
    monkeypatch.setattr(progressive_control_pilot, "source_view",
                        lambda _source, _map, name: nullcontext(originals[name]))
    monkeypatch.setattr(progressive_control_pilot, "slurm_cgroup_peak",
                        lambda *_: (2 * 1024**3, "cgroup_v2_memory.peak"))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 1)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda *_: SimpleNamespace(total_memory=1000))
    monkeypatch.setenv("LP_CONTROL_PILOT", "1")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "14336")
    report = tmp_path / "s01_w4.json"
    result = progressive_control_pilot.run("s01", 4, tmp_path, report, "a" * 40)
    assert result["status"] == "control_layer_pilot_validated"
    assert result["traversal_metadata_status"] == "unavailable"
    assert result["traversed_transformer_blocks"] is None
    assert result["traversed_transformer_block_count"] is None
    assert result["quantized_projection_count"] == 7
    assert result["unselected_projection_source_exact_count"] == 245
    assert report.is_file() and not list(tmp_path.glob("*.failure.*.json"))


def test_post_quantization_failure_records_memory_without_success_report(tmp_path, monkeypatch):
    from types import SimpleNamespace

    def fail(_branch, _bits, _source, _report, _sha, progress):
        progress.update(quantization_completed=True,
                        stage="post_quantization_validation",
                        selected_projection_checks_completed=3)
        raise ValueError("selected projection differs")

    device = types.ModuleType("auto_round.utils.device")
    device.memory_monitor = SimpleNamespace(peak_ram=14.71, peak_vram={"0": 2.0})
    monkeypatch.setitem(sys.modules, "auto_round.utils.device", device)
    monkeypatch.setitem(sys.modules, "resource", SimpleNamespace(RUSAGE_SELF=0,
        getrusage=lambda *_: SimpleNamespace(ru_maxrss=1024)))
    monkeypatch.setattr(progressive_control_pilot, "_run", fail)
    monkeypatch.setattr(progressive_control_pilot, "slurm_cgroup_peak",
                        lambda *_: (13 * 1024**3, "cgroup_v2_memory.peak"))
    monkeypatch.setenv("SLURM_JOB_ID", "54811")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "14336")
    report = tmp_path / "s01_w4.json"
    with pytest.raises(ValueError, match="selected projection differs"):
        progressive_control_pilot.run("s01", 4, tmp_path, report, "a" * 40)
    assert not report.exists()
    diagnostic = json.loads((tmp_path / "s01_w4.failure.54811.json").read_text())
    assert diagnostic["status"] == "control_layer_pilot_failed"
    assert diagnostic["selected_projection_checks_completed"] == 3
    assert diagnostic["quantization_completed"] is True
    assert diagnostic["process_peak_rss_kib"] == 1024
    assert diagnostic["cgroup_peak_bytes"] == 13 * 1024**3
    assert diagnostic["host_allocation_bytes"] == 14 * 1024**3
    assert not diagnostic["host_memory_10pct_headroom_passed"]
    assert diagnostic["autoround_peak_ram"]["value"] == 14.71


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
def test_validate_control_checks_all_projections_and_exclusion(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import save_file
    from lp_llm import progressive_control as control
    source, export = tmp_path / "source", tmp_path / "export"
    source.mkdir(); export.mkdir()
    names = [f"model.layers.{i}.self_attn.q_proj.weight" for i in range(252)]
    original = {name: torch.ones((1, 128), dtype=torch.float16) for name in names}
    original["model.layers.0.input_layernorm.weight"] = torch.ones(1, dtype=torch.float16)
    changed = {name: tensor.clone() for name, tensor in original.items()}
    for name in names:
        changed[name][0, 0] = 2
    save_file(original, str(source / "model.safetensors"))
    save_file(changed, str(export / "model.safetensors"))
    mapping = {name: "model.safetensors" for name in original}
    monkeypatch.setattr(control, "validate_output", lambda *_:
                        {"resolved_checkpoint_path": str(export)})
    monkeypatch.setattr(control, "_weight_map", lambda *_: mapping)
    monkeypatch.setattr(control, "_eligible", lambda name: name in names)
    result = control.validate_control(source, export, 4)
    assert result["quantized_projection_count"] == 252
    assert result["source_exact_excluded_tensor_count"] == 1
    assert len(result["projection_tensor_sha256"]) == 252
