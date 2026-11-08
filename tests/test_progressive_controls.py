import sys
import types
import json

import pytest

from lp_llm.progressive_control_worker import resolved_recipe
from lp_llm import progressive_control


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
