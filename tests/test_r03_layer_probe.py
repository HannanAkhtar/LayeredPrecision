import importlib.util
import types
from pathlib import Path

import pytest
import torch


PATH = Path(__file__).resolve().parents[1] / "hpc/quantizer_followups/r03_layer_probe.py"
spec = importlib.util.spec_from_file_location("r03_layer_probe_test", PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def _branch(count):
    return types.SimpleNamespace(model=types.SimpleNamespace(
        layers=torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(count)])),
        config=types.SimpleNamespace(num_hidden_layers=count))


def test_representative_probe_reduces_both_branches_to_one_layer():
    shell, turtle = _branch(36), _branch(36)
    original_shell_second = shell.model.layers[1]
    original_turtle_second = turtle.model.layers[1]
    probe.retain_first_layer(types.SimpleNamespace(model=shell, turtle_model=turtle))
    assert len(shell.model.layers) == len(turtle.model.layers) == 1
    assert shell.config.num_hidden_layers == turtle.config.num_hidden_layers == 1
    assert original_shell_second not in shell.model.layers
    assert original_turtle_second not in turtle.model.layers


def test_representative_probe_refuses_missing_turtle_or_wrong_layer_count():
    with pytest.raises(RuntimeError, match="offloaded turtle"):
        probe.retain_first_layer(types.SimpleNamespace(model=_branch(36), turtle_model=None))
    with pytest.raises(RuntimeError, match="Expected 36"):
        probe.retain_first_layer(types.SimpleNamespace(model=_branch(35), turtle_model=_branch(36)))


def test_probe_cannot_invoke_full_result_and_uses_pinned_helpers():
    code = PATH.read_text()
    assert 'R03_LAYER_PROBE") != "1"' in code
    assert "prepare_r03_calibration(args.source)" in code
    assert "make_r03_quantize_config()" in code
    assert "len(names) != 7" in code
    assert "stream_gptq_qdq" not in code
    assert "quantizer_followups.run_one" not in code
