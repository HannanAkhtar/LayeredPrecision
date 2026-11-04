"""Execute the actual pinned AutoRound source, isolated from its package imports.

The three vendored files are byte-identical Git blobs from Intel AutoRound
6c824f741be358983baba87aff83f9c5aa0a01e7 (Apache-2.0). AST extraction
loads the unmodified target function bodies without requiring the full package.
"""

import pytest
import subprocess
import sys
import types
from pathlib import Path
import torch

from lp_llm.quantize import autoround_rtn_sym_quantize
from lp_llm.rtn_reproduction import quantize_base
from lp_llm.upstream_reference import functions


@pytest.fixture(scope="module")
def upstream():
    return functions()


def _signed_from_reference(weight, scale, group_size=128):
    rows, cols = weight.shape
    if cols < group_size:
        grouped = weight
    else:
        pad = (-cols) % group_size
        grouped = torch.nn.functional.pad(weight, (0, pad)).reshape(-1, group_size)
    codes = grouped.div(scale).round_().clamp_(-2, 1)
    return codes.reshape(rows, -1)[:, :cols].int()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("family", ["plain", "optimized"])
def test_actual_pinned_upstream_scale_codes_qdq(upstream, dtype, family):
    torch.manual_seed(331)
    x = torch.randn(6, 259, dtype=dtype)
    x[0] = 0
    x[1] = 2.5
    x[2] = -2.5
    x[3] = x[3].abs() + 0.1
    x[4] = -x[4].abs() - 0.1
    x[5, 0], x[5, 1] = 50, -100
    if family == "plain":
        reference, scale, _ = upstream["quant_tensor_rtn_sym"](x, bits=2, group_size=128)
        if dtype == torch.float32:
            params, actual = autoround_rtn_sym_quantize(x, 2, "group", 128)
        else:
            mode = "autoround_plain_fp16" if dtype == torch.float16 else "autoround_plain_bf16"
            params, actual = quantize_base(x, mode)
    else:
        reference, scale, _ = upstream["quant_tensor_opt_rtn_sym"](x, bits=2, group_size=128)
        # B3 is deliberately FP16; the upstream function is also tested for
        # FP32/BF16 to characterize its dtype behavior independently.
        if dtype != torch.float16:
            assert torch.isfinite(reference).all()
            return
        params, actual = quantize_base(x, "autoround_opt_fp16")
    signed = _signed_from_reference(x, scale)
    assert torch.equal(params["scale"], scale.reshape(6, 3))
    assert torch.equal(params["codes"].int() - 2, signed)
    assert torch.equal(actual, reference)
    delta = (actual.float() - reference.float()).abs()
    assert delta.max().item() == 0
    assert torch.sqrt(delta.square().mean()).item() == 0


def test_input_dtype_reaches_quantizer_statistics(monkeypatch):
    x = torch.tensor([[0.10001, -0.30007, 0.98753] * 43], dtype=torch.bfloat16)
    seen = []
    original_clamp = torch.clamp
    def capture_dtype(tensor, *args, **kwargs):
        seen.append(tensor.dtype)
        return original_clamp(tensor, *args, **kwargs)
    monkeypatch.setattr(torch, "clamp", capture_dtype)
    fp16, a = quantize_base(x, "autoround_plain_fp16")
    bf16, b = quantize_base(x, "autoround_plain_bf16")
    assert a.dtype == torch.float16 and b.dtype == torch.bfloat16
    assert fp16["qdq_dtype"] != bf16["qdq_dtype"]
    assert torch.float16 in seen and torch.bfloat16 in seen
    # BF16 source values in FP16's normal range are often exactly representable
    # in FP16, so distinct computation dtypes need not change these codes.
    assert torch.equal(a.float(), b.float())


@pytest.mark.parametrize("order", ["reference_first", "transformers_first"])
def test_reference_and_transformers_import_orders_in_fresh_process(order):
    script = """
import importlib.util
import sys
import torch
from lp_llm.upstream_reference import functions

def reference():
    upstream = functions()
    x = torch.tensor([[-2., -1., 0., 1.] * 32])
    for name in ('quant_tensor_rtn_sym', 'quant_tensor_opt_rtn_sym'):
        for _ in range(2):
            qdq, scale, _ = upstream[name](x, bits=2, group_size=128)
            assert torch.isfinite(qdq).all() and torch.isfinite(scale).all()
    assert all(name not in sys.modules for name in
               ('auto_round', 'auto_round.data_type', 'auto_round.data_type.gguf'))
    importlib.util.find_spec('auto_round')

if sys.argv[1] == 'reference_first':
    reference()
    import transformers
else:
    import transformers
    reference()
"""
    result = subprocess.run([sys.executable, "-c", script, order],
                            cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_reference_import_scope_restores_entries_on_error_and_preserves_real_modules(monkeypatch):
    upstream = functions()
    names = ("auto_round", "auto_round.data_type", "auto_round.data_type.gguf")
    original = {name: sys.modules.get(name) for name in names}
    with pytest.raises((RuntimeError, IndexError)):
        upstream["quant_tensor_opt_rtn_sym"](torch.empty(1, 0), bits=2, group_size=128)
    assert {name: sys.modules.get(name) for name in names} == original

    real = {name: types.ModuleType(name) for name in names}
    real[names[-1]]._imatrix_handle_zero = lambda *args: None
    for name, module in real.items():
        monkeypatch.setitem(sys.modules, name, module)
    x = torch.tensor([[-2., -1., 0., 1.] * 32])
    upstream["quant_tensor_opt_rtn_sym"](x, bits=2, group_size=128)
    assert all(sys.modules[name] is real[name] for name in names)
