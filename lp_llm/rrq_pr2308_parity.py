"""CPU source-extracted parity probe for pinned PR base-plane control flow."""

from __future__ import annotations

import ast
import json
import math
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import torch

from .rrq_pr2308 import PR_COMMIT, config_for, lp_revision, lp_worktree_dirty, source_audit
from .storage import atomic_json


def _extract(path: Path, name: str, namespace: dict, *, class_name: str | None = None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = tree.body
    if class_name:
        klass = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name)
        body = klass.body
    node = next(n for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@contextmanager
def _gguf_scope(helper):
    names = ("auto_round", "auto_round.data_type", "auto_round.data_type.gguf")
    if "auto_round" in sys.modules:
        yield
        return
    absent = object()
    old = {name: sys.modules.get(name, absent) for name in names}
    try:
        for name in names:
            if name not in sys.modules:
                module = types.ModuleType(name)
                if name.endswith(".gguf"):
                    module._imatrix_handle_zero = helper
                sys.modules[name] = module
        yield
    finally:
        for name in reversed(names):
            if old[name] is absent:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old[name]


def pinned_functions(checkout: Path) -> dict:
    source_audit(checkout)
    root = checkout / "auto_round"
    ns = {"torch": torch, "ceil": math.ceil,
          "envs": types.SimpleNamespace(AR_SEARCH_SCALE_RATIO=None),
          "logger": types.SimpleNamespace(warning_once=lambda *_: None),
          "transformers": types.SimpleNamespace(pytorch_utils=types.SimpleNamespace(Conv1D=type("Conv1D", (), {})))}
    _extract(root / "utils/common.py", "get_reciprocal", ns)
    for name in ("reshape_pad_tensor_by_group_size", "revert_tensor_by_pad"):
        _extract(root / "data_type/utils.py", name, ns)
    helper = _extract(root / "data_type/gguf.py", "_imatrix_handle_zero", ns)
    _extract(root / "data_type/int.py", "search_scales", ns)
    optimized = _extract(root / "data_type/int.py", "quant_tensor_opt_rtn_sym", ns)
    _extract(root / "data_type/int.py", "quant_tensor_rtn_sym", ns)
    for name in ("_quantize_layer_rrq", "_normalize_scale_zp"):
        _extract(root / "algorithms/quantization/rrq/quantizer.py", name, ns,
                 class_name="RRQRTNQuantizer")

    def opt(*args, **kwargs):
        with _gguf_scope(helper):
            return optimized(*args, **kwargs)

    ns["optimized"] = opt
    return ns


def _codes(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    groups = weight.float().reshape(weight.shape[0], scale.shape[1], -1)
    return torch.round(groups / scale.float()).clamp(-2, 1).to(torch.int8)


def single_probe(ns: dict, dtype: torch.dtype, weight: torch.Tensor,
                 imatrix: torch.Tensor) -> dict:
    weight = weight.to(dtype)
    weighted, weighted_scale, _ = ns["optimized"](weight.clone(), bits=2, group_size=128,
                                                   imatrix=imatrix.clone())
    unweighted, unweighted_scale, _ = ns["optimized"](weight.clone(), bits=2, group_size=128)
    layer = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False, dtype=dtype)
    layer.weight.data.copy_(weight)
    layer.group_size, layer.sym, layer.global_name = 128, True, "fixture.linear"
    layer.imatrix = imatrix.clone()
    calls = []

    class Fixture:
        config = types.SimpleNamespace(bits=2)
        num_planes = 2
        _rrq_scale_dtype = torch.float16
        model_context = object()
        compress_context = object()
        _normalize_scale_zp = ns["_normalize_scale_zp"]

        def _get_quant_func(self, bits, group_size):
            return ns["optimized"]

        def _quantize_layer_via_rtn(self, target, disable_opt_rtn=None):
            calls.append(disable_opt_rtn)
            qdq, scale, zp = ns["optimized"](target.weight.detach().clone(), bits=2,
                                             group_size=128, imatrix=target.imatrix)
            target.weight.data.copy_(qdq)
            target.scale = scale.reshape(target.weight.shape[0], -1).to(torch.float16)
            target.zp = zp

        def _pack_plane(self, dequant, scale, zp, bits, group_size, in_features):
            # PR02 only tests plane zero. The residual pack is checked separately.
            return (torch.zeros(1, dtype=torch.int32), torch.ones(1), torch.zeros(1, dtype=torch.int32))

    ns["_quantize_layer_rrq"](Fixture(), layer)
    actual = layer.weight.detach()
    expected = weighted.to(dtype)
    if calls != [False] or not torch.equal(actual, expected):
        raise ValueError("RRQ PR base does not equal standard importance-weighted optimized RTN")
    if not torch.equal(layer.scale.float(), weighted_scale.reshape(layer.scale.shape).to(torch.float16).float()):
        raise ValueError("RRQ PR base scale differs from optimized RTN")
    code_equal = torch.equal(_codes(weight, weighted_scale.reshape(weight.shape[0], -1, 1)),
                             _codes(weight, layer.scale.unsqueeze(-1)))
    if not code_equal:
        raise ValueError("RRQ PR base codes differ from optimized RTN")
    delta = actual.float() - expected.float()
    weighted_error = (actual.float() - weight.float()).square() * imatrix.reshape(1, -1)
    unweighted_weighted_error = (unweighted.float() - weight.float()).square() * imatrix.reshape(1, -1)
    return {"dtype": str(dtype), "base_delegated_to_standard_rtn": calls == [False],
            "weighted_codes_exact": code_equal, "weighted_scales_exact": True,
            "weighted_qdq_exact_fraction": float((actual == expected).float().mean()),
            "weighted_qdq_max_abs": float(delta.abs().max()),
            "weighted_qdq_mse": float(delta.square().mean()),
            "weighted_reconstruction_mse": float(weighted_error.mean()),
            "importance_weighted_mse_unweighted_control": float(unweighted_weighted_error.mean()),
            "unweighted_reconstruction_mse": float(((unweighted.float() - weight.float()).square()).mean()),
            "weighted_vs_unweighted_scale_exact_fraction": float((weighted_scale == unweighted_scale).float().mean()),
            "weighted_vs_unweighted_qdq_exact_fraction": float((weighted == unweighted).float().mean())}


def run_parity(checkout: Path, result: Path) -> dict:
    ns = pinned_functions(checkout)
    generator = torch.Generator().manual_seed(2308)
    weight = torch.randn((3, 256), generator=generator) * 0.2
    # Column importance changes the optimized scale optimum on this fixture.
    imatrix = torch.exp(torch.linspace(-4, 4, 256))
    cases = [single_probe(ns, dtype, weight, imatrix) for dtype in (torch.float16, torch.bfloat16)]
    if all(row["weighted_vs_unweighted_scale_exact_fraction"] == 1 for row in cases):
        raise ValueError("PR02 fixture did not exercise importance-weighted scale selection")
    record = {"status": "completed_validated", "experiment_id": "pr02", "source_revision": PR_COMMIT,
              "lp_git_commit": lp_revision(), "config_sha256": config_for("pr02")["config_sha256"],
              "lp_worktree_dirty": lp_worktree_dirty(),
              "source_checkout": str(checkout.resolve()),
              "method": "AST-extracted pinned PR _quantize_layer_rrq on minimal Linear; standard base delegate "
                        "uses pinned PR opt_rtn_int_sym; residual packing is separately tested",
              "limitation": "Does not execute full compressor WrapperLinear or model calibration",
              "cases": cases}
    atomic_json(result / "parity.json", record)
    return record
