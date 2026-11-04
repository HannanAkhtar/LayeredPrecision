"""Load exact pinned AutoRound functions for diagnostics, without package installation.

This module executes only selected function AST nodes from byte-verified
Apache-2.0 source fixtures. It is never needed by checkpoint decoding.
"""

from __future__ import annotations

import ast
from contextlib import contextmanager
from functools import wraps
import hashlib
import sys
import types
from pathlib import Path

import torch


FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
SOURCES = {
    "autoround_int_6c824f7.py": "f65c3991693836c8b9c14a7ae9df9e4aa7582ad3",
    "autoround_data_type_utils_6c824f7.py": "fb679e89b0f4fb9e1c7d652fee4201016a409ffa",
    "autoround_common_6c824f7.py": "2a932d91a28df5d93862d9efdbeb7488123e3b0b",
}


def _extract(path: Path, names: tuple[str, ...], namespace: dict):
    raw = path.read_bytes()
    git_hash = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if git_hash != SOURCES[path.name]:
        raise ValueError(f"Pinned AutoRound source fixture changed: {path}")
    nodes = [node for node in ast.parse(raw.decode()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise ValueError(f"Missing pinned function in {path}")
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


@contextmanager
def _gguf_import_scope():
    """Supply the unused imatrix helper only while a reference call runs."""
    names = ("auto_round", "auto_round.data_type", "auto_round.data_type.gguf")
    if "auto_round" in sys.modules:
        # Let an installed AutoRound resolve its own import normally.
        yield
        return
    absent = object()
    original = {name: sys.modules.get(name, absent) for name in names}
    try:
        for name in names:
            if name not in sys.modules:
                module = types.ModuleType(name)
                if name.endswith(".gguf"):
                    module._imatrix_handle_zero = lambda *args: None
                sys.modules[name] = module
        yield
    finally:
        for name in reversed(names):
            if original[name] is absent:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original[name]


def functions() -> dict:
    namespace = {"torch": torch, "envs": types.SimpleNamespace(AR_SEARCH_SCALE_RATIO=None)}
    _extract(FIXTURES / "autoround_common_6c824f7.py", ("get_reciprocal",), namespace)
    from math import ceil
    namespace["ceil"] = ceil
    _extract(FIXTURES / "autoround_data_type_utils_6c824f7.py",
             ("reshape_pad_tensor_by_group_size", "revert_tensor_by_pad"), namespace)
    _extract(FIXTURES / "autoround_int_6c824f7.py",
             ("search_scales", "quant_tensor_rtn_sym", "quant_tensor_opt_rtn_sym"), namespace)
    # The optimized function imports this helper unconditionally, but calls it
    # only when imatrix is supplied. Phase 1 deliberately uses imatrix=None.
    raw_optimized = namespace["quant_tensor_opt_rtn_sym"]

    @wraps(raw_optimized)
    def optimized(*args, **kwargs):
        with _gguf_import_scope():
            return raw_optimized(*args, **kwargs)

    namespace["quant_tensor_opt_rtn_sym"] = optimized
    return namespace


def direct_qdq(weight: torch.Tensor, algorithm: str, group_size: int = 128):
    fn = functions()["quant_tensor_rtn_sym" if algorithm == "plain" else "quant_tensor_opt_rtn_sym"]
    qdq, scale, _ = fn(weight, bits=2, group_size=group_size)
    return qdq, scale
