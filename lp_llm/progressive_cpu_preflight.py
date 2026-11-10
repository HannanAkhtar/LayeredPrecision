"""CPU-only import and version gate for the prerequisite Slurm job."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


MODULES = ("torch", "transformers", "numpy", "safetensors", "datasets", "yaml",
           "lp_llm.progressive_plan",
           "lp_llm.progressive_lp", "lp_llm.progressive_support",
           "lp_llm.progressive_prerequisites", "lp_llm.quantizer_followups")


def check(environment: Path, *, executable: Path | None = None) -> dict:
    environment = environment.resolve()
    executable = (executable or Path(sys.executable)).resolve()
    if Path(sys.prefix).resolve() != environment or executable != Path(sys.executable).resolve():
        raise RuntimeError("CPU preflight is not running with the LP environment interpreter")
    versions = {}
    for name in MODULES:
        module = importlib.import_module(name)
        if name in ("torch", "transformers", "numpy", "safetensors", "datasets", "yaml"):
            path = Path(module.__file__).resolve()
            if environment not in path.parents:
                raise RuntimeError(f"{name} resolves outside the LP environment: {path}")
            versions[name] = getattr(module, "__version__", None)
    if not str(versions["torch"]).startswith("2.9.1+cu128"):
        raise RuntimeError(f"Unexpected LP PyTorch version: {versions['torch']}")
    if versions["transformers"] != "4.57.1":
        raise RuntimeError(f"Unexpected LP Transformers version: {versions['transformers']}")
    return {"status": "cpu_imports_ready", "python": str(executable),
            "environment": str(environment), "versions": versions,
            "cuda_initialized": False}


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m lp_llm.progressive_cpu_preflight ENV PYTHON")
    print("[CPU PREREQUISITE IMPORTS READY] " +
          json.dumps(check(Path(sys.argv[1]), executable=Path(sys.argv[2])), sort_keys=True))


if __name__ == "__main__":
    main()
