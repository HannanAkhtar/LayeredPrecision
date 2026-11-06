"""Read actual imports and package versions from a reference subprocess."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


PROBE = """import importlib, importlib.metadata, json, sys
names = sys.argv[1:]
result = {'python': sys.executable, 'modules': {}}
for name in names:
    module = importlib.import_module(name)
    package = name.split('.')[0]
    try:
        distribution = {'cpuinfo': 'py-cpuinfo'}.get(package, package)
        version = (getattr(importlib.import_module(package), '__version__', None)
                   if package == 'auto_round' else importlib.metadata.version(distribution))
    except importlib.metadata.PackageNotFoundError:
        version = getattr(module, '__version__', None)
    result['modules'][name] = {'file': getattr(module, '__file__', None), 'version': version}
print(json.dumps(result))
"""


def probe(python: str, modules: list[str], *, env: dict[str, str] | None = None) -> dict:
    completed = subprocess.run([python, "-c", PROBE, *modules], env=env,
                               text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"Reference import probe failed for {modules}: {completed.stderr[-2000:]}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Reference import probe returned invalid JSON: {completed.stdout[-1000:]}") from error


def source_environment(checkout: Path, overlay: Path | None = None,
                       *, base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    paths = [str(Path(checkout).resolve())]
    if overlay is not None and Path(overlay).is_dir():
        paths.append(str(Path(overlay).resolve()))
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env
