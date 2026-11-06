"""Plan and optionally install small AutoRound dependencies outside the LP environment."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from .rrq_runtime import probe, source_environment
from .storage import atomic_json


DIRECT = ("pydantic", "cpuinfo", "filelock", "requests")
PINS = {"pydantic": "pydantic==2.12.5", "pydantic_core": "pydantic-core==2.41.5",
        "annotated_types": "annotated-types==0.7.0", "typing_inspection": "typing-inspection==0.4.2",
        "typing_extensions": "typing-extensions==4.15.0", "cpuinfo": "py-cpuinfo==9.0.0",
        "filelock": "filelock==3.20.0", "requests": "requests==2.32.5",
        "charset_normalizer": "charset-normalizer==3.4.4", "idna": "idna==3.10",
        "urllib3": "urllib3==2.5.0", "certifi": "certifi==2025.10.5"}
PYDANTIC_DEPS = ("pydantic_core", "annotated_types", "typing_inspection", "typing_extensions")
REQUESTS_DEPS = ("charset_normalizer", "idna", "urllib3", "certifi")
MINIMUM = {"pydantic": (2,), "annotated_types": (0, 6),
           "typing_inspection": (0, 4, 2), "typing_extensions": (4, 14, 1)}
DIST_NAMES = {"annotated_types": "annotated-types", "typing_inspection": "typing-inspection",
              "typing_extensions": "typing-extensions"}


def available(module: str) -> bool:
    try:
        importlib.import_module(module)
        if module in MINIMUM:
            version = importlib.metadata.version(DIST_NAMES.get(module, module))
            numbers = tuple(int(x) for x in re.findall(r"\d+", version)[:3])
            required = MINIMUM[module]
            return numbers[:len(required)] >= required
        return True
    except (ImportError, ValueError, RuntimeError, AttributeError):
        return False


def planned_packages() -> list[str]:
    selected = []
    if not available("pydantic"):
        selected.extend(("pydantic", "pydantic_core"))
    selected.extend(name for name in PYDANTIC_DEPS[1:] if not available(name))
    for name in ("cpuinfo", "filelock", "requests"):
        if not available(name):
            selected.append(name)
    if "requests" in selected:
        selected.extend(name for name in REQUESTS_DEPS if not available(name))
    return [PINS[name] for name in dict.fromkeys(selected)]


def validate_overlay(target: Path, python: str = sys.executable) -> dict:
    marker = target / "OVERLAY_COMPLETE.json"
    if not marker.is_file():
        raise ValueError(f"Incomplete dependency overlay: {target}")
    details = json.loads(marker.read_text())
    if not isinstance(details.get("packages"), list) or not isinstance(details.get("validated_versions"), dict):
        raise ValueError(f"Invalid dependency overlay marker: {marker}")
    result = probe(python, list(dict.fromkeys((*DIRECT, *PYDANTIC_DEPS, *REQUESTS_DEPS))),
                   env=source_environment(target))
    paths = result["modules"]
    if int(paths["pydantic"]["version"].split(".")[0]) < 2:
        raise ValueError("AutoRound requires Pydantic v2")
    for pin in details["packages"]:
        module = next((name for name, expected in PINS.items() if expected == pin), None)
        entry = paths.get(module, {})
        imported = entry.get("file")
        if (module is None or not imported or
                not Path(imported).resolve().is_relative_to(target.resolve()) or
                entry.get("version") != pin.split("==", 1)[1]):
            raise ValueError(f"Overlay package did not resolve from target: {pin}")
    return result


def install(target: Path, python: str = sys.executable) -> dict:
    target = Path(target)
    if target.exists():
        return {"status": "already_complete", "imports": validate_overlay(target, python)}
    pins = planned_packages()
    if not pins:
        return {"status": "existing_environment_sufficient", "packages": []}
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="extra_deps.stage-", dir=target.parent))
    command = [python, "-m", "pip", "install", "--target", str(staging),
               "--no-deps", "--only-binary=:all:", *pins]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"Overlay install failed; inspect {staging}: {completed.stderr[-2000:]}")
    # Check all direct imports and Pydantic's genuinely required dependencies.
    modules = list(dict.fromkeys((*DIRECT, *PYDANTIC_DEPS, *REQUESTS_DEPS)))
    imports = probe(python, modules, env=source_environment(staging))
    if int(imports["modules"]["pydantic"]["version"].split(".")[0]) < 2:
        raise ValueError(f"Overlay Pydantic v2 validation failed: {staging}")
    for pin in pins:
        module = next(name for name, expected in PINS.items() if expected == pin)
        entry = imports["modules"].get(module, {})
        imported = entry.get("file")
        if (not imported or not Path(imported).resolve().is_relative_to(staging.resolve()) or
                entry.get("version") != pin.split("==", 1)[1]):
            raise ValueError(f"Package did not install into overlay: {pin}; inspect {staging}")
    atomic_json(staging / "OVERLAY_COMPLETE.json", {"packages": pins,
                "python": python, "validated_versions": {
                    name: entry["version"] for name, entry in imports["modules"].items()}})
    os.replace(staging, target)
    return {"status": "complete", "packages": pins, "target": str(target), "imports": imports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "install"))
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "install":
        value = install(args.target)
    else:
        pins = planned_packages()
        value = {"missing_supplemental_packages": pins,
                 "overlay": str(args.target), "overlay_ready": False}
        if args.target.exists():
            value["overlay_imports"] = validate_overlay(args.target)
            value["overlay_ready"] = True
        if pins and not value["overlay_ready"]:
            value["install_command"] = (f"{sys.executable} -m lp_llm.rrq_overlay install "
                                        f"--target {args.target}")
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
