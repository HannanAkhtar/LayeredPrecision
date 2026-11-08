"""Resolve supplemental GPTQ dependencies and verify the isolated interpreter."""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import inspect
import json
import subprocess
import sys
import sysconfig
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 LP environments
    import tomli as tomllib
from pathlib import Path
from urllib.parse import urlparse, unquote
from urllib.request import url2pathname

from packaging.requirements import Requirement

REVISION = "ed96f2e04ec868b04f969733a242ce3b5bb7cb1e"
MARKER = "quantizer_followups_ready.json"


def editable_distribution(environment: Path, source: Path) -> tuple[metadata.Distribution, Path]:
    """Find PEP 660 dist-info in this interpreter's venv, not its source tree."""
    environment, source = environment.resolve(), source.resolve()
    if Path(sys.prefix).resolve() != environment:
        raise RuntimeError(f"Wrong active interpreter: {sys.prefix}")
    site = Path(sysconfig.get_path("purelib")).resolve()
    if not site.is_relative_to(environment):
        raise RuntimeError(f"GPTQ site-packages escapes isolated environment: {site}")
    matches = [dist for dist in metadata.distributions(path=[str(site)])
               if dist.metadata.get("Name", "").lower().replace("_", "-") == "gptqmodel"]
    if len(matches) != 1 or matches[0].version != "5.7.0":
        raise RuntimeError("Pinned GPTQModel 5.7.0 dist-info is absent from isolated environment")
    direct = json.loads(matches[0].read_text("direct_url.json") or "{}")
    parsed = urlparse(direct.get("url", ""))
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost") or direct.get("dir_info", {}).get("editable") is not True:
        raise RuntimeError("GPTQModel is not a local PEP 660 editable install")
    editable_source = Path(url2pathname(unquote(parsed.path))).resolve()
    if editable_source != source:
        raise RuntimeError(f"GPTQModel editable source differs from pinned checkout: {editable_source}")
    return matches[0], site


def missing_extras(source: Path, pins: Path) -> list[str]:
    project = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    requirements = [Requirement(line) for line in project["dependencies"]]
    pinned = {Requirement(line).name.lower().replace("_", "-"): line for line in pins.read_text().splitlines()
              if line.strip() and not line.lstrip().startswith("#")}
    missing = []
    for requirement in requirements:
        name = requirement.name.lower().replace("_", "-")
        if name in ("torch", "transformers"):
            continue
        try:
            version = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            version = None
        if version is None or version not in requirement.specifier:
            if name not in pinned:
                raise RuntimeError(f"No supplemental pin for {requirement}")
            pin_version = pinned[name].split("==", 1)[-1]
            if pin_version not in requirement.specifier:
                raise RuntimeError(f"Supplemental pin {pinned[name]} violates {requirement}")
            missing.append(pinned[name])
    return missing


def verify(environment: Path, lp_environment: Path, source: Path, mode: str) -> dict:
    import torch
    import transformers
    import datasets
    import gptqmodel
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.quantization import QuantizeConfig
    from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear

    environment, lp_environment, source = (path.resolve() for path in
                                            (environment, lp_environment, source))
    if Path(sys.prefix).resolve() != environment:
        raise RuntimeError(f"Wrong active interpreter: {sys.prefix}")
    expected_core = lp_environment if mode == "light" else environment
    if not Path(torch.__file__).resolve().is_relative_to(expected_core):
        raise RuntimeError(f"Torch did not come from LP environment: {torch.__file__}")
    if not Path(transformers.__file__).resolve().is_relative_to(expected_core):
        raise RuntimeError(f"Transformers did not come from LP environment: {transformers.__file__}")
    if not Path(gptqmodel.__file__).resolve().is_relative_to(source):
        raise RuntimeError(f"GPTQModel did not come from pinned source: {gptqmodel.__file__}")
    installed, installed_site = editable_distribution(environment, source)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise RuntimeError(f"GPTQModel source revision differs: {revision}")
    if not torch.__version__.startswith("2.9.1+cu128") or transformers.__version__ != "4.57.1":
        raise RuntimeError("Inherited Torch/Transformers versions differ from pinned LP versions")
    requirements = [Requirement(line) for line in
                    tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]]
    allowed_sites = (installed_site, Path(torch.__file__).resolve().parents[1])
    allowed = {}
    for site in allowed_sites:
        for dist in metadata.distributions(path=[str(site)]):
            name = dist.metadata.get("Name", "").lower().replace("_", "-")
            allowed.setdefault(name, set()).add(dist.version)
    dependency_versions = {}
    for requirement in requirements:
        name = requirement.name.lower().replace("_", "-")
        version = metadata.version(requirement.name)
        if version not in requirement.specifier or version not in allowed.get(name, set()):
            raise RuntimeError(f"GPTQ dependency outside isolated/LP sites or incompatible: {name}=={version}")
        dependency_versions[name] = version
    qcfg = QuantizeConfig(bits=2, group_size=128, sym=True, desc_act=False,
                          act_group_aware=True, damp_percent=0.05,
                          damp_auto_increment=0.01, static_groups=False,
                          true_sequential=True)
    if (qcfg.bits, qcfg.group_size, qcfg.sym, getattr(qcfg.format, "value", qcfg.format)) != (2, 128, True, "gptq"):
        raise RuntimeError("Pinned GPTQ quantization API/configuration differs")
    if (BACKEND.TORCH.value != "torch" or not callable(TorchQuantLinear.dequantize_weight) or
            "backend" not in inspect.signature(GPTQModel.load).parameters or
            "backend" not in inspect.signature(GPTQModel.quantize).parameters):
        raise RuntimeError("Pinned GPTQ load/quantize/decode API differs")
    return {"environment": str(environment), "lp_environment": str(lp_environment), "mode": mode,
            "source": str(source), "revision": REVISION, "python": str(Path(sys.executable).resolve()),
            "torch": torch.__version__, "torch_file": str(Path(torch.__file__).resolve()),
            "transformers": transformers.__version__,
            "transformers_file": str(Path(transformers.__file__).resolve()),
            "gptqmodel_file": str(Path(gptqmodel.__file__).resolve()),
            "gptqmodel_distribution_site": str(installed_site),
            "gptqmodel_version": installed.version,
            "datasets": datasets.__version__, "dependency_versions": dependency_versions,
            "backend": "TORCH", "bits": 2, "group_size": 128}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("missing", "installed", "initialize", "verify"))
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--lp-env", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pins", type=Path, required=True)
    parser.add_argument("--mode", choices=("light", "clone"), required=True)
    args = parser.parse_args()
    if args.action == "missing":
        print("\n".join(missing_extras(args.source, args.pins)))
        return
    if args.action == "installed":
        installed, site = editable_distribution(args.env, args.source)
        print(json.dumps({"version": installed.version, "dist_info_site": str(site)}))
        return
    current = verify(args.env, args.lp_env, args.source, args.mode)
    if missing_extras(args.source, args.pins):
        raise RuntimeError("GPTQ supplemental dependencies remain missing or incompatible")
    marker = args.env / MARKER
    if args.action == "initialize":
        if marker.exists():
            raise RuntimeError("Environment marker already exists")
        marker.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    elif not marker.is_file() or json.loads(marker.read_text()) != current:
        raise RuntimeError("GPTQ environment is incomplete or its imports changed")
    print(json.dumps(current, indent=2))


if __name__ == "__main__":
    main()
