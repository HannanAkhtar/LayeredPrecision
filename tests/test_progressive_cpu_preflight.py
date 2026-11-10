from pathlib import Path
from types import SimpleNamespace

import pytest

from lp_llm import progressive_cpu_preflight as preflight


def test_cpu_preflight_pins_interpreter_versions_and_package_paths(tmp_path, monkeypatch):
    env = tmp_path / "lp_llm"
    python = env / "bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(preflight.sys, "prefix", str(env))
    monkeypatch.setattr(preflight.sys, "executable", str(python))
    versions = {"torch": "2.9.1+cu128", "transformers": "4.57.1"}
    def imported(name):
        return SimpleNamespace(__file__=str(env / "lib" / (name + ".py")),
                               __version__=versions.get(name, "1.0"))
    monkeypatch.setattr(preflight.importlib, "import_module", imported)
    report = preflight.check(env, executable=python)
    assert report["status"] == "cpu_imports_ready"
    assert set(preflight.MODULES) >= {"torch", "safetensors", "numpy",
                                    "lp_llm.progressive_prerequisites"}
    versions["torch"] = "2.8.0"
    with pytest.raises(RuntimeError, match="PyTorch version"):
        preflight.check(env, executable=python)
    versions["torch"] = "2.9.1+cu128"
    monkeypatch.setattr(preflight.importlib, "import_module",
                        lambda name: SimpleNamespace(__file__="/outside/torch.py",
                                                     __version__=versions.get(name, "1.0")))
    with pytest.raises(RuntimeError, match="outside the LP environment"):
        preflight.check(env, executable=python)


def test_cpu_job_uses_direct_interpreter_and_never_initializes_gpu():
    script = (Path(__file__).resolve().parents[1] /
              "hpc/progressive_llm/verify_prerequisites.sbatch").read_text()
    assert 'LP_PYTHON="${LP_PYTHON:-$LP_ROOT/envs/lp_llm/bin/python}"' in script
    assert "lp_llm.progressive_cpu_preflight" in script
    assert "source /opt/miniconda" not in script
    assert 'source "$LP_ROOT/env.sh"' not in script
    assert "conda activate" not in script
    assert "#SBATCH --gres" not in script
    assert "torch.cuda" not in script
