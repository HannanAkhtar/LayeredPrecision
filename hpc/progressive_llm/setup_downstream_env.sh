#!/usr/bin/env bash
set -euo pipefail
LP_ROOT=${LP_ROOT:-/shared/makhtar/lp_llm}
export LP_ROOT
PARENT=$LP_ROOT/envs/lp_llm/bin/python
VENV=$LP_ROOT/envs/downstream
SOURCE=$LP_ROOT/references/downstream/lm-evaluation-harness
export SOURCE
PIN=d021bf846218f1bb3bdc0603864789329476d464
[[ -x "$PARENT" ]] || { echo 'Original LP Python missing' >&2; exit 2; }
"$PARENT" - <<'PY'
import torch, transformers
assert torch.__version__ == '2.9.1+cu128', torch.__version__
assert transformers.__version__ == '4.57.1', transformers.__version__
PY
mkdir -p "$(dirname "$SOURCE")"
if [[ ! -d "$SOURCE/.git" ]]; then
  git clone https://github.com/EleutherAI/lm-evaluation-harness.git "$SOURCE"
  git -C "$SOURCE" checkout --detach "$PIN"
fi
[[ $(git -C "$SOURCE" rev-parse HEAD) == "$PIN" ]] || {
  echo "Harness checkout must be pinned to $PIN before setup" >&2; exit 2;
}
if [[ ! -x "$VENV/bin/python" ]]; then
  "$PARENT" -m venv --system-site-packages "$VENV"
fi
"$VENV/bin/python" -m pip install -e "$SOURCE"
"$VENV/bin/python" -m pip check
"$VENV/bin/python" - <<'PY'
import importlib.metadata as m
import os, pathlib, sys, torch, transformers, lm_eval
from lm_eval import simple_evaluate
assert m.version('lm_eval') == '0.4.9.1'
assert torch.__version__ == '2.9.1+cu128'
assert transformers.__version__ == '4.57.1'
assert pathlib.Path(sys.prefix).name == 'downstream'
parent = pathlib.Path(os.environ['LP_ROOT'])/'envs/lp_llm'
assert pathlib.Path(torch.__file__).resolve().is_relative_to(parent)
assert pathlib.Path(transformers.__file__).resolve().is_relative_to(parent)
assert pathlib.Path(lm_eval.__file__).resolve().is_relative_to(pathlib.Path(os.environ['SOURCE']))
print('[DOWNSTREAM ENV READY]', sys.executable)
PY
