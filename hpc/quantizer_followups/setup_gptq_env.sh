#!/usr/bin/env bash
set -euo pipefail
LP_ROOT="${LP_ROOT:-/shared/makhtar/lp_llm}"
GPTQ_ENV="$LP_ROOT/envs/gptq_followups_v570"
GPTQ_SOURCE="${GPTQ_SOURCE:-$LP_ROOT/references/quantizer_followups/GPTQModel}"
[[ -x "$LP_ROOT/envs/lp_llm/bin/python" ]] || { echo "ERROR: existing LP env missing" >&2; exit 1; }
[[ "$(git -C "$GPTQ_SOURCE" rev-parse HEAD)" == ed96f2e04ec868b04f969733a242ce3b5bb7cb1e ]] || {
  echo "ERROR: GPTQModel checkout revision mismatch" >&2; exit 1;
}
git -C "$GPTQ_SOURCE" diff --quiet HEAD || { echo "ERROR: GPTQModel checkout has local changes" >&2; exit 1; }
source /opt/miniconda/etc/profile.d/conda.sh
if [[ ! -x "$GPTQ_ENV/bin/python" ]]; then
  conda create --yes --copy --prefix "$GPTQ_ENV" --clone "$LP_ROOT/envs/lp_llm"
fi
# Keep the CUDA 12.8-compatible Torch and Transformers from the cloned LP env.
# Resolve any missing GPTQModel dependencies in this clone only, without allowing
# pip to replace the core stack or touch the original LP environment.
"$GPTQ_ENV/bin/python" -m pip install --no-deps -r "$(dirname "$0")/gptq_v570_extras.txt"
"$GPTQ_ENV/bin/python" -m pip install --no-deps --no-build-isolation --editable "$GPTQ_SOURCE"
PYTHONPATH="$GPTQ_SOURCE" "$GPTQ_ENV/bin/python" - <<'PY'
import torch, transformers, datasets, gptqmodel
from gptqmodel import BACKEND, GPTQModel
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
assert torch.__version__.startswith('2.9.1+cu128'), torch.__version__
assert transformers.__version__ == '4.57.1', transformers.__version__
assert BACKEND.TORCH and QuantizeConfig and TorchQuantLinear
print('GPTQModel:', gptqmodel.__file__)
print('Torch:', torch.__version__, 'Transformers:', transformers.__version__)
PY
"$GPTQ_ENV/bin/python" -m pip check || {
  echo "ERROR: GPTQ clone has missing or incompatible dependencies; install only the reported packages in $GPTQ_ENV with --no-deps, then rerun this script" >&2
  exit 1
}
"$GPTQ_ENV/bin/python" -m pip freeze > "$GPTQ_ENV/quantizer_followups_requirements_lock.txt"
echo "[GPTQ ENV READY] $GPTQ_ENV"
