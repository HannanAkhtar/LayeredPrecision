#!/usr/bin/env bash
set -euo pipefail
LP_ROOT="${LP_ROOT:-/shared/makhtar/lp_llm}"
GPTQ_ENV="$LP_ROOT/envs/gptq_followups"
GPTQ_SOURCE="${GPTQ_SOURCE:-$LP_ROOT/references/quantizer_followups/GPTQModel}"
[[ -x "$LP_ROOT/envs/lp_llm/bin/python" ]] || { echo "ERROR: existing LP env missing" >&2; exit 1; }
[[ "$(git -C "$GPTQ_SOURCE" rev-parse HEAD)" == 70579a771423767da2af57b67d2d3adf014eebb6 ]] || {
  echo "ERROR: GPTQModel checkout revision mismatch" >&2; exit 1;
}
source /opt/miniconda/etc/profile.d/conda.sh
if [[ ! -x "$GPTQ_ENV/bin/python" ]]; then
  conda create --yes --prefix "$GPTQ_ENV" python=3.11 pip
fi
# GPTQModel's pinned source needs a newer Torch/Transformers stack. Install it
# only in this separate environment; the LP environment remains untouched.
"$GPTQ_ENV/bin/python" -m pip install --editable "$GPTQ_SOURCE"
PYTHONPATH="$GPTQ_SOURCE" "$GPTQ_ENV/bin/python" - <<'PY'
import torch, transformers, datasets, gptqmodel
from gptqmodel import GPTQModel
print('GPTQModel:', gptqmodel.__file__)
print('Torch:', torch.__version__, 'Transformers:', transformers.__version__)
PY
echo "[GPTQ ENV READY] $GPTQ_ENV"
