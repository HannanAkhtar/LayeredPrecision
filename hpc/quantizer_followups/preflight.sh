#!/usr/bin/env bash
set -euo pipefail
: "${EXPERIMENT_ID:?Set EXPERIMENT_ID to s01, r01, r02, or r03}"
case "$EXPERIMENT_ID" in s01|r01|r02|r03) ;; *) echo "ERROR: invalid experiment ID" >&2; exit 1;; esac
LP_ROOT="${LP_ROOT:-/shared/makhtar/lp_llm}"
LP_CODE="${LP_CODE:-$LP_ROOT/code}"
LP_SOURCE="${LP_SOURCE:-$LP_ROOT/models/qwen3-8b-base}"
RRQ_PR_SOURCE="${RRQ_PR_SOURCE:-$LP_ROOT/references/rrq_pr2308/auto-round}"
GPTQ_SOURCE="${GPTQ_SOURCE:-$LP_ROOT/references/quantizer_followups/GPTQModel}"
MATGPTQ_SOURCE="${MATGPTQ_SOURCE:-$LP_ROOT/references/rrq_w2/MatGPTQ}"
LP_PYTHON="${LP_PYTHON:-$LP_ROOT/envs/lp_llm/bin/python}"
GPTQ_ENV_MODE="${GPTQ_ENV_MODE:-light}"
case "$GPTQ_ENV_MODE" in
  light) GPTQ_ENV="$LP_ROOT/envs/gptq_followups_v570_light" ;;
  clone) GPTQ_ENV="$LP_ROOT/envs/gptq_followups_v570" ;;
  *) echo "ERROR: GPTQ_ENV_MODE must be light or clone" >&2; exit 1 ;;
esac
GPTQ_ENV="${GPTQ_ENV_OVERRIDE:-$GPTQ_ENV}"
GPTQ_PYTHON="${GPTQ_PYTHON:-$GPTQ_ENV/bin/python}"
[[ -x "$LP_PYTHON" && -f "$LP_SOURCE/config.json" && -f "$LP_SOURCE/model.safetensors.index.json" ]] || {
  echo "ERROR: LP Python or pinned source checkpoint is missing" >&2; exit 1;
}
[[ -d "$LP_CODE/lp_llm" ]] || { echo "ERROR: LP code missing" >&2; exit 1; }
[[ "$(git -C "$MATGPTQ_SOURCE" rev-parse HEAD)" == 0e5bd93bc53381f7195db870f5408ae04906bb3c ]] || {
  echo "ERROR: MatGPTQ revision mismatch" >&2; exit 1;
}
if [[ "$EXPERIMENT_ID" == r03 ]]; then
  UPSTREAM_SOURCE="$GPTQ_SOURCE"
  PYTHON="$GPTQ_PYTHON"
  revision=ed96f2e04ec868b04f969733a242ce3b5bb7cb1e
else
  UPSTREAM_SOURCE="$RRQ_PR_SOURCE"
  PYTHON="$LP_PYTHON"
  revision=7985c1b0f4eef65fb717511668fda7e347e584c3
fi
reference_path="$UPSTREAM_SOURCE"
if [[ "$EXPERIMENT_ID" != r03 && -n "${RRQ_PR_OVERLAY:-}" ]]; then
  [[ -d "$RRQ_PR_OVERLAY" ]] || { echo "ERROR: AutoRound overlay missing" >&2; exit 1; }
  "$LP_PYTHON" -c 'from pathlib import Path; from lp_llm.rrq_overlay import validate_overlay; import sys; validate_overlay(Path(sys.argv[1]), sys.argv[2])' "$RRQ_PR_OVERLAY" "$LP_PYTHON"
  reference_path="$reference_path:$RRQ_PR_OVERLAY"
fi
[[ -x "$PYTHON" && "$(git -C "$UPSTREAM_SOURCE" rev-parse HEAD)" == "$revision" ]] || {
  echo "ERROR: pinned $EXPERIMENT_ID reference or isolated Python missing" >&2; exit 1;
}
git -C "$UPSTREAM_SOURCE" diff --quiet HEAD || { echo "ERROR: reference source modified" >&2; exit 1; }
if [[ -n "${SLURM_MEM_PER_NODE:-}" && "$SLURM_MEM_PER_NODE" -lt 12000 ]]; then
  echo "ERROR: allocation below 12 GB" >&2; exit 1
fi
command -v nvidia-smi >/dev/null || { echo "ERROR: GPU unavailable" >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
"$PYTHON" - <<'PY'
import torch, transformers, datasets, safetensors
assert torch.cuda.is_available(), 'CUDA unavailable'
print('Torch', torch.__version__, 'Transformers', transformers.__version__)
PY
if [[ "$EXPERIMENT_ID" == r03 ]]; then
  PYTHONPATH="$GPTQ_SOURCE" "$PYTHON" "$LP_CODE/hpc/quantizer_followups/gptq_env_check.py" verify \
    --env "$GPTQ_ENV" --lp-env "$LP_ROOT/envs/lp_llm" --source "$GPTQ_SOURCE" \
    --pins "$LP_CODE/hpc/quantizer_followups/gptq_v570_extras.txt" --mode "$GPTQ_ENV_MODE"
  PYTHONPATH="$reference_path${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -c 'import torch, transformers, gptqmodel; from gptqmodel import BACKEND, GPTQModel; from gptqmodel.quantization import QuantizeConfig; from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear; assert torch.__version__.startswith("2.9.1+cu128"), torch.__version__; assert transformers.__version__ == "4.57.1", transformers.__version__; assert BACKEND.TORCH and GPTQModel and QuantizeConfig and TorchQuantLinear'
else
  PYTHONPATH="$reference_path${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -c 'import auto_round; from auto_round import AutoRound'
fi
echo "[PREFLIGHT COMPLETE] $EXPERIMENT_ID"
