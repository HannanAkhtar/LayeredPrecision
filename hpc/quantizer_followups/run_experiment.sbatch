#!/usr/bin/env bash
#SBATCH --job-name=quant-followup
#SBATCH --account=acc-makhtar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=/shared/makhtar/lp_llm/logs/%x-%j.out
set -euo pipefail
: "${EXPERIMENT_ID:?Set EXPERIMENT_ID to s01, r01, r02, or r03}"
LP_ROOT="${LP_ROOT:-/shared/makhtar/lp_llm}"
LP_CODE="${LP_CODE:-$LP_ROOT/code}"
LP_SOURCE="${LP_SOURCE:-$LP_ROOT/models/qwen3-8b-base}"
RESULT_ROOT="${RESULT_ROOT:-$LP_ROOT/results/qwen3_8b_base}"
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
source /opt/miniconda/etc/profile.d/conda.sh
conda activate "$LP_ROOT/envs/lp_llm"
source "$LP_ROOT/env.sh"
unset TRANSFORMERS_CACHE
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
cd "$LP_CODE"
bash hpc/quantizer_followups/preflight.sh
mkdir -p "$RESULT_ROOT/quantizer_followups/.locks"
exec 9>"$RESULT_ROOT/quantizer_followups/.locks/$EXPERIMENT_ID.lock"
flock -n 9 || { echo "ERROR: $EXPERIMENT_ID already has an active writer" >&2; exit 1; }
if [[ "$EXPERIMENT_ID" == r03 ]]; then
  upstream="$GPTQ_SOURCE"; python="$GPTQ_PYTHON"
else
  upstream="$RRQ_PR_SOURCE"; python="$LP_PYTHON"
fi
if [[ "${FOLLOWUP_SMOKE:-0}" == 1 ]]; then
  reference_path="$upstream"
  if [[ "$EXPERIMENT_ID" != r03 && -n "${RRQ_PR_OVERLAY:-}" ]]; then
    reference_path="$reference_path:$RRQ_PR_OVERLAY"
  fi
  PYTHONPATH="$reference_path:$LP_CODE${PYTHONPATH:+:$PYTHONPATH}" \
    "$python" hpc/quantizer_followups/smoke.py --experiment-id "$EXPERIMENT_ID" --source "$LP_SOURCE"
  echo "[SMOKE COMPLETE] $EXPERIMENT_ID $(date -Is)"
  exit 0
fi
args=(--experiment-id "$EXPERIMENT_ID" --stage "${FOLLOWUP_STAGE:-all}"
      --source "$LP_SOURCE" --result-root "$RESULT_ROOT"
      --upstream-source "$upstream" --matgptq-source "$MATGPTQ_SOURCE" --python "$python")
if [[ -n "${RRQ_PR_OVERLAY:-}" && "$EXPERIMENT_ID" != r03 ]]; then
  args+=(--overlay "$RRQ_PR_OVERLAY")
fi
echo "[START] $EXPERIMENT_ID $(date -Is)"
"$LP_PYTHON" -m lp_llm.quantizer_followups "${args[@]}"
echo "[COMPLETE] $EXPERIMENT_ID $(date -Is)"
