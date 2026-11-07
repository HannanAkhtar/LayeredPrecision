#!/usr/bin/env bash
#SBATCH --job-name=rrq-pr2308
#SBATCH --account=acc-makhtar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=/shared/makhtar/lp_llm/logs/%x-%j.out
set -euo pipefail

: "${EXPERIMENT_ID:?Set EXPERIMENT_ID=pr03 or pr04}"
case "$EXPERIMENT_ID" in pr03|pr04) ;; *) echo "ERROR: only PR03/PR04 are GPU jobs" >&2; exit 1;; esac
LP_ROOT="${LP_ROOT:-/shared/makhtar/lp_llm}"
LP_CODE="${LP_CODE:-$LP_ROOT/code}"
LP_SOURCE="${LP_SOURCE:-$LP_ROOT/models/qwen3-8b-base}"
RESULT_ROOT="${RESULT_ROOT:-$LP_ROOT/results/qwen3_8b_base}"
RRQ_PR_SOURCE="${RRQ_PR_SOURCE:-$LP_ROOT/references/rrq_pr2308/auto-round}"
MATGPTQ_SOURCE="${MATGPTQ_SOURCE:-$LP_ROOT/references/rrq_w2/MatGPTQ}"
LP_PYTHON="${LP_PYTHON:-$LP_ROOT/envs/lp_llm/bin/python}"
source /opt/miniconda/etc/profile.d/conda.sh
conda activate "$LP_ROOT/envs/lp_llm"
source "$LP_ROOT/env.sh"
unset TRANSFORMERS_CACHE
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
cd "$LP_CODE"
bash hpc/rrq_pr2308/preflight.sh
echo "[START] $EXPERIMENT_ID $(date -Is)"
args=(--experiment-id "$EXPERIMENT_ID" --source "$LP_SOURCE" --result-root "$RESULT_ROOT"
      --rrq-source "$RRQ_PR_SOURCE" --matgptq-source "$MATGPTQ_SOURCE" --python "$LP_PYTHON")
if [[ -n "${RRQ_PR_OVERLAY:-}" ]]; then args+=(--overlay "$RRQ_PR_OVERLAY"); fi
"$LP_PYTHON" -m lp_llm.rrq_pr2308 "${args[@]}"
echo "[COMPLETE] $EXPERIMENT_ID $(date -Is)"
