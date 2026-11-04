#!/usr/bin/env bash
#SBATCH --job-name=rtn-phase1-full
#SBATCH --account=acc-makhtar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=/shared/makhtar/lp_llm/logs/%x-%j.out
set -euo pipefail

source /opt/miniconda/etc/profile.d/conda.sh
conda activate /shared/makhtar/lp_llm/envs/lp_llm
source /shared/makhtar/lp_llm/env.sh
unset TRANSFORMERS_CACHE
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

LP_ROOT="${LP_ROOT:-/shared/makhtar/lp_llm}"
: "${LP_CODE:?Set LP_CODE to the Phase 1 branch checkout}"
: "${RTN_EXPERIMENTS:?Set RTN_EXPERIMENTS to the screened IDs selected for full evaluation}"
LP_SOURCE="${LP_SOURCE:-$LP_ROOT/models/qwen3-8b-base}"
RESULT_ROOT="${RESULT_ROOT:-$LP_ROOT/results/qwen3_8b_base}"
EVAL_CODE="${EVAL_CODE:-$LP_CODE}"
[[ -f "$EVAL_CODE/lp_llm/evaluate_matgptq.py" ]] || {
    echo "ERROR: existing untracked MatGPTQ evaluator is missing" >&2; exit 1;
}
cd "$LP_CODE"

for experiment in $RTN_EXPERIMENTS; do
    case "$experiment" in
        b1_plain_fp16|b2_plain_bf16|b3_opt_fp16) ;;
        *) echo "ERROR: invalid experiment ID: $experiment" >&2; exit 1 ;;
    esac
    config="$LP_CODE/configs/llm/rtn_reproduction/$experiment.yaml"
    echo "[START] full matched WikiText-2 evaluation: $experiment"
    python -m lp_llm.run_rtn_reproduction --config "$config" --source "$LP_SOURCE" \
        --result-root "$RESULT_ROOT" --stage full --evaluator-code "$EVAL_CODE"
    python -m lp_llm.aggregate_rtn_reproduction --result-root "$RESULT_ROOT"
    echo "[COMPLETE] $experiment"
done
echo "[SUCCESS] Selected Phase 1 full evaluations completed"
