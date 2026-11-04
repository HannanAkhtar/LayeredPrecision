#!/usr/bin/env bash
#SBATCH --job-name=rtn-b3-fp16-checkpoint
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
: "${LP_CODE:?Set LP_CODE to the deployed Phase 1 checkout}"
LP_SOURCE="${LP_SOURCE:-$LP_ROOT/models/qwen3-8b-base}"
RESULT_ROOT="${RESULT_ROOT:-$LP_ROOT/results/qwen3_8b_base}"
[[ -f "$RESULT_ROOT/rtn_reproduction/b3_opt_fp16/artifact/COMPLETE.json" ]] || {
    echo "ERROR: existing B3 artifact is incomplete" >&2; exit 1;
}
[[ -f "$RESULT_ROOT/rtn_reproduction/b3_opt_fp16/checkpoint/COMPLETE.json" ]] || {
    echo "ERROR: existing B3 BF16 checkpoint is incomplete" >&2; exit 1;
}
cd "$LP_CODE"
python -m lp_llm.run_b3_materialization --source "$LP_SOURCE" \
    --result-root "$RESULT_ROOT" --device cuda:0
python -m lp_llm.aggregate_rtn_reproduction --result-root "$RESULT_ROOT"
echo "[SUCCESS] B3 FP16 checkpoint materialization diagnostic completed"
