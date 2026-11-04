#!/usr/bin/env bash
#SBATCH --job-name=rtn-phase1-screen
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
LP_SOURCE="${LP_SOURCE:-$LP_ROOT/models/qwen3-8b-base}"
RESULT_ROOT="${RESULT_ROOT:-$LP_ROOT/results/qwen3_8b_base}"
[[ -f "$LP_SOURCE/model.safetensors.index.json" ]] || { echo "ERROR: original checkpoint missing" >&2; exit 1; }
mkdir -p "$RESULT_ROOT/rtn_reproduction"
cd "$LP_CODE"

for experiment in b1_plain_fp16 b2_plain_bf16 b3_opt_fp16; do
    config="$LP_CODE/configs/llm/rtn_reproduction/$experiment.yaml"
    [[ -f "$config" ]] || { echo "ERROR: missing $config" >&2; exit 1; }
    echo "[START] $experiment pinned reference parity, 2-bit checkpoint, logits, diagnostic perplexity"
    python -m lp_llm.run_rtn_reproduction --config "$config" --source "$LP_SOURCE" \
        --result-root "$RESULT_ROOT" --stage screen --device cuda:0
    python -m lp_llm.aggregate_rtn_reproduction --result-root "$RESULT_ROOT"
    echo "[COMPLETE] $experiment"
done
echo "[SUCCESS] Phase 1 screening completed"
