#!/usr/bin/env bash
#SBATCH --job-name=s01-fp16-repair
#SBATCH --account=acc-makhtar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=/shared/makhtar/lp_llm/logs/%x-%j.out
set -euo pipefail
LP_ROOT="${LP_ROOT:-/shared/makhtar/lp_llm}"
LP_CODE="${LP_CODE:-$LP_ROOT/code}"
LP_SOURCE="${LP_SOURCE:-$LP_ROOT/models/qwen3-8b-base}"
RESULT_ROOT="${RESULT_ROOT:-$LP_ROOT/results/qwen3_8b_base}"
RRQ_PR_SOURCE="${RRQ_PR_SOURCE:-$LP_ROOT/references/rrq_pr2308/auto-round}"
MATGPTQ_SOURCE="${MATGPTQ_SOURCE:-$LP_ROOT/references/rrq_w2/MatGPTQ}"
LP_PYTHON="${LP_PYTHON:-$LP_ROOT/envs/lp_llm/bin/python}"
stage="${S01_REPAIR_STAGE:-encode}"
[[ "$stage" == encode || "$stage" == all ]] || { echo "ERROR: S01_REPAIR_STAGE must be encode or all" >&2; exit 1; }
source /opt/miniconda/etc/profile.d/conda.sh
conda activate "$LP_ROOT/envs/lp_llm"
source "$LP_ROOT/env.sh"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
cd "$LP_CODE"
EXPERIMENT_ID=s01 bash hpc/quantizer_followups/preflight.sh
mkdir -p "$RESULT_ROOT/quantizer_followups/.locks"
exec 9>"$RESULT_ROOT/quantizer_followups/.locks/s01_fp16_load_repair.lock"
flock -n 9 || { echo "ERROR: S01 repair already has an active writer" >&2; exit 1; }
echo "[START] S01 FP16-load repair stage=$stage $(date -Is)"
"$LP_PYTHON" -m lp_llm.quantizer_followups_repair \
  --source "$LP_SOURCE" \
  --raw-result "$RESULT_ROOT/quantizer_followups/s01" \
  --repaired-result "$RESULT_ROOT/quantizer_followups/s01_fp16_load_repair" \
  --upstream-source "$RRQ_PR_SOURCE" \
  --matgptq-source "$MATGPTQ_SOURCE" \
  --stage "$stage"
echo "[COMPLETE] S01 FP16-load repair stage=$stage $(date -Is)"
