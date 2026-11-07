#!/usr/bin/env bash
#SBATCH --job-name=rrq-w2
#SBATCH --account=acc-makhtar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=/shared/makhtar/lp_llm/logs/%x-%j.out
set -euo pipefail

: "${EXPERIMENT_ID:?Set EXPERIMENT_ID=j01..j14}"
case "$EXPERIMENT_ID" in
  j0[1-7]|j09|j1[0-4]) ;;
  j08) echo 'ERROR: J08 recipe unavailable; no job submitted' >&2; exit 1 ;;
  *) echo "ERROR: invalid experiment: $EXPERIMENT_ID" >&2; exit 1 ;;
esac
LP_ROOT="${LP_ROOT:-/shared/makhtar/lp_llm}"
LP_CODE="${LP_CODE:-$LP_ROOT/code}"
LP_SOURCE="${LP_SOURCE:-$LP_ROOT/models/qwen3-8b-base}"
RESULT_ROOT="${RESULT_ROOT:-$LP_ROOT/results/qwen3_8b_base}"
REF_ROOT="${RRQ_REF_ROOT:-$LP_ROOT/references/rrq_w2}"
LP_PYTHON="${RRQ_PYTHON:-$LP_ROOT/envs/lp_llm/bin/python}"
variant="${RRQ_AUTOROUND_VARIANT:-current}"
case "$variant" in
  current)
    AUTO_SOURCE="$REF_ROOT/auto-round" ;;
  preprint_candidate)
    AUTO_SOURCE="$REF_ROOT/auto-round-preprint-candidate" ;;
  *) echo "ERROR: unknown AutoRound variant: $variant" >&2; exit 1 ;;
esac
source /opt/miniconda/etc/profile.d/conda.sh
conda activate /shared/makhtar/lp_llm/envs/lp_llm
source /shared/makhtar/lp_llm/env.sh
unset TRANSFORMERS_CACHE
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
cd "$LP_CODE"
bash hpc/rrq_w2/preflight.sh
stage="${RRQ_STAGE:-screen}"
echo "[START] $EXPERIMENT_ID stage=$stage"
recovery_args=()
if [[ "${RRQ_REUSE_EXPORT_ONLY:-0}" == 1 ]]; then
  recovery_args+=(--reuse-export-only)
fi
"$LP_PYTHON" -m lp_llm.rrq_runner --experiment-id "$EXPERIMENT_ID" --source "$LP_SOURCE" \
  --result-root "$RESULT_ROOT" --matgptq-source "$REF_ROOT/MatGPTQ" \
  --autoround-source "$AUTO_SOURCE" --autoround-variant "$variant" --stage "$stage" "${recovery_args[@]}"
echo "[COMPLETE] $EXPERIMENT_ID stage=$stage"
