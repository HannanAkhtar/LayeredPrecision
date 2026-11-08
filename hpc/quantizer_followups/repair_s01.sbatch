#!/usr/bin/env bash
#SBATCH --job-name=followup-fp16-repair
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
REPAIR_EXPERIMENT_ID="${REPAIR_EXPERIMENT_ID:-s01}"
case "$REPAIR_EXPERIMENT_ID" in s01|r01|r02) ;; *) echo "ERROR: invalid REPAIR_EXPERIMENT_ID" >&2; exit 1;; esac
[[ "$stage" == encode || "$stage" == all ]] || { echo "ERROR: S01_REPAIR_STAGE must be encode or all" >&2; exit 1; }
source /opt/miniconda/etc/profile.d/conda.sh
conda activate "$LP_ROOT/envs/lp_llm"
source "$LP_ROOT/env.sh"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
FOLLOWUP_CODE_SHA="${FOLLOWUP_CODE_SHA:-$(git -C "$LP_CODE" rev-parse HEAD)}"
[[ "$FOLLOWUP_CODE_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "ERROR: invalid FOLLOWUP_CODE_SHA" >&2; exit 1; }
git -C "$LP_CODE" cat-file -e "$FOLLOWUP_CODE_SHA^{commit}" || {
  echo "ERROR: pinned code commit unavailable: $FOLLOWUP_CODE_SHA" >&2; exit 1;
}
CODE_STAGE=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/lp-s01-repair-${SLURM_JOB_ID:-manual}.XXXXXXXX")
trap 'rm -rf -- "$CODE_STAGE"' EXIT
git -C "$LP_CODE" archive --format=tar --output="$CODE_STAGE/code.tar" "$FOLLOWUP_CODE_SHA"
tar -xf "$CODE_STAGE/code.tar" -C "$CODE_STAGE"
rm -- "$CODE_STAGE/code.tar"
[[ -f "$CODE_STAGE/hpc/quantizer_followups/preflight.sh" && -f "$CODE_STAGE/lp_llm/quantizer_followups_repair.py" ]] || {
  echo "ERROR: staged S01 code incomplete for $FOLLOWUP_CODE_SHA" >&2; exit 1;
}
export LP_CODE="$CODE_STAGE" PYTHONPATH="$CODE_STAGE${PYTHONPATH:+:$PYTHONPATH}"
echo "[CODE] commit=$FOLLOWUP_CODE_SHA stage=$CODE_STAGE"
cd "$CODE_STAGE"
EXPERIMENT_ID="$REPAIR_EXPERIMENT_ID" bash hpc/quantizer_followups/preflight.sh
mkdir -p "$RESULT_ROOT/quantizer_followups/.locks"
exec 8>"$RESULT_ROOT/quantizer_followups/.locks/$REPAIR_EXPERIMENT_ID.lock"
flock -n 8 || { echo "ERROR: original $REPAIR_EXPERIMENT_ID has an active writer" >&2; exit 1; }
exec 9>"$RESULT_ROOT/quantizer_followups/.locks/${REPAIR_EXPERIMENT_ID}_fp16_load_repair.lock"
flock -n 9 || { echo "ERROR: $REPAIR_EXPERIMENT_ID repair already has an active writer" >&2; exit 1; }
echo "[START] $REPAIR_EXPERIMENT_ID FP16-load repair stage=$stage $(date -Is)"
"$LP_PYTHON" -m lp_llm.quantizer_followups_repair \
  --experiment-id "$REPAIR_EXPERIMENT_ID" \
  --source "$LP_SOURCE" \
  --raw-result "$RESULT_ROOT/quantizer_followups/$REPAIR_EXPERIMENT_ID" \
  --repaired-result "$RESULT_ROOT/quantizer_followups/${REPAIR_EXPERIMENT_ID}_fp16_load_repair" \
  --upstream-source "$RRQ_PR_SOURCE" \
  --matgptq-source "$MATGPTQ_SOURCE" \
  --stage "$stage"
echo "[COMPLETE] $REPAIR_EXPERIMENT_ID FP16-load repair stage=$stage $(date -Is)"
