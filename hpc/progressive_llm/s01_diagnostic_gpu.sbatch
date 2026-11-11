#!/usr/bin/env bash
#SBATCH --job-name=s01-residual-diagnostic
#SBATCH --account=acc-makhtar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --time=12:00:00
#SBATCH --output=/shared/makhtar/lp_llm/logs/%x-%j.out
set -euo pipefail
LP_ROOT=${LP_ROOT:-/shared/makhtar/lp_llm}
LP_CODE=${LP_CODE:-$LP_ROOT/code}
LP_PYTHON=$LP_ROOT/envs/lp_llm/bin/python
RESULT_ROOT=$LP_ROOT/results/qwen3_8b_base
CODE_SHA=${FOLLOWUP_CODE_SHA:?Set reviewed code SHA}
ACTION=${S01_DIAG_ACTION:?Set baseline or sweep}
[[ "$ACTION" == baseline || "$ACTION" == sweep ]] || { echo 'ERROR: invalid diagnostic action' >&2; exit 1; }
[[ "$CODE_SHA" =~ ^[0-9a-f]{40}$ && -x "$LP_PYTHON" && -n "${SLURM_JOB_GPUS:-}" ]] || exit 1
git -C "$LP_CODE" cat-file -e "$CODE_SHA^{commit}"
STAGE=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/s01-diagnostic-${SLURM_JOB_ID}.XXXXXXXX")
TIMES=$STAGE/times
mkdir -p "$TIMES" "$RESULT_ROOT/progressive_llm/job_resources" "$RESULT_ROOT/progressive_llm/.locks"
finish() {
  status=$?; trap - EXIT; set +e
  "$LP_PYTHON" -m lp_llm.progressive_job_resources \
    --output "$RESULT_ROOT/progressive_llm/job_resources/s01_${ACTION}-${SLURM_JOB_ID}.json" \
    --time-dir "$TIMES" --job-id "$SLURM_JOB_ID" \
    --allocation-mib "${SLURM_MEM_PER_NODE:-0}" --exit-code "$status" --point "s01_$ACTION"
  measured=$?; rm -rf -- "$STAGE"
  if (( status == 0 && measured != 0 )); then status=$measured; fi
  exit "$status"
}
trap finish EXIT
git -C "$LP_CODE" archive --format=tar --output="$STAGE/code.tar" "$CODE_SHA"
tar -xf "$STAGE/code.tar" -C "$STAGE"; rm -- "$STAGE/code.tar"
export PYTHONPATH="$STAGE${PYTHONPATH:+:$PYTHONPATH}"
source "$LP_ROOT/env.sh"
unset TRANSFORMERS_CACHE
cd "$STAGE"
exec 9>"$RESULT_ROOT/progressive_llm/.locks/s01-${ACTION}-$CODE_SHA.lock"
flock -n 9 || { echo 'ERROR: diagnostic action already running' >&2; exit 1; }
BLOCKS=${S01_DIAG_BLOCKS:-8}
/usr/bin/time -f '%M' -o "$TIMES/$ACTION.rss_kib" \
  "$LP_PYTHON" -m lp_llm.s01_residual_diagnostic "$ACTION" \
    --source "$LP_ROOT/models/qwen3-8b-base" \
    --base "$RESULT_ROOT/quantizer_followups/s01_fp16_load_repair/export" \
    --result-root "$RESULT_ROOT" --code-sha "$CODE_SHA" \
    --matgptq "$LP_ROOT/references/rrq_w2/MatGPTQ" --blocks "$BLOCKS"
echo "[S01 DIAGNOSTIC COMPLETE] action=$ACTION code=$CODE_SHA blocks=$BLOCKS"
