#!/usr/bin/env bash
#SBATCH --job-name=lp-control-action
#SBATCH --account=acc-makhtar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --time=12:00:00
#SBATCH --output=/shared/makhtar/lp_llm/logs/%x-%j.out
set -euo pipefail
ACTION=${MATRIX_ACTION:?Set encode, evaluate, or recover_evaluate}
BRANCH=${MATRIX_BRANCH:?Set s01 or r02}
BITS=${MATRIX_BITS:?Set 4, 6 or 8}
[[ "$BRANCH" == s01 || "$BRANCH" == r02 ]] || exit 1
[[ "$BITS" == 4 || "$BITS" == 6 || "$BITS" == 8 ]] || exit 1
[[ "$ACTION" == encode || "$ACTION" == evaluate || "$ACTION" == recover_evaluate ]] || exit 1
[[ "$ACTION" != recover_evaluate || ( "$BRANCH" == s01 && "$BITS" == 4 ) ]] || exit 1
LP_ROOT=${LP_ROOT:-/shared/makhtar/lp_llm}
LP_CODE=${LP_CODE:-$LP_ROOT/code}
LP_PYTHON=$LP_ROOT/envs/lp_llm/bin/python
RESULT_ROOT=$LP_ROOT/results/qwen3_8b_base
CODE_SHA=${FOLLOWUP_CODE_SHA:?Set reviewed SHA}
[[ "$CODE_SHA" =~ ^[0-9a-f]{40}$ && -x "$LP_PYTHON" && -n "${SLURM_JOB_GPUS:-}" ]] || exit 1
git -C "$LP_CODE" cat-file -e "$CODE_SHA^{commit}"
STAGE=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/lp-control-${SLURM_JOB_ID}.XXXXXXXX")
TIMES=$STAGE/times
mkdir -p "$TIMES" "$RESULT_ROOT/progressive_llm/job_resources" "$RESULT_ROOT/progressive_llm/.locks"
finish() {
  status=$?; trap - EXIT; set +e
  "$LP_PYTHON" -m lp_llm.progressive_job_resources \
    --output "$RESULT_ROOT/progressive_llm/job_resources/control-${BRANCH}-w${BITS}-${ACTION}-${SLURM_JOB_ID}.json" \
    --time-dir "$TIMES" --job-id "$SLURM_JOB_ID" --allocation-mib "${SLURM_MEM_PER_NODE:-0}" \
    --exit-code "$status" --point "control-${BRANCH}-w${BITS}-${ACTION}"
  measured=$?; rm -rf -- "$STAGE"
  if (( status == 0 && measured != 0 )); then status=$measured; fi
  exit "$status"
}
trap finish EXIT
git -C "$LP_CODE" archive --format=tar --output="$STAGE/code.tar" "$CODE_SHA"
tar -xf "$STAGE/code.tar" -C "$STAGE"; rm -- "$STAGE/code.tar"
export PYTHONPATH="$STAGE${PYTHONPATH:+:$PYTHONPATH}" FOLLOWUP_CODE_SHA="$CODE_SHA"
source "$LP_ROOT/env.sh"; unset TRANSFORMERS_CACHE
cd "$STAGE"
CONTROL=$RESULT_ROOT/progressive_llm/controls/${BRANCH}_w${BITS}
exec 9>"$RESULT_ROOT/progressive_llm/.locks/control-${BRANCH}-w${BITS}-${ACTION}.lock"
flock -n 9 || { echo 'ERROR: control action already running' >&2; exit 1; }
if [[ "$ACTION" == recover_evaluate ]]; then
  AUDIT=$RESULT_ROOT/progressive_llm/scaled_signed_v1/control_recovery/s01_w4/audit.json
  [[ -s "$AUDIT" ]] || { echo 'ERROR: CPU W4 recovery audit required' >&2; exit 1; }
  /usr/bin/time -f '%M' -o "$TIMES/recover_evaluate.rss_kib" \
    "$LP_PYTHON" -m lp_llm.progressive_control_recovery evaluate \
      --source "$LP_ROOT/models/qwen3-8b-base" --control "$CONTROL" \
      --checkout "$LP_ROOT/references/rrq_pr2308/auto-round" \
      --matgptq "$LP_ROOT/references/rrq_w2/MatGPTQ" --audit "$AUDIT" \
      --result "$RESULT_ROOT/progressive_llm/scaled_signed_v1/control_recovery/s01_w4/evaluation" \
      --code-sha "$CODE_SHA" --code-repo "$LP_CODE"
else
  if [[ "$ACTION" == encode ]]; then
    if [[ ! -s "$CONTROL/encoding_complete.json" ]]; then
      [[ ! -e "$CONTROL/export.partial" && ! -e "$CONTROL/quantization_process.json" ]] || {
        echo 'ERROR: existing partial or attempted quantization requires inspection, not requantization' >&2; exit 1; }
    fi
  else
    [[ -s "$CONTROL/encoding_complete.json" ]] || {
      echo 'ERROR: validated control encoding required before evaluation' >&2; exit 1; }
  fi
  /usr/bin/time -f '%M' -o "$TIMES/$ACTION.rss_kib" \
    "$LP_PYTHON" -m lp_llm.progressive_control --branch "$BRANCH" --bits "$BITS" \
      --source "$LP_ROOT/models/qwen3-8b-base" --result "$CONTROL" \
      --checkout "$LP_ROOT/references/rrq_pr2308/auto-round" \
      --matgptq "$LP_ROOT/references/rrq_w2/MatGPTQ" --stage "$ACTION"
fi
echo "[CONTROL ACTION NUMERICALLY COMPLETE; RESOURCE CHECK PENDING] $BRANCH W$BITS $ACTION code=$CODE_SHA"
