#!/usr/bin/env bash
#SBATCH --job-name=lp-w4-recover-audit
#SBATCH --account=acc-makhtar
#SBATCH --partition=hpc
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/shared/makhtar/lp_llm/logs/%x-%j.out
set -euo pipefail
LP_ROOT=${LP_ROOT:-/shared/makhtar/lp_llm}
LP_CODE=${LP_CODE:-$LP_ROOT/code}
LP_PYTHON=$LP_ROOT/envs/lp_llm/bin/python
RESULT_ROOT=$LP_ROOT/results/qwen3_8b_base
CODE_SHA=${FOLLOWUP_CODE_SHA:?Set reviewed SHA}
[[ "$CODE_SHA" =~ ^[0-9a-f]{40}$ && -x "$LP_PYTHON" && -z "${SLURM_JOB_GPUS:-}" ]] || exit 1
git -C "$LP_CODE" cat-file -e "$CODE_SHA^{commit}"
STAGE=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/lp-w4-audit-${SLURM_JOB_ID}.XXXXXXXX")
TIMES=$STAGE/times
mkdir -p "$TIMES" "$RESULT_ROOT/progressive_llm/job_resources" "$RESULT_ROOT/progressive_llm/.locks"
finish() {
  status=$?; trap - EXIT; set +e
  "$LP_PYTHON" -m lp_llm.progressive_job_resources \
    --output "$RESULT_ROOT/progressive_llm/job_resources/w4-recover-audit-${SLURM_JOB_ID}.json" \
    --time-dir "$TIMES" --job-id "$SLURM_JOB_ID" --allocation-mib "${SLURM_MEM_PER_NODE:-0}" \
    --exit-code "$status" --point w4-recover-audit
  measured=$?; rm -rf -- "$STAGE"
  if (( status == 0 && measured != 0 )); then status=$measured; fi
  exit "$status"
}
trap finish EXIT
git -C "$LP_CODE" archive --format=tar --output="$STAGE/code.tar" "$CODE_SHA"
tar -xf "$STAGE/code.tar" -C "$STAGE"; rm -- "$STAGE/code.tar"
export PYTHONPATH="$STAGE${PYTHONPATH:+:$PYTHONPATH}" FOLLOWUP_CODE_SHA="$CODE_SHA"
cd "$STAGE"
exec 9>"$RESULT_ROOT/progressive_llm/.locks/w4-recover-audit-$CODE_SHA.lock"
flock -n 9 || { echo 'ERROR: W4 audit already running' >&2; exit 1; }
OUT=$RESULT_ROOT/progressive_llm/scaled_signed_v1/control_recovery/s01_w4/audit.json
/usr/bin/time -f '%M' -o "$TIMES/audit.rss_kib" \
  "$LP_PYTHON" -m lp_llm.progressive_control_recovery audit \
    --source "$LP_ROOT/models/qwen3-8b-base" \
    --control "$RESULT_ROOT/progressive_llm/controls/s01_w4" \
    --checkout "$LP_ROOT/references/rrq_pr2308/auto-round" \
    --audit "$OUT" --code-sha "$CODE_SHA"
echo "[W4 PARTIAL AUDITED; RESOURCE CHECK PENDING] code=$CODE_SHA"
