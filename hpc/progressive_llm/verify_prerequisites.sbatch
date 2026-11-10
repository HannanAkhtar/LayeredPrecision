#!/usr/bin/env bash
#SBATCH --job-name=lp-llm-prerequisites
#SBATCH --account=acc-makhtar
#SBATCH --partition=hpc
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/shared/makhtar/lp_llm/logs/%x-%j.out
set -euo pipefail
LP_ROOT="${LP_ROOT:-/shared/makhtar/lp_llm}"
LP_CODE="${LP_CODE:-$LP_ROOT/code}"
LP_SOURCE="${LP_SOURCE:-$LP_ROOT/models/qwen3-8b-base}"
RESULT_ROOT="${RESULT_ROOT:-$LP_ROOT/results/qwen3_8b_base}"
LP_PYTHON="${LP_PYTHON:-$LP_ROOT/envs/lp_llm/bin/python}"
RRQ_PR_SOURCE="${RRQ_PR_SOURCE:-$LP_ROOT/references/rrq_pr2308/auto-round}"
FOLLOWUP_CODE_SHA="${FOLLOWUP_CODE_SHA:?Set the reviewed code SHA}"
[[ "${SLURM_JOB_PARTITION:-}" == hpc && -z "${SLURM_JOB_GPUS:-}" &&
   "${SLURM_MEM_PER_NODE:-0}" -ge 32768 ]] || {
  echo 'ERROR: prerequisite verification needs CPU-only hpc partition and 32 GiB RAM' >&2
  exit 1
}
[[ "$FOLLOWUP_CODE_SHA" =~ ^[0-9a-f]{40}$ && -x "$LP_PYTHON" ]] || exit 1
git -C "$LP_CODE" cat-file -e "$FOLLOWUP_CODE_SHA^{commit}"
# This audit only reads local artifacts. The pinned LP interpreter is sufficient;
# Conda activation and env.sh are not required on CPU nodes.
STAGE=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/lp-prerequisites-${SLURM_JOB_ID}.XXXXXXXX")
trap 'rm -rf -- "$STAGE"' EXIT
git -C "$LP_CODE" archive --format=tar --output="$STAGE/code.tar" "$FOLLOWUP_CODE_SHA"
tar -xf "$STAGE/code.tar" -C "$STAGE"
rm -- "$STAGE/code.tar"
export PYTHONPATH="$STAGE${PYTHONPATH:+:$PYTHONPATH}"
unset LP_PREREQUISITE_REPORT
cd "$STAGE"
REPORT="$RESULT_ROOT/progressive_llm/prerequisites/prerequisites_$FOLLOWUP_CODE_SHA.json"
mkdir -p "$(dirname "$REPORT")" "$RESULT_ROOT/progressive_llm/.locks"
finish() {
  status=$?
  trap - EXIT
  if (( status != 0 )); then
    "$LP_PYTHON" - "$REPORT" "$SLURM_JOB_ID" "$status" <<'PY' || true
import sys
from pathlib import Path
from lp_llm.storage import atomic_json
report, job_id, exit_code = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
failure = report.with_name(report.stem + f".failure.{job_id}.json")
if not failure.exists():
    atomic_json(failure, {"status": "prerequisites_failed", "job_id": job_id,
                          "exit_code": exit_code, "error": "See dedicated Slurm and audit logs"})
PY
  fi
  rm -rf -- "$STAGE"
  exit "$status"
}
trap finish EXIT
exec 9>"$RESULT_ROOT/progressive_llm/.locks/prerequisites-$FOLLOWUP_CODE_SHA.lock"
flock -n 9 || { echo 'ERROR: prerequisite verification already running' >&2; exit 1; }
echo "[PREREQUISITE VERIFICATION STARTED] job=$SLURM_JOB_ID report=$REPORT"
"$LP_PYTHON" -m lp_llm.progressive_cpu_preflight \
  "$LP_ROOT/envs/lp_llm" "$LP_PYTHON" || {
    echo 'ERROR: LP interpreter or CPU prerequisite imports are unavailable on this node' >&2
    exit 1
  }
SUPPORT="$RESULT_ROOT/progressive_llm/pilots/control_support_$FOLLOWUP_CODE_SHA.json"
"$LP_PYTHON" -m lp_llm.progressive_support \
  --checkout "$RRQ_PR_SOURCE" --output "$SUPPORT" \
  --code-sha "$FOLLOWUP_CODE_SHA" >"$REPORT.support.log" 2>&1 || {
    tail -n 80 "$REPORT.support.log" >&2; exit 1;
  }
echo '[CONTROL SUPPORT AUDIT COMPLETE]'
"$LP_PYTHON" -m lp_llm.progressive_prerequisites full \
  --source "$LP_SOURCE" --result-root "$RESULT_ROOT" \
  --code-sha "$FOLLOWUP_CODE_SHA" --code-repo "$LP_CODE" --job-id "$SLURM_JOB_ID" \
  >"$REPORT.log" 2>&1 || { tail -n 80 "$REPORT.log" >&2; exit 1; }
tail -n 8 "$REPORT.log"
