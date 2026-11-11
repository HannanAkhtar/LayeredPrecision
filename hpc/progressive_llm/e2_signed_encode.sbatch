#!/usr/bin/env bash
#SBATCH --job-name=lp-e2-signed-encode
#SBATCH --account=acc-makhtar
#SBATCH --partition=hpc
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/shared/makhtar/lp_llm/logs/%x-%j.out
set -euo pipefail
LP_ROOT=${LP_ROOT:-/shared/makhtar/lp_llm}
LP_CODE=${LP_CODE:-$LP_ROOT/code}
LP_PYTHON=$LP_ROOT/envs/lp_llm/bin/python
RESULT_ROOT=$LP_ROOT/results/qwen3_8b_base
FOLLOWUP_CODE_SHA=${FOLLOWUP_CODE_SHA:?Set pinned reviewed commit}
[[ "$FOLLOWUP_CODE_SHA" =~ ^[0-9a-f]{40}$ && -x "$LP_PYTHON" ]] || exit 1
git -C "$LP_CODE" cat-file -e "$FOLLOWUP_CODE_SHA^{commit}"
STAGE=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/lp-e2-encode-${SLURM_JOB_ID}.XXXXXXXX")
TIMES=$STAGE/times
mkdir -p "$TIMES" "$RESULT_ROOT/progressive_llm/job_resources" "$RESULT_ROOT/progressive_llm/.locks"
finish() {
  status=$?
  trap - EXIT
  set +e
  "$LP_PYTHON" -m lp_llm.progressive_job_resources \
    --output "$RESULT_ROOT/progressive_llm/job_resources/s01_e2_signed_encode-${SLURM_JOB_ID}.json" \
    --time-dir "$TIMES" --job-id "$SLURM_JOB_ID" \
    --allocation-mib "${SLURM_MEM_PER_NODE:-0}" --exit-code "$status" --point s01_e2_signed_encode
  measured=$?
  rm -rf -- "$STAGE"
  if (( status == 0 && measured != 0 )); then status=$measured; fi
  exit "$status"
}
trap finish EXIT
git -C "$LP_CODE" archive --format=tar --output="$STAGE/code.tar" "$FOLLOWUP_CODE_SHA"
tar -xf "$STAGE/code.tar" -C "$STAGE"
rm -- "$STAGE/code.tar"
export PYTHONPATH="$STAGE${PYTHONPATH:+:$PYTHONPATH}" FOLLOWUP_CODE_SHA
cd "$STAGE"
exec 9>"$RESULT_ROOT/progressive_llm/.locks/s01-e2-signed-encode.lock"
flock -n 9 || { echo 'ERROR: signed E2 encoding already running' >&2; exit 1; }
ROOT=$RESULT_ROOT/progressive_llm/ablations/s01_autoround_signed_rtn
mkdir -p "$ROOT"
/usr/bin/time -f '%M' -o "$TIMES/encode.rss_kib" \
  "$LP_PYTHON" -m lp_llm.progressive_run encode --branch s01 --depth 1 \
    --quantizer autoround_signed_rtn --source "$LP_ROOT/models/qwen3-8b-base" \
    --base "$RESULT_ROOT/quantizer_followups/s01_fp16_load_repair/export" --root "$ROOT"
echo "[SIGNED E2 ENCODE COMPLETE] code=$FOLLOWUP_CODE_SHA"
