#!/usr/bin/env bash
#SBATCH --job-name=lp-scaled-encode
#SBATCH --account=acc-makhtar
#SBATCH --partition=hpc
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/shared/makhtar/lp_llm/logs/%x-%j.out
set -euo pipefail
BRANCH=${MATRIX_BRANCH:?Set s01 or r02}
DEPTH=${MATRIX_DEPTH:?Set 1, 2 or 3}
[[ "$BRANCH" == s01 || "$BRANCH" == r02 ]] || exit 1
[[ "$DEPTH" =~ ^[123]$ && !( "$BRANCH" == s01 && "$DEPTH" == 1 ) ]] || exit 1
LP_ROOT=${LP_ROOT:-/shared/makhtar/lp_llm}
LP_CODE=${LP_CODE:-$LP_ROOT/code}
LP_PYTHON=$LP_ROOT/envs/lp_llm/bin/python
RESULT_ROOT=$LP_ROOT/results/qwen3_8b_base
FOLLOWUP_CODE_SHA=${FOLLOWUP_CODE_SHA:?Set reviewed commit}
[[ "$FOLLOWUP_CODE_SHA" =~ ^[0-9a-f]{40}$ && -x "$LP_PYTHON" && -z "${SLURM_JOB_GPUS:-}" ]] || exit 1
git -C "$LP_CODE" cat-file -e "$FOLLOWUP_CODE_SHA^{commit}"
STAGE=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/scaled-encode-${SLURM_JOB_ID}.XXXXXXXX")
TIMES=$STAGE/times
mkdir -p "$TIMES" "$RESULT_ROOT/progressive_llm/job_resources" "$RESULT_ROOT/progressive_llm/.locks"
finish() {
  status=$?; trap - EXIT; set +e
  "$LP_PYTHON" -m lp_llm.progressive_job_resources \
    --output "$RESULT_ROOT/progressive_llm/job_resources/scaled-${BRANCH}-e${DEPTH}-${SLURM_JOB_ID}.json" \
    --time-dir "$TIMES" --job-id "$SLURM_JOB_ID" --allocation-mib "${SLURM_MEM_PER_NODE:-0}" \
    --exit-code "$status" --point "scaled-${BRANCH}-e${DEPTH}"
  measured=$?; rm -rf -- "$STAGE"
  if (( status == 0 && measured != 0 )); then status=$measured; fi
  exit "$status"
}
trap finish EXIT
git -C "$LP_CODE" archive --format=tar --output="$STAGE/code.tar" "$FOLLOWUP_CODE_SHA"
tar -xf "$STAGE/code.tar" -C "$STAGE"; rm -- "$STAGE/code.tar"
export PYTHONPATH="$STAGE${PYTHONPATH:+:$PYTHONPATH}" FOLLOWUP_CODE_SHA
cd "$STAGE"
ROOT=$RESULT_ROOT/progressive_llm/scaled_signed_v1/$BRANCH
mkdir -p "$ROOT"
exec 9>"$RESULT_ROOT/progressive_llm/.locks/scaled-${BRANCH}-e${DEPTH}.lock"
flock -n 9 || { echo 'ERROR: scaled layer writer already running' >&2; exit 1; }
/usr/bin/time -f '%M' -o "$TIMES/encode.rss_kib" \
  "$LP_PYTHON" -m lp_llm.progressive_scaled encode --branch "$BRANCH" --depth "$DEPTH" \
    --source "$LP_ROOT/models/qwen3-8b-base" \
    --base "$RESULT_ROOT/quantizer_followups/${BRANCH}_fp16_load_repair/export" --root "$ROOT"
echo "[SCALED LAYER NUMERICALLY COMPLETE; RESOURCE CHECK PENDING] branch=$BRANCH depth=$DEPTH code=$FOLLOWUP_CODE_SHA"
