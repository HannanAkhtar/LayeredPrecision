#!/usr/bin/env bash
#SBATCH --job-name=lp-llm-point
#SBATCH --account=acc-makhtar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=/shared/makhtar/lp_llm/logs/%x-%j.out
set -euo pipefail
: "${LP_MATRIX_ID:?Set LP_MATRIX_ID from the planned 12-point manifest}"
[[ "${LP_MATRIX_ALLOW:-0}" == 1 ]] || { echo 'ERROR: reviewed full-matrix gate missing' >&2; exit 1; }
LP_ROOT="${LP_ROOT:-/shared/makhtar/lp_llm}"
LP_CODE="${LP_CODE:-$LP_ROOT/code}"
LP_SOURCE="${LP_SOURCE:-$LP_ROOT/models/qwen3-8b-base}"
RESULT_ROOT="${RESULT_ROOT:-$LP_ROOT/results/qwen3_8b_base}"
LP_PYTHON="${LP_PYTHON:-$LP_ROOT/envs/lp_llm/bin/python}"
RRQ_PR_SOURCE="${RRQ_PR_SOURCE:-$LP_ROOT/references/rrq_pr2308/auto-round}"
MATGPTQ_SOURCE="${MATGPTQ_SOURCE:-$LP_ROOT/references/rrq_w2/MatGPTQ}"
source /opt/miniconda/etc/profile.d/conda.sh
conda activate "$LP_ROOT/envs/lp_llm"
source "$LP_ROOT/env.sh"
unset TRANSFORMERS_CACHE
FOLLOWUP_CODE_SHA="${FOLLOWUP_CODE_SHA:-$(git -C "$LP_CODE" rev-parse HEAD)}"
export FOLLOWUP_CODE_SHA
[[ "$FOLLOWUP_CODE_SHA" =~ ^[0-9a-f]{40}$ && -x "$LP_PYTHON" ]] || { echo 'ERROR: code SHA or LP Python invalid' >&2; exit 1; }
git -C "$LP_CODE" cat-file -e "$FOLLOWUP_CODE_SHA^{commit}"
CODE_STAGE=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/lp-matrix-${SLURM_JOB_ID:-manual}.XXXXXXXX")
RESOURCE_DIR="$RESULT_ROOT/progressive_llm/job_resources"
TIME_DIR="$CODE_STAGE/resource_times"
mkdir -p "$RESOURCE_DIR" "$TIME_DIR"
[[ -x /usr/bin/time ]] || { echo 'ERROR: /usr/bin/time is required for peak RSS' >&2; exit 1; }
finish() {
  status=$?
  trap - EXIT
  set +e
  "$LP_PYTHON" -m lp_llm.progressive_job_resources \
    --output "$RESOURCE_DIR/$LP_MATRIX_ID-${SLURM_JOB_ID}.json" \
    --time-dir "$TIME_DIR" --job-id "${SLURM_JOB_ID:-}" \
    --allocation-mib "${SLURM_MEM_PER_NODE:-0}" --exit-code "$status" --point "$LP_MATRIX_ID"
  measured=$?
  rm -rf -- "$CODE_STAGE"
  if (( measured != 0 && status == 0 )); then status=$measured; fi
  if (( status == 0 )); then
    echo "[LP MATRIX POINT VALIDATED] id=$LP_MATRIX_ID commit=$FOLLOWUP_CODE_SHA"
  fi
  exit "$status"
}
trap finish EXIT
measure() {
  stage=$1
  shift
  /usr/bin/time -f '%M' -o "$TIME_DIR/$stage.rss_kib" "$@"
}
git -C "$LP_CODE" archive --format=tar --output="$CODE_STAGE/code.tar" "$FOLLOWUP_CODE_SHA"
tar -xf "$CODE_STAGE/code.tar" -C "$CODE_STAGE"
rm -- "$CODE_STAGE/code.tar"
export PYTHONPATH="$CODE_STAGE${PYTHONPATH:+:$PYTHONPATH}"
cd "$CODE_STAGE"
"$LP_PYTHON" -m lp_llm.progressive_resources --source "$LP_SOURCE" \
  --shared "$RESULT_ROOT" --scratch "$CODE_STAGE" --mode point \
  --slots "${LP_MATRIX_SLOTS:-14}" --strict
"$LP_PYTHON" -m lp_llm.progressive_plan verify-prerequisites --source "$LP_SOURCE" \
  --result-root "$RESULT_ROOT" --code-sha "$FOLLOWUP_CODE_SHA" --code-repo "$LP_CODE"
mkdir -p "$RESULT_ROOT/progressive_llm/.locks"
exec 9>"$RESULT_ROOT/progressive_llm/.locks/$LP_MATRIX_ID.lock"
flock -n 9 || { echo 'ERROR: point writer already active' >&2; exit 1; }
case "$LP_MATRIX_ID" in
  s01_lp_2p2|s01_lp_2p2p2|s01_lp_2p2p2p2|r02_lp_2p2|r02_lp_2p2p2|r02_lp_2p2p2p2)
    branch="${LP_MATRIX_ID%%_*}"
    case "$LP_MATRIX_ID" in *_2p2p2p2) depth=3;; *_2p2p2) depth=2;; *_2p2) depth=1;; esac
    base="$RESULT_ROOT/quantizer_followups/${branch}_fp16_load_repair/export"
    root="$RESULT_ROOT/progressive_llm/$branch"
    result="$RESULT_ROOT/progressive_llm/evaluation/$LP_MATRIX_ID"
    measure encode "$LP_PYTHON" -m lp_llm.progressive_run encode --branch "$branch" --depth "$depth" \
      --source "$LP_SOURCE" --base "$base" --root "$root"
    measure evaluate "$LP_PYTHON" -m lp_llm.progressive_run evaluate --branch "$branch" --depth "$depth" \
      --source "$LP_SOURCE" --base "$base" --root "$root" \
      --result "$result" --matgptq "$MATGPTQ_SOURCE"
    ;;
  s01_w4|s01_w6|s01_w8|r02_w4|r02_w6|r02_w8)
    branch="${LP_MATRIX_ID%%_*}"
    bits="${LP_MATRIX_ID##*w}"
    result="$RESULT_ROOT/progressive_llm/controls/$LP_MATRIX_ID"
    measure encode "$LP_PYTHON" -m lp_llm.progressive_control --branch "$branch" --bits "$bits" \
      --source "$LP_SOURCE" --result "$result" --checkout "$RRQ_PR_SOURCE" \
      --matgptq "$MATGPTQ_SOURCE" --stage encode
    measure evaluate "$LP_PYTHON" -m lp_llm.progressive_control --branch "$branch" --bits "$bits" \
      --source "$LP_SOURCE" --result "$result" --checkout "$RRQ_PR_SOURCE" \
      --matgptq "$MATGPTQ_SOURCE" --stage evaluate
    ;;
  *) echo "ERROR: unknown matrix point $LP_MATRIX_ID" >&2; exit 1;;
esac
