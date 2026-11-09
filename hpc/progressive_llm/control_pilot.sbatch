#!/usr/bin/env bash
#SBATCH --job-name=lp-control-pilot
#SBATCH --account=acc-makhtar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=18G
#SBATCH --time=04:00:00
#SBATCH --output=/shared/makhtar/lp_llm/logs/%x-%j.out
set -euo pipefail
: "${LP_CONTROL_ID:?Set LP_CONTROL_ID=s01_w4, etc.}"
case "$LP_CONTROL_ID" in s01_w4|s01_w6|s01_w8|r02_w4|r02_w6|r02_w8) ;; *) echo 'ERROR: invalid control pilot ID' >&2; exit 1;; esac
LP_ROOT="${LP_ROOT:-/shared/makhtar/lp_llm}"
LP_CODE="${LP_CODE:-$LP_ROOT/code}"
LP_SOURCE="${LP_SOURCE:-$LP_ROOT/models/qwen3-8b-base}"
RESULT_ROOT="${RESULT_ROOT:-$LP_ROOT/results/qwen3_8b_base}"
RRQ_PR_SOURCE="${RRQ_PR_SOURCE:-$LP_ROOT/references/rrq_pr2308/auto-round}"
LP_PYTHON="${LP_PYTHON:-$LP_ROOT/envs/lp_llm/bin/python}"
source /opt/miniconda/etc/profile.d/conda.sh
conda activate "$LP_ROOT/envs/lp_llm"
source "$LP_ROOT/env.sh"
unset TRANSFORMERS_CACHE
FOLLOWUP_CODE_SHA="${FOLLOWUP_CODE_SHA:-$(git -C "$LP_CODE" rev-parse HEAD)}"
export FOLLOWUP_CODE_SHA
[[ "$FOLLOWUP_CODE_SHA" =~ ^[0-9a-f]{40}$ && -x "$LP_PYTHON" ]] || { echo 'ERROR: code SHA or LP Python invalid' >&2; exit 1; }
[[ "$(git -C "$RRQ_PR_SOURCE" rev-parse HEAD)" == 7985c1b0f4eef65fb717511668fda7e347e584c3 ]] || { echo 'ERROR: AutoRound revision differs' >&2; exit 1; }
git -C "$RRQ_PR_SOURCE" diff --quiet HEAD || { echo 'ERROR: AutoRound source modified' >&2; exit 1; }
git -C "$LP_CODE" cat-file -e "$FOLLOWUP_CODE_SHA^{commit}"
CODE_STAGE=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/lp-control-pilot-${SLURM_JOB_ID:-manual}.XXXXXXXX")
trap 'rm -rf -- "$CODE_STAGE"' EXIT
git -C "$LP_CODE" archive --format=tar --output="$CODE_STAGE/code.tar" "$FOLLOWUP_CODE_SHA"
tar -xf "$CODE_STAGE/code.tar" -C "$CODE_STAGE"
rm -- "$CODE_STAGE/code.tar"
export PYTHONPATH="$RRQ_PR_SOURCE:$CODE_STAGE${PYTHONPATH:+:$PYTHONPATH}" LP_CONTROL_PILOT=1
cd "$CODE_STAGE"
"$LP_PYTHON" -m lp_llm.progressive_resources --source "$LP_SOURCE" \
  --shared "$RESULT_ROOT" --scratch "$CODE_STAGE" --mode control_pilot --strict
branch="${LP_CONTROL_ID%%_*}"
bits="${LP_CONTROL_ID##*w}"
report="$RESULT_ROOT/progressive_llm/pilots/$LP_CONTROL_ID.json"
mkdir -p "$RESULT_ROOT/progressive_llm/pilots" "$RESULT_ROOT/progressive_llm/.locks"
exec 9>"$RESULT_ROOT/progressive_llm/.locks/pilot-$LP_CONTROL_ID.lock"
flock -n 9 || { echo 'ERROR: control pilot writer already active' >&2; exit 1; }
"$LP_PYTHON" -m lp_llm.progressive_control_pilot --branch "$branch" --bits "$bits" \
  --source "$LP_SOURCE" --report "$report" --code-sha "$FOLLOWUP_CODE_SHA"
echo "[CONTROL PILOT VALIDATED] id=$LP_CONTROL_ID commit=$FOLLOWUP_CODE_SHA"
