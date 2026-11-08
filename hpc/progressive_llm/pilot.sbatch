#!/usr/bin/env bash
#SBATCH --job-name=lp-llm-pilot
#SBATCH --account=acc-makhtar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=/shared/makhtar/lp_llm/logs/%x-%j.out
set -euo pipefail
: "${LP_BRANCH:?Set LP_BRANCH=s01 or r02}"
[[ "$LP_BRANCH" == s01 || "$LP_BRANCH" == r02 ]] || { echo 'ERROR: invalid LP branch' >&2; exit 1; }
LP_ROOT="${LP_ROOT:-/shared/makhtar/lp_llm}"
LP_CODE="${LP_CODE:-$LP_ROOT/code}"
LP_SOURCE="${LP_SOURCE:-$LP_ROOT/models/qwen3-8b-base}"
RESULT_ROOT="${RESULT_ROOT:-$LP_ROOT/results/qwen3_8b_base}"
LP_PYTHON="${LP_PYTHON:-$LP_ROOT/envs/lp_llm/bin/python}"
MATGPTQ_SOURCE="${MATGPTQ_SOURCE:-$LP_ROOT/references/rrq_w2/MatGPTQ}"
source /opt/miniconda/etc/profile.d/conda.sh
conda activate "$LP_ROOT/envs/lp_llm"
source "$LP_ROOT/env.sh"
unset TRANSFORMERS_CACHE
FOLLOWUP_CODE_SHA="${FOLLOWUP_CODE_SHA:-$(git -C "$LP_CODE" rev-parse HEAD)}"
export FOLLOWUP_CODE_SHA
[[ "$FOLLOWUP_CODE_SHA" =~ ^[0-9a-f]{40}$ && -x "$LP_PYTHON" ]] || { echo 'ERROR: code SHA or LP Python invalid' >&2; exit 1; }
git -C "$LP_CODE" cat-file -e "$FOLLOWUP_CODE_SHA^{commit}"
CODE_STAGE=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/lp-matrix-pilot-${SLURM_JOB_ID:-manual}.XXXXXXXX")
trap 'rm -rf -- "$CODE_STAGE"' EXIT
git -C "$LP_CODE" archive --format=tar --output="$CODE_STAGE/code.tar" "$FOLLOWUP_CODE_SHA"
tar -xf "$CODE_STAGE/code.tar" -C "$CODE_STAGE"
rm -- "$CODE_STAGE/code.tar"
export PYTHONPATH="$CODE_STAGE${PYTHONPATH:+:$PYTHONPATH}"
cd "$CODE_STAGE"
"$LP_PYTHON" -m lp_llm.progressive_resources --source "$LP_SOURCE" \
  --shared "$RESULT_ROOT" --scratch "$CODE_STAGE" --mode pilot --strict
base="$RESULT_ROOT/quantizer_followups/${LP_BRANCH}_fp16_load_repair/export"
pilot="$RESULT_ROOT/progressive_llm/pilots/$LP_BRANCH"
mkdir -p "$RESULT_ROOT/progressive_llm/.locks"
exec 9>"$RESULT_ROOT/progressive_llm/.locks/pilot-$LP_BRANCH.lock"
flock -n 9 || { echo 'ERROR: pilot writer already active' >&2; exit 1; }
"$LP_PYTHON" -m lp_llm.progressive_pilot --branch "$LP_BRANCH" --source "$LP_SOURCE" \
  --base "$base" --root "$pilot" --report "$pilot/report.json" --code-sha "$FOLLOWUP_CODE_SHA"
"$LP_PYTHON" -m lp_llm.progressive_base_parity --branch "$LP_BRANCH" \
  --source "$LP_SOURCE" --base "$base" --root "$pilot" \
  --result "$pilot/base_parity" --matgptq "$MATGPTQ_SOURCE" --code-sha "$FOLLOWUP_CODE_SHA"
echo "[LP PILOT VALIDATED] branch=$LP_BRANCH commit=$FOLLOWUP_CODE_SHA"
