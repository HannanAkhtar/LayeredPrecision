#!/usr/bin/env bash
#SBATCH --job-name=r03-capacity
#SBATCH --account=acc-makhtar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --output=/shared/makhtar/lp_llm/logs/%x-%j.out
set -euo pipefail
LP_ROOT="${LP_ROOT:-/shared/makhtar/lp_llm}"
LP_CODE="${LP_CODE:-$LP_ROOT/code}"
LP_SOURCE="${LP_SOURCE:-$LP_ROOT/models/qwen3-8b-base}"
RESULT_ROOT="${RESULT_ROOT:-$LP_ROOT/results/qwen3_8b_base}"
LP_PYTHON="${LP_PYTHON:-$LP_ROOT/envs/lp_llm/bin/python}"
FOLLOWUP_CODE_SHA="${FOLLOWUP_CODE_SHA:-$(git -C "$LP_CODE" rev-parse HEAD)}"
[[ "$FOLLOWUP_CODE_SHA" =~ ^[0-9a-f]{40}$ && -x "$LP_PYTHON" ]] || {
  echo "ERROR: pinned code SHA or LP Python is invalid" >&2; exit 1;
}
git -C "$LP_CODE" cat-file -e "$FOLLOWUP_CODE_SHA^{commit}" || {
  echo "ERROR: pinned code commit unavailable" >&2; exit 1;
}
CODE_STAGE=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/lp-r03-capacity-${SLURM_JOB_ID:-manual}.XXXXXXXX")
trap 'rm -rf -- "$CODE_STAGE"' EXIT
git -C "$LP_CODE" archive --format=tar --output="$CODE_STAGE/code.tar" "$FOLLOWUP_CODE_SHA"
tar -xf "$CODE_STAGE/code.tar" -C "$CODE_STAGE"
rm -- "$CODE_STAGE/code.tar"
[[ -f "$CODE_STAGE/lp_llm/r03_resources.py" ]] || {
  echo "ERROR: staged R03 capacity tool is missing" >&2; exit 1;
}
cd "$CODE_STAGE"
PYTHONPATH="$CODE_STAGE" "$LP_PYTHON" -m lp_llm.r03_resources \
  --source "$LP_SOURCE" --result-root "$RESULT_ROOT" --scratch "$CODE_STAGE" --strict
echo "[R03 CAPACITY CHECK COMPLETE] commit=$FOLLOWUP_CODE_SHA"
