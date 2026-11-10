#!/usr/bin/env bash
#SBATCH --job-name=lp-stage-a-recover
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
  echo 'ERROR: Stage A recovery audit needs CPU-only hpc partition and 32 GiB RAM' >&2
  exit 1
}
[[ "$FOLLOWUP_CODE_SHA" =~ ^[0-9a-f]{40}$ && -x "$LP_PYTHON" ]] || exit 1
git -C "$LP_CODE" cat-file -e "$FOLLOWUP_CODE_SHA^{commit}"
STAGE=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/lp-stage-a-recover-${SLURM_JOB_ID}.XXXXXXXX")
trap 'rm -rf -- "$STAGE"' EXIT
git -C "$LP_CODE" archive --format=tar --output="$STAGE/code.tar" "$FOLLOWUP_CODE_SHA"
tar -xf "$STAGE/code.tar" -C "$STAGE"
rm -- "$STAGE/code.tar"
export PYTHONPATH="$STAGE${PYTHONPATH:+:$PYTHONPATH}"
export LP_CODE LP_SOURCE RESULT_ROOT FOLLOWUP_CODE_SHA
export LP_PREREQUISITE_REPORT="$RESULT_ROOT/progressive_llm/prerequisites/prerequisites_$FOLLOWUP_CODE_SHA.json"
cd "$STAGE"
"$LP_PYTHON" -m lp_llm.progressive_cpu_preflight "$LP_ROOT/envs/lp_llm" "$LP_PYTHON"
"$LP_PYTHON" -m lp_llm.progressive_prerequisites check --source "$LP_SOURCE" \
  --result-root "$RESULT_ROOT" --code-sha "$FOLLOWUP_CODE_SHA" --code-repo "$LP_CODE"
OUT="$RESULT_ROOT/progressive_llm/recovery/$FOLLOWUP_CODE_SHA/stage_a_scientific_audit.json"
mkdir -p "$(dirname "$OUT")" "$RESULT_ROOT/progressive_llm/.locks"
exec 9>"$RESULT_ROOT/progressive_llm/.locks/stage-a-recovery-$FOLLOWUP_CODE_SHA.lock"
flock -n 9 || { echo 'ERROR: Stage A recovery audit already running' >&2; exit 1; }
"$LP_PYTHON" -m lp_llm.progressive_stage_a_recovery audit \
  --source "$LP_SOURCE" --result-root "$RESULT_ROOT" --code-sha "$FOLLOWUP_CODE_SHA" \
  --code-repo "$LP_CODE" --autoround "$RRQ_PR_SOURCE" --output "$OUT"
