#!/usr/bin/env bash
#SBATCH --job-name=lp-w4-recover-eval
#SBATCH --account=acc-makhtar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --time=12:00:00
#SBATCH --output=/shared/makhtar/lp_llm/logs/%x-%j.out
set -euo pipefail
LP_ROOT="${LP_ROOT:-/shared/makhtar/lp_llm}"
LP_CODE="${LP_CODE:-$LP_ROOT/code}"
LP_SOURCE="${LP_SOURCE:-$LP_ROOT/models/qwen3-8b-base}"
RESULT_ROOT="${RESULT_ROOT:-$LP_ROOT/results/qwen3_8b_base}"
LP_PYTHON="${LP_PYTHON:-$LP_ROOT/envs/lp_llm/bin/python}"
RRQ_PR_SOURCE="${RRQ_PR_SOURCE:-$LP_ROOT/references/rrq_pr2308/auto-round}"
MATGPTQ_SOURCE="${MATGPTQ_SOURCE:-$LP_ROOT/references/rrq_w2/MatGPTQ}"
FOLLOWUP_CODE_SHA="${FOLLOWUP_CODE_SHA:?Set the reviewed code SHA}"
source /opt/miniconda/etc/profile.d/conda.sh
conda activate "$LP_ROOT/envs/lp_llm"
source "$LP_ROOT/env.sh"
unset TRANSFORMERS_CACHE
[[ "${SLURM_MEM_PER_NODE:-0}" -ge 15360 && -n "${SLURM_JOB_GPUS:-}" ]] || {
  echo 'ERROR: W4 recovery evaluation requires one GPU and 15 GiB host allocation' >&2
  exit 1
}
[[ "$FOLLOWUP_CODE_SHA" =~ ^[0-9a-f]{40}$ && -x "$LP_PYTHON" ]] || exit 1
git -C "$LP_CODE" cat-file -e "$FOLLOWUP_CODE_SHA^{commit}"
STAGE=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/lp-w4-recovery-${SLURM_JOB_ID}.XXXXXXXX")
TIME_DIR="$STAGE/resource_times"
mkdir -p "$TIME_DIR" "$RESULT_ROOT/progressive_llm/job_resources"
finish() {
  status=$?
  trap - EXIT
  set +e
  "$LP_PYTHON" -m lp_llm.progressive_job_resources \
    --output "$RESULT_ROOT/progressive_llm/job_resources/s01_w4_recovery-${SLURM_JOB_ID}.json" \
    --time-dir "$TIME_DIR" --job-id "$SLURM_JOB_ID" \
    --allocation-mib "${SLURM_MEM_PER_NODE:-0}" --exit-code "$status" --point s01_w4_recovery
  measured=$?
  rm -rf -- "$STAGE"
  if (( status == 0 && measured != 0 )); then status=$measured; fi
  exit "$status"
}
trap finish EXIT
git -C "$LP_CODE" archive --format=tar --output="$STAGE/code.tar" "$FOLLOWUP_CODE_SHA"
tar -xf "$STAGE/code.tar" -C "$STAGE"
rm -- "$STAGE/code.tar"
export PYTHONPATH="$STAGE${PYTHONPATH:+:$PYTHONPATH}"
export LP_CODE LP_SOURCE RESULT_ROOT FOLLOWUP_CODE_SHA
export LP_PREREQUISITE_REPORT="$RESULT_ROOT/progressive_llm/prerequisites/prerequisites_$FOLLOWUP_CODE_SHA.json"
cd "$STAGE"
"$LP_PYTHON" -m lp_llm.progressive_prerequisites check --source "$LP_SOURCE" \
  --result-root "$RESULT_ROOT" --code-sha "$FOLLOWUP_CODE_SHA" --code-repo "$LP_CODE"
OUT="$RESULT_ROOT/progressive_llm/recovery/$FOLLOWUP_CODE_SHA/stage_a_scientific_audit.json"
[[ -f "$OUT" ]] || { echo 'ERROR: CPU Stage A recovery audit is required first' >&2; exit 1; }
mkdir -p "$RESULT_ROOT/progressive_llm/.locks"
exec 9>"$RESULT_ROOT/progressive_llm/.locks/w4-recovery-$FOLLOWUP_CODE_SHA.lock"
flock -n 9 || { echo 'ERROR: W4 recovery evaluation already running' >&2; exit 1; }
/usr/bin/time -f '%M' -o "$TIME_DIR/evaluate.rss_kib" \
  "$LP_PYTHON" -m lp_llm.progressive_stage_a_recovery evaluate \
    --source "$LP_SOURCE" --result-root "$RESULT_ROOT" --code-sha "$FOLLOWUP_CODE_SHA" \
    --code-repo "$LP_CODE" --autoround "$RRQ_PR_SOURCE" \
    --matgptq "$MATGPTQ_SOURCE" --output "$OUT"
