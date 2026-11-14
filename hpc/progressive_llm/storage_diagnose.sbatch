#!/usr/bin/env bash
#SBATCH --job-name=lp-storage-diagnose
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
CODE_SHA=${FOLLOWUP_CODE_SHA:?Set reviewed SHA}
SAMPLE_SHA=${STORAGE_SAMPLE_SHA:?Set original sampled-audit SHA}
[[ "$CODE_SHA" =~ ^[0-9a-f]{40}$ && "$SAMPLE_SHA" =~ ^[0-9a-f]{40}$ &&
   -x "$LP_PYTHON" && -z "${SLURM_JOB_GPUS:-}" ]] || exit 2
git -C "$LP_CODE" cat-file -e "$CODE_SHA^{commit}"
ROOT=$LP_ROOT/results/qwen3_8b_base/progressive_llm/storage_feasibility
OUT=$ROOT/$CODE_SHA/diagnose.json
mkdir -p "$(dirname "$OUT")"
STAGE=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/lp-storage-diagnose-${SLURM_JOB_ID}.XXXXXXXX")
finish() {
  rc=$?; trap - EXIT; set +e
  "$LP_PYTHON" -m lp_llm.progressive_job_resources \
    --output "$ROOT/$CODE_SHA/diagnose-resources-${SLURM_JOB_ID}.json" \
    --time-dir "$STAGE/times" --job-id "$SLURM_JOB_ID" \
    --allocation-mib "${SLURM_MEM_PER_NODE:-0}" --exit-code "$rc" --point storage-diagnose
  measured=$?; rm -rf -- "$STAGE"
  if (( rc == 0 && measured != 0 )); then rc=$measured; fi
  exit "$rc"
}
trap finish EXIT
mkdir -p "$STAGE/times"
git -C "$LP_CODE" archive --format=tar --output="$STAGE/code.tar" "$CODE_SHA"
tar -xf "$STAGE/code.tar" -C "$STAGE"; rm -- "$STAGE/code.tar"
export PYTHONPATH="$STAGE${PYTHONPATH:+:$PYTHONPATH}"
cd "$STAGE"
exec 9>"$ROOT/$CODE_SHA/.diagnose.lock"
flock -n 9 || { echo 'Diagnostic already running' >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo 'Refusing to overwrite diagnostic' >&2; exit 1; }
/usr/bin/time -f '%M' -o "$STAGE/times/diagnose.rss_kib" \
  "$LP_PYTHON" -m lp_llm.progressive_storage_diagnose \
    --source "$LP_ROOT/models/qwen3-8b-base" \
    --result-root "$LP_ROOT/results/qwen3_8b_base" \
    --sample "$ROOT/$SAMPLE_SHA/sample-all.json" \
    --checkout "$LP_ROOT/references/rrq_pr2308/auto-round" \
    --output "$OUT" --code-sha "$CODE_SHA"
