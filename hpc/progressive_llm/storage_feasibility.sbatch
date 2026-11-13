#!/usr/bin/env bash
#SBATCH --job-name=lp-storage-audit
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
CODE_SHA=${FOLLOWUP_CODE_SHA:?Set reviewed commit SHA}
MODE=${STORAGE_MODE:-sample}
MODEL=${STORAGE_MODEL:-}
[[ "$MODE" == metadata || "$MODE" == sample || "$MODE" == full ]] || { echo 'Invalid STORAGE_MODE' >&2; exit 2; }
[[ "$CODE_SHA" =~ ^[0-9a-f]{40}$ && -x "$LP_PYTHON" && -z "${SLURM_JOB_GPUS:-}" ]] || exit 2
git -C "$LP_CODE" cat-file -e "$CODE_SHA^{commit}"
ROOT=$LP_ROOT/results/qwen3_8b_base/progressive_llm/storage_feasibility/$CODE_SHA
mkdir -p "$ROOT" "$LP_ROOT/logs"
STAGE=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/lp-storage-${SLURM_JOB_ID}.XXXXXXXX")
finish() {
  rc=$?; trap - EXIT; set +e
  "$LP_PYTHON" -m lp_llm.progressive_job_resources \
    --output "$ROOT/resources-${MODE}-${MODEL:-all}-${SLURM_JOB_ID}.json" \
    --time-dir "$STAGE/times" --job-id "$SLURM_JOB_ID" \
    --allocation-mib "${SLURM_MEM_PER_NODE:-0}" --exit-code "$rc" --point "storage-${MODE}-${MODEL:-all}"
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
exec 9>"$ROOT/.${MODE}-${MODEL:-all}.lock"
flock -n 9 || { echo 'Audit already running' >&2; exit 1; }
OUT=$ROOT/${MODE}-${MODEL:-all}.json
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite $OUT" >&2; exit 1; }
args=()
if [[ -n "$MODEL" ]]; then args+=(--model "$MODEL"); fi
/usr/bin/time -f '%M' -o "$STAGE/times/audit.rss_kib" \
  "$LP_PYTHON" -m lp_llm.progressive_storage_feasibility \
    --source "$LP_ROOT/models/qwen3-8b-base" \
    --result-root "$LP_ROOT/results/qwen3_8b_base" \
    --output "$OUT" --mode "$MODE" --code-sha "$CODE_SHA" "${args[@]}"
