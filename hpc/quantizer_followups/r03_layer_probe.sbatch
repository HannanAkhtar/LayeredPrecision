#!/usr/bin/env bash
#SBATCH --job-name=r03-layer-probe
#SBATCH --account=acc-makhtar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=/shared/makhtar/lp_llm/logs/%x-%j.out
set -euo pipefail
LP_ROOT="${LP_ROOT:-/shared/makhtar/lp_llm}"
LP_CODE="${LP_CODE:-$LP_ROOT/code}"
LP_SOURCE="${LP_SOURCE:-$LP_ROOT/models/qwen3-8b-base}"
RESULT_ROOT="${RESULT_ROOT:-$LP_ROOT/results/qwen3_8b_base}"
GPTQ_SOURCE="${GPTQ_SOURCE:-$LP_ROOT/references/quantizer_followups/GPTQModel}"
GPTQ_PYTHON="${GPTQ_PYTHON:-$LP_ROOT/envs/gptq_followups_v570_light/bin/python}"
LP_PYTHON="${LP_PYTHON:-$LP_ROOT/envs/lp_llm/bin/python}"
: "${R03_SHARED_QUOTA_FREE_GIB:?Set site-confirmed available shared quota in GiB}"
R03_OFFLOAD_ROOT="${R03_OFFLOAD_ROOT:-$LP_ROOT/tmp/r03_offload}"
export R03_OFFLOAD_ROOT R03_SHARED_QUOTA_FREE_GIB R03_LAYER_PROBE=1
FOLLOWUP_CODE_SHA="${FOLLOWUP_CODE_SHA:-$(git -C "$LP_CODE" rev-parse HEAD)}"
[[ "$FOLLOWUP_CODE_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "ERROR: invalid pinned code SHA" >&2; exit 1; }
git -C "$LP_CODE" cat-file -e "$FOLLOWUP_CODE_SHA^{commit}" || {
  echo "ERROR: pinned code commit unavailable" >&2; exit 1;
}
CODE_STAGE=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/lp-r03-layer-${SLURM_JOB_ID:-manual}.XXXXXXXX")
trap 'rm -rf -- "$CODE_STAGE"' EXIT
git -C "$LP_CODE" archive --format=tar --output="$CODE_STAGE/code.tar" "$FOLLOWUP_CODE_SHA"
tar -xf "$CODE_STAGE/code.tar" -C "$CODE_STAGE"
rm -- "$CODE_STAGE/code.tar"
[[ -f "$CODE_STAGE/hpc/quantizer_followups/r03_layer_probe.py" ]] || {
  echo "ERROR: staged R03 layer probe missing" >&2; exit 1;
}
export LP_CODE="$CODE_STAGE" PYTHONPATH="$CODE_STAGE${PYTHONPATH:+:$PYTHONPATH}"
cd "$CODE_STAGE"
EXPERIMENT_ID=r03 bash hpc/quantizer_followups/preflight.sh
"$LP_PYTHON" -m lp_llm.r03_resources --source "$LP_SOURCE" \
  --result-root "$RESULT_ROOT" --scratch "$CODE_STAGE" \
  --offload-root "$R03_OFFLOAD_ROOT" --strict
PYTHONPATH="$GPTQ_SOURCE:$CODE_STAGE${PYTHONPATH:+:$PYTHONPATH}" \
  "$GPTQ_PYTHON" hpc/quantizer_followups/r03_layer_probe.py --source "$LP_SOURCE" \
    --code-sha "$FOLLOWUP_CODE_SHA" \
    --report "$RESULT_ROOT/quantizer_followups/r03_feasibility/layer_probe-${SLURM_JOB_ID}.json"
echo "[R03 LAYER PROBE COMPLETE] commit=$FOLLOWUP_CODE_SHA"
