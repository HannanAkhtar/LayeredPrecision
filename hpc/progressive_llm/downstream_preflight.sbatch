#!/usr/bin/env bash
#SBATCH --job-name=lp-downstream-preflight
#SBATCH --account=acc-makhtar
#SBATCH --partition=hpc
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/shared/makhtar/lp_llm/logs/%x-%j.out
set -euo pipefail
LP_ROOT=${LP_ROOT:-/shared/makhtar/lp_llm}
LP_CODE=${LP_CODE:-$LP_ROOT/code}
SHA=${FOLLOWUP_CODE_SHA:?Pin the reviewed code SHA}
PYTHON=$LP_ROOT/envs/downstream/bin/python
[[ "$SHA" =~ ^[0-9a-f]{40}$ && -x "$PYTHON" && -z "${SLURM_JOB_GPUS:-}" ]] || exit 2
ROOT=$LP_ROOT/results/qwen3_8b_base/progressive_llm/downstream_zero_shot/$SHA
mkdir -p "$ROOT"
STAGE=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/lp-downstream-preflight-${SLURM_JOB_ID}.XXXXXXXX")
trap 'rm -rf -- "$STAGE"' EXIT
git -C "$LP_CODE" archive --format=tar --output="$STAGE/code.tar" "$SHA"
tar -xf "$STAGE/code.tar" -C "$STAGE"; rm -- "$STAGE/code.tar"
export PYTHONPATH="$STAGE${PYTHONPATH:+:$PYTHONPATH}" HF_HOME="$LP_ROOT/cache/downstream_hf"
export TOKENIZERS_PARALLELISM=false
cd "$STAGE"
exec 9>"$ROOT/.preflight.lock"
flock -n 9 || { echo 'Preflight already running' >&2; exit 1; }
"$PYTHON" -m lp_llm.progressive_downstream preflight \
  --source "$LP_ROOT/models/qwen3-8b-base" \
  --result-root "$LP_ROOT/results/qwen3_8b_base" --output-root "$ROOT" \
  --harness "$LP_ROOT/references/downstream/lm-evaluation-harness" --code-sha "$SHA"
