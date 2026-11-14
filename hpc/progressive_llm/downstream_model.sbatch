#!/usr/bin/env bash
#SBATCH --job-name=lp-downstream
#SBATCH --account=acc-makhtar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --time=24:00:00
#SBATCH --output=/shared/makhtar/lp_llm/logs/%x-%A_%a.out
set -euo pipefail
LP_ROOT=${LP_ROOT:-/shared/makhtar/lp_llm}
LP_CODE=${LP_CODE:-$LP_ROOT/code}
SHA=${FOLLOWUP_CODE_SHA:?Pin the reviewed code SHA}
[[ "$SHA" =~ ^[0-9a-f]{40}$ && -n "${SLURM_JOB_GPUS:-}" ]] || exit 2
PYTHON=$LP_ROOT/envs/downstream/bin/python
[[ -x "$PYTHON" ]] || { echo 'Isolated downstream environment missing' >&2; exit 2; }
ROOT=$LP_ROOT/results/qwen3_8b_base/progressive_llm/downstream_zero_shot/$SHA
PILOT=${DOWNSTREAM_PILOT:-0}
if [[ "$PILOT" == 1 ]]; then
  case ${SLURM_ARRAY_TASK_ID:?} in
    0) MODEL=original_fp16;; 1) MODEL=s01_lp6;; 2) MODEL=r02_w6;;
    *) echo 'Invalid pilot array index' >&2; exit 2;;
  esac
  EXTRA=(--pilot)
else
  MODELS=(original_fp16 s01_w2 s01_lp4 s01_lp6 s01_lp8 s01_w4 s01_w6 s01_w8 r02_w2 r02_lp4 r02_lp6 r02_lp8 r02_w4 r02_w6 r02_w8)
  MODEL=${MODELS[${SLURM_ARRAY_TASK_ID:?}]:-}
  [[ -n "$MODEL" ]] || exit 2
  EXTRA=()
fi
STAGE=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/lp-downstream-${SLURM_JOB_ID}.XXXXXXXX")
mkdir -p "$ROOT/job_resources" "$STAGE/times"
finish() {
  rc=$?; trap - EXIT; set +e
  "$PYTHON" -m lp_llm.progressive_job_resources \
    --output "$ROOT/job_resources/${MODEL}-${PILOT}-${SLURM_JOB_ID}.json" \
    --time-dir "$STAGE/times" --job-id "$SLURM_JOB_ID" \
    --allocation-mib "${SLURM_MEM_PER_NODE:-0}" --exit-code "$rc" --point "downstream-${MODEL}"
  measured=$?; rm -rf -- "$STAGE"
  if (( rc == 0 && measured != 0 )); then rc=$measured; fi
  exit "$rc"
}
trap finish EXIT
git -C "$LP_CODE" cat-file -e "$SHA^{commit}"
git -C "$LP_CODE" archive --format=tar --output="$STAGE/code.tar" "$SHA"
tar -xf "$STAGE/code.tar" -C "$STAGE"; rm -- "$STAGE/code.tar"
export PYTHONPATH="$STAGE${PYTHONPATH:+:$PYTHONPATH}" HF_HOME="$LP_ROOT/cache/downstream_hf"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$STAGE"
exec 9>"$ROOT/.${PILOT}-${MODEL}.lock"
flock -n 9 || { echo 'Model evaluation already running' >&2; exit 1; }
/usr/bin/time -f '%M' -o "$STAGE/times/evaluate.rss_kib" \
  "$PYTHON" -m lp_llm.progressive_downstream evaluate \
    --source "$LP_ROOT/models/qwen3-8b-base" \
    --result-root "$LP_ROOT/results/qwen3_8b_base" --output-root "$ROOT" \
    --harness "$LP_ROOT/references/downstream/lm-evaluation-harness" \
    --code-sha "$SHA" --model "$MODEL" --scratch "$STAGE" "${EXTRA[@]}"
