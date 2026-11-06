#!/usr/bin/env bash
set -euo pipefail

LP_ROOT="${LP_ROOT:-/shared/makhtar/lp_llm}"
LP_CODE="${LP_CODE:-$LP_ROOT/code}"
LP_SOURCE="${LP_SOURCE:-$LP_ROOT/models/qwen3-8b-base}"
REF_ROOT="${RRQ_REF_ROOT:-$LP_ROOT/references/rrq_w2}"
AUTO_ENV="${RRQ_AUTOROUND_ENV:-$LP_ROOT/envs/rrq_autoround}"
MAT_ENV="${RRQ_MATGPTQ_ENV:-$LP_ROOT/envs/rrq_matgptq}"
[[ -d "$LP_CODE/.git" && -f "$LP_SOURCE/config.json" ]] || { echo 'ERROR: code or model absent' >&2; exit 1; }
[[ -x "$AUTO_ENV/bin/auto-round-rtn" && -x "$AUTO_ENV/bin/auto-round-opt-rtn" ]] || {
  echo 'ERROR: isolated AutoRound CLI absent; run setup_reference_env.sh' >&2; exit 1;
}
[[ -x "$MAT_ENV/bin/python" ]] || { echo 'ERROR: isolated MatGPTQ environment absent' >&2; exit 1; }
[[ -f "$REF_ROOT/MatGPTQ/eval_ppl.py" ]] || { echo 'ERROR: MatGPTQ absent' >&2; exit 1; }
"$MAT_ENV/bin/python" "$REF_ROOT/MatGPTQ/eval_ppl.py" --help >/dev/null || {
  echo 'ERROR: pinned MatGPTQ evaluator imports or arguments are unavailable in reference environment' >&2; exit 1;
}
[[ "$(git -C "$REF_ROOT/MatGPTQ" rev-parse HEAD)" == 0e5bd93bc53381f7195db870f5408ae04906bb3c ]] || {
  echo 'ERROR: wrong MatGPTQ revision' >&2; exit 1;
}
[[ "$(git -C "$REF_ROOT/auto-round" rev-parse HEAD)" == 6c824f741be358983baba87aff83f9c5aa0a01e7 ]] || {
  echo 'ERROR: wrong AutoRound revision' >&2; exit 1;
}
command -v nvidia-smi >/dev/null || { echo 'ERROR: GPU driver absent' >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
[[ "$free_mb" =~ ^[0-9]+$ ]] || { echo 'ERROR: could not read free GPU memory' >&2; exit 1; }
if (( free_mb < ${RRQ_MIN_GPU_FREE_MB:-18000} )); then
  echo "ERROR: free GPU memory ${free_mb} MB below RRQ_MIN_GPU_FREE_MB=${RRQ_MIN_GPU_FREE_MB:-18000}" >&2
  exit 1
fi
if [[ "${EXPERIMENT_ID:-}" == j08 ]]; then
  echo 'ERROR: J08 unavailable until exact RRQ SignRoundV2 recipe is verified' >&2; exit 1
fi
if [[ -n "${SLURM_MEM_PER_NODE:-}" && "${SLURM_MEM_PER_NODE}" -lt "${RRQ_MIN_HOST_MEMORY_MB:-12000}" ]]; then
  echo 'ERROR: allocation below requested RRQ host-memory preflight' >&2; exit 1
fi
echo 'Preflight passed; verify GPU free memory and experiment-specific requirements before submission.'
