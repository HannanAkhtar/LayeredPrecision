#!/usr/bin/env bash
set -euo pipefail

LP_ROOT="${LP_ROOT:-/shared/makhtar/lp_llm}"
LP_CODE="${LP_CODE:-$LP_ROOT/code}"
LP_SOURCE="${LP_SOURCE:-$LP_ROOT/models/qwen3-8b-base}"
REF_ROOT="${RRQ_REF_ROOT:-$LP_ROOT/references/rrq_w2}"
LP_PYTHON="${RRQ_PYTHON:-$LP_ROOT/envs/lp_llm/bin/python}"
OVERLAY="$REF_ROOT/extra_deps"
[[ -d "$LP_CODE/.git" && -f "$LP_SOURCE/config.json" ]] || { echo 'ERROR: code or model absent' >&2; exit 1; }
[[ -x "$LP_PYTHON" ]] || { echo "ERROR: LP Python absent: $LP_PYTHON" >&2; exit 1; }
[[ -f "$REF_ROOT/MatGPTQ/eval_ppl.py" ]] || { echo 'ERROR: MatGPTQ absent' >&2; exit 1; }
"$LP_PYTHON" "$REF_ROOT/MatGPTQ/eval_ppl.py" --help >/dev/null || {
  echo 'ERROR: pinned MatGPTQ evaluator imports or arguments are unavailable in LP environment' >&2; exit 1;
}
[[ "$(git -C "$REF_ROOT/MatGPTQ" rev-parse HEAD)" == 0e5bd93bc53381f7195db870f5408ae04906bb3c ]] || {
  echo 'ERROR: wrong MatGPTQ revision' >&2; exit 1;
}
git -C "$REF_ROOT/MatGPTQ" diff --quiet HEAD || { echo 'ERROR: modified MatGPTQ source' >&2; exit 1; }
case "${EXPERIMENT_ID:-}" in
  j04|j05|j06|j07)
    case "${RRQ_AUTOROUND_VARIANT:-current}" in
      current)
        AUTO_SOURCE="$REF_ROOT/auto-round"
        AUTO_SHA=6c824f741be358983baba87aff83f9c5aa0a01e7 ;;
      preprint_candidate)
        AUTO_SOURCE="$REF_ROOT/auto-round-preprint-candidate"
        AUTO_SHA=24bfce00bb3bd053498232f0e8395118fe10a906 ;;
      *) echo 'ERROR: unknown AutoRound reference variant' >&2; exit 1 ;;
    esac
    [[ "$(git -C "$AUTO_SOURCE" rev-parse HEAD)" == "$AUTO_SHA" ]] || {
      echo "ERROR: wrong AutoRound revision at $AUTO_SOURCE" >&2; exit 1;
    }
    git -C "$AUTO_SOURCE" diff --quiet HEAD || { echo 'ERROR: modified AutoRound source' >&2; exit 1; }
    if [[ "${RRQ_REUSE_EXPORT_ONLY:-0}" != 1 ]]; then
      [[ ! -d "$OVERLAY" || -f "$OVERLAY/OVERLAY_COMPLETE.json" ]] || {
        echo "ERROR: incomplete dependency overlay: $OVERLAY" >&2; exit 1;
      }
      if [[ -d "$OVERLAY" ]]; then
        "$LP_PYTHON" -m lp_llm.rrq_overlay check --target "$OVERLAY" >/dev/null || {
          echo "ERROR: dependency overlay validation failed: $OVERLAY" >&2; exit 1;
        }
      fi
      export RRQ_AUTO_SOURCE="$AUTO_SOURCE"
      export PYTHONPATH="$AUTO_SOURCE:$OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
      "$LP_PYTHON" -c 'from auto_round.cli.main import run_rtn, run_opt_rtn; import auto_round, pathlib, os; expected=pathlib.Path(os.environ["RRQ_AUTO_SOURCE"]).resolve(); actual=pathlib.Path(auto_round.__file__).resolve(); assert actual.is_relative_to(expected), f"Wrong AutoRound import: {actual}"; print(actual)' \
        || { echo 'ERROR: pinned AutoRound import failed; check extra_deps' >&2; exit 1; }
    fi ;;
esac
command -v nvidia-smi >/dev/null || { echo 'ERROR: GPU driver absent' >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
free_mb="$("$LP_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.mem_get_info(0)[0] // (1024 * 1024))')" || {
  echo 'ERROR: LP environment cannot inspect its allocated CUDA GPU' >&2; exit 1;
}
[[ "$free_mb" =~ ^[0-9]+$ ]] || { echo 'ERROR: could not read free GPU memory' >&2; exit 1; }
if (( free_mb < ${RRQ_MIN_GPU_FREE_MB:-20000} )); then
  echo "ERROR: free GPU memory ${free_mb} MB below RRQ_MIN_GPU_FREE_MB=${RRQ_MIN_GPU_FREE_MB:-20000}" >&2
  exit 1
fi
if [[ "${EXPERIMENT_ID:-}" == j08 ]]; then
  echo 'ERROR: J08 unavailable until exact RRQ SignRoundV2 recipe is verified' >&2; exit 1
fi
minimum_host="${RRQ_MIN_HOST_MEMORY_MB:-12000}"
if [[ -n "${SLURM_MEM_PER_NODE:-}" && "${SLURM_MEM_PER_NODE}" -lt "$minimum_host" ]]; then
  echo 'ERROR: allocation below requested RRQ host-memory preflight' >&2; exit 1
fi
echo "Preflight passed: free_gpu_mb=$free_mb host_minimum_mb=$minimum_host"
