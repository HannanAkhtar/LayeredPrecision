#!/usr/bin/env bash
set -euo pipefail

LP_ROOT="${LP_ROOT:-/shared/makhtar/lp_llm}"
LP_CODE="${LP_CODE:-$LP_ROOT/code}"
LP_SOURCE="${LP_SOURCE:-$LP_ROOT/models/qwen3-8b-base}"
RRQ_PR_SOURCE="${RRQ_PR_SOURCE:-$LP_ROOT/references/rrq_pr2308/auto-round}"
MATGPTQ_SOURCE="${MATGPTQ_SOURCE:-$LP_ROOT/references/rrq_w2/MatGPTQ}"
LP_PYTHON="${LP_PYTHON:-$LP_ROOT/envs/lp_llm/bin/python}"
PR_REVISION=7985c1b0f4eef65fb717511668fda7e347e584c3
MATGPTQ_REVISION=0e5bd93bc53381f7195db870f5408ae04906bb3c

[[ -x "$LP_PYTHON" ]] || { echo "ERROR: existing LP Python absent: $LP_PYTHON" >&2; exit 1; }
[[ -f "$LP_SOURCE/config.json" && -f "$LP_SOURCE/model.safetensors.index.json" ]] || {
  echo "ERROR: pinned local Qwen3 source is incomplete: $LP_SOURCE" >&2; exit 1;
}
[[ -d "$LP_CODE/lp_llm" ]] || { echo "ERROR: LP code absent: $LP_CODE" >&2; exit 1; }
[[ "$(git -C "$RRQ_PR_SOURCE" rev-parse HEAD)" == "$PR_REVISION" ]] || {
  echo "ERROR: separate RRQ PR checkout has wrong revision" >&2; exit 1;
}
git -C "$RRQ_PR_SOURCE" diff --quiet HEAD || {
  echo "ERROR: RRQ PR checkout has modified tracked source" >&2; exit 1;
}
if [[ "${EXPERIMENT_ID:-}" == pr03 ]]; then
  source_path="$RRQ_PR_SOURCE"
  if [[ -n "${RRQ_PR_OVERLAY:-}" ]]; then
    [[ -d "$RRQ_PR_OVERLAY" ]] || { echo "ERROR: supplemental overlay is absent" >&2; exit 1; }
    source_path="$RRQ_PR_SOURCE:$RRQ_PR_OVERLAY"
  fi
  RRQ_PR_SOURCE="$RRQ_PR_SOURCE" PYTHONPATH="$source_path${PYTHONPATH:+:$PYTHONPATH}" \
    "$LP_PYTHON" - <<'PY' || { echo 'ERROR: pinned RRQ PR import failed; inspect supplemental dependencies' >&2; exit 1; }
import auto_round, os
from pathlib import Path
from auto_round import AutoRound, RRQConfig
actual = Path(auto_round.__file__).resolve()
expected = Path(os.environ['RRQ_PR_SOURCE']).resolve()
assert actual.is_relative_to(expected), f'Wrong AutoRound import: {actual}'
print('Pinned AutoRound import:', actual)
PY
fi
if [[ "${EXPERIMENT_ID:-}" == pr04 ]]; then
  [[ "$(git -C "$MATGPTQ_SOURCE" rev-parse HEAD)" == "$MATGPTQ_REVISION" ]] || {
    echo "ERROR: MatGPTQ checkout has wrong revision" >&2; exit 1;
  }
fi
echo "Host available KiB: $(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
minimum_host="${RRQ_PR_MIN_HOST_MB:-12000}"
if [[ -n "${SLURM_MEM_PER_NODE:-}" && "$SLURM_MEM_PER_NODE" -lt "$minimum_host" ]]; then
  echo "ERROR: Slurm host allocation below ${minimum_host} MB" >&2; exit 1
fi
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
  echo 'ERROR: GPU is not visible' >&2; exit 1
fi
free_gpu="$($LP_PYTHON -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.mem_get_info(0)[0] // (1024*1024))')"
if (( free_gpu < ${RRQ_PR_MIN_GPU_FREE_MB:-20000} )); then
  echo "ERROR: free GPU memory ${free_gpu} MB below required ${RRQ_PR_MIN_GPU_FREE_MB:-20000} MB" >&2
  exit 1
fi
"$LP_PYTHON" - <<'PY'
import sys, torch, transformers, safetensors
print('Python:', sys.executable, 'Torch:', torch.__version__,
      'Transformers:', transformers.__version__, 'Safetensors:', safetensors.__version__)
PY
