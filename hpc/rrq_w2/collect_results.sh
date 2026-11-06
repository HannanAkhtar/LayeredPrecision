#!/usr/bin/env bash
set -euo pipefail
LP_ROOT="${LP_ROOT:-/shared/makhtar/lp_llm}"
RESULT_ROOT="${RESULT_ROOT:-$LP_ROOT/results/qwen3_8b_base}"
python -m lp_llm.rrq_aggregate --result-root "$RESULT_ROOT"
