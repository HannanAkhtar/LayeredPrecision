#!/usr/bin/env bash
set -euo pipefail
: "${1:?Use first or second}"
LP_ROOT="${LP_ROOT:-/shared/makhtar/lp_llm}"
RESULT_ROOT="${RESULT_ROOT:-$LP_ROOT/results/qwen3_8b_base}"
mkdir -p "$RESULT_ROOT/quantizer_followups/.locks"
exec 9>"$RESULT_ROOT/quantizer_followups/.locks/submit.lock"
flock -n 9 || { echo 'ERROR: another follow-up submission is in progress' >&2; exit 1; }
case "$1" in first) ids=(s01 r01);; second) ids=(r02 r03);; *) echo 'ERROR: use first or second' >&2; exit 1;; esac
active=$(squeue -h -u "$USER" -o '%i %j' | awk '$2 == "quant-followup" {n++} END {print n+0}')
if (( active != 0 )); then echo "ERROR: wait for existing quant-followup jobs ($active)" >&2; exit 1; fi
for id in "${ids[@]}"; do
  result="$RESULT_ROOT/quantizer_followups/$id"
  if [[ -e "$result/full.json" || -e "$result/failure.json" ]]; then
    echo "ERROR: $id has completed or failed output; inspect before submitting" >&2; exit 1
  fi
done
for id in "${ids[@]}"; do
  echo "Submitting $id"
  sbatch --export=ALL,EXPERIMENT_ID="$id" hpc/quantizer_followups/run_experiment.sbatch
done
