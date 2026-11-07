#!/usr/bin/env bash
set -euo pipefail
: "${1:?Specify pr03 or pr04; submit only after reviewing PR01/PR02 and the prior stage}"
case "$1" in pr03|pr04) ;; *) echo "ERROR: unknown experiment: $1" >&2; exit 1;; esac
mkdir -p "${LP_ROOT:-/shared/makhtar/lp_llm}/logs"
EXPERIMENT_ID="$1" sbatch --export=ALL,EXPERIMENT_ID="$1" hpc/rrq_pr2308/run_experiment.sbatch
