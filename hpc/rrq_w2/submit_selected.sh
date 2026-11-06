#!/usr/bin/env bash
set -euo pipefail

[[ "$#" -ge 1 ]] || { echo 'Usage: submit_selected.sh j01 [j02 ...]' >&2; exit 2; }
limit="${RRQ_CONCURRENCY:-2}"
[[ "$limit" =~ ^[1-9][0-9]*$ ]] || { echo 'ERROR: RRQ_CONCURRENCY must be positive' >&2; exit 2; }
declare -A seen=()
for id in "$@"; do
  case "$id" in
    j0[1-7]|j09|j1[0-4]) ;;
    *) echo "ERROR: unavailable or invalid experiment: $id" >&2; exit 2 ;;
  esac
  [[ -z "${seen[$id]:-}" ]] || { echo "ERROR: duplicate experiment: $id" >&2; exit 2; }
  seen[$id]=1
done
for id in "$@"; do
  while [[ "$(squeue -h -u "$USER" -n rrq-w2 | wc -l)" -ge "$limit" ]]; do sleep 30; done
  sbatch --export="ALL,EXPERIMENT_ID=$id,RRQ_STAGE=${RRQ_STAGE:-screen}" hpc/rrq_w2/run_experiment.sbatch
done
