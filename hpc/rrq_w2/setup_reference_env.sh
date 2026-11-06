#!/usr/bin/env bash
set -euo pipefail

LP_ROOT="${LP_ROOT:-/shared/makhtar/lp_llm}"
LP_CODE="${LP_CODE:-$LP_ROOT/code}"
REF_ROOT="${RRQ_REF_ROOT:-$LP_ROOT/references/rrq_w2}"
LP_PYTHON="${RRQ_PYTHON:-$LP_ROOT/envs/lp_llm/bin/python}"
MATGPTQ_SHA=0e5bd93bc53381f7195db870f5408ae04906bb3c
AUTOROUND_SHA=6c824f741be358983baba87aff83f9c5aa0a01e7
HISTORICAL_SHA=24bfce00bb3bd053498232f0e8395118fe10a906

[[ -x "$LP_PYTHON" ]] || { echo "ERROR: LP Python absent: $LP_PYTHON" >&2; exit 1; }
[[ -d "$LP_CODE/.git" ]] || { echo "ERROR: LP code checkout absent: $LP_CODE" >&2; exit 1; }
mkdir -p "$REF_ROOT"

checkout() {
  local name="$1" revision="$2" directory="$3" url="$4"
  if [[ ! -d "$directory/.git" ]]; then
    [[ ! -e "$directory" ]] || { echo "ERROR: non-Git reference directory: $directory" >&2; exit 1; }
    git clone "$url" "$directory"
  fi
  git -C "$directory" fetch origin "$revision"
  git -C "$directory" checkout --detach "$revision"
  [[ "$(git -C "$directory" rev-parse HEAD)" == "$revision" ]] || {
    echo "ERROR: wrong $name revision at $directory" >&2; exit 1;
  }
  git -C "$directory" diff --quiet HEAD || { echo "ERROR: modified $name source at $directory" >&2; exit 1; }
  echo "Verified $name $revision at $directory"
}

checkout MatGPTQ "$MATGPTQ_SHA" "$REF_ROOT/MatGPTQ" https://github.com/IST-DASLab/MatGPTQ.git
if [[ "${RRQ_SETUP_AUTOROUND:-1}" == 1 ]]; then
  checkout AutoRound "$AUTOROUND_SHA" "$REF_ROOT/auto-round" https://github.com/intel/auto-round.git
fi
if [[ "${RRQ_SETUP_HISTORICAL:-0}" == 1 ]]; then
  checkout AutoRound-preprint-candidate "$HISTORICAL_SHA" "$REF_ROOT/auto-round-preprint-candidate" \
    https://github.com/intel/auto-round.git
fi

cd "$LP_CODE"
"$LP_PYTHON" "$REF_ROOT/MatGPTQ/eval_ppl.py" --help >/dev/null || {
  echo 'ERROR: pinned MatGPTQ evaluator cannot import in LP environment' >&2; exit 1;
}
"$LP_PYTHON" -m lp_llm.rrq_overlay check --target "$REF_ROOT/extra_deps"
echo 'Setup only verified source and imports. No Conda clone or package installation was performed.'
