#!/usr/bin/env bash
set -euo pipefail

LP_ROOT="${LP_ROOT:-/shared/makhtar/lp_llm}"
REF_ROOT="${RRQ_REF_ROOT:-$LP_ROOT/references/rrq_w2}"
MATGPTQ_SHA=0e5bd93bc53381f7195db870f5408ae04906bb3c
AUTOROUND_SHA=6c824f741be358983baba87aff83f9c5aa0a01e7
HISTORICAL_SHA=24bfce00bb3bd053498232f0e8395118fe10a906
mkdir -p "$REF_ROOT"
if [[ ! -d "$REF_ROOT/MatGPTQ/.git" ]]; then
  git clone https://github.com/IST-DASLab/MatGPTQ.git "$REF_ROOT/MatGPTQ"
fi
git -C "$REF_ROOT/MatGPTQ" fetch origin "$MATGPTQ_SHA"
git -C "$REF_ROOT/MatGPTQ" checkout --detach "$MATGPTQ_SHA"
[[ "$(git -C "$REF_ROOT/MatGPTQ" rev-parse HEAD)" == "$MATGPTQ_SHA" ]]

source /opt/miniconda/etc/profile.d/conda.sh
MAT_ENV="${RRQ_MATGPTQ_ENV:-$LP_ROOT/envs/rrq_matgptq}"
if [[ ! -x "$MAT_ENV/bin/python" ]]; then
  conda create -y -p "$MAT_ENV" --clone /shared/makhtar/lp_llm/envs/lp_llm
fi
"$MAT_ENV/bin/python" -m pip install -r "$REF_ROOT/MatGPTQ/requirements.txt"
"$MAT_ENV/bin/python" -m pip check
if [[ "${RRQ_SETUP_AUTOROUND:-1}" == 1 ]]; then
  AUTO_ENV="${RRQ_AUTOROUND_ENV:-$LP_ROOT/envs/rrq_autoround}"
  if [[ ! -d "$REF_ROOT/auto-round/.git" ]]; then
    git clone https://github.com/intel/auto-round.git "$REF_ROOT/auto-round"
  fi
  git -C "$REF_ROOT/auto-round" fetch origin "$AUTOROUND_SHA"
  git -C "$REF_ROOT/auto-round" checkout --detach "$AUTOROUND_SHA"
  [[ "$(git -C "$REF_ROOT/auto-round" rev-parse HEAD)" == "$AUTOROUND_SHA" ]]
  if [[ ! -x "$AUTO_ENV/bin/python" ]]; then
    conda create -y -p "$AUTO_ENV" --clone /shared/makhtar/lp_llm/envs/lp_llm
  fi
  "$AUTO_ENV/bin/python" -m pip install --no-build-isolation -e "$REF_ROOT/auto-round"
  "$AUTO_ENV/bin/python" -m pip check
fi
if [[ "${RRQ_SETUP_HISTORICAL:-0}" == 1 ]]; then
  HIST_ENV="${RRQ_AUTOROUND_HISTORICAL_ENV:-$LP_ROOT/envs/rrq_autoround_historical}"
  if [[ ! -d "$REF_ROOT/auto-round-preprint-candidate/.git" ]]; then
    git clone https://github.com/intel/auto-round.git "$REF_ROOT/auto-round-preprint-candidate"
  fi
  git -C "$REF_ROOT/auto-round-preprint-candidate" fetch origin "$HISTORICAL_SHA"
  git -C "$REF_ROOT/auto-round-preprint-candidate" checkout --detach "$HISTORICAL_SHA"
  [[ "$(git -C "$REF_ROOT/auto-round-preprint-candidate" rev-parse HEAD)" == "$HISTORICAL_SHA" ]]
  if [[ ! -x "$HIST_ENV/bin/python" ]]; then
    conda create -y -p "$HIST_ENV" --clone /shared/makhtar/lp_llm/envs/lp_llm
  fi
  "$HIST_ENV/bin/python" -m pip install --no-build-isolation -e "$REF_ROOT/auto-round-preprint-candidate"
  "$HIST_ENV/bin/python" -m pip check
fi
echo "Requested pinned reference environments ready under $LP_ROOT/envs"
