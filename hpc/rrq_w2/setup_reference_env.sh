#!/usr/bin/env bash
set -euo pipefail

LP_ROOT="${LP_ROOT:-/shared/makhtar/lp_llm}"
REF_ROOT="${RRQ_REF_ROOT:-$LP_ROOT/references/rrq_w2}"
MATGPTQ_SHA=0e5bd93bc53381f7195db870f5408ae04906bb3c
AUTOROUND_SHA=6c824f741be358983baba87aff83f9c5aa0a01e7
mkdir -p "$REF_ROOT"
if [[ ! -d "$REF_ROOT/MatGPTQ/.git" ]]; then
  git clone https://github.com/IST-DASLab/MatGPTQ.git "$REF_ROOT/MatGPTQ"
fi
if [[ ! -d "$REF_ROOT/auto-round/.git" ]]; then
  git clone https://github.com/intel/auto-round.git "$REF_ROOT/auto-round"
fi
git -C "$REF_ROOT/MatGPTQ" fetch origin "$MATGPTQ_SHA"
git -C "$REF_ROOT/MatGPTQ" checkout --detach "$MATGPTQ_SHA"
git -C "$REF_ROOT/auto-round" fetch origin "$AUTOROUND_SHA"
git -C "$REF_ROOT/auto-round" checkout --detach "$AUTOROUND_SHA"
[[ "$(git -C "$REF_ROOT/MatGPTQ" rev-parse HEAD)" == "$MATGPTQ_SHA" ]]
[[ "$(git -C "$REF_ROOT/auto-round" rev-parse HEAD)" == "$AUTOROUND_SHA" ]]

source /opt/miniconda/etc/profile.d/conda.sh
AUTO_ENV="${RRQ_AUTOROUND_ENV:-$LP_ROOT/envs/rrq_autoround}"
MAT_ENV="${RRQ_MATGPTQ_ENV:-$LP_ROOT/envs/rrq_matgptq}"
if [[ ! -x "$AUTO_ENV/bin/python" ]]; then
  conda create -y -p "$AUTO_ENV" --clone /shared/makhtar/lp_llm/envs/lp_llm
fi
"$AUTO_ENV/bin/python" -m pip install --no-build-isolation -e "$REF_ROOT/auto-round"
"$AUTO_ENV/bin/python" -m pip check
if [[ ! -x "$MAT_ENV/bin/python" ]]; then
  conda create -y -p "$MAT_ENV" --clone /shared/makhtar/lp_llm/envs/lp_llm
fi
"$MAT_ENV/bin/python" -m pip install -r "$REF_ROOT/MatGPTQ/requirements.txt"
"$MAT_ENV/bin/python" -m pip check
echo "Pinned sources and isolated AutoRound/MatGPTQ environments ready: $REF_ROOT"
