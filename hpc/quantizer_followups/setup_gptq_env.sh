#!/usr/bin/env bash
set -euo pipefail
LP_ROOT="${LP_ROOT:-/shared/makhtar/lp_llm}"
LP_CODE="${LP_CODE:-$LP_ROOT/code}"
GPTQ_SOURCE="${GPTQ_SOURCE:-$LP_ROOT/references/quantizer_followups/GPTQModel}"
GPTQ_ENV_MODE="${GPTQ_ENV_MODE:-light}"
case "$GPTQ_ENV_MODE" in
  light) GPTQ_ENV="$LP_ROOT/envs/gptq_followups_v570_light" ;;
  clone) GPTQ_ENV="$LP_ROOT/envs/gptq_followups_v570" ;;
  *) echo "ERROR: GPTQ_ENV_MODE must be light or clone" >&2; exit 1 ;;
esac
GPTQ_ENV="${GPTQ_ENV_OVERRIDE:-$GPTQ_ENV}"
LP_ENV="$LP_ROOT/envs/lp_llm"
LP_PYTHON="$LP_ENV/bin/python"
CHECK="$LP_CODE/hpc/quantizer_followups/gptq_env_check.py"
PINS="$LP_CODE/hpc/quantizer_followups/gptq_v570_extras.txt"
[[ -x "$LP_PYTHON" && -f "$CHECK" && -f "$PINS" ]] || {
  echo "ERROR: LP Python or GPTQ setup files missing" >&2; exit 1;
}
[[ "$(git -C "$GPTQ_SOURCE" rev-parse HEAD)" == ed96f2e04ec868b04f969733a242ce3b5bb7cb1e ]] || {
  echo "ERROR: GPTQModel checkout revision mismatch" >&2; exit 1;
}
git -C "$GPTQ_SOURCE" diff --quiet HEAD || { echo "ERROR: GPTQModel checkout has local changes" >&2; exit 1; }
if [[ -e "$GPTQ_ENV" ]]; then
  [[ -f "$GPTQ_ENV/quantizer_followups_ready.json" && -x "$GPTQ_ENV/bin/python" ]] || {
    echo "ERROR: incomplete GPTQ environment at $GPTQ_ENV; inspect it before retrying" >&2; exit 1;
  }
  PYTHONPATH="$GPTQ_SOURCE" "$GPTQ_ENV/bin/python" "$CHECK" verify \
    --env "$GPTQ_ENV" --lp-env "$LP_ENV" --source "$GPTQ_SOURCE" --pins "$PINS" --mode "$GPTQ_ENV_MODE"
  echo "[GPTQ ENV READY] $GPTQ_ENV"
  exit 0
fi
if [[ "$GPTQ_ENV_MODE" == light ]]; then
  "$LP_PYTHON" -m venv --system-site-packages "$GPTQ_ENV"
else
  source /opt/miniconda/etc/profile.d/conda.sh
  conda create --yes --copy --prefix "$GPTQ_ENV" --clone "$LP_ENV"
fi
PYTHON="$GPTQ_ENV/bin/python"
missing_output=$("$PYTHON" "$CHECK" missing --env "$GPTQ_ENV" --lp-env "$LP_ENV" \
  --source "$GPTQ_SOURCE" --pins "$PINS" --mode "$GPTQ_ENV_MODE")
if [[ -n "$missing_output" ]]; then
  mapfile -t extras <<< "$missing_output"
  "$PYTHON" -m pip install --no-deps "${extras[@]}"
fi
"$PYTHON" -m pip install --no-deps --no-build-isolation --editable "$GPTQ_SOURCE"
"$PYTHON" -m pip check || { echo "ERROR: isolated GPTQ environment has dependency conflicts" >&2; exit 1; }
"$PYTHON" -m pip freeze > "$GPTQ_ENV/quantizer_followups_requirements_lock.txt"
PYTHONPATH="$GPTQ_SOURCE" "$PYTHON" "$CHECK" initialize \
  --env "$GPTQ_ENV" --lp-env "$LP_ENV" --source "$GPTQ_SOURCE" --pins "$PINS" --mode "$GPTQ_ENV_MODE"
echo "[GPTQ ENV READY] $GPTQ_ENV"
