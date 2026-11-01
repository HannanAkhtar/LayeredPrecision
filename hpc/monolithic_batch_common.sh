#!/usr/bin/env bash
# Sourced by the two monolithic Slurm jobs after activating the existing env.

stage_missing_or_fail() {
  local stage="$1"
  shift
  local status
  if python -m lp_llm.validate_stage "$stage" "$@"; then
    echo "[$bits bits] Verified $stage; skipping completed stage."
    return 1
  else
    status=$?
  fi
  if [[ "$status" -ne 3 ]]; then
    echo "ERROR [$bits bits] $stage is present but invalid (exit $status); stopping." >&2
    exit 1
  fi
  return 0
}

run_monolithic_batch() {
  : "${LP_CODE:?Set LP_CODE to the deployed repository}"
  : "${MONO_CONFIG:?Set MONO_CONFIG}"
  : "${MONO_GRANULARITY:?Set MONO_GRANULARITY}"
  : "${MONO_OUTPUT_ROOT:?Set MONO_OUTPUT_ROOT}"
  LP_SOURCE="${LP_SOURCE:-/shared/makhtar/lp_llm/models/qwen3-8b-base}"
  [[ -f "$LP_SOURCE/model.safetensors.index.json" ]] || {
    echo "ERROR: original model index is missing: $LP_SOURCE" >&2; exit 1;
  }
  [[ -f "$MONO_CONFIG" ]] || { echo "ERROR: config is missing: $MONO_CONFIG" >&2; exit 1; }
  cd "$LP_CODE"
  local bits artifact checkpoint result started
  for bits in 2 4 6 8; do
    artifact="$MONO_OUTPUT_ROOT/bits-$bits"
    checkpoint="$artifact/checkpoint"
    result="$artifact/wikitext.json"
    started=$SECONDS
    echo "[$bits bits] Starting monolithic $MONO_GRANULARITY experiment."

    if stage_missing_or_fail encoding --artifact "$artifact" --source "$LP_SOURCE" \
        --bits "$bits" --granularity "$MONO_GRANULARITY"; then
      python -m lp_llm.encode --source "$LP_SOURCE" --config "$MONO_CONFIG" \
        --bits "$bits" --output "$artifact" || {
          echo "ERROR [$bits bits] encoding failed." >&2; exit 1;
        }
      python -m lp_llm.validate_stage encoding --artifact "$artifact" --source "$LP_SOURCE" \
        --bits "$bits" --granularity "$MONO_GRANULARITY" || {
          echo "ERROR [$bits bits] encoding completion validation failed." >&2; exit 1;
        }
    fi

    if stage_missing_or_fail reconstruction --artifact "$artifact" --checkpoint "$checkpoint"; then
      python -m lp_llm.reconstruct --artifact "$artifact" --depth 1 --output "$checkpoint" || {
        echo "ERROR [$bits bits] reconstruction failed." >&2; exit 1;
      }
      python -m lp_llm.validate_stage reconstruction --artifact "$artifact" \
        --checkpoint "$checkpoint" || {
          echo "ERROR [$bits bits] reconstruction completion validation failed." >&2; exit 1;
        }
    fi

    if stage_missing_or_fail evaluation --checkpoint "$checkpoint" --result "$result"; then
      python -m lp_llm.evaluate wikitext --checkpoint "$checkpoint" --output "$result" || {
        echo "ERROR [$bits bits] WikiText evaluation failed." >&2; exit 1;
      }
      python -m lp_llm.validate_stage evaluation --checkpoint "$checkpoint" --result "$result" || {
        echo "ERROR [$bits bits] WikiText result validation failed." >&2; exit 1;
      }
    fi
    echo "[$bits bits] Complete in $((SECONDS-started)) seconds: $result"
  done
  echo "All monolithic $MONO_GRANULARITY operating points completed."
}
