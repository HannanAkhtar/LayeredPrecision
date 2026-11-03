#!/usr/bin/env bash
#SBATCH --job-name=lp-ar-rtn-g128
#SBATCH --account=acc-makhtar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=/shared/makhtar/lp_llm/logs/%x-%j.out
set -euo pipefail

source /opt/miniconda/etc/profile.d/conda.sh
conda activate /shared/makhtar/lp_llm/envs/lp_llm
source /shared/makhtar/lp_llm/env.sh
unset TRANSFORMERS_CACHE
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

LP_ROOT="${LP_ROOT:-/shared/makhtar/lp_llm}"
: "${LP_CODE:?Set LP_CODE to the AutoRound RTN experiment checkout}"
EVAL_CODE="${EVAL_CODE:-$LP_CODE}"
LP_SOURCE="${LP_SOURCE:-$LP_ROOT/models/qwen3-8b-base}"
RESULT_ROOT="${RESULT_ROOT:-$LP_ROOT/results/qwen3_8b_base}"
LP_PHASE="${LP_PHASE:-pilot}"
ARTIFACT="$RESULT_ROOT/lp_autoround_rtn_g128_2_2_2_2"
DECODED="$RESULT_ROOT/decoded"
PPL_DIR="$RESULT_ROOT/matgptq_protocol"
CONFIG="$LP_CODE/configs/llm/qwen3_lp_autoround_rtn_group128.yaml"
EVALUATOR="$EVAL_CODE/lp_llm/evaluate_matgptq.py"

[[ "$LP_PHASE" == pilot || "$LP_PHASE" == full ]] || { echo "ERROR: LP_PHASE must be pilot or full" >&2; exit 1; }
[[ -f "$CONFIG" && -f "$LP_CODE/lp_llm/validate_autoround_stage.py" ]] || {
    echo "ERROR: AutoRound RTN checkout/configuration missing" >&2; exit 1;
}
[[ -f "$EVALUATOR" ]] || { echo "ERROR: untracked MatGPTQ evaluator missing: $EVALUATOR" >&2; exit 1; }
[[ -f "$LP_SOURCE/model.safetensors.index.json" ]] || { echo "ERROR: source checkpoint missing" >&2; exit 1; }
mkdir -p "$DECODED" "$PPL_DIR"
cd "$LP_CODE"

# A validator returns 3 for unfinished work and 2 for corrupt or conflicting
# work. Only the former is resumed. Every completed stage is checked again.
stage_status() {
    local rc
    if python -m lp_llm.validate_autoround_stage "$@"; then
        return 0
    else
        rc=$?
        [[ "$rc" == 3 ]] && return 3
        echo "ERROR: invalid $1 stage; refusing to skip or overwrite it" >&2
        exit 1
    fi
}

if [[ "$LP_PHASE" == pilot ]]; then
    if stage_status encoding --artifact "$ARTIFACT" --source "$LP_SOURCE"; then
        echo "[SKIP] validated encoding"
    else
        echo "[START] encode AutoRound RTN G128 LP hierarchy"
        python -m lp_llm.encode --source "$LP_SOURCE" --config "$CONFIG" --output "$ARTIFACT"
        stage_status encoding --artifact "$ARTIFACT" --source "$LP_SOURCE" || {
            echo "ERROR: encoding did not complete" >&2; exit 1;
        }
    fi
    depths=(1)
else
    stage_status encoding --artifact "$ARTIFACT" --source "$LP_SOURCE" || {
        echo "ERROR: full phase requires a completed pilot encoding" >&2; exit 1;
    }
    pilot_checkpoint="$DECODED/lp_autoround_rtn_g128_2bit"
    pilot_result="$PPL_DIR/lp_autoround_rtn_g128_2.json"
    stage_status reconstruction --artifact "$ARTIFACT" --checkpoint "$pilot_checkpoint" --depth 1 || {
        echo "ERROR: full phase requires a completed pilot reconstruction" >&2; exit 1;
    }
    stage_status evaluation --artifact "$ARTIFACT" --checkpoint "$pilot_checkpoint" \
        --result "$pilot_result" --evaluator "$EVALUATOR" --depth 1 || {
        echo "ERROR: full phase requires a validated pilot evaluation" >&2; exit 1;
    }
    depths=(2 3 4)
fi

for depth in "${depths[@]}"; do
    bits=$((depth * 2))
    checkpoint="$DECODED/lp_autoround_rtn_g128_${bits}bit"
    result="$PPL_DIR/lp_autoround_rtn_g128_${bits}.json"
    echo "[START] depth=$depth nominal_bits=$bits"
    if stage_status reconstruction --artifact "$ARTIFACT" --checkpoint "$checkpoint" --depth "$depth"; then
        echo "[SKIP] validated reconstruction: $checkpoint"
    else
        python -m lp_llm.reconstruct --artifact "$ARTIFACT" --depth "$depth" --output "$checkpoint"
        stage_status reconstruction --artifact "$ARTIFACT" --checkpoint "$checkpoint" --depth "$depth" || {
            echo "ERROR: reconstruction did not complete" >&2; exit 1;
        }
    fi
    if stage_status evaluation --artifact "$ARTIFACT" --checkpoint "$checkpoint" \
            --result "$result" --evaluator "$EVALUATOR" --depth "$depth"; then
        echo "[SKIP] validated perplexity: $result"
    else
        echo "[START] MatGPTQ-style WikiText-2 FP16 non-overlapping 2048-token evaluation"
        (cd "$EVAL_CODE" && python -m lp_llm.evaluate_matgptq --checkpoint "$checkpoint" \
            --output "$result" --sequence-length 2048 --dtype float16)
        python -m lp_llm.validate_autoround_stage mark-evaluation --artifact "$ARTIFACT" \
            --checkpoint "$checkpoint" --result "$result" --evaluator "$EVALUATOR" --depth "$depth"
        stage_status evaluation --artifact "$ARTIFACT" --checkpoint "$checkpoint" \
            --result "$result" --evaluator "$EVALUATOR" --depth "$depth"
    fi
    echo "[COMPLETE] nominal_bits=$bits result=$result"
done
report_args=(--artifact "$ARTIFACT" --result-root "$RESULT_ROOT"
             --output "$ARTIFACT/autoround_comparison.json")
if [[ -n "${ASYM_ARTIFACT:-}" ]]; then
    report_args+=(--asymmetric-artifact "$ASYM_ARTIFACT")
fi
if [[ -n "${ASYM_RESULT_PREFIX:-}" ]]; then
    report_args+=(--asymmetric-result-prefix "$ASYM_RESULT_PREFIX")
fi
python -m lp_llm.report_autoround "${report_args[@]}"
echo "[SUCCESS] AutoRound RTN G128 $LP_PHASE phase completed"
