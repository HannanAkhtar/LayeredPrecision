#!/usr/bin/env bash
#SBATCH --job-name=lp-sym-g128
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
: "${LP_CODE:?Point LP_CODE to the symmetric-rtn experiment checkout}"
EVAL_CODE="${EVAL_CODE:-$LP_CODE}"
LP_SOURCE="${LP_SOURCE:-$LP_ROOT/models/qwen3-8b-base}"
RESULT_ROOT="${RESULT_ROOT:-$LP_ROOT/results/qwen3_8b_base}"
ARTIFACT="$RESULT_ROOT/lp_symmetric_rtn_g128_2_2_2_2"
OUT_DIR="$RESULT_ROOT/decoded"
PPL_DIR="$RESULT_ROOT/matgptq_protocol"
CONFIG="$LP_CODE/configs/llm/qwen3_lp_symmetric_rtn_group128.yaml"

[[ -f "$LP_CODE/lp_llm/quantize.py" ]] || { echo "Missing experiment checkout" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "Missing symmetric configuration" >&2; exit 1; }
[[ -f "$EVAL_CODE/lp_llm/evaluate_matgptq.py" ]] || {
    echo "Missing MatGPTQ-style evaluator in $EVAL_CODE" >&2; exit 1;
}
[[ -f "$LP_SOURCE/model.safetensors.index.json" ]] || { echo "Missing model" >&2; exit 1; }
# A shared checkout is safe only after previous jobs have finished. The
# experiment uses separate output paths and does not overwrite prior artifacts.
mkdir -p "$OUT_DIR" "$PPL_DIR"
cd "$LP_CODE"
echo "[INFO] symmetric experiment source: $LP_CODE"
echo "[INFO] quantizer: symmetric_rtn, groups=128, prefix widths=2+2+2+2"
echo "[INFO] This is a controlled symmetric RTN ablation, NOT an exact RRQ replication."

# Resumable encoder checks source and implementation identities before skipping.
python -m lp_llm.encode --source "$LP_SOURCE" --config "$CONFIG" --output "$ARTIFACT"
test -f "$ARTIFACT/COMPLETE.json" || { echo "Encoding incomplete" >&2; exit 1; }

for depth in 1 2 3 4; do
    bits=$((depth * 2))
    checkpoint="$OUT_DIR/lp_symmetric_rtn_g128_${bits}bit"
    result="$PPL_DIR/lp_symmetric_rtn_g128_${bits}.json"
    echo "[START] depth=$depth nominal_bits=$bits"

    # Reconstruct is resumable; it validates individual completed shards.
    python -m lp_llm.reconstruct --artifact "$ARTIFACT" --depth "$depth" --output "$checkpoint"
    test -f "$checkpoint/COMPLETE.json" || { echo "Reconstruction incomplete" >&2; exit 1; }

    # The MatGPTQ-style evaluator is imported from EVAL_CODE.
    cd "$EVAL_CODE"
    python -m lp_llm.evaluate_matgptq --checkpoint "$checkpoint" --output "$result" \
        --sequence-length 2048 --dtype float16
    test -f "$result" || { echo "Evaluation output missing" >&2; exit 1; }
    echo "[COMPLETE] nominal_bits=$bits result=$result"
    cd "$LP_CODE"
done
echo "[SUCCESS] Four symmetric-RTN G128 prefixes evaluated."
