#!/usr/bin/env bash
#SBATCH --job-name=lp-quantize
#SBATCH --account=acc-makhtar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/shared/makhtar/lp_llm/logs/%x-%j.out
set -euo pipefail
source /opt/miniconda/etc/profile.d/conda.sh
conda activate /shared/makhtar/lp_llm/envs/lp_llm
source /shared/makhtar/lp_llm/env.sh
unset TRANSFORMERS_CACHE
: "${LP_CODE:?Set LP_CODE to the deployed repository}"
: "${LP_CONFIG:?Set LP_CONFIG}"
: "${LP_OUTPUT:?Set LP_OUTPUT}"
LP_SOURCE="${LP_SOURCE:-/shared/makhtar/lp_llm/models/qwen3-8b-base}"
cd "$LP_CODE"
start=$SECONDS
extra=()
if [[ -n "${LP_BITS:-}" ]]; then extra+=(--bits "$LP_BITS"); fi
if [[ -n "${LP_GRANULARITY:-}" ]]; then extra+=(--granularity "$LP_GRANULARITY"); fi
if [[ -n "${LP_GROUP_SIZE:-}" ]]; then extra+=(--group-size "$LP_GROUP_SIZE"); fi
if [[ -n "${LP_METHOD:-}" ]]; then extra+=(--method "$LP_METHOD"); fi
python -m lp_llm.encode --source "$LP_SOURCE" --config "$LP_CONFIG" --output "$LP_OUTPUT" "${extra[@]}"
echo "Quantization completed in $((SECONDS-start)) seconds: $LP_OUTPUT/COMPLETE.json"
