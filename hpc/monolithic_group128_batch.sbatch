#!/usr/bin/env bash
#SBATCH --job-name=mono-group128
#SBATCH --account=acc-makhtar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=/shared/makhtar/lp_llm/logs/%x-%j.out
set -euo pipefail
source /opt/miniconda/etc/profile.d/conda.sh
conda activate /shared/makhtar/lp_llm/envs/lp_llm
source /shared/makhtar/lp_llm/env.sh
unset TRANSFORMERS_CACHE
LP_ROOT="${LP_ROOT:-/shared/makhtar/lp_llm}"
: "${LP_CODE:?Set LP_CODE to the deployed repository}"
MONO_CONFIG="$LP_CODE/configs/llm/qwen3_monolithic_group128.yaml"
MONO_GRANULARITY=group
MONO_OUTPUT_ROOT="${MONO_OUTPUT_ROOT:-$LP_ROOT/results/qwen3_monolithic_group128}"
source "$LP_CODE/hpc/monolithic_batch_common.sh"
run_monolithic_batch
