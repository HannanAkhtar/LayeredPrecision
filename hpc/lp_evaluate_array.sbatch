#!/usr/bin/env bash
#SBATCH --job-name=lp-eval-array
#SBATCH --account=acc-makhtar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/shared/makhtar/lp_llm/logs/%x-%A_%a.out
#SBATCH --array=1-4%1
set -euo pipefail
source /opt/miniconda/etc/profile.d/conda.sh
conda activate /shared/makhtar/lp_llm/envs/lp_llm
source /shared/makhtar/lp_llm/env.sh
unset TRANSFORMERS_CACHE
: "${LP_CODE:?Set LP_CODE}"
: "${LP_CHECKPOINT_ROOT:?Set LP_CHECKPOINT_ROOT}"
: "${LP_RESULT_ROOT:?Set LP_RESULT_ROOT}"
LP_PROTOCOL="${LP_PROTOCOL:-wikitext}"
cd "$LP_CODE"
python -m lp_llm.evaluate "$LP_PROTOCOL" \
  --checkpoint "$LP_CHECKPOINT_ROOT/prefix-${SLURM_ARRAY_TASK_ID}" \
  --output "$LP_RESULT_ROOT/${LP_PROTOCOL}-${SLURM_ARRAY_TASK_ID}.json"
