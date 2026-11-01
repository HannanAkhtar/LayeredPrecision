#!/usr/bin/env bash
#SBATCH --job-name=lp-evaluate
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
: "${LP_CODE:?Set LP_CODE}"
: "${LP_CHECKPOINT:?Set LP_CHECKPOINT}"
: "${LP_RESULT:?Set LP_RESULT}"
LP_PROTOCOL="${LP_PROTOCOL:-wikitext}"
cd "$LP_CODE"
start=$SECONDS
python -m lp_llm.evaluate "$LP_PROTOCOL" --checkpoint "$LP_CHECKPOINT" --output "$LP_RESULT"
echo "Evaluation completed in $((SECONDS-start)) seconds: $LP_RESULT"
