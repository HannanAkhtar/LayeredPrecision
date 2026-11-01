#!/usr/bin/env bash
#SBATCH --job-name=lp-reconstruct
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
: "${LP_CODE:?Set LP_CODE}"
: "${LP_ARTIFACT:?Set LP_ARTIFACT}"
: "${LP_OUTPUT:?Set LP_OUTPUT}"
: "${LP_DEPTH:?Set LP_DEPTH}"
cd "$LP_CODE"
start=$SECONDS
python -m lp_llm.reconstruct --artifact "$LP_ARTIFACT" --depth "$LP_DEPTH" --output "$LP_OUTPUT"
echo "Reconstruction completed in $((SECONDS-start)) seconds: $LP_OUTPUT/COMPLETE.json"
