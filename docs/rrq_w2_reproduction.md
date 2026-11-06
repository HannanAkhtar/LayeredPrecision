# RRQ W2 reproduction on AWS

This branch prepares J01–J14. It contains no new Qwen3-8B measurements. RRQ's published 16.75 perplexity is a reference claim, not a target or a measured result here. The existing B0–B3 checkpoints, configs, results, and submitted B3 full evaluation are untouched.

## Protocol and identities

The [pinned MatGPTQ evaluator](https://github.com/IST-DASLab/MatGPTQ/tree/0e5bd93bc53381f7195db870f5408ae04906bb3c) supplies the official full evaluation. It uses WikiText-2 raw test, `"\n\n"` concatenation, the slow tokenizer, no special tokens, 2048-token nonoverlapping blocks, drops the tail, and scores 2047 predictions per block at FP16 inference. Its script prints perplexity to **two decimal places**; the stored cross-entropy is `log` of this rounded number, with that limitation labeled in JSON. J01 separately runs an independent scorer over the same checkpoint and token construction; the official and independent values must be interpreted at the official display precision. The source model revision in all configs is `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`; checkpoint content hashes and token hashes are recorded at execution.

The [pinned AutoRound revision](https://github.com/intel/auto-round/tree/6c824f741be358983baba87aff83f9c5aa0a01e7) provides the actual `auto-round-rtn` and `auto-round-opt-rtn` CLI recipes. These generate independent fake-format QDQ checkpoints. `W2A16` is the G128 scheme, while `W2A16G64` and `W2A16G32` set the smaller groups. Plain RTN uses no scale optimization or calibration; optimized RTN uses AutoRound's scale-search path and no gradient updates. The runner records the upstream CLI command and changed versus unchanged tensor names by exact FP16 comparison with the source. This reports observed coverage; a weight that happens to survive quantization exactly cannot be distinguished by this method alone. Review the AutoRound log and coverage before interpreting J04/J05 as matched to the seven-projection LP policy.

J08 is **unavailable**. The exact RRQ SignRoundV2 calibration dataset, sample count, iterations, learning rate, optimizer, coverage, and historical implementation recipe could not be established. Its proposal is to use an isolated pinned official AutoRound calibration workflow only after those fields and adequate GPU/host memory have been verified. No substitute quantizer is run as an RRQ reproduction.

J09–J14 restore selected original BF16 tensors into the verified B3 FP16 QDQ checkpoint, without re-quantizing the remaining B3 tensors. Their default screen is the existing eight-block diagnostic, which is distinct from the optional full MatGPTQ evaluation. The compact hybrid footprint counts the B3 files still needed for decoding, the restored BF16 Safetensors files (including headers), and the hybrid reference manifest. The on-disk full QDQ checkpoint footprint is reported separately. The hybrids have mixed effective precision and must not be described as uniformly 2-bit.

Official AutoRound's `fake` export is a QDQ checkpoint, so it has a measured QDQ Safetensors footprint but no claimed packed effective bpp. Its result table cell remains empty until an independent packed representation is built and measured. The dataset loader in the historical MatGPTQ script does not pin a Hub revision; the run records the resolved dataset version and fingerprint instead.

## Deploy and verify

Run these commands **on AWS**; they have not been run locally. The setup script clones the LP environment into separate AutoRound and MatGPTQ environments, then installs each pinned upstream dependency set only into its clone. The batch runner and official evaluator use the MatGPTQ clone, while the AutoRound CLI uses its own clone. The original LP environment is untouched.

```bash
export LP_ROOT=/shared/makhtar/lp_llm
export LP_CODE=/shared/makhtar/lp_llm/code
export LP_SOURCE=/shared/makhtar/lp_llm/models/qwen3-8b-base
export RESULT_ROOT=/shared/makhtar/lp_llm/results/qwen3_8b_base
cd "$LP_CODE"
git fetch origin experiment/rrq-w2-reproduction
git switch experiment/rrq-w2-reproduction
git pull --ff-only origin experiment/rrq-w2-reproduction
git rev-parse HEAD
source /opt/miniconda/etc/profile.d/conda.sh
conda activate /shared/makhtar/lp_llm/envs/lp_llm
source /shared/makhtar/lp_llm/env.sh
unset TRANSFORMERS_CACHE
python -m pytest tests -q
bash hpc/rrq_w2/setup_reference_env.sh
"$LP_ROOT/envs/rrq_matgptq/bin/python" -m pytest tests/test_rrq_w2.py -q
git -C "$LP_ROOT/references/rrq_w2/MatGPTQ" rev-parse HEAD
git -C "$LP_ROOT/references/rrq_w2/auto-round" rev-parse HEAD
bash hpc/rrq_w2/preflight.sh
```

Run the preflight on a GPU allocation, for example `srun --account=acc-makhtar --partition=gpu --gres=gpu:1 --mem=12G --time=00:15:00 --pty bash`, or let the batch script run it within each allocation. It requires at least 18 GB free GPU memory by default; raise `RRQ_MIN_GPU_FREE_MB` for a larger model or recipe. Use `sbatch --mem=... --cpus-per-task=... --time=...` to override the 12 GB, 4 CPU, 12 hour defaults per job when preflight or resource monitoring indicates more is needed. Check QoS limits before increasing time. AutoRound's full 8B export may need more than 12 GB host memory; the default preserves the established A10G allocation and is intentionally overridden when required.

J01–J05 establish evaluation and official pipeline comparability. Inspect them before interpreting J06–J14. J03 evaluates the **existing** BF16 B3 checkpoint using official MatGPTQ; it does not rerun the previously submitted LP B3 full evaluator.

```bash
cd /shared/makhtar/lp_llm/code
RRQ_CONCURRENCY=2 bash hpc/rrq_w2/submit_selected.sh j01 j02 j03 j04 j05
# equivalent single job:
sbatch --export=ALL,EXPERIMENT_ID=j04 hpc/rrq_w2/run_experiment.sbatch
squeue -u "$USER" -o '%.18i %.24j %.2t %.10M %R'
ls -lh /shared/makhtar/lp_llm/logs/rrq-w2-*.out
python -m lp_llm.rrq_aggregate --result-root "$RESULT_ROOT"
cat "$RESULT_ROOT/rrq_w2_reproduction/aggregate.md"
```

For J06–J14, J08 remains blocked and must be excluded:

```bash
RRQ_CONCURRENCY=2 bash hpc/rrq_w2/submit_selected.sh j06 j07 j09 j10 j11 j12 j13 j14
```

To resume, submit the **same** job ID and stage again. Valid completed stage markers and content identities are reused. An incomplete or corrupt official AutoRound export stops with an explicit error and requires inspection; it is never overwritten automatically. Hybrid tensor markers are checked before reuse. To request a hybrid full evaluation after its diagnostic:

```bash
sbatch --export=ALL,EXPERIMENT_ID=j09,RRQ_STAGE=full hpc/rrq_w2/run_experiment.sbatch
```

Each job writes under `$RESULT_ROOT/rrq_w2_reproduction/jNN/`. Inspect `run.json`, `encoding_complete.json` where applicable, `diagnostic.json`, `full.json`, `autoround.log`, and `matgptq.log`. Aggregate with:

```bash
bash hpc/rrq_w2/collect_results.sh
cat "$RESULT_ROOT/rrq_w2_reproduction/aggregate.md"
```

Null table cells mean no measurement exists. The JSON results retain config hashes, checkpoint identities, dataset fingerprint and token hash where full evaluation ran, inference/checkpoint dtype, logs, elapsed time, and sampled GPU memory usage. The `nvidia-smi` sample includes other processes on the busiest GPU and is distinct from the runner's PyTorch allocation counter.
