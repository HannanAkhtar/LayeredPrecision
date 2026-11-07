# Qwen3-8B W2 quantizer follow-ups

This adds S01 SignRoundV2, R01 symmetric NeUQI optimized RTN, R02 asymmetric NeUQI optimized RTN, and R03 GPTQModel GPTQ. No new 8B measurements are claimed. Existing J01–J14 and PR03/PR04 checkpoints and result directories are untouched.

R01/R02 use the same pinned AutoRound PR source as PR04. In this source, asymmetric zero-iteration optimized RTN requires `enable_neuqi=True`; otherwise the dispatcher runs plain asymmetric min/max RTN. Thus R01 versus PR04 tests NeUQI on the same revision; R02 versus R01 tests symmetry at fixed NeUQI settings. The [numerical audit](numerical_audit.md) records the older J05 revision and remaining confounders.

## Prepare AUS checkouts

Run on a login node; these commands do not submit jobs:

```bash
cd /shared/makhtar/lp_llm/code
git fetch origin experiment/rrq-w2-reproduction
git switch experiment/rrq-w2-reproduction
git pull --ff-only origin experiment/rrq-w2-reproduction
test "$(git -C /shared/makhtar/lp_llm/references/rrq_pr2308/auto-round rev-parse HEAD)" = 7985c1b0f4eef65fb717511668fda7e347e584c3
test "$(git -C /shared/makhtar/lp_llm/references/rrq_w2/MatGPTQ rev-parse HEAD)" = 0e5bd93bc53381f7195db870f5408ae04906bb3c
mkdir -p /shared/makhtar/lp_llm/references/quantizer_followups
git clone https://github.com/ModelCloud/GPTQModel.git /shared/makhtar/lp_llm/references/quantizer_followups/GPTQModel
git -C /shared/makhtar/lp_llm/references/quantizer_followups/GPTQModel checkout --detach 70579a771423767da2af57b67d2d3adf014eebb6
bash hpc/quantizer_followups/setup_gptq_env.sh
```

GPTQModel's pinned source requires newer Torch and Transformers versions than the validated LP environment. The setup script creates `/shared/makhtar/lp_llm/envs/gptq_followups` as a separate environment and installs that source's declared requirements there. It does not modify the LP environment. The GPTQ quantization subprocess uses the separate environment; the MatGPTQ evaluator uses the original LP Python. Verify that the new Torch/CUDA build supports the A10G driver during the smoke job. The reference source must remain unmodified. The AutoRound `RRQ_PR_OVERLAY`, if needed, is validated and applied only to its subprocess. Never export the overlay as a global `PYTHONPATH` before activating Conda.

## CPU checks and GPU smoke

```bash
/shared/makhtar/lp_llm/envs/lp_llm/bin/python -m pytest tests/test_quantizer_followups.py tests/test_rrq_pr2308.py -q
bash -n hpc/quantizer_followups/*.sh hpc/quantizer_followups/*.sbatch
mkdir -p /shared/makhtar/lp_llm/logs
```

Run one **smoke** job at a time before each full experiment. The smoke path runs actual 2-bit tuning or GPTQ quantization on a tiny one-layer Qwen3 model and reports allocated GPU memory and process RSS. It does not prove that an 8B model fits the allocation. Inspect the log and the method's projected 8B memory needs; if the 12 GB/23 GB A10G allocation is insufficient, stop and record the limitation without reducing samples, length, or model coverage.

Submit one smoke job at a time, wait for completion, and inspect its log before the next ID. For example:

```bash
sbatch --export=ALL,EXPERIMENT_ID=s01,FOLLOWUP_SMOKE=1 hpc/quantizer_followups/run_experiment.sbatch
squeue -u "$USER" -o '%.18i %.20j %.2t %.10M %.2D %R'
```

## Full jobs, at most two at a time

After smoke results and resource feasibility are reviewed, submit the first pair, wait for both to exit, then submit the second pair. The submission helper takes a lock and refuses while another follow-up job is queued or running. Each job has its own lock and output directory.

```bash
cd /shared/makhtar/lp_llm/code
bash hpc/quantizer_followups/submit_pair.sh first    # S01 + R01
squeue -u "$USER" -o '%.18i %.20j %.2t %.10M %.2D %R'
# After both jobs finish successfully and results are inspected:
bash hpc/quantizer_followups/submit_pair.sh second   # R02 + R03
```

To resume a completed encoding without repeating quantization, rerun the same experiment ID. `FOLLOWUP_STAGE=encode` validates or completes only the encoding stage. Example:

```bash
sbatch --export=ALL,EXPERIMENT_ID=r01,FOLLOWUP_STAGE=encode hpc/quantizer_followups/run_experiment.sbatch
sbatch --export=ALL,EXPERIMENT_ID=r01 hpc/quantizer_followups/run_experiment.sbatch
```

Never submit duplicate IDs simultaneously. Do not delete an unmarked export: the runner recovers it only if `quantization_process.json` records a successful upstream exit and full checkpoint validation passes. A mismatched shard or identity fails without overwrite.

## Inspect results

```bash
find /shared/makhtar/lp_llm/results/qwen3_8b_base/quantizer_followups -maxdepth 2 -type f -name '*.json' -print
/shared/makhtar/lp_llm/envs/lp_llm/bin/python -m lp_llm.quantizer_followups --aggregate --result-root /shared/makhtar/lp_llm/results/qwen3_8b_base
tail -n 100 /shared/makhtar/lp_llm/results/qwen3_8b_base/quantizer_followups/r01/quantization.log
```

Each `s01`/`r01`/`r02`/`r03` directory stores `run.json`, `source_audit.json`, `resolved_config.json`, `quantization.log`, `export/`, `checkpoint_validation.json`, `encoding_complete.json`, `evaluation.log`, and `full.json` after success. `failure.json` is written on runner errors. R03's `export/packed` is retained separately from its `export/qdq` for numerical and storage auditing. Runtime values are recorded only when observed.
