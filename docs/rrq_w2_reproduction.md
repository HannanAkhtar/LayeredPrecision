# RRQ W2 reproduction on AWS

This branch prepares J01–J14. It contains no new Qwen3-8B measurements. RRQ's published 16.75 perplexity is a reference claim, not a target or a measured result here. The existing B0–B3 checkpoints, configs, results, and submitted B3 full evaluation are untouched.

## Protocol and identities

The [pinned MatGPTQ evaluator](https://github.com/IST-DASLab/MatGPTQ/tree/0e5bd93bc53381f7195db870f5408ae04906bb3c) supplies the official full evaluation. It uses WikiText-2 raw test, `"\n\n"` concatenation, the slow tokenizer, no special tokens, 2048-token nonoverlapping blocks, drops the tail, and scores 2047 predictions per block at FP16 inference. Its script prints perplexity to **two decimal places**; the stored cross-entropy is `log` of this rounded number, with that limitation labeled in JSON. J01 separately runs an independent scorer over the same checkpoint and token construction; the official and independent values must be interpreted at the official display precision. The source model revision in all configs is `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`; checkpoint content hashes and token hashes are recorded at execution.

The default [AutoRound revision](https://github.com/intel/auto-round/commit/6c824f741be358983baba87aff83f9c5aa0a01e7) is dated **November 3, 2026**, after the August RRQ preprint. It remains the validated low-level B3 reference and is labeled a *current official AutoRound reproduction*, not a historical RRQ implementation. A separate [July 6, 2026 `v0.14.0` revision](https://github.com/intel/auto-round/commit/24bfce00bb3bd053498232f0e8395118fe10a906) supports the plain and optimized RTN CLI recipes with W2A16 and fake export. It is offered as `preprint_candidate`; it predates the preprint but has not been established as RRQ's exact revision or GPU-validated with Qwen3-8B. Its results use separate `j04/preprint_candidate/` through `j07/preprint_candidate/` directories.

Both revisions expose `run_rtn()` and `run_opt_rtn()` from `auto_round.cli.main`. The runner invokes those upstream functions in a **subprocess**, with the selected source checkout first on `PYTHONPATH`; no AutoRound package installation is required. `W2A16` is G128; `W2A16G64` and `W2A16G32` set the smaller groups. Commands explicitly fix zero tuning iterations, the recipe's configured `nsamples` (1 plain, 128 optimized), FP16 model loading, and fake QDQ export. The upstream RTN zero-shot path is expected to consume no calibration examples; `nsamples` is **not** an observed consumption count, and runtime consumption remains unverified until logs are inspected. Optimized symmetric 2-bit RTN uses the source's ±90 index scale search at 0.01 increments. The runner records the actual imported AutoRound file, runtime package versions, checkpoint dtypes, upstream exported module-selection metadata when present, and a separate changed-versus-unchanged FP16 weight diagnostic. The weight diagnostic alone cannot prove the selected module set: an eligible weight can remain exactly unchanged after quantization. If fake export omits module-selection metadata, the result explicitly reports that limitation. Review the AutoRound log and coverage before interpreting J04/J05 as matched to the seven-projection LP policy.

J08 is **unavailable**. The exact RRQ SignRoundV2 calibration dataset, sample count, iterations, learning rate, optimizer, coverage, and historical implementation recipe could not be established. Its proposal is to use an isolated pinned official AutoRound calibration workflow only after those fields and adequate GPU/host memory have been verified. No substitute quantizer is run as an RRQ reproduction.

J09–J14 restore selected original BF16 tensors into the verified B3 FP16 QDQ checkpoint, without re-quantizing the remaining B3 tensors. Their default screen is the existing eight-block diagnostic, distinct from optional full MatGPTQ evaluation. `estimated_logical_compact_bytes` counts referenced B3 decoder files, restored BF16 Safetensors shards, and the hybrid reference manifest. It is a **logical estimate**, not a physically measured self-contained compressed package. Actual QDQ checkpoint Safetensors and metadata bytes are reported separately. An independently deployable compressed representation would require copying its selected B3 streams, shared tensors, sidecars, base manifest, restored shards, and hybrid manifest into one package and measuring that package; this branch does not build one. The hybrids have mixed effective precision and must not be described as uniformly 2-bit.

Official AutoRound's `fake` export is a QDQ checkpoint, so it has a measured QDQ Safetensors footprint but no claimed packed effective bpp. Its result table cell remains empty until an independent packed representation is built and measured. The dataset loader in the historical MatGPTQ script does not pin a Hub revision; the run records the resolved dataset version and fingerprint instead.

## Deploy and verify

Run these commands **on AWS**; they have not been run locally. All jobs use `/shared/makhtar/lp_llm/envs/lp_llm/bin/python` by default (`RRQ_PYTHON` overrides it for compatibility testing). Setup fetches and checks source revisions and imports only. It never clones an environment or installs packages. The official MatGPTQ script runs from its pinned source using the LP environment's runtime dependencies. This reproduces the historical **source protocol**, not MatGPTQ's original dependency environment. Its `--help` import succeeds in the confirmed LP environment; full data loading, numerical parity and inference remain to be checked on AWS.

AutoRound's requirements include `pydantic`, `py-cpuinfo`, `filelock`, and `requests`. Check which are actually missing; the known missing package is `pydantic`. The explicit overlay installer uses pinned binary wheels and `pip --target` under `$LP_ROOT/references/rrq_w2/extra_deps`, with `--no-deps` to avoid installing or shadowing Torch, Transformers, CUDA, NumPy, or other working packages. It installs required Pydantic companions only where needed, validates imports, then atomically marks the overlay complete. It never writes to the LP environment. The interrupted `$LP_ROOT/envs/rrq_matgptq` directory is left untouched.

If the source import still fails because of a real dependency conflict, stop and record the import error. A manual fallback is a small `venv --system-site-packages` made with the LP Python, followed by installation of only the conflicting supplemental packages there and an import probe. Verify that it can see the LP Torch/CUDA packages before use. Do not clone the full Conda environment or silently switch reference implementations.

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
RRQ_SETUP_AUTOROUND=0 bash hpc/rrq_w2/setup_reference_env.sh
python -m pytest tests/test_rrq_w2.py -q
git -C "$LP_ROOT/references/rrq_w2/MatGPTQ" rev-parse HEAD
# J01–J03 can now be submitted without any AutoRound dependencies.
# Fetch the AutoRound checkout before J04/J05:
bash hpc/rrq_w2/setup_reference_env.sh
git -C "$LP_ROOT/references/rrq_w2/auto-round" rev-parse HEAD
python -m lp_llm.rrq_overlay check --target "$LP_ROOT/references/rrq_w2/extra_deps"
# Only if the check reports missing packages, explicitly install the overlay:
python -m lp_llm.rrq_overlay install --target "$LP_ROOT/references/rrq_w2/extra_deps"
```

The batch script runs preflight within its allocation. It checks the selected source revision and imports, the allocated GPU's free memory (20 GB default), and at least 12 GB host allocation. J01–J03 do not inspect AutoRound or its overlay. All jobs start from the previously successful **12 GB** request because A10G host RAM is limited. AutoRound 8B export may require more: set `RRQ_AUTOROUND_MEM` and `RRQ_MIN_HOST_MEMORY_MB` only after confirming the node can offer the requested memory. CPU, memory, and wall time can also be overridden with `sbatch` options within QoS limits. No installation occurs in GPU jobs. Neither preflight nor these defaults prove the 8B run will fit.

J01–J05 establish evaluation and official pipeline comparability. Inspect them before interpreting J06–J14. J03 evaluates the **existing** BF16 B3 checkpoint using official MatGPTQ; it does not rerun the previously submitted LP B3 full evaluator.

```bash
cd /shared/makhtar/lp_llm/code
RRQ_CONCURRENCY=2 bash hpc/rrq_w2/submit_selected.sh j01 j02 j03 j04 j05
# equivalent single job:
sbatch --mem=12G --export=ALL,EXPERIMENT_ID=j04 hpc/rrq_w2/run_experiment.sbatch
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

J01–J07 always create one `full.json` identity even if `RRQ_STAGE=screen` is supplied; rerunning with `RRQ_STAGE=full` resumes that result. J09–J14 retain separate `diagnostic.json` and `full.json`. To try the separate July AutoRound candidate after fetching its pinned source, run:

```bash
RRQ_SETUP_AUTOROUND=0 RRQ_SETUP_HISTORICAL=1 bash hpc/rrq_w2/setup_reference_env.sh
sbatch --mem=12G --export=ALL,EXPERIMENT_ID=j04,RRQ_AUTOROUND_VARIANT=preprint_candidate hpc/rrq_w2/run_experiment.sbatch
```

Each job writes under `$RESULT_ROOT/rrq_w2_reproduction/jNN/`. Inspect `run.json`, `encoding_complete.json` where applicable, `diagnostic.json`, `full.json`, `autoround.log`, and `matgptq.log`. Aggregate with:

```bash
bash hpc/rrq_w2/collect_results.sh
cat "$RESULT_ROOT/rrq_w2_reproduction/aggregate.md"
```

Null table cells mean no measurement exists. The JSON results retain config hashes, checkpoint identities, dataset fingerprint and token hash where full evaluation ran, inference/checkpoint dtype, logs, elapsed time, and sampled GPU memory usage. The `nvidia-smi` sample includes other processes on the busiest GPU and is distinct from the runner's PyTorch allocation counter.
