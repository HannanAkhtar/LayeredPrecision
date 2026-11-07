# RRQ PR #2308: Qwen3-8B-Base W2G128 reproduction

This suite uses the [pinned Intel PR source](https://github.com/intel/auto-round/tree/7985c1b0f4eef65fb717511668fda7e347e584c3) in a separate checkout. The existing J01–J14 directories, AutoRound checkout, and LP environment remain separate. New outputs go under `/shared/makhtar/lp_llm/results/qwen3_8b_base/rrq_pr2308/`.

## What the pinned source establishes (PR01)

`RRQConfig(iters=0, num_residual_planes=1)` creates a base plus one residual INT2 plane. `iters=0` removes sign gradient updates, but `need_calib=True` still sends the run through model calibration. Forward hooks sum squared FP32 inputs for each eligible module, and the block quantizer normalizes the resulting importance matrix. In normal compressor context, the first plane calls `_quantize_layer_via_rtn(layer, disable_opt_rtn=False)`, which uses the regular optimized RTN wrapper with zero iterations. `disable_opt_rtn` remains true on the RRQ *config* to prevent a dispatch coercion; the first plane overrides it to false. The default scale dtype is FP16 unless explicitly overridden. The exporter writes a packed base under `base/` and residual plane under `residual/`.

J05 is an official AutoRound optimized W2G128 RTN fake-QDQ export from revision `6c824f7`; its recorded log says 128 `NeelNanda/pile-10k` examples were prepared and cached. Its CLI requested FP16, but the exported J05 checkpoint contained BF16 tensors. The exact calibration sample order, importance matrices, and any differing module selection were not persisted. PR03 therefore records its actual imports, export dtypes and module list; PR04 measures weight and evaluation parity. The source path supports a parity *prediction*, not a measured 8B result.

PR02 extracts the pinned PR's actual `_quantize_layer_rrq` method into a minimal `Linear` fixture, with the same pinned `opt_rtn_int_sym` arithmetic behind its standard base delegate. FP16 and BF16 cases compare codes, scales, QDQ, exact fraction, max error, MSE, and weighted MSE. This does not execute full compressor calibration or `WrapperLinear`. Packed INT2 decoding is separately tested on CPU and verified against the official symmetric `qlinear_torch_zp` bit layout. The first full-model assertion remains PR03/PR04.

## Controlled matrix

| ID | Priority | Purpose | Relative compute | Launch rule |
|---|---:|---|---|---|
| PR01 | 1 | Pin and audit source and J05 path | negligible | Run before GPU |
| PR02 | 2 | CPU numerical base parity | negligible | Run before GPU |
| PR03 | 3 | Official 8B base plus one residual export | high | One GPU job after PR01/PR02 review |
| PR04 | 4 | Compare J05 and PR base; complete MatGPTQ PPL | medium | After validated PR03 |

Conditional PR05–PR12 and exact triggers are in [`experiment_plan.yaml`](../configs/rrq_pr2308/experiment_plan.yaml). PR05 replays calibration only if weights differ; PR06 tests dtype if the discrepancy persists; PR07 audits coverage if module sets differ; PR08 investigates packing if QDQ and packed inference disagree. PR09 checks four-plane base invariance only after base parity. PR10 is a 128 versus 512 sample ablation. PR11 reuses J03 for an unweighted control. PR12 requires an authoritative historical recipe. None is an automatic Slurm job.

## Prepare the separate source and run CPU checks

```bash
export LP_ROOT=/shared/makhtar/lp_llm
export LP_CODE=$LP_ROOT/code
export RRQ_PR_SOURCE=$LP_ROOT/references/rrq_pr2308/auto-round
mkdir -p "$LP_ROOT/references/rrq_pr2308" "$LP_ROOT/logs"
git clone https://github.com/intel/auto-round.git "$RRQ_PR_SOURCE"
git -C "$RRQ_PR_SOURCE" checkout --detach 7985c1b0f4eef65fb717511668fda7e347e584c3
git -C "$RRQ_PR_SOURCE" rev-parse HEAD
cd "$LP_CODE"
LP_PYTHON=$LP_ROOT/envs/lp_llm/bin/python
"$LP_PYTHON" -m lp_llm.rrq_pr2308 --experiment-id pr01 --rrq-source "$RRQ_PR_SOURCE" --result-root "$LP_ROOT/results/qwen3_8b_base"
"$LP_PYTHON" -m lp_llm.rrq_pr2308 --experiment-id pr02 --rrq-source "$RRQ_PR_SOURCE" --result-root "$LP_ROOT/results/qwen3_8b_base"
"$LP_PYTHON" -m pytest -q tests/test_rrq_pr2308.py
```

The PR source may require supplemental packages. Inspect the existing LP environment before adding anything. If needed, use a separate `RRQ_PR_OVERLAY` directory and install only compatible missing supplemental dependencies into it with `pip --target`; do not reinstall or replace PyTorch, CUDA, Transformers, or the original AutoRound reference. PR03 verifies the imported `auto_round.__file__` is inside `RRQ_PR_SOURCE`.

## Submit only the selected stages

```bash
cd /shared/makhtar/lp_llm/code
export LP_ROOT=/shared/makhtar/lp_llm
export LP_CODE=$LP_ROOT/code
export RRQ_PR_SOURCE=$LP_ROOT/references/rrq_pr2308/auto-round
mkdir -p "$LP_ROOT/logs"
EXPERIMENT_ID=pr03 sbatch --export=ALL,EXPERIMENT_ID=pr03 --mem=12G --cpus-per-task=4 hpc/rrq_pr2308/run_experiment.sbatch
# Inspect pr03/quantization_complete.json, module inventory, dtypes and logs first.
EXPERIMENT_ID=pr04 sbatch --export=ALL,EXPERIMENT_ID=pr04 --mem=12G --cpus-per-task=4 hpc/rrq_pr2308/run_experiment.sbatch
```

The default is one GPU, 12 GB host RAM, four CPUs and 12 hours on account `acc-makhtar`, partition `gpu`; override `--mem`, `--cpus-per-task` or `--time` if preflight or QoS requires it. The preflight prints available host and GPU memory and refuses an unavailable source, wrong revision, or failed import. PR03 refuses an unmarked partial official export and never overwrites its shards on resume. PR04 verifies each existing tensor against the expected decode after an interrupted QDQ write, fills missing shards, and rejects any mismatch; evaluation resumes from the QDQ completion identity. No job is submitted by these scripts without an explicit `sbatch` command.

PR04 decodes the official packed INT2 base using the PR's `qlinear_torch_zp` code, scale, and zero-point orientation, then writes a complete QDQ checkpoint with original nonquantized tensors copied exactly. It compares each quantized projection to J05 in FP32 after matched decode, and preserves the original tensor inventory. It checks the tokenizer SHA256 `8ffcee84dd4b684ec631cb52845e18c9c181ce6fdb8b26d65a4ffdc3d35fbda3` before the pinned MatGPTQ evaluator runs with FP16 inference and nonoverlapping 2048-token blocks. The evaluator reports rounded PPL; any cross-entropy derived from it is explicitly labeled approximate.

The one-call official exporter does not persist the numerical base immediately before packing. PR04 verifies the packed bit layout against the official decoder on CPU and records that a full-model prepacking comparison is unavailable; it does not claim that comparison passed.

```bash
$LP_ROOT/envs/lp_llm/bin/python -m lp_llm.rrq_pr2308_aggregate \
  --result-root "$LP_ROOT/results/qwen3_8b_base" \
  --output "$LP_ROOT/results/qwen3_8b_base/rrq_pr2308/summary.json"
```

## Interpretation and unresolved historical question

PR01/PR02 are source and small CPU findings. PR03/PR04, when run, reproduce the **November PR** on our pinned model and MatGPTQ protocol. PR10 and PR11 are calibration ablations. Only PR12, after a verified RRQ-era command/source/checkpoint is available, could claim a historically exact reproduction of the August paper's 16.75 PPL. The paper's reported number is a published reference, not a result measured by this suite. The historical calibration-free procedure, exact author revision, dataset/tokenizer/evaluation details, and checkpoint are still unresolved.
