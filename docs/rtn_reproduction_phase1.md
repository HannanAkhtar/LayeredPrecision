# Phase 1: Qwen3-8B-Base 2-bit RTN reproduction

This branch investigates the user-reported MatGPTQ-style WikiText-2 results:
BF16 6.99445, asymmetric G128 884,229.97, restricted symmetric G128
24,664,416.02, and the existing plain AutoRound-compatible FP32 LP base
9,730.84365. The [RRQ paper](https://arxiv.org/html/2608.04048) reports
16.75 for its all-RTN Qwen3-8B-Base 2-bit model. These are different
provenance records; no result is adjusted to match the paper.

## Verified source and numerical choices

The source revision is Intel AutoRound
[`6c824f741be358983baba87aff83f9c5aa0a01e7`](https://github.com/intel/auto-round/tree/6c824f741be358983baba87aff83f9c5aa0a01e7).
Byte-verified Apache-2.0 copies of its
[`int.py`](https://github.com/intel/auto-round/blob/6c824f741be358983baba87aff83f9c5aa0a01e7/auto_round/data_type/int.py)
[`data_type/utils.py`](https://github.com/intel/auto-round/blob/6c824f741be358983baba87aff83f9c5aa0a01e7/auto_round/data_type/utils.py),
and [`utils/common.py`](https://github.com/intel/auto-round/blob/6c824f741be358983baba87aff83f9c5aa0a01e7/auto_round/utils/common.py)
are retained in `tests/fixtures`. The tests execute the actual pinned
function AST nodes and compare scales, signed codes, and QDQ against our
implementations. The package is not needed by AWS inference or decoding.

The pinned [`get_quant_func`](https://github.com/intel/auto-round/blob/6c824f741be358983baba87aff83f9c5aa0a01e7/auto_round/data_type/utils.py#L107-L193)
selects `opt_rtn_int_sym` when `iters=0`, `sym=True`, and optimized RTN is
enabled. With `disable_opt_rtn=True`, it selects `rtn_int_sym`. The pinned
[`RTNConfig`](https://github.com/intel/auto-round/blob/6c824f741be358983baba87aff83f9c5aa0a01e7/auto_round/algorithms/quantization/rtn/config.py#L21-L119)
defaults to optimized RTN for this 2-bit weight-only setting unless the
plain variant is requested. `quant_tensor_sym` is the separate general
symmetric function used by other optimization paths; it is not used for B0–B3.
This describes the pinned revision, not necessarily the revision or dispatch
used for the RRQ paper. The paper states symmetric 2-bit RTN and group size
128 but does not identify its exact AutoRound code commit, numerical input
dtype, excluded parameter list, or checkpoint materialization path.

All four modes use row-local groups of 128, zero padding on each row's final
group, signed codes `[-2, -1, 0, 1]`, `torch.round` ties-to-even, and a
minimum absolute scale of `1e-5`. Packing maps signed code `q` to `q+2`.
For plain RTN, compare positive and negative extrema; use a negative scale
when the positive extremum is at least as large as the negative magnitude,
otherwise a positive scale. Convert the scale to FP16 before division.
Optimized RTN uses `search_scales` with the pinned 181-point 2-bit scale
grid, `imatrix=None`, no NeUQI, and a scale in the input dtype. Its search
minimizes per-group squared error before the same rounding and clipping.

| ID | Algorithm | Original source | Statistics / scale calculation | Stored scale | Rounding and QDQ | Packed decode | Saved checkpoint | Evaluation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | Existing plain RTN, unchanged | BF16 | FP32 | FP16 | FP32 | FP32 | BF16 | FP16 |
| B1 | Pinned plain RTN | BF16 | FP16 | FP16 | FP16 | FP32, then FP16 | BF16 | FP16 |
| B2 | Pinned plain RTN | BF16 | BF16 | FP16 | mixed BF16/FP16 division and multiplication in FP32, QDQ cast BF16 | FP32, then BF16 | BF16 | FP16 |
| B3 | Pinned optimized `search_scales` RTN | BF16 | FP16 | FP16 | FP16 | FP32, then FP16 | BF16 | FP16 |

B1–B3 quantize one 2-bit base only. Their decoder first reconstructs the
packed FP32 value, then casts to the reference QDQ dtype before BF16
serialization. The manifest records errors before and after that cast.
For ordinary BF16 values within FP16's representable range, the B1 and B2
plain RTN codes and scales can be identical. Their modes still perform
statistics and QDQ in their selected dtypes, which the tests verify. Any
measured difference must be reported, not assumed.

## Coverage and evaluation

The unchanged seven-projection policy quantizes `q_proj`, `k_proj`,
`v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj` in every transformer
block. Embeddings, LM head, norms, and other parameters remain BF16.
`rtn_audit` lists every quantized and excluded tensor, source dtype, counts
by projection and block, and source shard hashes. The production configs
require 6,945,767,424 quantized and 1,244,967,936 excluded parameters;
screening stops if the source checkpoint differs. The exact AutoRound and
RRQ tensor selection on their original runs remains unverified.

The diagnostic evaluator uses the first eight complete non-overlapping
2048-token blocks of WikiText-2 raw test text joined by `\n\n`, with no
special tokens. It records the selected token hash, weighted NLL,
cross-entropy, perplexity, and per-block values. The full result uses the
existing **untracked** AWS `lp_llm/evaluate_matgptq.py` and its established
FP16, 2048-token protocol. That evaluator is not added to Git or modified.
Its exact local source and block-tail behavior must be checked on AWS before
interpreting comparisons with the diagnostic subset.

## Local verification

```bash
python -m pytest tests/test_llm_rtn_upstream.py tests/test_llm_rtn_phase1.py -q
python -m pytest tests -q
bash -n hpc/rtn_reproduction_screen.sbatch hpc/rtn_reproduction_full.sbatch
```

## AWS deployment and execution

Transfer the reviewed branch files to the AWS checkout using your normal
deployment process. Keep the existing untracked evaluator at its path; this
branch has no tracked file with that name. The scripts only activate the
existing Conda environment and never install or replace packages. Before
submission, inspect `lp_llm/evaluate_matgptq.py` to confirm its established
`--checkpoint`, `--output`, `--sequence-length`, and `--dtype` CLI and the
JSON perplexity key. Confirm the selected Qwen model revision and source
checkpoint location. No jobs are submitted by these instructions.

```bash
export LP_ROOT=/shared/makhtar/lp_llm
export LP_CODE=$LP_ROOT/paper_release
test -f "$LP_ROOT/models/qwen3-8b-base/model.safetensors.index.json"
test -f "$LP_CODE/lp_llm/evaluate_matgptq.py"
mkdir -p "$LP_ROOT/logs" "$LP_ROOT/results/qwen3_8b_base/rtn_reproduction"
sbatch --export=ALL,LP_CODE="$LP_CODE",LP_ROOT="$LP_ROOT" \
  "$LP_CODE/hpc/rtn_reproduction_screen.sbatch"
```

The screen job runs pinned numerical parity, coverage audit, one-base
checkpoint construction, fixed-token logit comparison, and diagnostic
perplexity for B1, B2, then B3 in one allocation. It can be resubmitted:
completed stages check identities and continue. Read
`results/qwen3_8b_base/rtn_reproduction/comparison.md` and each
`<experiment_id>/result.json` before choosing full runs. A high diagnostic
perplexity does not automatically exclude a mode.

```bash
# Example selection after screening; edit RTN_EXPERIMENTS as warranted.
sbatch --export=ALL,LP_CODE="$LP_CODE",LP_ROOT="$LP_ROOT",RTN_EXPERIMENTS="b1_plain_fp16 b3_opt_fp16" \
  "$LP_CODE/hpc/rtn_reproduction_full.sbatch"
```

Both jobs default to one GPU, four CPUs, 12 GB host RAM, and 12 hours on the
existing account and partition. Override resource requests with `sbatch
--mem=... --cpus-per-task=... --time=...` within the QoS limits. No script
regenerates the existing B0 hierarchy or changes its result. `comparison.json`
preserves B0's 9,730.84365 user-reported full PPL if its original JSON is
unavailable and labels that provenance explicitly. RRQ's 16.75 remains an
external literature value until its exact implementation and protocol can
be established.

## B3 checkpoint materialization follow-up

The initial B3 run reported exact upstream-versus-LP in-memory logits, but
BF16-saved/reloaded logit RMSE 0.2338584661 and maximum error 1.75, with
eight-block diagnostic perplexity 315.02235. The existing decoder rounds
the FP16 optimized RTN QDQ values to BF16 when saving. This is a plausible
source of the discrepancy; the full Qwen3-8B weight/logit confirmation must
run against the existing AWS artifact.

The `--checkpoint-dtype float16` reconstruction option is restricted to B3.
It writes to a **new** checkpoint directory, records the materialization
dtype in the reconstruction identity, stores all parameters as FP16, and
updates the copied model config to FP16. The existing BF16 checkpoint,
quantized streams, scale metadata, and diagnostic JSON remain in place. A
different dtype cannot resume into the same reconstruction directory.

The follow-up compares each eligible saved tensor with exact pinned upstream
FP16 QDQ, reporting exact equality, maximum absolute error, and RMSE for
packed decoding, BF16 saving, and FP16 saving. It also checks that the BF16
checkpoint equals the upstream values explicitly cast to BF16. Logits are
measured for upstream in memory, LP in memory, the old BF16 checkpoint, and
the new FP16 checkpoint. Both checkpoints use the same eight complete
non-overlapping 2048-token WikiText-2 blocks and FP16 inference. The runner
checks their selected-token hashes and scored-token counts before reporting
perplexity and mean cross-entropy. No full WikiText-2 evaluation is invoked.

```bash
export LP_ROOT=/shared/makhtar/lp_llm
export LP_CODE=$LP_ROOT/paper_release
sbatch --export=ALL,LP_CODE="$LP_CODE",LP_ROOT="$LP_ROOT" \
  "$LP_CODE/hpc/rtn_b3_materialization.sbatch"
```

New results are under
`results/qwen3_8b_base/rtn_reproduction/b3_opt_fp16/materialization_fp16/`:
`checkpoint/`, `weights.json`, `logits.json`, `diagnostic.json`, and
`comparison.json`. The consolidated `comparison.md` adds a separate B3
FP16-checkpoint row; the algorithm, quantized codes, artifact storage, MSE,
and SQNR are the same for both B3 rows. FP16 and BF16 each occupy two bytes
per parameter. The report records checkpoint and model-loading dtypes
separately because equal byte counts do not imply equal logits.
