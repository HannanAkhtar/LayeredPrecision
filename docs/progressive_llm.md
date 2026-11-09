# Progressive LP-LLM matrix (S01 and R02)

The [tracking issue](https://github.com/HannanAkhtar/LayeredPrecision/issues/1)
defines six new LP prefixes and six independent AutoRound controls. This code
does **not** submit any full-model job. The original, raw and repaired W2
checkpoints remain read-only. The existing S01 and R02 WikiText-2 results are
the W2 endpoints: 12.47 and 32.22. J01 is 7.01. Each new point has its own
result namespace and one combined encode → independent validation → MatGPTQ
evaluation allocation. A failed validation exits nonzero before evaluation.

## Encoding and numerical rule

The common enhancement is deterministic calibration-free signed ternary RTN
with a 2-bit stored code. Each output row is grouped into 128 input columns;
the final partial group is independent. In FP32, the per-group scale is the
maximum absolute residual. The grid is `{-s, 0, +s}`; code 0 is unused, codes
1/2/3 denote negative/zero/positive, and the zero point is implicit code 2.
`torch.round` uses ties-to-even, followed by clipping to `[-1,+1]`. The scale
is stored as FP32 so near-zero nonzero residuals and constants remain
representable. An all-zero group has zero scale and decodes to zero. There is
no calibration, learned clipping or SignRoundV2 optimization in an E2 layer.

For each bounded 512-row block, the source and repaired base are read from
Safetensors slices. The source-minus-previous residual and each accumulated
prefix use FP32. Encoder output must equal an independent decoder exactly;
packed codes and scales are read back and checked. A zero-enhancement endpoint
is the existing repaired base tensor with its original dtype. Newly
materialized eligible projections are cast to FP16 for MatGPTQ; excluded
tensors are copied source-exact. Prefixes depend on verified predecessor
chunks and cannot reuse a partial or corrupted layer. Source, base, code,
metadata, and QDQ shard hashes are recorded. Layerwise and overall FP32 MSE,
normalized MSE, and SQNR are recorded by encoding.

The repaired W2 base is an **external, mandatory dense checkpoint**. An LP
hierarchy comprises that checkpoint plus independently decodable E2 files;
it is not yet a standalone compact W2 base bitstream. Measured physical
reconstruction bytes count the base once and add each E2 code/metadata file.
For a hypothetical compact W2 base, the aggregator reports a separately
labeled modeled rate. Monolithic AutoRound `format=fake` exports QDQ weights,
not physical compact W4/W6/W8 streams; their compact rates are also modeled
and never labeled measured. Excluded tensors are counted once in each
hierarchy; simulcast controls count their complete separate checkpoints.

## Job graph and admission

The committed matrix is `configs/progressive_llm/matrix.json`. At the default
two-GPU cap, the planner uses two lanes:

```text
lane 0: s01_lp_2p2 -> s01_lp_2p2p2 -> s01_lp_2p2p2p2 -> s01_w4 -> s01_w6 -> s01_w8
lane 1: r02_lp_2p2 -> r02_lp_2p2p2 -> r02_lp_2p2p2p2 -> r02_w4 -> r02_w6 -> r02_w8
```

Scientific LP predecessor arrows use `afterok`. Independent control jobs use
`afterany` lane dependencies, which release the lane after a failed pilot or
control without treating that failure as scientific success. Their dependencies are scheduling tokens
that cap concurrent allocations. Twelve submitted jobs stay below the QoS
`GrpSubmitJobs=15` limit. The optional three-lane plan places all controls on
the third lane and still submits only twelve jobs. The manifest records every
job ID, code SHA, output directory, stage, predecessor and lane. Every job
archives one commit into local scratch and locks its own result.

Using 6,945,767,424 projection weights, an E2 code payload is
1,736,441,856 bytes (1.617 GiB); FP32 G128 scales add about 0.20 GiB per
stage. Two three-layer branches need roughly 11 GiB of new persistent
residual files. Six dense monolithic QDQ exports need about 92 GiB. Two
concurrent temporary LP evaluation checkpoints can add about 31 GiB; the
planner reserves another 10 GiB and 5% E2 metadata allowance. The read-only
preflight computes the exact modeled floor from source headers on AUS. The LP
pilot also independently materializes each full W2 endpoint into a temporary
checkpoint, then requires MatGPTQ perplexity parity at 12.47/32.22 before
removing that job-owned copy; its shared floor therefore includes one dense
checkpoint plus 5 GiB. It
requires 12 GiB Slurm host allocation, 20 GiB free A10G memory, 2 GiB local
scratch, shared capacity and an explicitly confirmed *remaining* project
quota, or the explicit `LP_MATRIX_ALLOW_UNKNOWN_QUOTA=1` operator override.
The override records `unknown_operator_authorized` and bypasses only the
formal quota check. Actual filesystem free space, GPU, host RAM, scratch,
pilot, and checkpoint checks still apply. NFS `df` does not establish quota.
Neither the tiny tests nor these floors predict full-model peaks.
The matrix submission checks the entire remaining matrix floor once. Each
running point checks its own remaining output and temporary checkpoint floor,
so prior completed points do not cause a false quota failure. Under the
unknown-quota override, external concurrent writes remain an unmeasured risk;
the actual filesystem free-space check is repeated for each point.

## AUS pilot sequence

Pull and pin the commit supplied with the implementation. These commands do
not start any full-model point:

```bash
cd /shared/makhtar/lp_llm/code
set -euo pipefail
git pull --ff-only origin experiment/rrq-w2-reproduction
CODE_SHA=$(git rev-parse HEAD)
LP_ROOT=/shared/makhtar/lp_llm
RESULT_ROOT="$LP_ROOT/results/qwen3_8b_base"
mkdir -p "$LP_ROOT/logs" "$RESULT_ROOT/progressive_llm/pilots"
unset LP_MATRIX_SHARED_QUOTA_FREE_GIB
export LP_MATRIX_ALLOW_UNKNOWN_QUOTA=1
# The completed S01/R02 LP pilots (54161/54162) are preserved and validated
# against their original b650efb commit. Do not submit them again.
scancel 54166 54167 54168 || echo 'One or more old pilot jobs already left the queue'
PYTHONPATH="$LP_ROOT/references/rrq_pr2308/auto-round:$PWD" \
  "$LP_ROOT/envs/lp_llm/bin/python" -m lp_llm.progressive_support \
  --checkout "$LP_ROOT/references/rrq_pr2308/auto-round" \
  --output "$RESULT_ROOT/progressive_llm/pilots/control_support_$CODE_SHA.json" --code-sha "$CODE_SHA"
ids=(s01_w4 s01_w6 s01_w8 r02_w4 r02_w6 r02_w8)
declare -a lane_job=()
declare -a pilot_jobs=()
for index in "${!ids[@]}"; do
  lane=$((index % 3))
  dependency=()
  if [[ -n "${lane_job[$lane]:-}" ]]; then
    dependency=("--dependency=afterany:${lane_job[$lane]}")
  fi
  lane_job[$lane]=$(sbatch --parsable "${dependency[@]}" \
    --export=ALL,FOLLOWUP_CODE_SHA="$CODE_SHA",LP_CONTROL_ID="${ids[$index]}" \
    hpc/progressive_llm/control_pilot.sbatch)
  pilot_jobs+=("${lane_job[$lane]}")
  echo "${ids[$index]} pilot job ${lane_job[$lane]} lane $lane"
done
squeue -u "$USER" -o '%.18i %.30j %.2t %.20R'
pilot_job_ids=$(IFS=,; echo "${pilot_jobs[*]}")
# Run the following only after the six pilots have finished (check squeue).
sacct -j "$pilot_job_ids" --format=JobID,State,ExitCode
# Inspect all eight JSON reports before enabling full submission:
for id in "${ids[@]}"; do
  cat "$RESULT_ROOT/progressive_llm/pilots/$id.json"
done
for branch in s01 r02; do
  cat "$RESULT_ROOT/progressive_llm/pilots/$branch/report.json"
  cat "$RESULT_ROOT/progressive_llm/pilots/$branch/base_parity/base_parity.json"
done
"$LP_ROOT/envs/lp_llm/bin/python" -m lp_llm.progressive_plan verify-pilots \
  --source "$LP_ROOT/models/qwen3-8b-base" --result-root "$RESULT_ROOT" \
  --code-sha "$CODE_SHA"
"$LP_ROOT/envs/lp_llm/bin/python" -m lp_llm.progressive_plan dry-run \
  --result-root "$RESULT_ROOT" --code-sha "$CODE_SHA" --slots 3 \
  --manifest "$RESULT_ROOT/progressive_llm/planned_jobs.json"
```

After all eight pilot reports have been inspected and `verify-pilots` succeeds,
explicitly enable the twelve full points in three GPU lanes. The planner
checks the existing QoS 15-job limit and records each job ID and admission:

```bash
export LP_MATRIX_SUBMIT=1 LP_MATRIX_PILOTS_REVIEWED=1
"$LP_ROOT/envs/lp_llm/bin/python" -m lp_llm.progressive_plan submit \
  --source "$LP_ROOT/models/qwen3-8b-base" --result-root "$RESULT_ROOT" \
  --code-sha "$CODE_SHA" --slots 3 \
  --manifest "$RESULT_ROOT/progressive_llm/planned_jobs.json" \
  --script hpc/progressive_llm/run_point.sbatch
```

The first two pilots use all seven real first-layer matrices, independently
verify the first and second enhancement, and score a separately materialized
zero-enhancement W2 checkpoint. Six control pilots exercise actual
one-layer quantization with the pinned 128-sample calibration and algorithm.
Each requires all seven selected projections to change and all 245 projections
outside layer 0 to remain source-equivalent at their loaded dtype. A meta or
unavailable outside weight fails the pilot. The support audit records dispatch
separately. If any upstream control mode
fails, exclude it and document that mode. A separately identified common
affine RTN control could be proposed, but it must not be presented as
SignRoundV2 or NeUQI. Pilot scripts need A10G access and were not run in the
local development environment. Review each measured RSS, GPU peak, runtime,
seven-projection coverage and storage report before enabling matrix submission.

Once results exist, aggregate without inventing missing values:

```bash
"$LP_ROOT/envs/lp_llm/bin/python" -m lp_llm.progressive_aggregate \
  --source "$LP_ROOT/models/qwen3-8b-base" --result-root "$RESULT_ROOT" \
  --output "$RESULT_ROOT/progressive_llm/analysis"
```

This produces CSV, JSON and three PNG plots. The established and new points
retain the same official MatGPTQ WikiText-2 protocol: 2048 non-overlapping
tokens, batch one, FP16 inference, 298862 scored tokens, and token SHA256
`8ffcee84dd4b684ec631cb52845e18c9c181ce6fdb8b26d65a4ffdc3d35fbda3`.
No zero-shot downstream benchmark is included.
