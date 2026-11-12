# S01/R02 scaled signed LP matrix on AUS

This runbook submits no jobs by itself. Run one stage at a time and inspect its
Slurm and scientific reports before advancing. The scaled E2 policy is signed
G128 RTN with α=0.50 at every depth; it is a fixed experiment, not a claim of
optimality for R02 or deeper layers. The historical S01 signed E1 remains at
`progressive_llm/ablations/s01_autoround_signed_rtn/layer1` with its original
producer commit and failed encoding-job resource status.

## 0. Deploy and inventory from the login node

```bash
cd /shared/makhtar/lp_llm/code
set -euo pipefail
git pull --ff-only origin experiment/rrq-w2-reproduction
CODE_SHA=$(git rev-parse HEAD)
test "$CODE_SHA" = "${FOLLOWUP_CODE_SHA:?Set the reviewed 40-character commit SHA}"
LP_ROOT=/shared/makhtar/lp_llm
RESULT_ROOT=$LP_ROOT/results/qwen3_8b_base
SOURCE=$LP_ROOT/models/qwen3-8b-base
INVENTORY=$RESULT_ROOT/progressive_llm/inventory/$CODE_SHA.json
"$LP_ROOT/envs/lp_llm/bin/python" -m lp_llm.progressive_matrix_inventory \
  --source "$SOURCE" --result-root "$RESULT_ROOT" --output "$INVENTORY"
"$LP_ROOT/envs/lp_llm/bin/python" -m json.tool "$INVENTORY" | less
```

The inventory is fast and read-only. Check `filesystem_free_bytes`, the
explicitly unknown `quota_status`, missing indexed shards, producer commits,
quantization process records, completion markers, and evaluation statuses.
`recoverable_candidate` is **not** validation. Do not launch a control
quantizer against an attempted or partial result.

## 1. First S01 LP4 full evaluation

Require `s01_w2` to have a completed evaluation report, and `s01_signed_e1`
to have a complete layer marker with producer SHA
`78b67800d786ba8bf03534818ae77137066c6ad8`. The encoding job's failed
resource gate remains historical evidence. The evaluation job verifies all
252 signed E1 chunks against source and base identities, creates a separate
reference manifest, and never re-encodes E1.

```bash
JOB=$(sbatch --parsable --export=ALL,FOLLOWUP_CODE_SHA="$CODE_SHA",MATRIX_BRANCH=s01,MATRIX_DEPTH=1 \
  hpc/progressive_llm/scaled_evaluate.sbatch)
echo "S01 LP4 job: $JOB"
sacct -j "$JOB" --format=JobID,State,ExitCode,ReqMem,MaxRSS
```

After `COMPLETED, 0:0`, inspect:

```bash
ROOT=$RESULT_ROOT/progressive_llm/scaled_signed_v1/s01
"$LP_ROOT/envs/lp_llm/bin/python" -m json.tool "$ROOT/layer1_reference.json" | head -n 35
"$LP_ROOT/envs/lp_llm/bin/python" -m json.tool "$ROOT/evaluation/depth1/full.json" | head -n 65
tail -n 20 "$ROOT/evaluation/depth1/evaluation.log"
"$LP_ROOT/envs/lp_llm/bin/python" -m json.tool \
  "$RESULT_ROOT/progressive_llm/job_resources/scaled-s01-d1-$JOB.json" | head -n 50
```

Require `completed_validated`, 252 projections, the pinned test token hash
`8ffcee84dd4b684ec631cb52845e18c9c181ce6fdb8b26d65a4ffdc3d35fbda3`,
298862 scored tokens, finite PPL, and memory within the 15 GiB allocation.
Preserve a scientifically complete result even if its later resource check
fails; report that status separately. **Stop for review before S01 LP6.**

## 2. Remaining scaled LP stages

For each row, submit encoding on the 48 GiB `hpc` partition. Check its layer
marker, projection count, source/base identity, MSE, and resource report.
Then submit evaluation on one 15 GiB GPU and inspect `full.json`, the MatGPTQ
log, corpus hash, and Slurm status. Do not submit the next row before the
current row has been reviewed.

| Branch | Depth | Nominal bits | Layer marker | Evaluation report |
|---|---:|---:|---|---|
| s01 | 2 | 6 | `s01/layer2/encoding_complete.json` | `s01/evaluation/depth2/full.json` |
| s01 | 3 | 8 | `s01/layer3/encoding_complete.json` | `s01/evaluation/depth3/full.json` |
| r02 | 1 | 4 | `r02/layer1/encoding_complete.json` | `r02/evaluation/depth1/full.json` |
| r02 | 2 | 6 | `r02/layer2/encoding_complete.json` | `r02/evaluation/depth2/full.json` |
| r02 | 3 | 8 | `r02/layer3/encoding_complete.json` | `r02/evaluation/depth3/full.json` |

Example for **one** row, after setting `BRANCH` and `DEPTH` from the table:

```bash
BRANCH=s01 DEPTH=2
ENCODE_JOB=$(sbatch --parsable --export=ALL,FOLLOWUP_CODE_SHA="$CODE_SHA",MATRIX_BRANCH="$BRANCH",MATRIX_DEPTH="$DEPTH" \
  hpc/progressive_llm/scaled_encode.sbatch)
echo "encoding: $ENCODE_JOB"
sacct -j "$ENCODE_JOB" --format=JobID,State,ExitCode,ReqMem,MaxRSS
# Check layer$DEPTH/encoding_complete.json and job_resources/scaled-$BRANCH-e$DEPTH-$ENCODE_JOB.json.
EVAL_JOB=$(sbatch --parsable --export=ALL,FOLLOWUP_CODE_SHA="$CODE_SHA",MATRIX_BRANCH="$BRANCH",MATRIX_DEPTH="$DEPTH" \
  hpc/progressive_llm/scaled_evaluate.sbatch)
echo "evaluation: $EVAL_JOB"
sacct -j "$EVAL_JOB" --format=JobID,State,ExitCode,ReqMem,MaxRSS
# Check evaluation/depth$DEPTH/full.json, evaluation.log, and job_resources/scaled-$BRANCH-d$DEPTH-$EVAL_JOB.json.
```

## 3. Monolithic controls

Use the inventory first. For a **missing, never-attempted** W4/W6/W8 control,
submit `encode`; the script refuses any prior partial or process record. Review
its `encoding_complete.json`, repair report, 252 projection hashes, resolved
recipe, and resource record. Then submit `evaluate`; that action refuses a
missing completion marker. The existing `progressive_control` recipes retain
the pinned S01 SignRoundV2 and R02 OptimizedRTN/NeUQI settings.

```bash
BRANCH=r02 BITS=4
ENCODE_JOB=$(sbatch --parsable --export=ALL,FOLLOWUP_CODE_SHA="$CODE_SHA",MATRIX_ACTION=encode,MATRIX_BRANCH="$BRANCH",MATRIX_BITS="$BITS" \
  hpc/progressive_llm/control_matrix_action.sbatch)
echo "control encoding: $ENCODE_JOB"
# After independent review of the validated encoding:
EVAL_JOB=$(sbatch --parsable --export=ALL,FOLLOWUP_CODE_SHA="$CODE_SHA",MATRIX_ACTION=evaluate,MATRIX_BRANCH="$BRANCH",MATRIX_BITS="$BITS" \
  hpc/progressive_llm/control_matrix_action.sbatch)
echo "control evaluation: $EVAL_JOB"
```

For the historical **S01 W4 partial export only**, first inspect the inventory.
If its process and repair evidence appear present, submit the 48 GiB CPU audit.
It performs all tensor and group checks without moving or changing the export.
Review `audit.json`, its SHA-256 seal, and the CPU resource report. Only then
submit the GPU evaluation, which hashes the audited files, builds hardlinks
to verified weight shards plus source tokenizer copies, and evaluates the
original partial checkpoint through that view.

```bash
AUDIT_JOB=$(sbatch --parsable --export=ALL,FOLLOWUP_CODE_SHA="$CODE_SHA" \
  hpc/progressive_llm/control_recover_audit.sbatch)
echo "W4 recovery audit: $AUDIT_JOB"
sacct -j "$AUDIT_JOB" --format=JobID,State,ExitCode,ReqMem,MaxRSS
# Require completed_validated in scaled_signed_v1/control_recovery/s01_w4/audit.json.
W4_JOB=$(sbatch --parsable --export=ALL,FOLLOWUP_CODE_SHA="$CODE_SHA",MATRIX_ACTION=recover_evaluate,MATRIX_BRANCH=s01,MATRIX_BITS=4 \
  hpc/progressive_llm/control_matrix_action.sbatch)
echo "W4 recovery evaluation: $W4_JOB"
```

An incomplete or mismatched partial must be preserved for inspection. No
recovery action silently invokes AutoRound. Keep independent heavy jobs to
one or two at a time until resource reports have been reviewed.

## 4. Aggregate reported matrix results

Run only after reviewing all scientific and Slurm reports. Missing rows stay
explicitly missing; fake-QDQ checkpoint bytes are not packed control bytes.

```bash
OUT=$RESULT_ROOT/progressive_llm/scaled_signed_v1/matrix_$CODE_SHA.json
"$LP_ROOT/envs/lp_llm/bin/python" -m lp_llm.progressive_scaled_aggregate \
  --result-root "$RESULT_ROOT" --output "$OUT"
"$LP_ROOT/envs/lp_llm/bin/python" -m json.tool "$OUT" | less
```
