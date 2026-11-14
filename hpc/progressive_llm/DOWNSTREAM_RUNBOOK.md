# Qwen3-8B-Base downstream zero-shot evaluation

The harness is pinned to `d021bf846218f1bb3bdc0603864789329476d464`
(`lm_eval` 0.4.9.1). Its task YAMLs select ARC-Easy/Challenge test splits,
and HellaSwag, PIQA, WinoGrande validation splits. Raw `acc` is the primary
metric; `acc_norm` is retained separately where available. Pilot runs use
eight examples on ARC-Easy and ARC-Challenge and are never full results.

## Prepare the isolated environment

```bash
export LP_ROOT=/shared/makhtar/lp_llm
export RESULT_ROOT=$LP_ROOT/results/qwen3_8b_base
cd "$LP_ROOT/code"
git pull --ff-only origin experiment/rrq-w2-reproduction
export FOLLOWUP_CODE_SHA=$(git rev-parse HEAD)
mkdir -p "$LP_ROOT/references/downstream"
if [[ ! -d "$LP_ROOT/references/downstream/lm-evaluation-harness/.git" ]]; then
  git clone https://github.com/EleutherAI/lm-evaluation-harness.git \
    "$LP_ROOT/references/downstream/lm-evaluation-harness"
  git -C "$LP_ROOT/references/downstream/lm-evaluation-harness" checkout --detach \
    d021bf846218f1bb3bdc0603864789329476d464
fi
test "$(git -C "$LP_ROOT/references/downstream/lm-evaluation-harness" rev-parse HEAD)" = \
  d021bf846218f1bb3bdc0603864789329476d464
bash hpc/progressive_llm/setup_downstream_env.sh
```

The setup script leaves `envs/lp_llm` unchanged. Review
its `[DOWNSTREAM ENV READY]` output before preflight.

## Pre-cache and verify on CPU

```bash
PREFLIGHT=$(sbatch --parsable --export=ALL,FOLLOWUP_CODE_SHA="$FOLLOWUP_CODE_SHA" \
  hpc/progressive_llm/downstream_preflight.sbatch)
echo "$PREFLIGHT"
sacct -j "$PREFLIGHT" --format=JobID,State,ExitCode,ReqMem,MaxRSS
ROOT=$RESULT_ROOT/progressive_llm/downstream_zero_shot/$FOLLOWUP_CODE_SHA
"$LP_ROOT/envs/downstream/bin/python" -m json.tool "$ROOT/manifest.json" | head -n 100
"$LP_ROOT/envs/downstream/bin/python" -m json.tool "$ROOT/preflight_complete.json" | head -n 60
"$LP_ROOT/envs/downstream/bin/python" -m json.tool "$ROOT/job_resources/preflight-$PREFLIGHT.json" | head -n 80
```

The preflight writes the pinned environment versions and dataset fingerprints,
then seals all 15 verified model identities. An incomplete preflight can be
rerun; the seal is absent until all checks pass. The 48 GiB CPU allocation
assumes sequential checkpoint verification with source/base identities cached
only within this one process, followed by an independent rehash. NFS page
cache may count against the Slurm cgroup; inspect the resource report and
increase or split the CPU audit if it exceeds 48 GiB. Require exit `0:0`
and inspect the manifest, seal and resource report before submitting pilots.

## Three concurrent pilots

```bash
PILOTS=$(sbatch --parsable --array=0-2%3 \
  --export=ALL,FOLLOWUP_CODE_SHA="$FOLLOWUP_CODE_SHA",DOWNSTREAM_PILOT=1 \
  hpc/progressive_llm/downstream_model.sbatch)
echo "$PILOTS"
sacct -j "$PILOTS" --format=JobID,State,ExitCode,ReqMem,MaxRSS
for MODEL in original_fp16 s01_lp6 r02_w6; do
  "$LP_ROOT/envs/downstream/bin/python" -m json.tool "$ROOT/pilot/$MODEL/full.json" | head -n 80
done
```

Review scientific results and each `$ROOT/job_resources/*-1-*.json`. The LP
pilot needs at least 18 GiB free node-local scratch for a temporary checkpoint.
Do not submit the full array until all three pilots pass and resource usage is
within the 15 GiB allocation.

## Fifteen independent full jobs

Choose `N` from measured pilot resource use, available GPUs and site policy.
Jobs have no inter-model dependency; Slurm may queue pending array elements.

```bash
N=3
FULL=$(sbatch --parsable --array="0-14%$N" \
  --export=ALL,FOLLOWUP_CODE_SHA="$FOLLOWUP_CODE_SHA",DOWNSTREAM_PILOT=0 \
  hpc/progressive_llm/downstream_model.sbatch)
echo "$FULL"
sacct -j "$FULL" --format=JobID,State,ExitCode,ReqMem,MaxRSS
```

Each model has an independent lock and output directory. Completed reports
are idempotent; incomplete directories require inspection before retry.

## Aggregate

After all scientific and resource reports are reviewed:

```bash
"$LP_ROOT/envs/downstream/bin/python" -m lp_llm.progressive_downstream aggregate \
  --source "$LP_ROOT/models/qwen3-8b-base" --result-root "$RESULT_ROOT" \
  --output-root "$ROOT" \
  --harness "$LP_ROOT/references/downstream/lm-evaluation-harness" \
  --code-sha "$FOLLOWUP_CODE_SHA"
cat "$ROOT/aggregate.md"
```

Missing and invalid models remain explicit. Aggregate output is new and
cannot overwrite an existing report. WikiText-2 perplexity is read from
completed reports; it is not rerun.
