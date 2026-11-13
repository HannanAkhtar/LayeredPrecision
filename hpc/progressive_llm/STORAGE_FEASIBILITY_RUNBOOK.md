# Read-only fake-QDQ grid feasibility audit

No command below runs quantization or WikiText-2 evaluation. The CPU audit
reads the completed S01/R02 checkpoints and writes new reports only.

```bash
export LP_ROOT=/shared/makhtar/lp_llm
export RESULT_ROOT=$LP_ROOT/results/qwen3_8b_base
cd "$LP_ROOT/code"
git pull --ff-only origin experiment/rrq-w2-reproduction
export FOLLOWUP_CODE_SHA=$(git rev-parse HEAD)
# Compare with the reviewed SHA supplied with the patch before submitting.
git cat-file -e "$FOLLOWUP_CODE_SHA^{commit}"
```

Metadata-only audit, directly on an HPC CPU node (or with `srun -p hpc`):

```bash
OUT=$RESULT_ROOT/progressive_llm/storage_feasibility/$FOLLOWUP_CODE_SHA
mkdir -p "$OUT"
"$LP_ROOT/envs/lp_llm/bin/python" -m lp_llm.progressive_storage_feasibility \
  --source "$LP_ROOT/models/qwen3-8b-base" --result-root "$RESULT_ROOT" \
  --mode metadata --code-sha "$FOLLOWUP_CODE_SHA" --output "$OUT/metadata-all.json"
```

The sampled audit runs all eight checkpoints in one CPU-only Slurm job. It
creates one report and one JSONL group file per checkpoint. A failed or partial
output is preserved for inspection; do not overwrite it.

```bash
JOB=$(sbatch --parsable --export=ALL,FOLLOWUP_CODE_SHA="$FOLLOWUP_CODE_SHA",STORAGE_MODE=sample \
  hpc/progressive_llm/storage_feasibility.sbatch)
echo "$JOB"
sacct -j "$JOB" --format=JobID,State,ExitCode,ReqMem,MaxRSS
tail -n 80 "$LP_ROOT/logs/lp-storage-audit-$JOB.out"
```

Inspect sampled exactness, storage components, and examples:

```bash
OUT=$RESULT_ROOT/progressive_llm/storage_feasibility/$FOLLOWUP_CODE_SHA
"$LP_ROOT/envs/lp_llm/bin/python" - <<'PY'
import json, os
p=os.path.join(os.environ['RESULT_ROOT'],'progressive_llm/storage_feasibility',
               os.environ['FOLLOWUP_CODE_SHA'],'sample-all.json')
for name, model in json.load(open(p))['models'].items():
 print(name, 'groups=', model['group_count'], 'counts=', model['counts'],
       'codes=', model['storage']['code_bytes'],
       'excluded=', model['storage']['excluded_tensor_payload_bytes'])
PY
head -n 2 "$OUT/sample-all.s01_w4.groups.jsonl" | "$LP_ROOT/envs/lp_llm/bin/python" -m json.tool --json-lines
grep 'not_found_under_search\|not_representable_proven' "$OUT/sample-all.s01_w4.groups.jsonl" | head -n 2
"$LP_ROOT/envs/lp_llm/bin/python" -m json.tool "$OUT/resources-sample-all-$JOB.json"
```

Review sample results before any full audit. A sampled exact construction is
evidence for that group only. A bounded search miss is not impossibility.
Optional full audit of one model, never submitted automatically:

```bash
MODEL=s01_w2
JOB=$(sbatch --parsable --export=ALL,FOLLOWUP_CODE_SHA="$FOLLOWUP_CODE_SHA",STORAGE_MODE=full,STORAGE_MODEL="$MODEL" \
  hpc/progressive_llm/storage_feasibility.sbatch)
echo "$JOB"
```

`metadata-all.json`, `sample-all.json`, and `full-MODEL.json` are separate.
Their `.md` summaries and `.figures/` SVG diagnostics are also new outputs.
The JSONL files hold group-level evidence. Storage totals are analytical
payload estimates excluding future serialized framing, not measured packed
checkpoints. The existing fake-QDQ file bytes are measured independently.
