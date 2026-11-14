# Experiment 2: high-precision grid recovery diagnosis

The pinned AutoRound checkout is `7985c1b0f4eef65fb717511668fda7e347e584c3`.
In its `auto_round/data_type/int.py`, SignRoundV2 uses signed codes
`[-2^(b-1), 2^(b-1)-1]` and `(scale * q).to(tensor.dtype)`.
In `auto_round/data_type/neuqi.py`, optimized asymmetric RTN casts scale and
zero point to FP16, clamps `q` to `[0, 2^b-1]`, and computes
`(scale * (q-zp)).to(tensor.dtype)`. Both workers request FP16 model loading,
group size 128, and FP16 scales. The fake-QDQ export may store BF16 tensors.

The prior audit's FP32-scale alternative multiplied in FP32 and cast directly
to the exported dtype. This can differ from FP16 QDQ followed by BF16 export.
The diagnostic records both operation orders. The corrected R02 matcher also
checks the full feasible integer-zero-point interval instead of accepting the
first effective-code match, which can be wrong when several codes round to
one stored value. Its interval-guided candidates
come from FP16 rounding boundaries. They are a research heuristic, not a
complete solver or a proof of impossibility. It retains the old 256 search,
then tests 1024 and 4096 on only two original misses and one original success
from each W4/W6/W8 model.

```bash
export LP_ROOT=/shared/makhtar/lp_llm
export RESULT_ROOT=$LP_ROOT/results/qwen3_8b_base
cd "$LP_ROOT/code"
git pull --ff-only origin experiment/rrq-w2-reproduction
export FOLLOWUP_CODE_SHA=$(git rev-parse HEAD)
export STORAGE_SAMPLE_SHA=5b8c3dee6e612027e98f3f0768a47ae6b5be6370
test -s "$RESULT_ROOT/progressive_llm/storage_feasibility/$STORAGE_SAMPLE_SHA/sample-all.json"
JOB=$(sbatch --parsable --export=ALL,FOLLOWUP_CODE_SHA="$FOLLOWUP_CODE_SHA",STORAGE_SAMPLE_SHA="$STORAGE_SAMPLE_SHA" \
  hpc/progressive_llm/storage_diagnose.sbatch)
echo "$JOB"
sacct -j "$JOB" --format=JobID,State,ExitCode,ReqMem,MaxRSS
```

Inspect the immutable new output without reading all checkpoint tensors:

```bash
OUT=$RESULT_ROOT/progressive_llm/storage_feasibility/$FOLLOWUP_CODE_SHA/diagnose.json
"$LP_ROOT/envs/lp_llm/bin/python" - <<'PY'
import json, os
from pathlib import Path
p=Path(os.environ['RESULT_ROOT'])/'progressive_llm/storage_feasibility'/os.environ['FOLLOWUP_CODE_SHA']/'diagnose.json'
d=json.loads(p.read_text())
print('oracle:', len(d['oracle']), 'known-grid examples')
print('oracle pinned exact:', sum(x['pinned']=='exact_constructive' for x in d['oracle']))
for limit in (256,1024,4096):
 total=exact=0
 for rows in d['models'].values():
  for row in rows:
   total+=1
   exact+=row['checks']['fp16'][f'pinned_fp16_qdq_{limit}']['outcome']=='exact_constructive'
 print('FP16-scale pinned search', limit, exact, '/', total)
for model, rows in d['models'].items():
 for row in rows:
  old=row['original_outcomes']['fp16']['outcome']
  new=row['checks']['fp16']['pinned_fp16_qdq_4096']['outcome']
  direct=row['checks']['fp16']['legacy_4096']['outcome']
  if old!='exact_constructive' or direct!=new:
   print(model, row['coordinate']['tensor'], row['coordinate']['row'],
         row['coordinate']['group'], 'old=',old, 'direct=',direct,
         'pinned=',new, 'seconds=',
         row['checks']['fp16']['pinned_fp16_qdq_4096']['cpu_seconds'])
PY
"$LP_ROOT/envs/lp_llm/bin/python" -m json.tool \
  "$RESULT_ROOT/progressive_llm/storage_feasibility/$FOLLOWUP_CODE_SHA/diagnose-resources-$JOB.json"
```

`diagnose.json` includes all search limits, decoder variants, CPU times,
constructive matches, and unresolved cases. No full-model packing or audit
is launched by this script.
