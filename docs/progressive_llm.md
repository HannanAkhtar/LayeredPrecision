# Qwen3-8B progressive LP matrix

The twelve points are defined in `configs/progressive_llm/matrix.json`: three
LP prefixes and W4/W6/W8 independent controls for each S01 and R02 base.
Controls use pinned AutoRound `7985c1b0f4eef65fb717511668fda7e347e584c3`
through `quantize_and_save()`. Full export validation checks 252 changed
projections, source-exact exclusions, finite weights, group grids, model
inventory, and checkpoint hashes. The checkpoint is then evaluated with the
established MatGPTQ WikiText-2 protocol. Distortion and actual dense-QDQ
checkpoint bytes are recorded. Compact rates are modeled, not measured.

The two historical LP pilots, repaired W2 bases, their source identities,
and 12.47/32.22 perplexity parity remain required. A pinned control-dispatch
audit is also required. The six failed first-layer control pilots are not a
full-model gate and their artifacts remain untouched.

## Scheduling and capacity

The planner counts all active jobs under account `acc-makhtar` before each
`sbatch` and admits at most 14. An interrupted submission leaves an intent
file and fails closed on resume until the operator reconciles it with Slurm.
Successful submissions are recorded immediately in the manifest. The two
LP chains use `afterok` for their scientific predecessors. Independent
controls have no scheduling dependencies; Slurm chooses available GPUs.

The full-point script retains 12G host memory and one A10G. This has not
been proven sufficient for every full-model control. Each job writes
`progressive_llm/job_resources/<point>-<job>.json` with stage process peak
RSS, Slurm cgroup peak, allocation, and 10% headroom status. Stage A must be
reviewed before Stage B. The first-layer pilot's 14G request does not change
full-point resources. A 12G job that exceeds 10.8 GiB peak fails its resource
gate; investigate before submitting more work.

Storage admission models all six persistent dense controls and six LP
residual stages. At most two temporary LP evaluation checkpoints can coexist
because there are two scientific LP chains; no fourteen-checkpoint reserve
is imposed. Actual filesystem free space and the site quota are separate
checks. The previously authorized `LP_MATRIX_ALLOW_UNKNOWN_QUOTA=1` skips
only formal quota confirmation.

## AUS Stage A: two real matrix points

Run these commands from the trusted Git checkout. They submit only `s01_w4`
and `s01_lp_2p2`; both results count toward the twelve-point matrix.

```bash
cd /shared/makhtar/lp_llm/code
set -euo pipefail
git pull --ff-only origin experiment/rrq-w2-reproduction
CODE_SHA=$(git rev-parse HEAD)
LP_ROOT=/shared/makhtar/lp_llm
RESULT_ROOT="$LP_ROOT/results/qwen3_8b_base"
MANIFEST="$RESULT_ROOT/progressive_llm/matrix_jobs.json"
mkdir -p "$LP_ROOT/logs" "$RESULT_ROOT/progressive_llm/pilots"
unset LP_MATRIX_SHARED_QUOTA_FREE_GIB
export LP_MATRIX_ALLOW_UNKNOWN_QUOTA=1
PYTHONPATH="$LP_ROOT/references/rrq_pr2308/auto-round:$PWD" \
  "$LP_ROOT/envs/lp_llm/bin/python" -m lp_llm.progressive_support \
  --checkout "$LP_ROOT/references/rrq_pr2308/auto-round" \
  --output "$RESULT_ROOT/progressive_llm/pilots/control_support_$CODE_SHA.json" \
  --code-sha "$CODE_SHA"
"$LP_ROOT/envs/lp_llm/bin/python" -m lp_llm.progressive_plan verify-prerequisites \
  --source "$LP_ROOT/models/qwen3-8b-base" --result-root "$RESULT_ROOT" \
  --code-sha "$CODE_SHA" --code-repo "$PWD"
"$LP_ROOT/envs/lp_llm/bin/python" -m lp_llm.progressive_plan dry-run \
  --result-root "$RESULT_ROOT" --code-sha "$CODE_SHA" --slots 14
export LP_MATRIX_SUBMIT=1 LP_MATRIX_PILOTS_REVIEWED=1
"$LP_ROOT/envs/lp_llm/bin/python" -m lp_llm.progressive_plan submit \
  --source "$LP_ROOT/models/qwen3-8b-base" --result-root "$RESULT_ROOT" \
  --code-sha "$CODE_SHA" --code-repo "$PWD" --slots 14 \
  --manifest "$MANIFEST" --script hpc/progressive_llm/run_point.sbatch \
  --only s01_w4 s01_lp_2p2
```

After both jobs finish, check Slurm state and validate the actual artifacts:

```bash
"$LP_ROOT/envs/lp_llm/bin/python" - <<'PY'
import json, math
from pathlib import Path
from lp_llm.progressive_lp import _files
root = Path('/shared/makhtar/lp_llm/results/qwen3_8b_base')
manifest = json.loads((root / 'progressive_llm/matrix_jobs.json').read_text())
jobs = manifest['job_ids']
source = _files(Path('/shared/makhtar/lp_llm/models/qwen3-8b-base'))
for point in ('s01_w4', 's01_lp_2p2'):
    resource = json.loads((root / f'progressive_llm/job_resources/{point}-{jobs[point]}.json').read_text())
    assert resource['host_memory_10pct_headroom_passed'], (point, resource)
    assert {'encode', 'evaluate'} <= set(resource['stage_process_peak_rss_kib'])
    path = (root / f'progressive_llm/controls/{point}/full.json' if point == 's01_w4'
            else root / f'progressive_llm/evaluation/{point}/full.json')
    result = json.loads(path.read_text())
    assert result['status'] == 'completed_validated'
    assert result['identity']['source'] == source
    assert math.isfinite(result['evaluation']['perplexity'])
    assert result['corpus']['token_sha256'] == '8ffcee84dd4b684ec631cb52845e18c9c181ce6fdb8b26d65a4ffdc3d35fbda3'
    if point == 's01_w4':
        assert result['identity']['code_sha'] == manifest['code_sha']
        marker = json.loads((path.parent / 'encoding_complete.json').read_text())
        assert marker['status'] == 'completed_validated'
        assert marker['checkpoint_validation'] == result['checkpoint_validation']
        assert (path.parent / 'excluded_repair.json').is_file()
        check = result['checkpoint_validation']
        assert check['quantized_projection_count'] == 252
        assert check['source_exact_excluded_tensor_count'] > 0
        assert check['sampled_group_grid_checks'] > 0
        assert check['actual_checkpoint_file_bytes'] > 0
        assert check['compact_encoded_bytes'] is None
        assert math.isfinite(check['distortion']['mse_fp32'])
    else:
        assert result['code_sha'] == manifest['code_sha']
        assert result['materialization']['projection_count'] == 252
        assert result['storage']
    print(point, jobs[point], result['evaluation']['perplexity'])
PY
```

Run `sacct -j <first-job>,<second-job> --format=JobID,State,ExitCode,ReqMem,MaxRSS`
and require `COMPLETED` and `0:0` for each. Inspect the encoding marker,
repair report, evaluation log, source identity and resource record. If a
point fails, preserve its diagnostics and allow the independent point to
finish. Do not submit Stage B based on Slurm exit status alone.

## AUS Stage B: remaining ten points

After Stage A artifacts pass review, resume the same manifest. The two
recorded jobs are reused. If account capacity is insufficient, rerun this
same command later to submit the remaining missing points.

```bash
cd /shared/makhtar/lp_llm/code
set -euo pipefail
CODE_SHA=$(git rev-parse HEAD)
LP_ROOT=/shared/makhtar/lp_llm
RESULT_ROOT="$LP_ROOT/results/qwen3_8b_base"
unset LP_MATRIX_SHARED_QUOTA_FREE_GIB
export LP_MATRIX_ALLOW_UNKNOWN_QUOTA=1 LP_MATRIX_SUBMIT=1 LP_MATRIX_PILOTS_REVIEWED=1
"$LP_ROOT/envs/lp_llm/bin/python" -m lp_llm.progressive_plan submit \
  --source "$LP_ROOT/models/qwen3-8b-base" --result-root "$RESULT_ROOT" \
  --code-sha "$CODE_SHA" --code-repo "$PWD" --slots 14 \
  --manifest "$RESULT_ROOT/progressive_llm/matrix_jobs.json" \
  --script hpc/progressive_llm/run_point.sbatch
```

Review all twelve `full.json` results and resource records before running
`python -m lp_llm.progressive_aggregate`. No job is submitted by this
documentation or the dry-run action alone.
