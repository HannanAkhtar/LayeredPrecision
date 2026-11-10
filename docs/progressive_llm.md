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

The full-point script now requests 15G host memory and one A10G. AUS
`sbatch --test-only` accepted 15G and rejected 16G, 18G, and 20G.
The original Stage A jobs used 12G and retain that allocation in their
resource records. Each new job writes
`progressive_llm/job_resources/<point>-<job>.json` with stage process peak
RSS, Slurm cgroup peak, allocation, and exact headroom fraction. A full job
must remain within its 15 GiB allocation; 10% headroom is recorded but is no
longer a gate. Stage A must be reviewed before Stage B. The first-layer
pilot's 14G request remains separate.

Storage admission models all six persistent dense controls and six LP
residual stages. At most two temporary LP evaluation checkpoints can coexist
because there are two scientific LP chains; no fourteen-checkpoint reserve
is imposed. Actual filesystem free space and the site quota are separate
checks. The previously authorized `LP_MATRIX_ALLOW_UNKNOWN_QUOTA=1` skips
only formal quota confirmation.

## AUS prerequisite CPU audit

The `hpc` partition has 96 CPUs and 373555 MiB per node on AUS; the `cpu`
partition has only 15564 MiB. The audit therefore requests 4 CPUs, 32 GiB,
and no GPU on `hpc`. It stages the requested Git commit, runs the pinned
AutoRound dispatch audit and full repaired-base/pilot verification, and writes
`progressive_llm/prerequisites/prerequisites_<CODE_SHA>.json` only on success.
It invokes the existing LP interpreter directly; the CPU node needs neither
Conda shell initialization nor the GPU job's site-local `env.sh`. A CPU-only
preflight checks the pinned Torch and Transformers versions and prerequisite
imports before reading checkpoints.
An existing report does not skip the full CPU audit; a successful repeat saves
the previous report under a `superseded.<job_id>.json` name. Failures write a
separate `failure.<job_id>.json` diagnostic.

```bash
cd /shared/makhtar/lp_llm/code
set -euo pipefail
git pull --ff-only origin experiment/rrq-w2-reproduction
CODE_SHA=$(git rev-parse HEAD)
LP_ROOT=/shared/makhtar/lp_llm
RESULT_ROOT="$LP_ROOT/results/qwen3_8b_base"
mkdir -p "$LP_ROOT/logs" "$RESULT_ROOT/progressive_llm/prerequisites"
sinfo -h -p hpc -o '%P %c %m %l'
sbatch --export=ALL,FOLLOWUP_CODE_SHA="$CODE_SHA" \
  hpc/progressive_llm/verify_prerequisites.sbatch
```

Monitor the returned job ID and inspect its report:

```bash
JOB_ID=<returned-job-id>
squeue -j "$JOB_ID" -o '%.18i %.9T %.10M %.12R'
sacct -j "$JOB_ID" --format=JobID,State,ExitCode,ReqMem,MaxRSS,Elapsed
tail -n 40 "$LP_ROOT/logs/lp-llm-prerequisites-$JOB_ID.out"
tail -n 40 "$RESULT_ROOT/progressive_llm/prerequisites/prerequisites_$CODE_SHA.json.log"
"$LP_ROOT/envs/lp_llm/bin/python" -m lp_llm.progressive_prerequisites check \
  --source "$LP_ROOT/models/qwen3-8b-base" --result-root "$RESULT_ROOT" \
  --code-sha "$CODE_SHA" --code-repo "$PWD"
```

Require Slurm `COMPLETED`, exit code `0:0`, and a successful lightweight
report check before Stage A. The report contains verified SHA-256 identities;
the lightweight gate checks current size, timestamps, inode, path, and file
inventory without rereading checkpoint tensors. It records the filesystem
device ID for diagnostics but does not compare that node-local ID across AUS
nodes. These metadata checks catch
ordinary changes but cannot prove that a file owner with write access did not
restore metadata after modifying contents. If file integrity is uncertain,
rerun the full CPU audit before submitting more jobs. Any metadata mismatch
fails closed. Preserve the checkpoint directories as read-only operational
inputs while the matrix runs.

## AUS Stage A: two real matrix points

**Recovery of jobs 55385/55386:** Do not rerun the commands below for those
existing points. The LP `full.json` is scientifically complete at PPL 13.38,
but its 12G Slurm job failed the prior memory rule. The W4 `export.partial`
contains successful quantization and repaired exclusions; it still needs
independent validation and evaluation. After a fresh prerequisite CPU audit
for the recovery commit, submit `recover_stage_a_cpu.sbatch`. It verifies the
historical failed Slurm jobs, producing commit, unchanged numerical code,
source identity, LP chunks and evaluation log, W4 pinned recipe and process,
all 252 projections, source-exact exclusions, and projection hashes. It writes
`progressive_llm/recovery/<CODE_SHA>/stage_a_scientific_audit.json` without
moving or rewriting the original export.

Only after that report passes, `recover_w4_evaluate.sbatch` can evaluate the
original partial export under a new 15G job. It stores results in a separate
recovery directory. The old job IDs and 12G allocation remain recorded as
failed. `progressive_stage_a_recovery verify` reports the W4 recovery job's
actual memory use and Slurm status. Stage B remains blocked by the original
LP job's failed resource certification; scientific acceptance does not turn
that failed job into a successful 15G run. No new LP encoding or AutoRound
quantization is performed by these recovery scripts.

After the new SHA's CPU prerequisite report passes `progressive_prerequisites
check`, submit the read-only recovery audit and inspect its Slurm result:

```bash
sbatch --export=ALL,FOLLOWUP_CODE_SHA="$CODE_SHA" \
  hpc/progressive_llm/recover_stage_a_cpu.sbatch
sacct -j <recovery-cpu-job-id> --format=JobID,State,ExitCode,ReqMem,MaxRSS
RECOVERY="$RESULT_ROOT/progressive_llm/recovery/$CODE_SHA/stage_a_scientific_audit.json"
test -s "$RECOVERY"
```

Require `COMPLETED` and `0:0`. Then submit only the W4 evaluation job, which
does not launch AutoRound and does not move `export.partial`:

```bash
sbatch --test-only --export=ALL,FOLLOWUP_CODE_SHA="$CODE_SHA" \
  hpc/progressive_llm/recover_w4_evaluate.sbatch
sbatch --export=ALL,FOLLOWUP_CODE_SHA="$CODE_SHA" \
  hpc/progressive_llm/recover_w4_evaluate.sbatch
sacct -j <recovery-gpu-job-id> --format=JobID,State,ExitCode,ReqMem,MaxRSS
"$LP_ROOT/envs/lp_llm/bin/python" -m lp_llm.progressive_stage_a_recovery verify \
  --source "$LP_ROOT/models/qwen3-8b-base" --result-root "$RESULT_ROOT" \
  --code-sha "$CODE_SHA" --code-repo "$PWD" \
  --autoround "$LP_ROOT/references/rrq_pr2308/auto-round" --output "$RECOVERY"
```

The final verification explicitly reports that Stage B remains blocked by
the historical LP job's failed resource gate. It cannot convert that Slurm
job into a completed job or claim the original computation ran with 15G.

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
unset LP_MATRIX_SHARED_QUOTA_FREE_GIB
export LP_MATRIX_ALLOW_UNKNOWN_QUOTA=1
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
root = Path('/shared/makhtar/lp_llm/results/qwen3_8b_base')
manifest = json.loads((root / 'progressive_llm/matrix_jobs.json').read_text())
jobs = manifest['job_ids']
source = json.loads((root / ('progressive_llm/prerequisites/prerequisites_' +
                          manifest['code_sha'] + '.json')).read_text())['source_identity']
for point in ('s01_w4', 's01_lp_2p2'):
    resource = json.loads((root / f'progressive_llm/job_resources/{point}-{jobs[point]}.json').read_text())
    assert resource['host_memory_within_allocation'], (point, resource)
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
same command later to submit the remaining missing points. The planner also
checks both Stage A Slurm exit codes, `full.json` identities, coverage,
perplexity, corpus hash, storage/distortion fields, and cgroup/RSS headroom
before it submits any of the other ten points.

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
