# Qwen3-8B W2 quantizer follow-ups

This adds S01 SignRoundV2, R01 symmetric NeUQI optimized RTN, R02 asymmetric NeUQI optimized RTN, and R03 GPTQModel GPTQ. No new 8B measurements are claimed. Existing J01–J14 and PR03/PR04 checkpoints and result directories are untouched.

The tiny R03 GPU smoke passed on AUS (job 53793). Capacity job 53833 rejected the old nonstreaming workflow. The full R03 path now streams dense QDQ tensors and uses job-owned shared offloading, but remains gated by a fresh read-only preflight and separate largest-layer GPU probe. See [R03 feasibility](r03_feasibility.md) before any full submission.

R01/R02 use the same pinned AutoRound PR source as PR04. In this source, asymmetric zero-iteration optimized RTN requires `enable_neuqi=True`; otherwise the dispatcher runs plain asymmetric min/max RTN. Thus R01 versus PR04 tests NeUQI on the same revision; R02 versus R01 tests symmetry at fixed NeUQI settings. The [numerical audit](numerical_audit.md) records the older J05 revision and remaining confounders.

## Prepare AUS checkouts

Run on a login node; these commands do not submit jobs:

```bash
cd /shared/makhtar/lp_llm/code
git fetch origin experiment/rrq-w2-reproduction
git switch experiment/rrq-w2-reproduction
git pull --ff-only origin experiment/rrq-w2-reproduction
test "$(git -C /shared/makhtar/lp_llm/references/rrq_pr2308/auto-round rev-parse HEAD)" = 7985c1b0f4eef65fb717511668fda7e347e584c3
test "$(git -C /shared/makhtar/lp_llm/references/rrq_w2/MatGPTQ rev-parse HEAD)" = 0e5bd93bc53381f7195db870f5408ae04906bb3c
mkdir -p /shared/makhtar/lp_llm/references/quantizer_followups
git -C /shared/makhtar/lp_llm/references/quantizer_followups/GPTQModel fetch origin
git -C /shared/makhtar/lp_llm/references/quantizer_followups/GPTQModel checkout --detach ed96f2e04ec868b04f969733a242ce3b5bb7cb1e
bash hpc/quantizer_followups/setup_gptq_env.sh
```

The pinned GPTQModel v5.7.0 revision declares Torch >=2.8.0 and Transformers >=4.57.1, compatible with the validated LP versions (Torch 2.9.1+cu128, Transformers 4.57.1). Setup now defaults to a small `venv --system-site-packages` at `/shared/makhtar/lp_llm/envs/gptq_followups_v570_light`; Torch and Transformers must resolve inside the unchanged LP environment. Only missing supplemental requirements and GPTQModel are installed into the new venv with `--no-deps`. Import paths, dependency versions, the W2G128 API, and an environment completion marker are checked before smoke. An existing directory without a valid marker is rejected; inspect it rather than rerunning setup into it. The earlier full copy remains available with `GPTQ_ENV_MODE=clone bash hpc/quantizer_followups/setup_gptq_env.sh` and `GPTQ_ENV_MODE=clone` on the R03 smoke job. The original LP environment runs MatGPTQ evaluation. The AutoRound `RRQ_PR_OVERLAY`, if needed, is validated and applied only to its subprocess.

On AUS, after pulling this patch, run:

```bash
cd /shared/makhtar/lp_llm/code
git pull --ff-only origin experiment/rrq-w2-reproduction
bash hpc/quantizer_followups/setup_gptq_env.sh
PYTHONPATH=/shared/makhtar/lp_llm/references/quantizer_followups/GPTQModel \
  /shared/makhtar/lp_llm/envs/gptq_followups_v570_light/bin/python \
  hpc/quantizer_followups/gptq_env_check.py verify \
  --env /shared/makhtar/lp_llm/envs/gptq_followups_v570_light \
  --lp-env /shared/makhtar/lp_llm/envs/lp_llm \
  --source /shared/makhtar/lp_llm/references/quantizer_followups/GPTQModel \
  --pins hpc/quantizer_followups/gptq_v570_extras.txt --mode light
sbatch --export=ALL,EXPERIMENT_ID=r03,FOLLOWUP_SMOKE=1 hpc/quantizer_followups/run_experiment.sbatch
```

The pinned GPTQModel `pyproject.toml` requires `device-smi>=0.5.3` (its `requirements.txt` still says `>=0.5.2`); the supplemental pin is now `0.5.7`, matching the working AUS lightweight environment. Setup accepts that existing version without reinstalling it. If lightweight setup stops before the marker, rerun the same `bash hpc/quantizer_followups/setup_gptq_env.sh`. It verifies the existing venv, installs only missing/incompatible supplemental packages, skips GPTQModel when its pinned distribution is already installed there, and finishes the checks without rebuilding the venv. A malformed venv is rejected. The interrupted full clone at `envs/gptq_followups_v570` is not touched by light setup. If it is incomplete, clone mode rejects it. To keep it for inspection and build a fresh clone elsewhere, set `GPTQ_ENV_MODE=clone GPTQ_ENV_OVERRIDE=/shared/makhtar/lp_llm/envs/gptq_followups_v570_fallback` for both setup and smoke.

The readiness check understands PEP 660 editable installs: it finds GPTQModel 5.7.0's `dist-info` in the venv, reads `direct_url.json` to verify the external pinned source checkout, checks its Git revision, and verifies dependency versions and origins. The editable source directory is expected to be outside the venv. GPTQModel 5.7.0 exposes `quantize()` on the `Qwen3QModel` instance returned by `GPTQModel.load()`, not on the `GPTQModel` facade class. Setup checks that instance class's quantization signature and Torch kernel W2G128 support; both full and tiny-smoke paths assert the loaded instance type before calling the same `model.quantize(..., backend=BACKEND.TORCH)` entry point.

Batch jobs stage the tracked repository code from `FOLLOWUP_CODE_SHA` into a private directory on the compute node before running preflight or importing `lp_llm`. A later `git pull` in the shared checkout cannot replace a script opened by that running job. The selected commit and stage path appear in the Slurm log. Pass the commit captured at submission; if omitted, the batch captures `HEAD` at job start. The staging change does not alter model exports or experiment identities.

The pinned v5.7.0 HF exporter creates empty bias parameters for biasless Qwen3 projections. R03 uses its `TorchQuantLinear.dequantize_weight()` on the packed checkpoint, preserves the original bias inventory, and verifies all 252 FP16 projection tensors against the pre-save decode. This is a materialization workaround, recorded in the resolved configuration; quantization, codes, scales and grouping remain upstream. R03 has **not** passed an AUS GPU smoke test. Do not submit its full experiment until the smoke log is reviewed.

## Recover completed S01/R01/R02 exports

The existing successful `quantization_process.json` records let the runner validate and evaluate without launching AutoRound again. Do not delete or edit the exports. R01/R02's pinned `OptimizedRTNConfig` has no `iters` field; JSON `null` now represents the verified zero-iteration dispatch. For R01/R02, first validate, then evaluate:

```bash
cd /shared/makhtar/lp_llm/code
git pull --ff-only origin experiment/rrq-w2-reproduction
CODE_SHA=$(git rev-parse HEAD)
sbatch --export=ALL,EXPERIMENT_ID=r01,FOLLOWUP_STAGE=encode,FOLLOWUP_REUSE_EXPORT_ONLY=1,FOLLOWUP_CODE_SHA="$CODE_SHA" hpc/quantizer_followups/run_experiment.sbatch
sbatch --export=ALL,EXPERIMENT_ID=r02,FOLLOWUP_STAGE=encode,FOLLOWUP_REUSE_EXPORT_ONLY=1,FOLLOWUP_CODE_SHA="$CODE_SHA" hpc/quantizer_followups/run_experiment.sbatch
# After both encoding markers pass validation, run evaluation one job at a time:
sbatch --export=ALL,EXPERIMENT_ID=r01,FOLLOWUP_REUSE_EXPORT_ONLY=1,FOLLOWUP_CODE_SHA="$CODE_SHA" hpc/quantizer_followups/run_experiment.sbatch
sbatch --export=ALL,EXPERIMENT_ID=r02,FOLLOWUP_REUSE_EXPORT_ONLY=1,FOLLOWUP_CODE_SHA="$CODE_SHA" hpc/quantizer_followups/run_experiment.sbatch
```

For S01, first measure every excluded tensor against the source after casting to its export dtype:

```bash
LP_ROOT=/shared/makhtar/lp_llm
"$LP_ROOT/envs/lp_llm/bin/python" -m lp_llm.quantizer_followups_audit \
  --source "$LP_ROOT/models/qwen3-8b-base" \
  --export-root "$LP_ROOT/results/qwen3_8b_base/quantizer_followups/s01/export" \
  --output "$LP_ROOT/results/qwen3_8b_base/quantizer_followups/s01/excluded_audit.json"
```

S01 validation remains strict. The audit does not change the export or create a completion marker. A separate repaired variant requires evidence about why the excluded tensor changed.

If every changed excluded tensor has `fp16_roundtrip_mismatches: 0`, the separately identified `s01_fp16_load_repair` path can restore only those excluded source bytes. It copies the raw checkpoint, verifies all 252 quantized projection tensors are byte-identical to the raw export, validates the repaired full checkpoint, and then evaluates it with the same MatGPTQ protocol. The original S01 export and result files are never changed. Review the audit and ensure enough disk space for a second roughly 16 GB checkpoint before running:

```bash
sbatch --export=ALL,S01_REPAIR_STAGE=encode,FOLLOWUP_CODE_SHA="$CODE_SHA" hpc/quantizer_followups/repair_s01.sbatch
# Inspect s01_fp16_load_repair/encoding_complete.json and the job log.
sbatch --export=ALL,S01_REPAIR_STAGE=all,FOLLOWUP_CODE_SHA="$CODE_SHA" hpc/quantizer_followups/repair_s01.sbatch
```

The repair refuses an unexplained excluded-tensor change or a conflicting partial/repaired checkpoint. The result is a separate materialization variant, not a replacement for the official S01 export.

R01 and R02 use the same guarded path with `REPAIR_EXPERIMENT_ID=r01` or `r02`. Their existing exports showed the same 12 excluded-tensor BF16→FP16→BF16 changes as S01. Each repaired variant has its own `run.json`, `encoding_complete.json`, `integrity_report.json`, checkpoint and evaluation result under `${id}_fp16_load_repair`. The report records exact excluded-source agreement and SHA-256 for all 252 quantized projections, compared byte-for-byte with the untouched raw export. Rerunning `S01_REPAIR_STAGE=encode` on the existing S01 repaired result verifies its marker and creates this explicit report without requantization.

After pulling the recovery commit, rerun S01's encode stage to verify its existing completion marker and write the explicit report. Run R01 and R02 through the same encode-only repair, then inspect the three reports before evaluation. These commands read the successful quantization records and exports; the repair script has no quantization entry point:

```bash
CODE_SHA=$(git rev-parse HEAD)
for id in s01 r01 r02; do
  sbatch --export=ALL,REPAIR_EXPERIMENT_ID="$id",S01_REPAIR_STAGE=encode,FOLLOWUP_CODE_SHA="$CODE_SHA" hpc/quantizer_followups/repair_s01.sbatch
done
# Wait for each job, then verify each result before its evaluation stage:
for id in s01 r01 r02; do
  ID="$id" python - <<'PY'
import json, os
from pathlib import Path
root = Path('/shared/makhtar/lp_llm/results/qwen3_8b_base/quantizer_followups')
result = root / (os.environ['ID'] + '_fp16_load_repair')
marker = json.loads((result / 'encoding_complete.json').read_text())
report = json.loads((result / 'integrity_report.json').read_text())
assert marker['status'] == report['status'] == 'completed_validated'
assert marker['identity'] == report['identity']
assert report['excluded_source_exact'] and report['projection_byte_identical']
assert report['checkpoint_validation']['unchanged_tensor_count'] == report['checkpoint_validation']['tensor_count'] - 252
assert len(report['projection_tensor_sha256']) == 252
assert marker['projection_tensor_sha256'] == report['projection_tensor_sha256']
print(result.name, 'excluded source exact; 252 quantized projections byte-identical')
PY
done
# Submit only after the corresponding report passes:
sbatch --export=ALL,REPAIR_EXPERIMENT_ID=s01,S01_REPAIR_STAGE=all,FOLLOWUP_CODE_SHA="$CODE_SHA" hpc/quantizer_followups/repair_s01.sbatch
sbatch --export=ALL,REPAIR_EXPERIMENT_ID=r01,S01_REPAIR_STAGE=all,FOLLOWUP_CODE_SHA="$CODE_SHA" hpc/quantizer_followups/repair_s01.sbatch
sbatch --export=ALL,REPAIR_EXPERIMENT_ID=r02,S01_REPAIR_STAGE=all,FOLLOWUP_CODE_SHA="$CODE_SHA" hpc/quantizer_followups/repair_s01.sbatch
```

The pinned GPTQModel 5.7.0 packed module creates `wf_unsqueeze_zero` and `wf_unsqueeze_neg_one` in `post_init()`. Immediately after quantization, the shell model may also contain meta placeholders while packed weights live in its turtle/offload state; the AUS tiny smoke observed a meta `qweight`. GPTQModel's save path synchronizes those states, and its loader initializes packed Torch modules. The R03 smoke therefore saves and reloads first, requires all packed fields to be materialized, independently decodes W2 codes/scales/zero points, and compares all seven dense FP16 projections exactly against the official Torch decoder's QDQ checkpoint. The full R03 runner applies the same check to all 252 projections. A missing packed field, meta tensor, or mismatch fails before a success marker.

## CPU checks and GPU smoke

```bash
/shared/makhtar/lp_llm/envs/lp_llm/bin/python -m pytest tests/test_quantizer_followups.py tests/test_rrq_pr2308.py -q
bash -n hpc/quantizer_followups/*.sh hpc/quantizer_followups/*.sbatch
mkdir -p /shared/makhtar/lp_llm/logs
```

Run one **smoke** job at a time before each full experiment. The smoke path runs actual 2-bit tuning or GPTQ quantization on a tiny one-layer Qwen3 model and reports allocated GPU memory and process RSS. It does not prove that an 8B model fits the allocation. Inspect the log and the method's projected 8B memory needs; if the 12 GB/23 GB A10G allocation is insufficient, stop and record the limitation without reducing samples, length, or model coverage.

Submit one smoke job at a time, wait for completion, and inspect its log before the next ID. For example:

```bash
sbatch --export=ALL,EXPERIMENT_ID=s01,FOLLOWUP_SMOKE=1 hpc/quantizer_followups/run_experiment.sbatch
sbatch --export=ALL,EXPERIMENT_ID=r03,FOLLOWUP_SMOKE=1 hpc/quantizer_followups/run_experiment.sbatch
squeue -u "$USER" -o '%.18i %.20j %.2t %.10M %.2D %R'
```

## Full jobs, at most two at a time

After smoke results and resource feasibility are reviewed, submit the first pair, wait for both to exit, then submit the second pair. The submission helper takes a lock and refuses while another follow-up job is queued or running. Each job has its own lock and output directory.

```bash
cd /shared/makhtar/lp_llm/code
bash hpc/quantizer_followups/submit_pair.sh first    # S01 + R01
squeue -u "$USER" -o '%.18i %.20j %.2t %.10M %.2D %R'
# After both jobs finish successfully and results are inspected:
bash hpc/quantizer_followups/submit_pair.sh second   # R02 + R03
```

To resume a completed encoding without repeating quantization, rerun the same experiment ID. `FOLLOWUP_STAGE=encode` validates or completes only the encoding stage. Example:

```bash
sbatch --export=ALL,EXPERIMENT_ID=r01,FOLLOWUP_STAGE=encode hpc/quantizer_followups/run_experiment.sbatch
sbatch --export=ALL,EXPERIMENT_ID=r01 hpc/quantizer_followups/run_experiment.sbatch
```

Never submit duplicate IDs simultaneously. Do not delete an unmarked export: the runner recovers it only if `quantization_process.json` records a successful upstream exit and full checkpoint validation passes. A mismatched shard or identity fails without overwrite.

## Inspect results

```bash
find /shared/makhtar/lp_llm/results/qwen3_8b_base/quantizer_followups -maxdepth 2 -type f -name '*.json' -print
/shared/makhtar/lp_llm/envs/lp_llm/bin/python -m lp_llm.quantizer_followups --aggregate --result-root /shared/makhtar/lp_llm/results/qwen3_8b_base
tail -n 100 /shared/makhtar/lp_llm/results/qwen3_8b_base/quantizer_followups/r01/quantization.log
```

Each `s01`/`r01`/`r02`/`r03` directory stores `run.json`, `source_audit.json`, `resolved_config.json`, `quantization.log`, `export/`, `checkpoint_validation.json`, `encoding_complete.json`, `evaluation.log`, and `full.json` after success. `failure.json` is written on runner errors. R03's `export/packed` is retained separately from its `export/qdq` for numerical and storage auditing. Runtime values are recorded only when observed.
