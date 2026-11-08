# R03 A10G feasibility and guarded layer probe

The tiny R03 smoke passed on AUS in job 53793 (46.9165 MiB PyTorch GPU peak,
2,267,360 KiB process RSS). Those are tiny-model observations only. Read-only
capacity job 53833 found 21.97 GiB free on its A10G and confirmed the old
workflow could not fit: 12 GiB host RAM was below its 20 GiB floor and local
`/tmp` had only 21.60 GiB free versus 40 GiB required. GPU nodes have only
15,564 MiB Slurm RealMemory.

## Revised workflow

R03 retains GPTQModel 5.7.0 revision
`ed96f2e04ec868b04f969733a242ce3b5bb7cb1e`, symmetric W2G128, Torch
backend, FP16 load, FP32 Hessian staging, `desc_act=False`,
`act_group_aware=True`, `static_groups=False`, `true_sequential=True`, and the
existing dampening settings. Full and probe paths share the exact calibration
builder: source slow tokenizer, `NeelNanda/pile-10k` shuffled with seed 42,
128 sequences of 2048 tokens with the established short-sample fallback, and
a token-ID SHA256 fingerprint. GPTQModel's Qwen3 tree selects seven
projections per layer, 252 total. Embeddings, untied head and norms are
excluded.

GPTQModel saves a **packed** checkpoint first. The worker reloads it and
streams the dense **QDQ** checkpoint one source-named tensor per Safetensors
shard, with the `format=pt` header needed by Hugging Face. It does not
accumulate all CPU FP16 projection weights. For each projection it compares
the independent INT2 decoder with the official Torch decoder exactly. The
checkpoint validator repeats all 252 comparisons, including `g_idx`, and
checks source inventory, exclusions, finiteness and hashes. An interrupted
QDQ export remains in `.partial` and is never automatically overwritten.
Packed and QDQ storage are measured and reported separately. MatGPTQ's
recorded WikiText-2 token hash, 2048-token protocol and FP16 inference are
unchanged.

GPTQModel's temporary offload path now points into a unique, job-owned
directory under `/shared/makhtar/lp_llm/tmp/r03_offload` (or another reviewed
`R03_OFFLOAD_ROOT` under `/shared/makhtar/lp_llm`). Normal exit or a Python
exception removes only the directory bearing that job's ownership marker.
A hard kill can leave an orphan; later jobs never delete it automatically.
Inspect an orphan before manual cleanup.

## Revised admission floors

These are capacity gates, **not predicted 8B peaks**. Source Safetensors
headers set the exact byte counts. For Qwen3-8B, the estimated packed tensor
payload is 4,353,779,712 bytes (about 4.05 GiB), including dense excluded
tensors. Dense FP16 QDQ tensor payload is about 15.26 GiB. Physical packed
files include headers and sidecars and are measured after export. The
streamed writer's largest single dense tensor is about 1.16 GiB. One FP16
128×2048×4096 activation cache is 2 GiB, and one FP32 12288² Hessian is
0.5625 GiB; overlapping GPTQ workspaces remain unmeasured.

| Resource | New floor |
| --- | ---: |
| Allocated A10G free memory | 20 GiB |
| Slurm host allocation | 10 GiB (current 12 GiB request passes this static check) |
| Local scratch for staged code/runtime | 2 GiB |
| Dedicated shared offload directory | 40 GiB free |
| Shared packed plus QDQ output | payload estimates plus 4 GiB, about 23.3 GiB |
| Declared available shared quota | offload plus output floors, about 63.3 GiB |

For paths on one filesystem, the preflight sums the relevant floors. It also
checks declared available shared quota. AUS has no `quota` command; `/shared`
is NFSv4 and reports 8.0E free via `df`, while current project usage is 666
GiB. None of those figures proves remaining user/project quota. With no
declared quota, the read-only preflight prints other measurements but fails
the quota gate; the layer probe and full worker fail closed. Obtain an
administrator or portal value for **available** quota before setting
`R03_SHARED_QUOTA_FREE_GIB`.

## AUS commands

The next capacity job reads headers and node metrics; it does not load weights
or write an experiment result. Create the dedicated empty parent once, then
submit it even if quota is still unknown. Expect a nonzero exit when the quota
gate remains unresolved; inspect its JSON in the Slurm log.

```bash
cd /shared/makhtar/lp_llm/code
git pull --ff-only origin experiment/rrq-w2-reproduction
CODE_SHA=$(git rev-parse HEAD)
LP_ROOT=/shared/makhtar/lp_llm
export R03_OFFLOAD_ROOT="$LP_ROOT/tmp/r03_offload"
mkdir -p "$R03_OFFLOAD_ROOT" "$LP_ROOT/logs"
sbatch --wait --export=ALL,FOLLOWUP_CODE_SHA="$CODE_SHA" \
  hpc/quantizer_followups/r03_feasibility.sbatch
```

Once an administrator or portal confirms the free quota, set it and rerun the
same capacity job. **Only after it passes**, submit the separate first-layer
probe:

```bash
read -r -p 'Site-confirmed available shared quota (GiB): ' R03_SHARED_QUOTA_FREE_GIB
export R03_SHARED_QUOTA_FREE_GIB
sbatch --wait --export=ALL,FOLLOWUP_CODE_SHA="$CODE_SHA" \
  hpc/quantizer_followups/r03_feasibility.sbatch
sbatch --wait --export=ALL,FOLLOWUP_CODE_SHA="$CODE_SHA" \
  hpc/quantizer_followups/r03_layer_probe.sbatch
```

The probe requests one A10G, four CPUs, 12 GiB and four hours. It loads the
pinned 8B source but keeps only its first real decoder layer in the in-memory
shell and turtle. The seven projections include the largest-shape
`down_proj` (12288 input channels). It runs the same 128×2048 calibration and
W2G128 Torch algorithm, packs, saves/reloads a temporary one-layer checkpoint
and checks all seven packed decodes exactly. It writes a separate report under
`quantizer_followups/r03_feasibility/layer_probe-<jobid>.json`, never under
`r03/`. Job-owned offload and temporary packed files are removed on normal
exit. No full 252-projection quantization is invoked.

Abort or decline full R03 if the probe reaches 90% of GPU or cgroup host
memory, runs out of shared quota, approaches storage floors, produces
nonfinite weights, fails packed parity, or leaves an unexplained partial
export. This first layer exercises the largest tensor dimensions but cannot
prove late-layer activation, total runtime or final QDQ export peaks.

Full R03 remains guarded by `R03_FULL_FEASIBILITY_OK=1`, a same-code and
same-source successful `R03_PROBE_REPORT` (including source-shard hashes and
the exact W2G128 Torch settings), and a fresh capacity preflight.
**Do not submit full R03 yet.** Review the probe measurements, quota and
remaining storage first. A later authorized first full run should use
`FOLLOWUP_STAGE=encode` and a QoS-approved wall time; evaluate only after
the 252-projection completion marker passes validation.
