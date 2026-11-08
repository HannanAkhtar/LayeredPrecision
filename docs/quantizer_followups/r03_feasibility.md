# R03 full-model feasibility gate

The AUS tiny Qwen3 GPTQ smoke passed (job 53793). It verified the seven packed
projections against an independent INT2 decoder after checkpoint reload.
Its reported GPU allocation peak was 46.9165 MiB and process RSS peak was
2,267,360 KiB (about 2.16 GiB). These are tiny-model observations, not a
measurement or scaling basis for Qwen3-8B peak memory or disk use.

## Pinned workflow

R03 uses GPTQModel 5.7.0 at `ed96f2e04ec868b04f969733a242ce3b5bb7cb1e`.
The source Qwen3-8B-Base revision is
`49e3418fbbbca6ecbdf9608b4d22e5a407081db4`. `QuantizeConfig` fixes
W2G128, `sym=True`, `desc_act=False`, `act_group_aware=True`, `static_groups=False`,
`true_sequential=True`, FP32 Hessian staging, 0.05 dampening and 0.01 automatic
dampening increment. The worker shuffles `NeelNanda/pile-10k` with seed 42,
uses the source slow tokenizer, selects 128 sequences of 2048 tokens (joining
short samples as needed), hashes the exact token IDs, and runs batch-one GPTQ
on the Torch backend. The Qwen3 module tree selects seven projections in each
of 36 layers. RMS norms, embeddings and the untied output head are excluded.

The worker requires 252 packed Torch projections before save, saves the packed
checkpoint, reloads it, and requires identical projection coverage. It
independently decodes every reloaded INT2 `qweight`/`qzeros`/`scales`/`g_idx`
tensor and SHA-compares all 252 decoded FP16 weights with the official Torch
decoder's dense QDQ export. The checkpoint validator then checks source/QDQ
inventory, architecture, all 252 changed projections, W2 group structure,
excluded source-cast weights, finite values, packed code/scale/zero coverage,
and file hashes. Resume compares those hashes with its completion marker.
MatGPTQ evaluation retains the recorded WikiText-2 token hash, 2048-token
nonoverlapping protocol and FP16 inference dtype.

## Capacity estimate and limits

The read-only preflight scans the source Safetensors headers. For the expected
Qwen3-8B shapes, approximately 6.946 billion projection weights contribute
1.617 GiB of 2-bit codes, 0.101 GiB of FP16 scales, 0.013 GiB of packed zero
points and 0.005 GiB of group indices. Excluded dense weights, mainly the
separate embedding and head, contribute about 2.32 GiB. The **estimated packed
tensor payload** is therefore about 4.05 GiB, before file headers, config,
tokenizer and any other sidecars. The actual packed checkpoint bytes will be
measured after export. The **dense FP16 QDQ representation** remains about
15.26 GiB of tensor payload; it is not a 2-bit deployment size.

One FP16 128×2048×4096 activation cache is 2 GiB. One FP32 12288² Hessian is
0.5625 GiB; decomposition, workspaces and other projection Hessians can
coexist. GPTQModel defaults to disk offloading and loads a CPU turtle model
alongside a meta shell. It captures initial inputs on CPU and processes layers
sequentially, but asynchronous submodule finalizers, cache replay, packing,
save/reload and the large excluded tensors prevent a reliable 8B peak estimate
from the tiny smoke. More decisively, the current QDQ materializer creates all
252 CPU FP16 projection `Linear` weights before `save_pretrained`; these alone
occupy about 12.94 GiB. The 12 GiB host allocation **cannot complete the
current full workflow**, irrespective of quantization offloading. A streaming
QDQ writer or a larger validated node allocation is required. The 23 GiB A10G
GPU peak remains unmeasured for the 8B path.

The preflight uses conservative **capacity floors**, not predicted peaks: at
least 20 GiB currently free on the allocated GPU, a 20 GiB host allocation for
the current nonstreaming QDQ phase,
40 GiB scratch free for offloaded state, and the estimated packed plus dense
QDQ payload plus 4 GiB (about 23.3 GiB) free for output. If scratch and output
share a filesystem, it requires their combined free space (about 63.3 GiB).
The disk floors allow for duplicate offload serialization but are not an
upper bound. The reported node available RAM and cgroup headroom are read-only
observations. Full quantization remains blocked pending a representative-layer
measurement of GPU peak, process-tree RSS and scratch growth.

An interactive reading on `ip-10-240-16-194` found 15 GiB total host RAM,
about 8.3 GiB available and only 20 GiB free on `/tmp`; `/shared` had ample
reported capacity. That is **not** a measurement from the GPU compute-node
allocation. If its local scratch has the same 20 GiB free, the preflight will
fail before quantization. Do not redirect offloading to `/shared` without a
separately reviewed path, capacity check and restart policy.

## Run only the read-only assessment

```bash
cd /shared/makhtar/lp_llm/code
git pull --ff-only origin experiment/rrq-w2-reproduction
CODE_SHA=$(git rev-parse HEAD)
mkdir -p /shared/makhtar/lp_llm/logs
sbatch --wait --export=ALL,FOLLOWUP_CODE_SHA="$CODE_SHA" \
  hpc/quantizer_followups/r03_feasibility.sbatch
```

The job requests one A10G, one CPU, 12 GiB host RAM and ten minutes. It reads
source headers, `nvidia-smi`, filesystem capacity and Linux memory counters.
It does not load the model, download data, write an experiment result or run
GPTQ. It prints a JSON report and exits nonzero if a capacity floor fails.
With the current 12 GiB request, failure of the host-QDQ floor is expected;
that is an actionable result, not a preflight malfunction. The GPU-node disk
and GPU measurements in the same report still determine whether those
resources also need work.

Before full R03, run a **separately identified representative-layer job** using
the pinned 8B source and 128×2048 calibration inputs. It should exercise the
largest `down_proj` (12288 input channels), Hessian staging, packing and disk
offloading while sampling GPU memory, process-tree RSS, cgroup memory and
scratch usage at short intervals. Preserve its outputs outside `r03/`; do not
use the full R03 result identity or launch the 252-projection runner. Abort if
the allocation approaches 90% of GPU or host memory, scratch/output free space
falls below the capacity floor, any tensor becomes nonfinite, or a packed
projection cannot be independently decoded. The representative job still
cannot prove every phase's peak; review its measurements with additional room
for final save/reload and QDQ export before approving a full allocation.

The full R03 batch path now requires `R03_FULL_FEASIBILITY_OK=1` **and** reruns
the capacity preflight. There is no safe full submission on the known 12 GiB
allocation. Do not set that flag until the host-QDQ blocker is resolved, the
representative-layer results have been reviewed, and a suitable allocation is
available. A later full submission must override the memory request with a
reviewed value of at least 20 GiB and the wall time with the QoS-approved
limit; start with `FOLLOWUP_STAGE=encode`. The existing 12 GiB/12-hour defaults
are not a feasibility claim. A failed or partial R03 export must be inspected,
not automatically overwritten. Evaluate only after the 252-projection
encoding marker passes validation.
