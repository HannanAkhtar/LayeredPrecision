# Layered Precision Reproducibility Code

The RRQ 2-bit Qwen3 reproduction branch has an [AWS execution guide](docs/rrq_w2_reproduction.md) with pinned upstream revisions, experiment mappings, and selected-job submission commands. No new full-model results are included in the branch.

This directory contains the minimal code used for the paper's post-training
Layered Precision (LP) experiments on ImageNet classification and per-signal
NGP image/audio models. Exploratory numerical formats, BN folding and
calibration branches, percentile experiments, DoReFa variants, and unrelated
model families are intentionally excluded.

## Experiment Contract

All reported quantization uses unsigned affine integer PTQ with an explicit
integer zero point:

```text
scale = (max - min) / (2^b - 1)
zero_point = round(-min / scale)
q = clamp(round(x / scale + zero_point), 0, 2^b - 1)
x_hat = scale * (q - zero_point)
```

Zero is included in every fitted range. LP quantizes the original tensor for
the base layer, then independently quantizes the reconstruction residual for
each enhancement layer. An LP prefix is the sum of the decoded base and the
available residual contributions.

This is weight-only PTQ. Activations and inference arithmetic remain FP32.

ImageNet uses one affine range per output channel (`axis=0`). Norm/BN and all
1D parameters are left in FP32. There is no BN folding or BN recalibration.
NGP uses one affine range for its packed 1D tiny-cuda-nn parameter tensor;
1D quantization is therefore required. Bias/norm names are excluded, although
tiny-cuda-nn's internal weights and biases are jointly contained in its packed
`net.params` tensor and are quantized together.

## Repository Layout

```text
layered_quantization.py       affine INT and LP residual coding
bitplane.py                   affine bit-plane truncation baseline
evaluate_imagenet.py          six-model ImageNet experiment
plot_imagenet_results.py      strict multi-model PNG/PDF plotting
ngp_models.py                 paper HashGrid image/audio architectures
ngp_data.py                   DIV2K and LibriSpeech preprocessing
ngp_metrics.py                image and audio reconstruction metrics
train_ngp_dataset.py          one NGP model per image/audio sample
evaluate_ngp_dataset.py       monolithic and LP dataset evaluation
storage_utils.py              entropy coding and complete accounting
analyze_storage.py            ImageNet/NGP storage experiments
tests/                        deterministic regression tests
```

## Environment

The validated environment used Python 3.11, PyTorch 2.10.0+cu128,
Torchvision 0.25.0+cu128, Torchaudio 2.10.0+cu128, NumPy 2.3.5,
Pillow 12.0.0, Matplotlib 3.10.7, TorchCodec 0.10.0, and CUDA 12.8.

On Linux:

```bash
conda env create -f environment.yml
conda activate dl

pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128

# Required only for NGP training/evaluation/storage analysis.
pip install 'git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch'

python -m unittest discover -s tests -v
```

The ImageNet workflow can run on CPU, although a CUDA GPU is strongly
recommended. The NGP workflow requires Linux, CUDA, and tiny-cuda-nn.

## Data Layout

ImageNet must use an `ImageFolder`-style validation set. Both WordNet class
names and integer class directories are supported:

```text
/data/imagenet/val/
  n01440764/*.JPEG
  n01443537/*.JPEG
  ...
```

For paper evaluation, use all 50,000 ImageNet validation images at their
original resolution. Each pretrained weight enum supplies its own resize,
crop, and normalization transform. The evaluator rejects a full run unless it
finds exactly 50,000 images, 1,000 classes, and labels in `[0, 999]`.

The image NGP preset expects DIV2K validation images under one root, for
example `/data/DIV2K/DIV2K_valid_HR`. The audio preset uses LibriSpeech
`dev-clean` under `/data/librispeech/LibriSpeech/dev-clean`. Torchaudio can
download the latter when `--download` is passed.

## ImageNet

The fixed paper presets are:

| Models | Monolithic points | LP path | Bit-plane source |
|---|---:|---:|---:|
| EfficientNet-B2, ResNet-50, ViT-B/16 | 5, 7, 9, 11 | 5, 2, 2, 2 | 11 |
| ConvNeXt-Base, VGG-16, Swin-V2-B | 4, 6, 8, 10 | 4, 2, 2, 2 | 10 |

The exact pretrained enums are EfficientNet-B2 V1, ResNet-50 V2, ViT-B/16 V1,
ConvNeXt-Base V1, VGG-16 V1, and Swin-V2-B V1.

Run FP32, monolithic PTQ, LP, and bit-plane coding in one command:

```bash
python evaluate_imagenet.py \
  --data-dir /data/imagenet/val \
  --models all \
  --methods monolithic lp bitplane \
  --outdir outputs/imagenet \
  --batch-size 64 \
  --num-workers 8
```

The evaluator writes each completed operating point immediately. To continue
an interrupted run, repeat the identical command with `--resume`. Resume is
refused when saved model, data, transform, method, or quantizer metadata does
not match.

The bit-plane comparison is a PTQ baseline, not Any-Precision QAT. A model is
first affine-quantized at the LP path's highest width. To retain fewer planes,
the integer code and zero point are right-shifted together and the scale is
multiplied by the corresponding power of two.

Plot one new output directory:

```bash
python plot_imagenet_results.py \
  --results outputs/imagenet \
  --outdir outputs/imagenet_plots
```

Multiple historical run folders can be merged safely:

```bash
python plot_imagenet_results.py \
  --results outputs/base4 outputs/base5 \
  --outdir outputs/imagenet_plots
```

Conflicting duplicate model/type/config rows are rejected rather than plotted
as multiple points at the same bit width.

## NGP Image Dataset

The image preset trains one RGB HashGrid model per DIV2K validation image:
10 levels, 2 features per level, log2 hash size 13, base resolution 16,
per-level scale 2, 32 MLP neurons, and 2 hidden layers. Images are resized with
Lanczos only when their longest side exceeds 512 pixels. The paper run uses
100 images, 5,000 full-signal steps, learning rate 0.01, and seed 123.

```bash
python train_ngp_dataset.py image \
  --root /data/DIV2K/DIV2K_valid_HR \
  --outdir checkpoints/div2k_valid \
  --max-samples 100 \
  --steps 5000 \
  --lr 0.01 \
  --seed 123

python evaluate_ngp_dataset.py \
  --checkpoints-root checkpoints/div2k_valid \
  --manifest checkpoints/div2k_valid/manifest.json \
  --data-root /data/DIV2K/DIV2K_valid_HR \
  --outdir outputs/div2k_valid \
  --monolithic-bits 6,7,8,9,10,12,14 \
  --layered-bits 6,2,2,2
```

Image monolithic points are 6, 7, 8, 9, 10, 12, and 14 bits. The LP path is
`[6, 2, 2, 2]`. Reported metrics are MSE, PSNR, and SSIM.

## NGP Audio Dataset

The audio preset trains one HashGrid model per randomly selected five-second,
16 kHz, peak-normalized LibriSpeech clip: 16 levels, 2 features per level,
log2 hash size 11, base resolution 8, per-level scale 2, 32 MLP neurons, and
2 hidden layers. The paper run uses 10 clips, 5,000 full-signal steps,
learning rate 0.01, and seed 123.

```bash
python train_ngp_dataset.py audio \
  --root /data/librispeech \
  --outdir checkpoints/librispeech_devclean \
  --max-samples 10 \
  --steps 5000 \
  --lr 0.01 \
  --seed 123 \
  --download

python evaluate_ngp_dataset.py \
  --checkpoints-root checkpoints/librispeech_devclean \
  --manifest checkpoints/librispeech_devclean/manifest.json \
  --data-root /data/librispeech \
  --outdir outputs/librispeech_devclean \
  --monolithic-bits 6,7,8,9,10,12,14 \
  --layered-bits 8,2,2,2
```

Audio monolithic points are 6, 7, 8, 9, 10, 12, and 14 bits. The LP path is
`[8, 2, 2, 2]`; the 8-bit base was selected so the reported base models do not
have negative SNR. Metrics are MSE, SNR, SI-SNR, and log-spectral distance.

Use `--resume` with the same command to continue NGP training or evaluation.
Training and evaluation metadata are checked before existing checkpoints or
rows are reused. NGP plots are dataset means and are written as both PNG and
PDF.

## Reference Checks

### Experiment 1: first LP residual quantizer

`evaluate_residual_ablation.py` evaluates ResNet-50 `IMAGENET1K_V2` with the
checkpoint's own preprocessing. It keeps the paper's channel-wise, axis-0,
unsigned affine 5-bit base and the original parameter exclusions. Only the
first 2-bit residual quantizer changes. The prespecified variants are
`symmetric_minmax`, `affine_minmax` (the unchanged LP reference),
`symmetric_percentile`, `affine_percentile`, and `affine_mse`. The default
clipping percentile is 99.5. Symmetric 2-bit codes are `-1, 0, 1` and yield
three distinct levels, including exact zero; affine codes are `0, 1, 2, 3`.
MSE calibration uses a deterministic 17-point scale grid from 0.5 to 1.5
times the affine min/max scale and checks all four integer zero points per
output channel. It uses weights only, never validation labels.

Install the PyTorch/Torchvision pair shown in the setup section above and the
remaining dependencies in `requirements.txt`. Run the synthetic checks first:

```bash
python -m unittest discover -s tests -p test_quantizer_variants.py
```

Use labeled ImageNet validation images in numeric class folders `test/0`
through `test/999` (50 images per class for the complete evaluation). The
official unlabeled ImageNet test split cannot produce accuracy metrics.

```bash
python evaluate_residual_ablation.py --data-dir test --outdir outputs/residual_ablation \
  --batch-size 64 --num-workers 8 --variants all
```

For a development-only run, add `--max-batches 2`. This uses the same first
two batches for every method; its accuracy is **not** a full ImageNet result.
Use `--resume` with identical settings to continue saved results. `--device`
defaults to automatic CUDA/CPU selection. The runner writes
`residual_ablation.csv`, `residual_ablation.json`, and PNG plots. The JSON
contains `schema_version`, run `metadata`, and a `results` array. Each result
records method, scope, top-1/top-5, cross-entropy, weighted residual and
parameter MSE, SQNR, code count, effective levels, channel scale/zero-point
statistics, calibration time, nominal bits, and reference discrepancy. CSV
contains the scalar comparison columns. The complete run first checks FP32,
5-bit LP base, original 5+2-bit LP, and monolithic 7-bit references. It stops
before the new variants if any full-set Top-1 reference differs by more than
0.1 percentage points. Plots and rows from a limited run are labeled `smoke`.
For each residual variant, a separate `.pt` artifact stores its integer codes,
per-channel scales, zero points, and clipping/calibration metadata. These are
reproducibility artifacts, not compressed-size measurements.
`base_5_codes_and_metadata.pt` stores the identical base codes and metadata
shared by all variants.
Reconstruction MSE and SQNR aggregate squared errors over all quantized weight
elements; excluded parameters stay FP32 and are not included in that denominator.

The CSV schema is one row per method with columns `method`, `status`,
`evaluation_scope`, `images`, `top1`, `top5`, `cross_entropy`, `residual_mse`,
`parameter_mse`, `sqnr_db`, `nominal_bits_per_quantized_weight`,
`stored_codes`, `effective_levels`, `calibration_seconds`,
`actual_storage_status`, `reference_top1`, and `reference_delta_top1`.
The JSON schema has top-level `schema_version`, `metadata`, and `results` keys.
Each result includes the CSV metrics plus `settings`, `quantizer_stats`, and
`quantizer_artifact` (the `.pt` filename, where applicable). Reference rows
have null residual metrics or quantizer statistics when those do not apply.

Actual coded storage is marked `pending`: the existing storage decoder assumes
unsigned affine reconstruction and does not represent the signed symmetric
codes or every clipped/optimized constant-channel case. Nominal 5/7-bit
precision is separate from coded bytes; tensor container sizes are not used
as compressed sizes.

The completed full-resolution runs produced the following Top-1 values. These
are useful checks for the dataset layout, transforms, and checkpoint versions:

| Model | FP32 | LP base | Deepest LP |
|---|---:|---:|---:|
| EfficientNet-B2 | 80.57 | 79.00 (5 bits) | 80.58 (11 bits) |
| ResNet-50 | 80.74 | 76.25 (5 bits) | 80.74 (11 bits) |
| ViT-B/16 | 81.00 | 80.72 (5 bits) | 81.00 (11 bits) |
| ConvNeXt-Base | 83.88 | 81.45 (4 bits) | 83.89 (10 bits) |
| VGG-16 | 71.47 | 69.88 (4 bits) | 71.48 (10 bits) |
| Swin-V2-B | 83.91 | 82.98 (4 bits) | 83.90 (10 bits) |

For 100 DIV2K models, mean LP PSNR/SSIM was 25.857 dB/0.7733 at
6 bits and 29.306 dB/0.8462 at 12 bits. For 10 LibriSpeech models, mean LP
SNR was 20.316 dB at 8 bits and 24.650 dB at 14 bits; no clip had negative
SNR at either LP endpoint.

## Storage Analysis

ImageNet storage:

```bash
python analyze_storage.py imagenet \
  --models all \
  --outdir outputs/storage_imagenet
```

NGP dataset storage:

```bash
python analyze_storage.py ngp \
  --checkpoints-root checkpoints/div2k_valid \
  --manifest checkpoints/div2k_valid/manifest.json \
  --outdir outputs/storage_div2k \
  --monolithic-bits 6,7,8,9,10,12,14 \
  --layered-bits 6,2,2,2
```

The analyzer reports three coding regimes: one Huffman codebook per precision
layer, one global Huffman codebook, and raw tight bit-packing. Two storage
definitions are kept deliberately:

- `storage_bytes`, `global_storage_bytes`, and `raw_storage_bytes` reproduce
  the historical symbol-stream accounting: payload, codebook, and stream
  headers only.
- `complete_*_storage_bytes` additionally include every scale, zero point,
  tensor shape, layer offset, and unquantized FP32 parameter/buffer needed to
  decode the state with the released model architecture.

By default, payload file sizes are computed exactly from the code lengths
without writing model-sized temporary streams. Add `--write-artifacts` to keep
the real payloads, codebooks, headers, quantizer metadata, and fixed FP32 state.
Artifact writing is much slower than size-only analysis.

`storage_utils.load_layered_state` is the reference decoder for all three
artifact regimes. It reconstructs a state dictionary that can be loaded into
the released architecture.

`simulcast_vs_lp.csv` compares one deepest LP stream against storing an
independent monolithic model at every LP operating point. NGP analysis also
writes per-dataset sum tables because every sample owns a separate model.

## Reproducibility Notes

- Given the same checkpoint, all PTQ code paths are deterministic and the LP
  base is numerically identical to an independently quantized monolithic model
  at the same width.
- ImageNet accuracy reproduction requires the same torchvision weights, full
  50,000-image validation set, class mapping, and package versions.
- NGP checkpoints store the final 5,000-step state, matching the original
  experiment. `best_loss` is diagnostic; it is not a separately saved model.
- Retraining tiny-cuda-nn can show small GPU/toolchain variation even with the
  same seed. Quantizing the same saved checkpoints is the strict comparison.
- Datasets, pretrained caches, checkpoints, and generated outputs are not part
  of this source directory.

## Qwen3-8B-Base LLM experiments

`lp_llm` extends the same unsigned affine INT quantizer to the pinned
`Qwen/Qwen3-8B-Base` revision
`49e3418fbbbca6ecbdf9608b4d22e5a407081db4`. It reads the Safetensors
index and maps one tensor at a time. It validates the complete Qwen3 key set,
BF16 dtype, and tensor shapes before writing anything. The selection policy is
in `lp_llm/model_adapter.py`; it quantizes the four attention projections and
three MLP projections in each transformer layer. Embeddings, the separate
untied LM head, RMSNorms, and all other tensors are copied exactly as shared
BF16 state. The quantizer itself has no Transformers dependency.

The first schedule is `[2,2,2,2]` with nominal 2, 4, 6, and 8 bit prefixes.
Arbitrary schedules such as `[5,2,2,2]` are accepted. `channel` fits one range
per output row. `group` fits one range per 128 consecutive input values within
each row; the final short group is fitted separately. Both modes include zero
in each range and use the existing `IntegerFormat` rounding, zero point, and
zero range behavior. Each eligible BF16 tensor is converted to FP32, and all
base, residual, and prefix arithmetic stays FP32. Only the reconstructed
checkpoint is cast to BF16. Per tensor metrics include FP32 parameter MSE,
BF16 cast MSE, SQNR, residual energy, and residual MSE; task metrics measure
the actual BF16 inference model.

Each eligible tensor owns one packed `.bin` code stream and one Safetensors
metadata file per LP layer. Metadata contains FP32 scales and integer zero
points. Zero is included in every fitted range, so a zero scale occurs only
for an all-zero region and decodes to zero without a stored minimum. Bits
within a byte are stored least significant first. The versioned manifest
records the model identity, selected parameters,
bit schedule, numerical conventions, software versions, stream hashes, and
source inventory. Excluded BF16 tensors and Hugging Face configuration and
tokenizer sidecars are copied into the artifact. A tensor completion marker is
written after its files; an interrupted run can repeat the same command. A
final `COMPLETE.json` is created only after every tensor finishes.

`storage_report.json` measures the bytes needed to retain each prefix:
packed codes, scale/zero point payload, other decoder metadata and framing,
and shared unquantized tensor files. The final prefix total is the cumulative
hierarchy size. These are physical file sizes, not nominal bit widths. Resume
markers and `run.json` are measured separately; the optional report file
itself is excluded because the decoder does not need these files.
Only the streams up to the requested prefix are read during reconstruction.
Each tensor is written to its own output Safetensors shard and an index is
generated after all shards finish. The original checkpoint is never modified
or needed to decode an artifact.

For local testing, use Python with PyTorch, Safetensors, NumPy, PyYAML, pytest,
and Transformers 4.57.1. `requirements-llm.txt` lists the added LLM packages;
it is separate from the validated ImageNet environment. The tiny Qwen3 test
requires a Transformers release with `Qwen3Config` and creates its own random
sharded checkpoint. It performs an AutoModel reload and forward pass for every
LP prefix in both granularity modes and for a monolithic model. No model
download is needed.

```bash
python -m pytest tests -q
```

On AWS, deploy this reviewed source tree to `$LP_ROOT/paper_release` after
publishing it yourself. The existing Conda environment and `$LP_ROOT/env.sh`
are used by the Slurm scripts. The scripts do not install packages. All four
scripts default to 12 GB host RAM and four CPUs, matching the earlier working
allocation. Slurm command line options such as `--mem=14G --cpus-per-task=8`
override these defaults when needed. First verify BF16 protocol parity, then
submit the first per channel LP quantization. Reconstruction and evaluation
are separate submissions after their preceding completion marker appears:

```bash
export LP_ROOT=/shared/makhtar/lp_llm
export LP_CODE=$LP_ROOT/paper_release
mkdir -p "$LP_ROOT/logs" "$LP_ROOT/results"
test -f "$LP_ROOT/models/qwen3-8b-base/model.safetensors.index.json"
test -f "$LP_CODE/configs/llm/qwen3_lp_channel.yaml"

sbatch --export=ALL,LP_CODE="$LP_CODE",LP_CHECKPOINT="$LP_ROOT/models/qwen3-8b-base",LP_RESULT="$LP_ROOT/results/qwen3_bf16_wikitext_parity.json",LP_PROTOCOL=wikitext \
  "$LP_CODE/hpc/lp_evaluate.sbatch"

# After the BF16 job completes, verify 299078 corpus tokens, 299077 scored
# tokens, and perplexity near 6.219182234107349 before quantized runs.
sbatch --export=ALL,LP_CODE="$LP_CODE",LP_CONFIG="$LP_CODE/configs/llm/qwen3_lp_channel.yaml",LP_SOURCE="$LP_ROOT/models/qwen3-8b-base",LP_OUTPUT="$LP_ROOT/results/qwen3_lp_channel" \
  "$LP_CODE/hpc/lp_quantize.sbatch"

# After qwen3_lp_channel/COMPLETE.json exists, choose one depth (1 to 4):
export LP_DEPTH=1
sbatch --export=ALL,LP_CODE="$LP_CODE",LP_ARTIFACT="$LP_ROOT/results/qwen3_lp_channel",LP_DEPTH="$LP_DEPTH",LP_OUTPUT="$LP_ROOT/results/qwen3_lp_channel/prefix-$LP_DEPTH" \
  "$LP_CODE/hpc/lp_reconstruct.sbatch"

# After prefix-1/COMPLETE.json exists:
sbatch --export=ALL,LP_CODE="$LP_CODE",LP_CHECKPOINT="$LP_ROOT/results/qwen3_lp_channel/prefix-$LP_DEPTH",LP_RESULT="$LP_ROOT/results/qwen3_lp_channel/wikitext-$LP_DEPTH.json",LP_PROTOCOL=wikitext \
  "$LP_CODE/hpc/lp_evaluate.sbatch"
```

Repeat the selected depth commands for 2, 3, and 4 after inspecting each
result. Use `LP_PROTOCOL=zeroshot` and a distinct result file for the five
zero shot tasks, batch size 1 and no few shot examples. To run the next G128
experiment, use `qwen3_lp_group128.yaml` and a distinct artifact directory.
For matched monolithic controls, use `qwen3_monolithic.yaml` or
`qwen3_monolithic_group128.yaml` and pass `LP_BITS=2`, `4`, `6`, or `8`, with
one distinct output directory per width. The same Slurm quantization and
reconstruction scripts apply; monolithic depth is always 1. The optional
array script evaluates already reconstructed `prefix-1` through `prefix-4`
directories and is not part of the first submission.

For the monolithic comparison with fewer simultaneous submissions, use the
consolidated runners below after the already submitted LP jobs. Each runner
uses one GPU allocation and processes 2, 4, 6, and 8 bits in order, performing
encode, reconstruction, and WikiText evaluation for each width. The defaults
are 12 GB RAM, four CPUs, and 24 hours (the same wall-time request as the
existing accepted quantization jobs); Slurm command-line options can override
these requests within the QoS limit. If the allocation ends before all widths
finish, resubmit the same script. Valid completed stages are skipped; a
present but invalid artifact stops the job with an error for inspection.
The runners use distinct result roots and do not touch existing LP results.

```bash
export LP_ROOT=/shared/makhtar/lp_llm
export LP_CODE=$LP_ROOT/paper_release
mkdir -p "$LP_ROOT/logs" "$LP_ROOT/results"
sbatch --export=ALL,LP_CODE="$LP_CODE",LP_ROOT="$LP_ROOT" \
  "$LP_CODE/hpc/monolithic_channel_batch.sbatch"
sbatch --export=ALL,LP_CODE="$LP_CODE",LP_ROOT="$LP_ROOT" \
  "$LP_CODE/hpc/monolithic_group128_batch.sbatch"
```

These are two jobs total. Submit them when the `gpu-long-makhtar-001`
`GrpSubmitJobs=15` limit has room. Outputs appear under
`$LP_ROOT/results/qwen3_monolithic_channel/bits-{2,4,6,8}` and the analogous
`qwen3_monolithic_group128` root. Each width contains the LP artifact, a
`checkpoint` directory, and `wikitext.json`. No packages are installed by
the batch scripts; they activate the existing Conda environment.

For the separate 2-bit RTN reproduction and diagnostic workflow (B0–B3),
including the consolidated screen and selected full-evaluation Slurm
commands, see [Phase 1 RTN reproduction](docs/rtn_reproduction_phase1.md).

### AutoRound-compatible plain symmetric RTN G128 LP experiment

The `autoround_rtn_sym` family is separate from the earlier `symmetric_rtn`
ternary-style ablation and from asymmetric affine quantization. It follows
[`quant_tensor_rtn_sym` at Intel AutoRound commit `6c824f741be358983baba87aff83f9c5aa0a01e7`](https://github.com/intel/auto-round/blob/6c824f741be358983baba87aff83f9c5aa0a01e7/auto_round/data_type/int.py#L124-L164),
with row-local 128-element groups and zero padding of each final partial
group. This is the plain RTN function; AutoRound also has distinct optimized
RTN and learned quantizers. The [RRQ paper](https://arxiv.org/html/2608.04048)
reports Qwen3-8B-Base all-RTN perplexities around 16.75, 8.39, 7.22, and
7.10 for 2/4/6/8 bits, but its exact AutoRound revision and evaluation setup
are not established as identical to this experiment. Those values are
external context, not validation targets.

For each group, let `Q = 2^(b-1)`, `n = -min(min(w), 0)`, and
`p = max(max(w), 0)`. The signed extremum is `n` if `n > p`, otherwise `-p`
(including ties). The scale is `float16(signed_extremum / Q)`, clipped to
absolute magnitude at least `1e-5` while retaining its sign; zero groups
receive a positive minimum scale. Codes are
`q = clamp(round_ties_to_even(w / scale), -Q, Q-1)`, and the FP32 decoded
contribution is `q * scale`. Packing stores `u = q + Q`, covering every
unsigned `b`-bit code; decoding uses `q = u - Q`. No group zero point is
stored. The encoder uses the FP32 decoded prefix to form the next residual;
the final checkpoint casts that decoded prefix to BF16. Scale conversion,
division, multiplication, and final input-dtype conversion match the pinned
reference function. The tests retain an instrumented, independently
verifiable copy of its arithmetic and compare scales, codes, and outputs for
FP32/BF16, outliers, skew, constants, zeros, and partial groups.

Use the existing Conda environment and the locally installed, **untracked**
`lp_llm/evaluate_matgptq.py` on the AWS checkout. The runner does not install
packages or add a tracked file at that path. It passes `--sequence-length 2048
--dtype float16` to that evaluator, whose protocol uses non-overlapping
WikiText-2 blocks. Run pilot first; submit full only after its output is
validated. Each phase uses one 12 GB, four CPU, 12-hour GPU allocation by
default. Override memory, CPU, or time with Slurm command-line flags if the
QoS permits. The jobs are not submitted by this repository.

```bash
export LP_ROOT=/shared/makhtar/lp_llm
export LP_CODE=$LP_ROOT/paper_release
mkdir -p "$LP_ROOT/logs" "$LP_ROOT/results/qwen3_8b_base"
sbatch --export=ALL,LP_CODE="$LP_CODE",LP_ROOT="$LP_ROOT",LP_PHASE=pilot \
  "$LP_CODE/hpc/lp_autoround_rtn_group128_batch.sbatch"
# Inspect the 2-bit result, then submit the remaining prefixes:
sbatch --export=ALL,LP_CODE="$LP_CODE",LP_ROOT="$LP_ROOT",LP_PHASE=full \
  "$LP_CODE/hpc/lp_autoround_rtn_group128_batch.sbatch"
```

The hierarchy is stored under
`results/qwen3_8b_base/lp_autoround_rtn_g128_2_2_2_2/`; reconstructed
prefixes and MatGPTQ protocol results use the corresponding
`decoded/lp_autoround_rtn_g128_{bits}bit/` and
`matgptq_protocol/lp_autoround_rtn_g128_{bits}.json` paths. The runner writes
`autoround_comparison.json` into the hierarchy directory with nominal and
effective bits, packed codes and metadata bytes, MSE/SQNR, perplexity, and
timings. It automatically includes the earlier ternary results when present.
To compare an asymmetric G128 run, export `ASYM_ARTIFACT` as its hierarchy
directory and `ASYM_RESULT_PREFIX` as the stem before `_{bits}.json` in
`matgptq_protocol`. Missing control results appear as null values. Resubmitting
a phase skips validated stages; a corrupt or conflicting completed stage
stops with an error. The full phase requires the pilot encoding and does not
re-submit or modify the existing LP jobs.

WikiText-2 uses `Salesforce/wikitext`, `wikitext-2-raw-v1` test text joined
with double newlines, no added special tokens, context 2048, stride 512,
and the original baseline window construction: each step ends at `i+stride`
and begins up to 2048 tokens before that end. Valid shifted labels are counted
for each window when weighting its loss. The reference BF16 run scored
299,077 of 299,078 tokens and reached perplexity 6.219182234107349.
Zero shot evaluation uses the existing Hugging Face backend of
`lm-evaluation-harness==0.4.13` for ARC Easy, ARC Challenge, PIQA,
WinoGrande, and HellaSwag. The complete harness result is retained, including
accuracy, normalized accuracy, standard errors, task versions, and settings.
The evaluation command accepts either an original Hugging Face Safetensors
checkpoint or a completed LP reconstruction. Completed JSON results are reused
only for the same checkpoint contents and protocol. Evaluation hashes every
Safetensors shard to establish that identity, adding bounded-memory disk I/O
before a run or a resume check.

Full 8B correctness, peak RAM, disk throughput, reconstructed model loading,
and task quality still require AWS validation. The local synthetic tests
exercise streaming behavior but cannot prove the 16 GB host RAM target or the
22 GiB GPU target. Huffman coding, other quantizer families, Llama selection,
and activation quantization are deferred.
