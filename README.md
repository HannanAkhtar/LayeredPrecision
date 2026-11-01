# Layered Precision Reproducibility Code

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
