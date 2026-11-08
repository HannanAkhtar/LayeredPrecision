# Qwen3-8B-Base W2G128 numerical audit

The comparison target is the verified J05/PR04 MatGPTQ result, 65.62 PPL. The RRQ paper's 16.75 RTN and 11.29 SignRoundV2 figures are published reference claims, not outcomes of these follow-up jobs. No new full-model experiment has run here.

## Pinned implementations and version effect

J04/J05 use AutoRound `6c824f741be358983baba87aff83f9c5aa0a01e7`. PR03/PR04 use AutoRound `7985c1b0f4eef65fb717511668fda7e347e584c3`. The latter source is also pinned for S01, R01 and R02. PR04's 65.62 agrees with J05, so it supplies the same-revision base comparator for R01. R02 shares R01's revision, calibration policy, and NeUQI search size/backend policy (`256/64`, `auto`). R02 versus R01 isolates the symmetric/asymmetric rule; R02 versus PR04 changes both NeUQI and symmetry. The AutoRound version difference versus historical J05 is unavoidable for NeUQI because its opt-in route is in the later PR source. The version gap should remain visible even though PR04/J05 output parity was reported.

R03 now pins independent GPTQModel v5.7.0 commit `ed96f2e04ec868b04f969733a242ce3b5bb7cb1e`, whose declared Torch >=2.8.0 and Transformers >=4.57.1 requirements fit the AUS CUDA 12.8-compatible Torch 2.9.1+cu128 / Transformers 4.57.1 environment. The prior pin `70579a771423767da2af57b67d2d3adf014eebb6` requires Torch >=2.14.1 and Transformers >=5.20.2 and failed AUS preflight because its CUDA build needs a newer driver. The v5.7 model definition explicitly lists seven Qwen3 projection types and excludes Q/K norms. Its official HF dequantizer assigns `nn.Parameter(module.bias)` even for biasless projections, creating empty bias tensors. The R03 materializer therefore uses the same pinned `TorchQuantLinear.dequantize_weight()` but only creates bias where the packed module has one. It verifies all 252 decoded projection tensors against the pre-save decode. This local materialization difference and the GPTQModel version change from the original R03 plan are recorded; R03 remains a different upstream method from AutoRound.

## Integer mapping and objective

| Path | W2 integer mapping | Scale/zero choice | Calibration and updates |
| --- | --- | --- | --- |
| J04 plain RTN | Signed `[-2,-1,0,1]` or sign-mirrored via negative scale; `zero=2` storage convention | Per output row and 128 input columns; min/max signed anchor, FP16 scale floor `1e-5` | No calibration; no gradients |
| J05 optimized RTN | Same signed grid | `search_scales`: signed max anchor; 181 candidates indexed -90..90 in 0.01 steps, activation-imatrix weighted squared reconstruction error | 128 pile-10k examples prepared/cached; zero gradient updates |
| PR03/PR04 RRQ base | Packed INT2 GPTQ-zero convention, decoded into the same signed grid | Activation-imatrix weighted W2 optimized RTN in RRQ compressor; PR04 compares decoded weights to J05 | Calibrated base; no gradient updates |
| R01 symmetric NeUQI | Signed `[-2,-1,0,1]`, with negative scale capable of the mirrored clamp orientation; return zero=2 | Two-stage log/additive scale search; fixed 256 coarse and 64 fine candidates; FP32 search and FP16 returned scale | Same requested 128 pile-10k examples; zero gradient updates |
| R02 asymmetric NeUQI | Codes `[0,1,2,3]`, affine `scale*(code-zero)` with integer zero in `[0,3]` | Joint scale/integer-zero search, fixed 256/64 grid, FP32 objective and FP16 exported scale/zero | Same requested 128 pile-10k examples; zero gradient updates |
| R03 GPTQ | Packed unsigned `[0,1,2,3]`, symmetric zero=2, decoded signed `[-2,-1,0,1]` | Group min/max initialized scales; Hessian-based error compensation, 128-column block, damp 0.05 with 0.01 increment | 128 pile-10k sequences; no gradient descent, but second-order activation information |

The mapping comes from pinned `auto_round/data_type/int.py`, `neuqi.py`, `utils.py`, `algorithms/quantization/rrq/quantizer.py`, and GPTQModel `quantization/quantizer.py` and `gptq.py`. All methods group along input columns within each output row. The symmetric grid is asymmetric in its number of negative and positive nonzero levels; `sym=True` does not imply `{-1.5,-0.5,0.5,1.5}`.

The pinned PR dispatcher explicitly removes asymmetric optimized RTN when `enable_neuqi=False` at zero iterations. Merely setting `sym=False` would run plain min/max RTN. R02 therefore enables NeUQI so its selected function is `opt_rtn_int_asym`; R01 selects `opt_rtn_int_sym_neuqi`. The executable dispatch test checks both selections and the fallback.

## SignRoundV2 and GPTQ constraints

`SignRoundV2Config` maps to `SignRoundV2Quantizer` in the pinned registry. At W2 symmetric, V2 selects `SignRoundOptimizedWrapperLinear` and suppresses the highest 0.1% of block-output error entries in its loss. Its source defaults include `iters=200`, min/max tuning enabled, quantized block inputs enabled, best-MSE checkpoint restoration enabled, and no LFQ. The default learning rate is `1/iters` for 200 steps, with `minmax_lr` falling back to it; the runner records resolved fields. These are audited implementation defaults. We have not established that they exactly match the paper's training recipe, particularly all loss filtering and data-order details; S01 should be called a pinned-implementation SignRoundV2 test until verified.

GPTQModel v5.7.0's `QuantizeConfig` supports 2-bit symmetric group 128 and Qwen3. This experiment pins `desc_act=False`, `act_group_aware=True`, `static_groups=False`, `true_sequential=True`, FP32 Hessian staging, `damp_percent=0.05`, `damp_auto_increment=0.01`, and block size 128. Its range initialization uses zero=2 for W2 symmetric and maps an all-zero group to a nonzero scale before quantization. The materializer decodes `TorchQuantLinear` into FP16 and removes `quantization_config` from the dense config. R03 still needs a completed AUS GPU smoke and full-model runtime and memory validation.

## Quantization-time versus exported QDQ

AutoRound's fake export writes dense FP16/BF16 tensors. It may omit scale/zero arrays and module-selection metadata. The checkpoint validator verifies complete tensor inventory, original architecture, excluded tensors after source-to-output dtype conversion, finite weights, 252 changed projection matrices, every output row/group's four-level affine structure, and representative symmetric signed-grid checks. Changed weights are evidence of quantization, not definitive module-selection metadata. An affine QDQ grid cannot recover the exact pre-export code, scale, or zero point in underdetermined groups. The full dense checkpoint bytes are physically measurable; a 2-bit compressed footprint is not measured from a fake export.

GPTQModel saves a packed model before its HF QDQ export, so packed bytes and its tensor names are inspectable. Its packed qweight, scales, qzeros and g_idx provide a path for an independent full-code audit on AUS. The Python API's QDQ export is the implementation's own dequantization path, but this local CPU pass did not load a full packed 8B model. A packed-kernel evaluation must not be assumed numerically identical to MatGPTQ FP16 QDQ evaluation.

## Most plausible sources of a 16.75 versus 65.62 gap

The largest algorithmic differences left are calibration-weighted scale selection, NeUQI's broader search, asymmetric zero placement, SignRoundV2 gradient tuning and loss filtering, and GPTQ Hessian correction. Dataset sample selection and model-loading/scale dtype can also shift a W2 result. An alternative reaching 16.75 would show that this accuracy is attainable under its own recipe; it would not validate the paper's claimed calibration-free RTN base.
