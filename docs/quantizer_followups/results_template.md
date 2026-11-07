# Results template

| Experiment | Method | Symmetric | Group | Calibration | PPL | Effective BPW | Status |
| --- | --- | --- | ---: | --- | ---: | ---: | --- |
| J01 | Original BF16 | — | — | — | 7.01 | — | validated prior result |
| J04 | Plain RTN | yes | 128 | no | 9928.00 | unknown | validated prior result |
| J05 | Optimized RTN | yes | 128 | yes | 65.62 | unknown | validated prior result |
| PR04 | RRQ PR base | yes | 128 | yes | 65.62 | see PR04 artifact | validated prior result |
| S01 | SignRoundV2 | yes | 128 | yes | pending | unknown | not run |
| R01 | NeUQI optimized RTN | yes | 128 | yes | pending | unknown | not run |
| R02 | Asymmetric NeUQI optimized RTN | no | 128 | yes | pending | unknown | not run |
| R03 | GPTQModel GPTQ | yes | 128 | yes | pending | pending packed audit | not run |

Record actual quantization/evaluation runtime, sampled process-tree RSS, GPU memory, checkpoint bytes, 252-module coverage, checkpoint dtype distribution, calibration fingerprint, and deviations. For S01/R01/R02, a fake-QDQ checkpoint's physical bytes describe dense inference storage, not a self-contained 2-bit packed representation. Do not publish an effective compressed BPW unless code, scale, zero and sidecar storage have all been accounted for.
