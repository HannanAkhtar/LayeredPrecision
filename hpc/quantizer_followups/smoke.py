"""Exercise the pinned upstream quantizer on a tiny local Qwen3 CUDA model."""

from __future__ import annotations

import argparse
import json
import resource
import tempfile
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", choices=("s01", "r01", "r02", "r03"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "GPU smoke test requires CUDA"
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    tiny = Qwen3ForCausalLM(Qwen3Config(vocab_size=256, hidden_size=128,
        intermediate_size=256, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32, max_position_embeddings=64,
        tie_word_embeddings=False)).eval().half()
    inputs = [{"input_ids": torch.randint(0, 256, (32,), dtype=torch.long)} for _ in range(2)]
    with tempfile.TemporaryDirectory(prefix=f"followup-{args.experiment_id}-") as temporary:
        tmp = Path(temporary)
        if args.experiment_id == "r03":
            from gptqmodel import BACKEND, GPTQModel
            from gptqmodel.quantization import GPTQConfig

            tiny.save_pretrained(tmp / "model")
            AutoTokenizer.from_pretrained(args.source, use_fast=False).save_pretrained(tmp / "model")
            qcfg = GPTQConfig(bits=2, group_size=128, sym=True, desc_act=False,
                act_group_aware=True, damp_percent=0.05, damp_auto_increment=0.01,
                static_groups=False, true_sequential=True)
            model = GPTQModel.load(str(tmp / "model"), qcfg, device="cuda:0", dtype=torch.float16)
            model.quantize(inputs, batch_size=1, backend=BACKEND.GPTQ_TORCH,
                           calibration_sort=None, calibration_data_min_length=32)
            from gptqmodel.nn_modules.qlinear.torch import TorchLinear
            quantized_modules = {name: module for name, module in model.model.named_modules()
                                 if isinstance(module, TorchLinear)}
            assert len(quantized_modules) == 7, f"Expected seven quantized projections, got {len(quantized_modules)}"
            sampled_name, sampled_module = next(iter(quantized_modules.items()))
            expected_qdq = sampled_module.dequantize_weight().T.detach().cpu().half()
            model.save(str(tmp / "packed"))
            assert (tmp / "packed/config.json").is_file()
            del model
            torch.cuda.empty_cache()
            GPTQModel.export(str(tmp / "packed"), str(tmp / "qdq"), format="hf")
            from safetensors import safe_open
            from lp_llm.rrq_pr2308_compare import _weight_map
            weight_name = sampled_name + ".weight"
            mapping = _weight_map(tmp / "qdq")
            with safe_open(tmp / "qdq" / mapping[weight_name], framework="pt", device="cpu") as handle:
                actual_qdq = handle.get_tensor(weight_name)
            assert torch.equal(actual_qdq, expected_qdq), "GPTQ packed and exported QDQ differ"
        else:
            import os
            if args.experiment_id in ("r01", "r02"):
                os.environ["AR_NEUQI_COARSE"] = "256"
                os.environ["AR_NEUQI_FINE"] = "64"
            from auto_round import AutoRound
            from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig
            from auto_round.algorithms.quantization.sign_round.config import SignRoundV2Config
            tokenizer = AutoTokenizer.from_pretrained(args.source, use_fast=False)
            alg = (SignRoundV2Config(bits=2, group_size=128, sym=True, data_type="int", iters=2)
                   if args.experiment_id == "s01" else
                   OptimizedRTNConfig(bits=2, group_size=128, sym=args.experiment_id == "r01",
                                      data_type="int", disable_opt_rtn=False, enable_neuqi=True))
            loader = torch.utils.data.DataLoader(inputs, batch_size=1)
            engine = AutoRound(model=tiny, tokenizer=tokenizer,
                scheme={"bits": 2, "group_size": 128, "sym": args.experiment_id != "r02",
                        "data_type": "int", "act_bits": 16},
                alg_configs=alg, dataset=loader, nsamples=2, seqlen=32, batch_size=1,
                model_dtype="float16", format="fake", device_map="cuda:0", low_gpu_mem_usage=True)
            engine.quantize_and_save(str(tmp / "export"))
            assert any((tmp / "export").rglob("*.safetensors"))
    print(json.dumps({"status": "smoke_quantization_completed", "experiment_id": args.experiment_id,
        "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "note": "Tiny one-layer exercise; not an 8B resource fit proof"}, indent=2))


if __name__ == "__main__":
    main()
