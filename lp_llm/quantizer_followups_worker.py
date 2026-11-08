"""Isolated upstream quantization subprocesses. Never import either reference in LP."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def autoround(config: dict, source: Path, export: Path, resolved: Path) -> None:
    import torch
    if config["id"] in ("r01", "r02"):
        os.environ["AR_NEUQI_COARSE"] = str(config["neuqi_coarse"])
        os.environ["AR_NEUQI_FINE"] = str(config["neuqi_fine"])
        os.environ["AR_NEUQI_BACKEND"] = config["neuqi_backend"]
    import auto_round
    from auto_round import AutoRound
    from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig
    from auto_round.algorithms.quantization.sign_round.config import SignRoundV2Config
    from auto_round.data_type.utils import get_quant_func
    from auto_round.algorithms.registry import resolve_pipeline_member

    kind = config["id"]
    if kind == "s01":
        algorithm = SignRoundV2Config(bits=2, group_size=128, sym=True, data_type="int", iters=200)
        expected_class = "SignRoundV2Quantizer"
        selected_function = None
    else:
        algorithm = OptimizedRTNConfig(bits=2, group_size=128, sym=config["symmetric"],
                                       data_type="int", disable_opt_rtn=False, enable_neuqi=True)
        function, selected_function = get_quant_func("int", 2, config["symmetric"],
                                                     disable_opt_rtn=False, group_size=128,
                                                     iters=0, enable_neuqi=True)
        expected = "opt_rtn_int_sym_neuqi" if kind == "r01" else "opt_rtn_int_asym"
        if selected_function != expected or function.__name__ not in {
                "quant_tensor_opt_rtn_sym_neuqi", "quant_tensor_opt_rtn_asym"}:
            raise RuntimeError(f"Wrong optimized RTN dispatch: {selected_function}, {function.__name__}")
        expected_class = "OptimizedRTNQuantizer"
    if kind == "s01":
        # Verify registry mapping, not only the config object's class.
        quantizer_cls = resolve_pipeline_member(algorithm)
        if quantizer_cls.__name__ != expected_class:
            raise RuntimeError(f"Wrong SignRoundV2 dispatch: {quantizer_cls}")
    record = {"reference_module": str(Path(auto_round.__file__).resolve()),
              "config_class": type(algorithm).__name__, "expected_quantizer_class": expected_class,
              "selected_quant_function": selected_function,
              "effective_scheme": {"bits": 2, "group_size": 128, "sym": config["symmetric"],
                                   "data_type": "int", "act_bits": 16},
              "requested_calibration": {"dataset": config["calibration_dataset"],
                  "nsamples": 128, "sequence_length": 2048, "batch_size": 8, "seed": 42},
              "algorithm": {"iters": algorithm.iters if hasattr(algorithm, "iters") else 0,
                            "enable_neuqi": algorithm.enable_neuqi,
                            "disable_opt_rtn": getattr(algorithm, "disable_opt_rtn", None),
                            "need_calib": algorithm.need_calib}}
    if kind in ("r01", "r02"):
        record["neuqi_grid"] = {"coarse": config["neuqi_coarse"], "fine": config["neuqi_fine"],
                                "backend": config["neuqi_backend"]}
    if kind == "s01":
        record["algorithm"].update({key: getattr(algorithm, key) for key in (
            "lr", "minmax_lr", "enable_minmax_tuning", "not_use_best_mse",
            "enable_quanted_input", "enable_alg_ext", "enable_lfq")})
    # Explicit scheme prevents an algorithm config from changing the quantization grid.
    engine = AutoRound(model=str(source), scheme={"bits": 2, "group_size": 128,
        "sym": config["symmetric"], "data_type": "int", "act_bits": 16},
        alg_configs=algorithm, format="fake", dataset=config["calibration_dataset"],
        nsamples=128, seqlen=2048, batch_size=8, seed=42, model_dtype="float16",
        scale_dtype="fp16", device_map="cuda:0", low_gpu_mem_usage=True)
    selected = [item for item in engine._alg_configs if isinstance(item, type(algorithm))]
    if len(selected) != 1 or resolve_pipeline_member(selected[0]).__name__ != expected_class:
        raise RuntimeError(f"AutoRound replaced algorithm config: {engine._alg_configs}")
    actual = selected[0]
    if (actual.bits != 2 or actual.group_size != 128 or actual.sym is not config["symmetric"] or
            getattr(actual, "iters", 0) != config["iterations"] or
            actual.enable_neuqi is not config["enable_neuqi"]):
        raise RuntimeError("AutoRound resolved settings differ from the pinned experiment")
    record["resolved_config_class"] = type(actual).__name__
    record["resolved_quantizer_class"] = resolve_pipeline_member(actual).__name__
    record["resolved_algorithm"] = {key: getattr(actual, key, None) for key in (
        "iters", "lr", "minmax_lr", "enable_minmax_tuning", "not_use_best_mse",
        "enable_quanted_input", "enable_alg_ext", "enable_lfq", "enable_neuqi",
        "disable_opt_rtn", "bits", "group_size", "sym", "scale_dtype")}
    loaded = getattr(engine, "model", None)
    first = next(loaded.parameters(), None) if isinstance(loaded, torch.nn.Module) else None
    record["observed_first_loaded_weight_dtype"] = str(first.dtype) if first is not None else None
    resolved.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    engine.quantize_and_save(str(export))


def materialize_gptq_qdq(packed_model, target: Path) -> int:
    """Decode pinned GPTQModel v5.7 weights without inventing bias parameters."""
    import torch
    from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear

    if target.exists():
        raise FileExistsError(f"QDQ target already exists: {target}")
    model = packed_model.model
    modules = dict(model.named_modules())
    names = [name for name, module in modules.items() if isinstance(module, TorchQuantLinear)]
    for name in names:
        module = modules[name]
        parent_name, field = name.rsplit(".", 1)
        parent = modules[parent_name]
        linear = torch.nn.Linear(module.in_features, module.out_features,
                                 bias=module.bias is not None, device="cpu", dtype=torch.float16)
        linear.weight = torch.nn.Parameter(module.dequantize_weight().T.detach().to(
            device="cpu", dtype=torch.float16).contiguous())
        if module.bias is not None:
            linear.bias = torch.nn.Parameter(module.bias.detach().to(device="cpu", dtype=torch.float16))
        setattr(parent, field, linear)
    if hasattr(model.config, "quantization_config"):
        del model.config.quantization_config
    model.save_pretrained(target)
    packed_model.tokenizer.save_pretrained(target)
    return len(names)


def gptq(config: dict, source: Path, export: Path, resolved: Path) -> None:
    import torch
    import gptqmodel
    from datasets import load_dataset
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.quantization import QuantizeConfig
    from transformers import AutoTokenizer

    qcfg = QuantizeConfig(bits=2, group_size=128, sym=True, desc_act=False,
                     act_group_aware=True, damp_percent=0.05,
                     damp_auto_increment=0.01, static_groups=False,
                     true_sequential=True)
    actual = {key: getattr(qcfg, key) for key in ("bits", "group_size", "sym", "desc_act",
        "act_group_aware", "damp_percent", "damp_auto_increment", "static_groups",
        "true_sequential", "format")}
    actual["format"] = getattr(actual["format"], "value", actual["format"])
    if actual["bits"] != 2 or actual["group_size"] != 128 or actual["sym"] is not True:
        raise RuntimeError(f"Wrong GPTQ configuration: {actual}")
    # Save the packed checkpoint first. The pinned upstream HF exporter adds an
    # empty bias Parameter to biasless Qwen3 projections, so materialize from
    # its decoder while preserving the original bias inventory.
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=False)
    dataset = load_dataset("NeelNanda/pile-10k", split="train")
    dataset = dataset.shuffle(seed=42)
    natural, short = [], []
    for row in dataset:
        value = row.get("text")
        if not isinstance(value, str):
            continue
        ids = tokenizer(value, truncation=True, max_length=2048)["input_ids"]
        if len(ids) >= 2048:
            natural.append(ids[:2048])
        else:
            short.append(ids)
    packed, buffer = [], []
    if len(natural) < 128:
        for ids in short:
            buffer.extend(ids)
            if len(buffer) >= 2048:
                packed.append(buffer[:2048])
                buffer = []
            if len(natural) + len(packed) >= 128:
                break
    calibration = [{"input_ids": ids, "attention_mask": [1] * 2048}
                   for ids in (natural + packed)[:128]]
    if len(calibration) != 128:
        raise RuntimeError(f"Only {len(calibration)} calibration samples reached 2048 tokens")
    import hashlib
    fingerprint = hashlib.sha256()
    for item in calibration:
        fingerprint.update(torch.tensor(item["input_ids"], dtype=torch.int32).numpy().tobytes())
    resolved.write_text(json.dumps({"reference_module": str(Path(gptqmodel.__file__).resolve()),
        "config": {key: str(value) for key, value in actual.items()},
        "hessian": str(qcfg.hessian), "block_size": 128,
        "calibration_sample_count": len(calibration), "calibration_token_sha256": fingerprint.hexdigest(),
        "calibration_construction": "HF shuffle(seed=42), tokenize/truncate to 2048, natural full sequences first, concatenate short samples as fallback; first 128",
        "backend": "TORCH", "model_dtype": "float16",
        "materializer": "pinned TorchQuantLinear.dequantize_weight, preserving bias inventory"}, indent=2), encoding="utf-8")
    model = GPTQModel.load(str(source), qcfg, device="cuda:0", dtype=torch.float16)
    loaded_first = next(model.model.parameters(), None)
    details = json.loads(resolved.read_text())
    details["observed_first_loaded_weight_dtype"] = str(loaded_first.dtype) if loaded_first is not None else None
    resolved.write_text(json.dumps(details, indent=2), encoding="utf-8")
    model.quantize(calibration, batch_size=1, backend=BACKEND.TORCH,
                   calibration_sort=None, calibration_data_min_length=2048)
    from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
    quantized_modules = {name: module for name, module in model.model.named_modules()
                         if isinstance(module, TorchQuantLinear)}
    if len(quantized_modules) != 252:
        raise RuntimeError(f"GPTQ quantized {len(quantized_modules)} modules instead of 252")
    expected_qdq_hashes = {}
    for name, module in quantized_modules.items():
        decoded = module.dequantize_weight().T.detach().to(device="cpu", dtype=torch.float16).contiguous()
        expected_qdq_hashes[name + ".weight"] = hashlib.sha256(decoded.numpy().tobytes()).hexdigest()
        del decoded
    packed = export / "packed"
    model.save(str(packed))
    del model
    torch.cuda.empty_cache()
    reloaded = GPTQModel.load(str(packed), backend=BACKEND.TORCH)
    if materialize_gptq_qdq(reloaded, export / "qdq") != 252:
        raise RuntimeError("GPTQ QDQ materialization did not cover 252 projections")
    del reloaded
    torch.cuda.empty_cache()
    from safetensors import safe_open
    qdq = export / "qdq"
    index = qdq / "model.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text())["weight_map"]
    else:
        with safe_open(qdq / "model.safetensors", framework="pt", device="cpu") as handle:
            mapping = {name: "model.safetensors" for name in handle.keys()}
    for name, expected_hash in expected_qdq_hashes.items():
        with safe_open(qdq / mapping[name], framework="pt", device="cpu") as handle:
            actual = handle.get_tensor(name)
        if actual.dtype != torch.float16 or hashlib.sha256(actual.numpy().tobytes()).hexdigest() != expected_hash:
            raise RuntimeError(f"Packed TorchQuantLinear and exported QDQ disagree: {name}")
    details = json.loads(resolved.read_text())
    details["packed_to_qdq_exact_projection_parity"] = True
    details["packed_to_qdq_projection_count"] = len(expected_qdq_hashes)
    details["packed_to_qdq_hashes"] = expected_qdq_hashes
    resolved.write_text(json.dumps(details, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("config", "source", "export", "resolved"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["id"] == "r03":
        gptq(config, args.source, args.export, args.resolved)
    else:
        autoround(config, args.source, args.export, args.resolved)


if __name__ == "__main__":
    main()
