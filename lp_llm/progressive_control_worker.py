"""Isolated pinned AutoRound W4/W6/W8 controls, never used for LP residuals."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


BITS = (4, 6, 8)


def resolved_recipe(branch: str, bits: int) -> tuple[object, dict]:
    if branch not in ("s01", "r02") or bits not in BITS:
        raise ValueError("Unsupported monolithic control identity")
    from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig
    from auto_round.algorithms.quantization.sign_round.config import SignRoundV2Config
    from auto_round.algorithms.registry import resolve_pipeline_member
    from auto_round.data_type.utils import get_quant_func

    if branch == "s01":
        algorithm = SignRoundV2Config(bits=bits, group_size=128, sym=True,
                                      data_type="int", iters=200)
        selected = resolve_pipeline_member(algorithm).__name__
        function = None
        if selected != "SignRoundV2Quantizer":
            raise ValueError("Pinned SignRoundV2 quantizer dispatch changed")
    else:
        algorithm = OptimizedRTNConfig(bits=bits, group_size=128, sym=False,
                                       data_type="int", disable_opt_rtn=False,
                                       enable_neuqi=True)
        function, selected = get_quant_func("int", bits, False,
                                            disable_opt_rtn=False, group_size=128,
                                            iters=0, enable_neuqi=True)
        if (selected != "opt_rtn_int_asym" or
                function.__name__ != "quant_tensor_opt_rtn_asym" or
                resolve_pipeline_member(algorithm).__name__ != "OptimizedRTNQuantizer"):
            raise ValueError("Pinned asymmetric NeUQI dispatch is unavailable")
    if (algorithm.bits != bits or algorithm.group_size != 128 or
            algorithm.sym is not (branch == "s01") or
            getattr(algorithm, "iters", 0) != (200 if branch == "s01" else 0)):
        raise ValueError("Pinned control settings changed")
    return algorithm, {"branch": branch, "bits": bits, "group_size": 128,
        "sym": branch == "s01", "algorithm_class": type(algorithm).__name__,
        "quantizer_class": resolve_pipeline_member(algorithm).__name__,
        "selected_quant_function": selected if branch == "r02" else None,
        "function_name": function.__name__ if function is not None else None,
        "iterations": getattr(algorithm, "iters", 0),
        "enable_neuqi": bool(getattr(algorithm, "enable_neuqi", False)),
        "need_calib": algorithm.need_calib,
        "data_type": "int", "model_dtype": "float16", "scale_dtype": "fp16",
        "calibration": {"dataset": "NeelNanda/pile-10k", "nsamples": 128,
                        "seqlen": 2048, "batch_size": 8, "seed": 42},
        "rounding": "upstream round_ste forward", "numerical_format": "integer_fake_qdq",
        "scale_selection": "SignRoundV2 optimized scale/round offsets" if branch == "s01"
                           else "NeUQI joint scale/integer-zero coarse-fine search",
        "algorithm_defaults": {key: getattr(algorithm, key, None) for key in
            ("lr", "minmax_lr", "enable_minmax_tuning", "not_use_best_mse",
             "enable_quanted_input", "enable_alg_ext", "enable_lfq", "disable_opt_rtn")},
        "neuqi_grid": {"coarse": 256, "fine": 64} if branch == "r02" else None}


def run(branch: str, bits: int, source: Path, export: Path, resolved: Path) -> None:
    if branch == "r02":
        os.environ.update(AR_NEUQI_COARSE="256", AR_NEUQI_FINE="64", AR_NEUQI_BACKEND="auto")
    import auto_round
    from auto_round import AutoRound
    from auto_round.algorithms.registry import resolve_pipeline_member

    algorithm, record = resolved_recipe(branch, bits)
    record["reference_module"] = str(Path(auto_round.__file__).resolve())
    engine = AutoRound(model=str(source), scheme={"bits": bits, "group_size": 128,
        "sym": branch == "s01", "data_type": "int", "act_bits": 16},
        alg_configs=algorithm, format="fake", dataset="NeelNanda/pile-10k",
        nsamples=128, seqlen=2048, batch_size=8, seed=42, model_dtype="float16",
        scale_dtype="fp16", device_map="cuda:0", low_gpu_mem_usage=True)
    selected = [item for item in engine._alg_configs if isinstance(item, type(algorithm))]
    if len(selected) != 1 or resolve_pipeline_member(selected[0]).__name__ != record["quantizer_class"]:
        raise ValueError("AutoRound replaced monolithic control algorithm")
    actual = selected[0]
    if (actual.bits != bits or actual.group_size != 128 or actual.sym is not record["sym"] or
            getattr(actual, "iters", 0) != record["iterations"] or
            bool(getattr(actual, "enable_neuqi", False)) is not record["enable_neuqi"]):
        raise ValueError("AutoRound resolved a different monolithic control")
    record["resolved"] = {key: getattr(actual, key, None) for key in
        ("bits", "group_size", "sym", "iters", "lr", "minmax_lr",
         "enable_minmax_tuning", "not_use_best_mse", "enable_quanted_input",
         "enable_alg_ext", "enable_lfq", "enable_neuqi", "disable_opt_rtn")}
    resolved.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    engine.quantize_and_save(str(export))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-only", action="store_true")
    parser.add_argument("--branch", choices=("s01", "r02"))
    parser.add_argument("--bits", type=int, choices=BITS)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--resolved", type=Path)
    args = parser.parse_args()
    if args.support_only:
        selected = [(branch, bits) for branch in ((args.branch,) if args.branch else ("s01", "r02"))
                    for bits in ((args.bits,) if args.bits else BITS)]
        print(json.dumps({f"{branch}_w{bits}": resolved_recipe(branch, bits)[1]
                          for branch, bits in selected}, indent=2))
        return
    if None in (args.branch, args.bits, args.source, args.export, args.resolved):
        parser.error("quantization requires --branch, --bits, --source, --export, --resolved")
    run(args.branch, args.bits, args.source, args.export, args.resolved)


if __name__ == "__main__":
    main()
