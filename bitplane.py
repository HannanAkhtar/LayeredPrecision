"""Affine bit-plane truncation baseline used in the ImageNet experiment."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from layered_quantization import IntegerFormat, integer_dtype, parameter_is_quantized


def _right_shift(values: torch.Tensor, shift: int) -> torch.Tensor:
    if shift <= 0:
        return values.to(torch.int64)
    return torch.div(values.to(torch.int64), 1 << shift, rounding_mode="floor")


def truncate_affine_params(source_params: dict, target_bits: int) -> dict:
    """Keep the most significant affine integer planes from one source code.

    This reproduces the paper baseline: integer codes and zero points are
    shifted together, while the source scale is enlarged by ``2**shift``.
    """
    source_bits = int(source_params["bits"])
    if target_bits <= 0 or target_bits > source_bits:
        raise ValueError(
            f"target_bits must be in [1, {source_bits}], got {target_bits}"
        )
    shift = source_bits - target_bits
    target = dict(source_params)
    target.update({
        "bits": target_bits,
        "source_bits": source_bits,
        "dropped_bits": shift,
        "bitplane_decode_mode": "shift",
        "quantized": _right_shift(source_params["quantized"], shift).to(
            integer_dtype(target_bits)
        ),
        "zero_point": _right_shift(source_params["zero_point"], shift).to(
            integer_dtype(target_bits)
        ),
        "scale": source_params["scale"] * (1 << shift),
    })
    return target


def build_bitplane_database(
    model: torch.nn.Module,
    source_bits: int,
    granularity: str,
    channel_axis: int,
    exclude_patterns: Sequence[str],
    include_1d_parameters: bool = False,
) -> dict:
    quantizer = IntegerFormat(granularity, channel_axis)
    per_parameter_info = {}
    total_params = 0

    for name, parameter in model.named_parameters():
        if not parameter_is_quantized(
            name,
            parameter,
            exclude_patterns,
            include_1d_parameters,
        ):
            continue
        total_params += parameter.numel()
        params, dequantized = quantizer.quantize(parameter.data, source_bits)
        signal_power = (parameter.data**2).mean()
        noise_power = ((parameter.data - dequantized) ** 2).mean()
        if noise_power > 0 and signal_power > 0:
            sqnr_db = 10 * torch.log10(signal_power / noise_power).item()
        else:
            sqnr_db = float("inf") if noise_power == 0 else 0.0
        per_parameter_info[name] = {
            "source_bits": source_bits,
            "source_format": "int",
            "source_quant_params": params,
            "source_mse_error": torch.nn.functional.mse_loss(
                parameter.data, dequantized
            ).item(),
            "source_sqnr_db": sqnr_db,
        }

    return {
        "source_bits": source_bits,
        "total_parameters": total_params,
        "quantized_parameters": total_params,
        "coverage": 1.0 if total_params else 0.0,
        "per_parameter_info": per_parameter_info,
        "config": {
            "source_bits": source_bits,
            "format": "int",
            "decode_mode": "shift",
            "exclude_patterns": list(exclude_patterns),
            "include_1d_parameters": bool(include_1d_parameters),
            "int_granularity": granularity,
            "int_channel_axis": channel_axis,
            "int_use_zero_point": True,
        },
    }


def apply_bitplane_reconstruction(
    model: torch.nn.Module,
    database: dict,
    target_bits: int,
) -> dict:
    """Decode one retained-bit operating point into ``model`` in place."""
    target_info = {
        "source_bits": database["source_bits"],
        "target_bits": target_bits,
        "decode_mode": "shift",
        "total_parameters": database["total_parameters"],
        "quantized_parameters": database["quantized_parameters"],
        "coverage": database["coverage"],
        "per_parameter_info": {},
    }
    for name, parameter in model.named_parameters():
        source = database["per_parameter_info"].get(name)
        if source is None:
            continue
        target_params = truncate_affine_params(
            source["source_quant_params"],
            target_bits,
        )
        reconstructed = IntegerFormat.dequantize(target_params).to(
            dtype=parameter.dtype,
            device=parameter.device,
        )
        parameter.data.copy_(reconstructed)
        target_info["per_parameter_info"][name] = {
            "source_bits": database["source_bits"],
            "target_bits": target_bits,
            "layers_info": [{
                "layer_idx": 0,
                "bits": target_bits,
                "format": "int",
                "quant_params": target_params,
            }],
        }
    return target_info
