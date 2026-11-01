#!/usr/bin/env python3
"""Affine integer post-training quantization for Layered Precision.

The paper experiments use one quantizer family only: unsigned affine INT with
an explicit integer zero point. A base layer quantizes the original tensor;
each enhancement layer independently quantizes the residual left by the
decoded prefix.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Sequence

import torch


def integer_dtype(bits: int) -> torch.dtype:
    """Return the smallest PyTorch integer container used by the experiments."""
    if bits <= 0:
        raise ValueError(f"bits must be positive, got {bits}")
    if bits <= 8:
        return torch.uint8
    if bits <= 16:
        return torch.uint16
    return torch.int32


class IntegerFormat:
    """Unsigned affine INT quantization with an explicit zero point.

    ``tensor`` uses one range for a complete parameter tensor. ``channel``
    uses one range per output-channel slice (axis 0 in the paper experiments).
    """

    VALID_GRANULARITIES = ("tensor", "channel")

    def __init__(self, granularity: str = "tensor", channel_axis: int = 0):
        granularity = granularity.lower()
        if granularity not in self.VALID_GRANULARITIES:
            raise ValueError(
                f"Unknown INT granularity {granularity!r}; "
                f"expected one of {self.VALID_GRANULARITIES}"
            )
        self.granularity = granularity
        self.channel_axis = channel_axis
        self.use_zero_point = True

    def _range(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int | None]:
        if self.granularity == "tensor" or tensor.dim() <= 1:
            t_min = tensor.min()
            t_max = tensor.max()
            axis = None
        else:
            axis = self.channel_axis
            if axis < 0:
                axis += tensor.dim()
            if axis < 0 or axis >= tensor.dim():
                raise ValueError(
                    f"channel_axis {self.channel_axis} is invalid for "
                    f"shape {tuple(tensor.shape)}"
                )
            reduce_dims = tuple(dim for dim in range(tensor.dim()) if dim != axis)
            t_min = tensor.amin(dim=reduce_dims, keepdim=True)
            t_max = tensor.amax(dim=reduce_dims, keepdim=True)

        # Including zero guarantees that the affine integer zero point is valid.
        zero = torch.zeros_like(t_min)
        return torch.minimum(t_min, zero), torch.maximum(t_max, zero), axis

    def quantize(
        self, tensor: torch.Tensor, bits: int
    ) -> tuple[dict, torch.Tensor]:
        qmax = 2**bits - 1
        quant_dtype = integer_dtype(bits)
        t_min, t_max, axis = self._range(tensor)
        scale = (t_max - t_min) / qmax
        non_constant = scale > 0

        safe_scale_base = torch.where(non_constant, scale, torch.ones_like(scale))
        zero_point = torch.where(
            non_constant,
            (-t_min / safe_scale_base).round().clamp(0, qmax),
            torch.zeros_like(scale),
        )

        if scale.dim() > 0:
            expanded_scale = scale.expand_as(tensor)
            expanded_zero_point = zero_point.expand_as(tensor)
            expanded_non_constant = non_constant.expand_as(tensor)
            expanded_min = t_min.expand_as(tensor)
        else:
            expanded_scale = scale
            expanded_zero_point = zero_point
            expanded_non_constant = non_constant
            expanded_min = t_min

        safe_scale = torch.where(
            expanded_non_constant,
            expanded_scale,
            torch.ones_like(expanded_scale),
        )
        quantized = (
            tensor / safe_scale + expanded_zero_point
        ).round().clamp(0, qmax)
        quantized = torch.where(
            expanded_non_constant,
            quantized,
            torch.zeros_like(quantized),
        )
        dequantized = expanded_scale * (quantized - expanded_zero_point)
        dequantized = torch.where(
            expanded_non_constant,
            dequantized,
            expanded_min,
        )

        selected_granularity = (
            "channel" if self.granularity == "channel" and tensor.dim() > 1
            else "tensor"
        )
        params = {
            "format": "int",
            "bits": bits,
            "scale": scale,
            "zero_point": zero_point.to(quant_dtype),
            "quantized": quantized.to(quant_dtype),
            "shape": tensor.shape,
            "t_min": t_min,
            "granularity": self.granularity,
            "selected_granularity": selected_granularity,
            "axis": axis,
            "channel_axis": self.channel_axis,
            "use_zero_point": True,
        }
        return params, dequantized

    @staticmethod
    def dequantize(params: dict) -> torch.Tensor:
        quantized = params["quantized"].float()
        scale = params["scale"]
        zero_point = params["zero_point"].float()
        t_min = params.get("t_min", torch.zeros_like(scale))

        if scale.dim() > 0:
            scale = scale.expand_as(quantized)
            zero_point = zero_point.expand_as(quantized)
            t_min = t_min.expand_as(quantized)

        reconstructed = scale * (quantized - zero_point)
        return torch.where(scale > 0, reconstructed, t_min)


class LayeredQuantizer:
    """Build and decode a base layer followed by residual INT layers."""

    def __init__(self, int_granularity: str = "tensor", int_channel_axis: int = 0):
        self.int_granularity = int_granularity
        self.int_channel_axis = int_channel_axis
        self.int_use_zero_point = True
        self.integer = IntegerFormat(int_granularity, int_channel_axis)

    def quantize_layered(
        self,
        weights: torch.Tensor,
        bits_per_layer: Sequence[int],
        formats_per_layer: Sequence[str] | None = None,
    ) -> dict:
        bits = list(bits_per_layer)
        formats = list(formats_per_layer or ["int"] * len(bits))
        if not bits:
            raise ValueError("At least one precision layer is required")
        if len(bits) != len(formats):
            raise ValueError("bits_per_layer and formats_per_layer must match")
        if any(fmt != "int" for fmt in formats):
            raise ValueError("The paper release supports affine INT only")

        residual = weights.clone()
        layers_info: list[dict] = []
        for layer_idx, layer_bits in enumerate(bits):
            params, contribution = self.integer.quantize(residual, layer_bits)
            signal_power = (residual**2).mean()
            noise_power = ((residual - contribution) ** 2).mean()
            if noise_power > 0 and signal_power > 0:
                sqnr_db = 10 * torch.log10(signal_power / noise_power).item()
            else:
                sqnr_db = float("inf") if noise_power == 0 else 0.0

            layers_info.append({
                "layer_idx": layer_idx,
                "bits": layer_bits,
                "format": "int",
                "quant_params": params,
                "mse_error": torch.nn.functional.mse_loss(
                    residual, contribution
                ).item(),
                "sqnr_db": sqnr_db,
            })
            if layer_idx + 1 < len(bits):
                residual = residual - contribution

        reconstructed = self.reconstruct_from_layers(layers_info)
        noise_power = ((weights - reconstructed) ** 2).mean()
        signal_power = (weights**2).mean()
        if noise_power > 0 and signal_power > 0:
            total_sqnr_db = 10 * torch.log10(signal_power / noise_power).item()
        else:
            total_sqnr_db = float("inf") if noise_power == 0 else 0.0

        return {
            "original_shape": weights.shape,
            "num_layers": len(bits),
            "layers_info": layers_info,
            "total_mse_error": torch.nn.functional.mse_loss(
                weights, reconstructed
            ).item(),
            "total_sqnr_db": total_sqnr_db,
            "total_bits": sum(bits),
        }

    @staticmethod
    def reconstruct_from_layers(layers_info: Sequence[dict]) -> torch.Tensor:
        if not layers_info:
            raise ValueError("Cannot reconstruct an empty precision prefix")
        contributions = [
            IntegerFormat.dequantize(layer["quant_params"])
            for layer in layers_info
        ]
        reconstructed = contributions[0]
        for contribution in contributions[1:]:
            reconstructed = reconstructed + contribution
        return reconstructed

    def apply_layered_quantization_to_model(
        self,
        model: torch.nn.Module,
        bits_per_layer: Sequence[int],
        formats_per_layer: Sequence[str] | None = None,
        exclude_patterns: Sequence[str] = ("bias", "norm"),
        include_1d_parameters: bool = False,
        inplace: bool = False,
    ) -> tuple[torch.nn.Module, dict]:
        target_model = model if inplace else deepcopy(model)
        all_quant_info: dict[str, dict] = {}
        total_params = 0

        for name, parameter in target_model.named_parameters():
            if not parameter_is_quantized(
                name,
                parameter,
                exclude_patterns,
                include_1d_parameters,
            ):
                continue
            total_params += parameter.numel()
            info = self.quantize_layered(
                parameter.data,
                bits_per_layer,
                formats_per_layer,
            )
            parameter.data.copy_(self.reconstruct_from_layers(info["layers_info"]))
            all_quant_info[name] = info

        config_formats = list(
            formats_per_layer or ["int"] * len(bits_per_layer)
        )
        summary = {
            "total_parameters": total_params,
            "quantized_parameters": total_params,
            "coverage": 1.0 if total_params else 0.0,
            "per_parameter_info": all_quant_info,
            "config": {
                "bits_per_layer": list(bits_per_layer),
                "formats_per_layer": config_formats,
                "exclude_patterns": list(exclude_patterns),
                "include_1d_parameters": bool(include_1d_parameters),
                "int_granularity": self.int_granularity,
                "int_channel_axis": self.int_channel_axis,
                "int_use_zero_point": True,
            },
        }
        return target_model, summary


def parameter_is_quantized(
    name: str,
    parameter: torch.nn.Parameter,
    exclude_patterns: Sequence[str],
    include_1d_parameters: bool,
) -> bool:
    lower_name = name.lower()
    if any(pattern.lower() in lower_name for pattern in exclude_patterns):
        return False
    return (
        parameter.requires_grad
        and parameter.numel() > 0
        and (parameter.dim() > 1 or (include_1d_parameters and parameter.dim() == 1))
    )


def apply_partial_reconstruction(
    model: torch.nn.Module,
    per_parameter_info: dict[str, dict],
    layers_to_use: int,
) -> None:
    """Load one decoded LP prefix into ``model`` in place."""
    if layers_to_use <= 0:
        raise ValueError("layers_to_use must be positive")
    for name, parameter in model.named_parameters():
        info = per_parameter_info.get(name)
        if info is None:
            continue
        layers = info["layers_info"][:layers_to_use]
        parameter.data.copy_(LayeredQuantizer.reconstruct_from_layers(layers))
