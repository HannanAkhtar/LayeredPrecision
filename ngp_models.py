"""tiny-cuda-nn HashGrid models used for the NGP image/audio experiments."""

from __future__ import annotations

import torch
import torch.nn as nn


PAPER_NGP_CONFIGS = {
    "image": {
        "n_levels": 10,
        "n_features_per_level": 2,
        "log2_hashmap_size": 13,
        "base_resolution": 16,
        "per_level_scale": 2.0,
        "n_neurons": 32,
        "n_hidden_layers": 2,
    },
    "audio": {
        "n_levels": 16,
        "n_features_per_level": 2,
        "log2_hashmap_size": 11,
        "base_resolution": 8,
        "per_level_scale": 2.0,
        "n_neurons": 32,
        "n_hidden_layers": 2,
    },
}


def _network(input_dims: int, output_dims: int, config: dict):
    try:
        import tinycudann as tcnn
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The NGP experiments require tiny-cuda-nn's Python bindings. "
            "Install tinycudann in the Linux CUDA environment first."
        ) from exc
    return tcnn.NetworkWithInputEncoding(
        n_input_dims=input_dims,
        n_output_dims=output_dims,
        encoding_config={
            "otype": "HashGrid",
            "n_levels": config["n_levels"],
            "n_features_per_level": config["n_features_per_level"],
            "log2_hashmap_size": config["log2_hashmap_size"],
            "base_resolution": config["base_resolution"],
            "per_level_scale": config["per_level_scale"],
        },
        network_config={
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": config["n_neurons"],
            "n_hidden_layers": config["n_hidden_layers"],
        },
    )


class HashGridAudioINR(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.net = _network(1, 1, config)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        values = self.net(coordinates.contiguous().float()).float()
        return torch.tanh(values)


class HashGridImageINR(nn.Module):
    def __init__(self, out_channels: int = 3, **config):
        super().__init__()
        self.net = _network(2, out_channels, config)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        values = self.net(coordinates.contiguous().float()).float()
        return torch.sigmoid(values)


def build_ngp_model(modality: str, config: dict, out_channels: int = 1) -> nn.Module:
    if modality == "audio":
        return HashGridAudioINR(**config)
    if modality == "image":
        return HashGridImageINR(out_channels=out_channels, **config)
    raise ValueError(f"Unknown NGP modality: {modality}")
