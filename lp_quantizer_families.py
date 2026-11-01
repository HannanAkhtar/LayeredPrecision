"""Prespecified four-code, per-output-channel LP residual quantizers.

A and P delegate to the original implementations. S and L are fixed signed
four-level grids. N is deterministic Lloyd-Max with quantile initialization.
No calibration uses images, labels, gradients, or task loss.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch

from quantizer_variants import dequantize_variant, quantize_variant


FAMILIES = ("A", "P", "S", "N", "L")
FAMILY_NAMES = {"A": "affine_minmax", "P": "affine_percentile",
                "S": "symmetric_midrise", "N": "lloyd_max", "L": "logarithmic_ratio4"}
FIXED_LEVELS = {"S": (-1.5, -0.5, 0.5, 1.5),
                "L": (-4.0, -1.0, 1.0, 4.0)}


def _codes_from_centers(matrix: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    thresholds = (centers[:, :-1] + centers[:, 1:]) * 0.5
    # Strict greater-than sends exact midpoint ties to the lower code.
    return ((matrix > thresholds[:, 0:1]).to(torch.uint8) +
            (matrix > thresholds[:, 1:2]).to(torch.uint8) +
            (matrix > thresholds[:, 2:3]).to(torch.uint8))


@torch.no_grad()
def quantize_family(residual: torch.Tensor, family: str, *, max_iterations: int = 20,
                    tolerance: float = 1e-6) -> tuple[dict, torch.Tensor]:
    if family not in FAMILIES:
        raise ValueError(f"Unknown family {family!r}")
    if residual.ndim < 2 or not residual.is_floating_point() or not torch.isfinite(residual).all():
        raise ValueError("Expected finite floating-point channel-axis-0 residual")
    if family in ("A", "P"):
        method = "affine_minmax" if family == "A" else "affine_percentile"
        params, decoded = quantize_variant(residual, method, bits=2, channel_axis=0,
                                           clip_percentile=99.5)
        params["family"] = family
        return params, decoded
    matrix = residual.reshape(residual.shape[0], -1)
    channels = matrix.shape[0]
    if family in ("S", "L"):
        maxabs = matrix.abs().amax(dim=1)
        scale = maxabs / (1.5 if family == "S" else 4.0)
        factors = torch.tensor(FIXED_LEVELS[family], device=residual.device,
                               dtype=residual.dtype)
        centers = scale[:, None] * factors[None, :]
        codes = _codes_from_centers(matrix, centers)
        decoded = centers.gather(1, codes.long()).reshape_as(residual)
        view = (channels,) + (1,) * (residual.ndim - 1)
        return {"family": family, "quantized": codes.reshape_as(residual),
                "scale": scale.reshape(view), "shape": tuple(residual.shape),
                "bits": 2, "axis": 0, "zero_exact_for_nonzero_scale": False,
                "global_magnitude_ratio": 4 if family == "L" else None}, decoded
    if max_iterations <= 0 or tolerance < 0:
        raise ValueError("Invalid Lloyd iteration settings")
    # Four quantile centroids provide reproducible channel-local initialization.
    ordered = matrix.sort(dim=1).values
    positions = torch.tensor([.125, .375, .625, .875], device=matrix.device,
                             dtype=torch.float64) * (matrix.shape[1] - 1)
    left = positions.floor().long()
    right = positions.ceil().long()
    fraction = (positions - left).to(matrix.dtype)
    centers = ordered[:, left] * (1 - fraction) + ordered[:, right] * fraction
    centers = centers.contiguous()
    history = []
    iterations = 0
    for iteration in range(max_iterations):
        codes = _codes_from_centers(matrix, centers)
        updated = []
        for code in range(4):
            selected = codes == code
            count = selected.sum(dim=1)
            total = torch.where(selected, matrix, 0).sum(dim=1)
            updated.append(torch.where(count > 0, total / count.clamp_min(1),
                                       centers[:, code]))
        new_centers = torch.stack(updated, dim=1).sort(dim=1).values
        new_codes = _codes_from_centers(matrix, new_centers)
        decoded = new_centers.gather(1, new_codes.long())
        objective = (matrix.double() - decoded.double()).square().sum().item()
        history.append(objective)
        change = (new_centers - centers).abs().amax(dim=1)
        channel_scale = matrix.abs().amax(dim=1)
        centers = new_centers
        iterations = iteration + 1
        if bool((change <= tolerance * channel_scale.clamp_min(torch.finfo(matrix.dtype).tiny)).all()):
            break
    codes = _codes_from_centers(matrix, centers).reshape_as(residual)
    decoded = centers.gather(1, codes.reshape(channels, -1).long()).reshape_as(residual)
    duplicate = (centers[:, 1:] == centers[:, :-1]).any(dim=1)
    return {"family": family, "quantized": codes, "centers": centers,
            "shape": tuple(residual.shape), "bits": 2, "axis": 0,
            "iterations": iterations, "objective_history": history,
            "duplicate_centroid_channels": int(duplicate.sum().item()),
            "converged": iterations < max_iterations}, decoded


def dequantize_family(params: dict, family: str | None = None) -> torch.Tensor:
    family = family or params.get("family")
    if family in ("A", "P"):
        return dequantize_variant(params)
    codes = params["quantized"]
    channels = codes.shape[0]
    if family in ("S", "L"):
        factors = torch.tensor(FIXED_LEVELS[family], device=codes.device,
                               dtype=params["scale"].dtype)
        centers = params["scale"].reshape(channels, 1) * factors[None, :]
    elif family == "N":
        centers = params["centers"]
    else:
        raise ValueError(f"Unknown family {family!r}")
    return centers.gather(1, codes.reshape(channels, -1).long()).reshape_as(codes)


def pack_two_bit_codes(codes: np.ndarray) -> bytes:
    flat = np.asarray(codes, dtype=np.uint8).reshape(-1)
    if np.any(flat > 3):
        raise ValueError("2-bit codes must be in [0, 3]")
    padded = np.pad(flat, (0, (-len(flat)) % 4))
    blocks = padded.reshape(-1, 4)
    return (blocks[:, 0] | (blocks[:, 1] << 2) |
            (blocks[:, 2] << 4) | (blocks[:, 3] << 6)).tobytes()


def unpack_two_bit_codes(payload: bytes, count: int) -> np.ndarray:
    data = np.frombuffer(payload, dtype=np.uint8)
    codes = np.stack([(data >> shift) & 3 for shift in (0, 2, 4, 6)], axis=1)
    return codes.reshape(-1)[:count].copy()


def write_binary_codebook(path: Path, family: str, artifacts: dict[str, dict]) -> dict:
    """Write deterministic, losslessly decodable packed codes and FP32 metadata.

    Format: LPQ2 magic, one family byte, uint16 tensor count; each tensor has
    uint16 UTF-8 name bytes, uint8 ndim, uint32 shape values, uint32 channels,
    uint64 code count, packed little-endian 2-bit codes, then per-channel
    metadata (A/P: FP32 scale + uint8 zero point; S/L: FP32 scale; N: four
    FP32 centers). The global L ratio is fixed by the family identifier.
    """
    if family not in FAMILIES:
        raise ValueError(family)
    raw_bytes = metadata_bytes = framing_bytes = 0
    with path.open("wb") as handle:
        header = b"LPQ2" + family.encode("ascii") + struct.pack("<H", len(artifacts))
        handle.write(header)
        framing_bytes += len(header)
        for name in sorted(artifacts):
            params = artifacts[name]
            codes = params["quantized"].detach().cpu().numpy()
            shape = codes.shape
            channels = shape[0]
            name_bytes = name.encode("utf-8")
            frame = (struct.pack("<H", len(name_bytes)) + name_bytes +
                     struct.pack("<B", len(shape)) + struct.pack("<" + "I" * len(shape), *shape) +
                     struct.pack("<IQ", channels, codes.size))
            handle.write(frame)
            framing_bytes += len(frame)
            payload = pack_two_bit_codes(codes)
            handle.write(payload)
            raw_bytes += len(payload)
            if family in ("A", "P"):
                scale = params["scale"].reshape(-1).detach().cpu().numpy().astype("<f4")
                zp = params["zero_point"].reshape(-1).detach().cpu().numpy().astype("u1")
                metadata = scale.tobytes() + zp.tobytes()
            elif family in ("S", "L"):
                metadata = params["scale"].reshape(-1).detach().cpu().numpy().astype("<f4").tobytes()
            else:
                metadata = params["centers"].detach().cpu().numpy().astype("<f4").tobytes()
            handle.write(metadata)
            metadata_bytes += len(metadata)
    size = path.stat().st_size
    if size != raw_bytes + metadata_bytes + framing_bytes:
        raise AssertionError("Binary byte accounting differs from physical file")
    return {"raw_code_bytes": raw_bytes, "metadata_bytes": metadata_bytes,
            "framing_bytes": framing_bytes, "physical_bytes": size,
            "unquantized_shared_model_bytes": None,
            "entropy_coded": False}


def read_binary_codebook(path: Path) -> tuple[str, dict[str, dict]]:
    data = memoryview(path.read_bytes())
    if bytes(data[:4]) != b"LPQ2":
        raise ValueError("Invalid codebook magic")
    family = bytes(data[4:5]).decode("ascii")
    if family not in FAMILIES:
        raise ValueError("Invalid family")
    position = 5
    count = struct.unpack_from("<H", data, position)[0]
    position += 2
    result = {}
    for _ in range(count):
        name_len = struct.unpack_from("<H", data, position)[0]
        position += 2
        name = bytes(data[position:position+name_len]).decode("utf-8")
        position += name_len
        ndim = data[position]
        position += 1
        shape = struct.unpack_from("<" + "I"*ndim, data, position)
        position += 4*ndim
        channels, numel = struct.unpack_from("<IQ", data, position)
        position += 12
        payload_len = (numel + 3)//4
        codes = unpack_two_bit_codes(data[position:position+payload_len], numel).reshape(shape)
        position += payload_len
        params = {"family": family, "quantized": torch.from_numpy(codes),
                  "shape": shape, "bits": 2, "axis": 0}
        if family in ("A", "P"):
            scale = np.frombuffer(data[position:position+4*channels], dtype="<f4").copy()
            position += 4*channels
            zp = np.frombuffer(data[position:position+channels], dtype="u1").copy()
            position += channels
            view = (channels,) + (1,)*(ndim-1)
            params["scale"] = torch.from_numpy(scale.reshape(view))
            params["zero_point"] = torch.from_numpy(zp.reshape(view))
            params["t_min"] = torch.zeros(view, dtype=torch.float32)
        elif family in ("S", "L"):
            scale = np.frombuffer(data[position:position+4*channels], dtype="<f4").copy()
            position += 4*channels
            params["scale"] = torch.from_numpy(scale.reshape((channels,)+(1,)*(ndim-1)))
        else:
            centers = np.frombuffer(data[position:position+16*channels], dtype="<f4").copy()
            position += 16*channels
            params["centers"] = torch.from_numpy(centers.reshape(channels, 4))
        result[name] = params
    if position != len(data):
        raise ValueError("Trailing or truncated codebook bytes")
    return family, result
