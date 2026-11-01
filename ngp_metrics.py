"""Reconstruction metrics used by the NGP dataset experiment."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def snr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    signal_power = torch.sum(target**2)
    noise_power = torch.sum((target - prediction) ** 2)
    if noise_power <= 0:
        return float("inf")
    return 10 * torch.log10(signal_power / noise_power).item()


def si_snr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    target = target.view(-1)
    prediction = prediction.view(-1)
    target_energy = torch.sum(target**2)
    if target_energy < eps:
        return float("-inf")
    alpha = torch.sum(prediction * target) / (target_energy + eps)
    projected = alpha * target
    noise = prediction - projected
    signal_power = torch.sum(projected**2)
    noise_power = torch.sum(noise**2)
    if noise_power < eps:
        return float("inf")
    return 10 * torch.log10(signal_power / noise_power).item()


def log_spectral_distance(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    target = target.cpu().view(-1)
    prediction = prediction.cpu().view(-1)
    n_fft, hop = 2048, 512
    if target.numel() < n_fft:
        n_fft = max(256, 2 ** int(math.floor(math.log2(max(target.numel(), 2)))))
        hop = max(64, n_fft // 4)
    window = torch.hann_window(n_fft)
    target_stft = torch.stft(
        target, n_fft=n_fft, hop_length=hop, window=window, return_complex=True
    )
    pred_stft = torch.stft(
        prediction, n_fft=n_fft, hop_length=hop, window=window, return_complex=True
    )
    target_log = 10 * torch.log10(torch.abs(target_stft) ** 2 + 1e-10)
    pred_log = 10 * torch.log10(torch.abs(pred_stft) ** 2 + 1e-10)
    floor = torch.max(target_log) - 50.0
    target_log = torch.clamp(target_log, min=floor)
    pred_log = torch.clamp(pred_log, min=floor)
    return torch.mean(
        torch.sqrt(torch.mean((target_log - pred_log) ** 2, dim=0))
    ).item()


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((prediction - target) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return -10.0 * math.log10(mse)


def ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    height: int,
    width: int,
) -> float:
    channels = prediction.shape[1]
    kernel_size, sigma = 11, 1.5
    coords = (
        torch.arange(kernel_size, dtype=torch.float32, device=prediction.device)
        - kernel_size // 2
    )
    gaussian = torch.exp(-(coords**2) / (2 * sigma**2))
    gaussian = gaussian / gaussian.sum()
    kernel = (
        (gaussian.unsqueeze(1) * gaussian.unsqueeze(0))
        .unsqueeze(0)
        .unsqueeze(0)
    )

    def blur(image):
        return F.conv2d(
            image,
            kernel.expand(image.shape[1], -1, -1, -1),
            padding=kernel_size // 2,
            groups=image.shape[1],
        )

    pred_image = prediction.reshape(height, width, channels).permute(2, 0, 1).unsqueeze(0)
    target_image = target.reshape(height, width, channels).permute(2, 0, 1).unsqueeze(0)
    mu_pred, mu_target = blur(pred_image), blur(target_image)
    pred_var = blur(pred_image**2) - mu_pred**2
    target_var = blur(target_image**2) - mu_target**2
    covariance = blur(pred_image * target_image) - mu_pred * mu_target
    numerator = (
        (2 * mu_pred * mu_target + 0.01**2)
        * (2 * covariance + 0.03**2)
    )
    denominator = (
        (mu_pred**2 + mu_target**2 + 0.01**2)
        * (pred_var + target_var + 0.03**2)
    )
    return (numerator / denominator).mean().item()
