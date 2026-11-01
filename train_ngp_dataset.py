#!/usr/bin/env python3
"""Train one paper-configured HashGrid INR per DIV2K image or audio clip."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from PIL import Image

from ngp_data import iter_images, iter_librispeech
from ngp_metrics import psnr, snr
from ngp_models import PAPER_NGP_CONFIGS, build_ngp_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the per-sample NGP models used in the paper."
    )
    subparsers = parser.add_subparsers(dest="modality", required=True)
    for modality in ("image", "audio"):
        sub = subparsers.add_parser(modality)
        sub.add_argument("--root", required=True)
        sub.add_argument("--outdir", required=True)
        sub.add_argument("--max-samples", type=int, default=None)
        sub.add_argument("--steps", type=int, default=5000)
        sub.add_argument("--lr", type=float, default=1e-2)
        sub.add_argument("--seed", type=int, default=123)
        sub.add_argument("--log-every", type=int, default=0)
        sub.add_argument("--resume", action="store_true")
        sub.add_argument("--save-preview", action="store_true")
    subparsers.choices["audio"].add_argument(
        "--download",
        action="store_true",
        help="Download LibriSpeech dev-clean through torchaudio",
    )
    return parser.parse_args()


def save_preview(prediction: torch.Tensor, sample: dict, path: Path) -> None:
    if "height" in sample:
        array = (
            prediction.detach()
            .cpu()
            .clamp(0, 1)
            .reshape(sample["height"], sample["width"], sample["channels"])
            .numpy()
        )
        Image.fromarray((array * 255).round().astype(np.uint8)).save(path)
    else:
        torchaudio.save(
            str(path),
            prediction.detach().cpu().view(1, -1),
            sample["sample_rate"],
        )


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, path)


def training_metadata(args: argparse.Namespace) -> dict:
    maximum = args.max_samples
    if maximum is None:
        maximum = 100 if args.modality == "image" else 10
    return {
        "schema_version": 1,
        "release": "paper_release",
        "modality": args.modality,
        "root": str(Path(args.root).resolve()),
        "max_samples": maximum,
        "steps": args.steps,
        "learning_rate": args.lr,
        "seed": args.seed,
        "ngp_config": PAPER_NGP_CONFIGS[args.modality],
        "image_max_side": 512 if args.modality == "image" else None,
        "audio_sample_rate": 16000 if args.modality == "audio" else None,
        "audio_duration_sec": 5.0 if args.modality == "audio" else None,
        "audio_normalization": "peak" if args.modality == "audio" else None,
        "torch_version": torch.__version__,
        "torchaudio_version": torchaudio.__version__,
    }


def prepare_training_run(args: argparse.Namespace, outdir: Path) -> None:
    metadata = training_metadata(args)
    path = outdir / "run_metadata.json"
    if args.resume and path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous != metadata:
            raise RuntimeError(
                "Cannot resume: saved training metadata differs. "
                "Use a new output directory."
            )
    elif args.resume and (
        (outdir / "manifest.json").exists()
        or next(outdir.rglob("best_model.pth"), None) is not None
    ):
        raise RuntimeError(
            "Cannot safely resume existing checkpoints without run_metadata.json"
        )
    atomic_json(path, metadata)


def write_manifest(rows: list[dict], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    atomic_json(outdir / "manifest.json", {"samples": rows})
    fields = ["sample_id", "status", "checkpoint", "loss", "best_loss", "metric"]
    path = outdir / "manifest.csv"
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    os.replace(tmp, path)


def train_one(
    sample: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    sample_dir = Path(args.outdir) / sample["sample_id"]
    checkpoint_path = sample_dir / "best_model.pth"
    if args.resume and checkpoint_path.exists():
        print(f"Skipping {sample['sample_id']} (checkpoint exists)")
        return {
            "sample_id": sample["sample_id"],
            "status": "skipped",
            "checkpoint": str(checkpoint_path),
        }

    sample_dir.mkdir(parents=True, exist_ok=True)
    config = dict(PAPER_NGP_CONFIGS[args.modality])
    model = build_ngp_model(
        args.modality,
        config,
        out_channels=int(sample.get("channels", 1)),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    grid = sample["grid"].to(device)
    target = sample["target"].to(device)
    log_every = args.log_every or max(1, args.steps // 10)
    best_loss = float("inf")
    best_step = 0

    print(
        f"\n[{sample['sample_id']}] {args.modality} "
        f"points={grid.shape[0]} steps={args.steps}"
    )
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(grid)
        loss = F.mse_loss(prediction, target)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_step = step
        if step % log_every == 0 or step == args.steps:
            with torch.no_grad():
                quality = (
                    snr(prediction, target)
                    if args.modality == "audio"
                    else psnr(prediction.clamp(0, 1), target.clamp(0, 1))
                )
            metric_name = "SNR" if args.modality == "audio" else "PSNR"
            print(
                f"  step {step}/{args.steps} loss={loss_value:.6g} "
                f"{metric_name}={quality:.2f} dB"
            )

    model.eval()
    with torch.no_grad():
        prediction = model(grid)
        final_loss = float(F.mse_loss(prediction, target).item())
        final_metric = (
            snr(prediction, target)
            if args.modality == "audio"
            else psnr(prediction.clamp(0, 1), target.clamp(0, 1))
        )
    if args.save_preview:
        suffix = ".wav" if args.modality == "audio" else ".png"
        save_preview(prediction, sample, sample_dir / f"preview{suffix}")

    sample_metadata = {
        key: value for key, value in sample.items()
        if key not in {"grid", "target"}
    }
    checkpoint = {
        "step": args.steps,
        "best_step": best_step,
        "loss": final_loss,
        "best_loss": best_loss,
        "modality": args.modality,
        "model": "ngp",
        "model_state_dict": model.state_dict(),
        "ngp_config": config,
        "args": vars(args),
        "sample": sample_metadata,
        "coordinate_range": [0.0, 1.0],
    }
    torch.save(checkpoint, checkpoint_path)
    atomic_json(sample_dir / "train_metrics.json", {
        "sample_id": sample["sample_id"],
        "checkpoint": str(checkpoint_path),
        "loss": final_loss,
        "best_loss": best_loss,
        "metric": final_metric,
        "metric_name": "snr_db" if args.modality == "audio" else "psnr_db",
    })

    del model, optimizer, grid, target, prediction
    torch.cuda.empty_cache()
    return {
        "sample_id": sample["sample_id"],
        "status": "success",
        "checkpoint": str(checkpoint_path),
        "loss": final_loss,
        "best_loss": best_loss,
        "metric": final_metric,
    }


def selected_samples(args: argparse.Namespace) -> Iterable[dict]:
    maximum = args.max_samples
    if args.modality == "image":
        return iter_images(
            Path(args.root),
            max_samples=100 if maximum is None else maximum,
            seed=args.seed,
        )
    return iter_librispeech(
        Path(args.root),
        max_samples=10 if maximum is None else maximum,
        seed=args.seed,
        download=args.download,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("NGP training requires CUDA and tiny-cuda-nn")
    if args.steps <= 0 or args.lr <= 0:
        raise ValueError("steps and learning rate must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max samples must be positive")
    if args.log_every < 0:
        raise ValueError("log frequency cannot be negative")
    root = Path(args.root)
    if not root.exists() and not (
        args.modality == "audio" and getattr(args, "download", False)
    ):
        raise FileNotFoundError(f"Dataset root not found: {root}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prepare_training_run(args, outdir)
    print(f"Device: {device}")
    print(f"Output: {outdir}")
    print(f"NGP config: {PAPER_NGP_CONFIGS[args.modality]}")

    rows: list[dict] = []
    for sample in selected_samples(args):
        rows.append(train_one(sample, args, device))
        write_manifest(rows, outdir)
    if not rows:
        raise RuntimeError("No usable samples found")
    write_manifest(rows, outdir)
    print(f"\nSaved {outdir / 'manifest.json'}")


if __name__ == "__main__":
    main()
