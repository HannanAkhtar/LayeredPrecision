"""Dataset selection and signal preprocessing for per-sample NGP training."""

from __future__ import annotations

import re
from pathlib import Path
from collections.abc import Iterable

import numpy as np
import torch
import torchaudio
from PIL import Image


AUDIO_EXTENSIONS = {".flac", ".wav", ".mp3", ".ogg", ".m4a"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def safe_id(text: str) -> str:
    text = text.replace("\\", "/")
    text = re.sub(r"[^A-Za-z0-9._/-]+", "_", text)
    return text.strip("/").replace("/", "__")[:160] or "sample"


def audio_to_target(
    waveform: torch.Tensor,
    sample_rate: int,
    target_rate: int = 16000,
    start_sec: float = 0.0,
    duration_sec: float = 5.0,
    allow_short: bool = False,
) -> dict | None:
    if sample_rate != target_rate:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, target_rate
        )
        sample_rate = target_rate
    waveform = waveform.mean(dim=0)
    start = max(0, int(round(start_sec * sample_rate)))
    requested = int(round(duration_sec * sample_rate))
    clip = waveform[start:min(waveform.numel(), start + requested)]
    if clip.numel() == 0 or (clip.numel() < requested and not allow_short):
        return None
    peak = clip.abs().max()
    if peak > 0:
        clip = clip / peak
    return {
        "grid": torch.linspace(0.0, 1.0, steps=clip.numel()).view(-1, 1).float(),
        "target": clip.view(-1, 1).float(),
        "sample_rate": int(sample_rate),
        "num_samples": int(clip.numel()),
        "normalization": "peak",
    }


def load_audio(
    path: Path,
    sample_rate: int = 16000,
    start_sec: float = 0.0,
    duration_sec: float = 5.0,
    allow_short: bool = False,
) -> dict | None:
    waveform, source_rate = torchaudio.load(str(path))
    return audio_to_target(
        waveform,
        source_rate,
        sample_rate,
        start_sec,
        duration_sec,
        allow_short,
    )


def iter_librispeech(
    root: Path,
    max_samples: int = 10,
    seed: int = 123,
    download: bool = False,
) -> Iterable[dict]:
    from torchaudio.datasets import LIBRISPEECH

    if download:
        root.mkdir(parents=True, exist_ok=True)
    dataset = LIBRISPEECH(root=str(root), url="dev-clean", download=download)
    indices = list(range(len(dataset)))
    np.random.default_rng(seed).shuffle(indices)
    yielded = 0
    for index in indices:
        rel_path, rate, transcript, speaker, chapter, utterance = (
            dataset.get_metadata(index)
        )
        waveform, _, _, _, _, _ = dataset[index]
        target = audio_to_target(waveform, rate)
        if target is None:
            continue
        source_path = Path(rel_path)
        if not source_path.is_absolute():
            source_path = root / "LibriSpeech" / source_path
        yield {
            **target,
            "sample_id": safe_id(
                f"ls_dev-clean_{speaker}_{chapter}_{utterance}"
            ),
            "source_path": str(source_path.resolve()),
            "source_rel_path": str(rel_path),
            "transcript": transcript,
            "speaker_id": int(speaker),
            "chapter_id": int(chapter),
            "utterance_id": int(utterance),
            "dataset": "librispeech/dev-clean",
            "start_sec": 0.0,
            "duration_sec": 5.0,
            "requested_sample_rate": 16000,
            "allow_short": False,
        }
        yielded += 1
        if max_samples and yielded >= max_samples:
            return


def iter_audio_folder(
    root: Path,
    max_samples: int,
    seed: int,
) -> Iterable[dict]:
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )
    np.random.default_rng(seed).shuffle(files)
    yielded = 0
    for path in files:
        target = load_audio(path)
        if target is None:
            continue
        yield {
            **target,
            "sample_id": safe_id(path.relative_to(root).as_posix()),
            "source_path": str(path.resolve()),
            "source_rel_path": path.relative_to(root).as_posix(),
            "dataset": "audio_folder",
            "start_sec": 0.0,
            "duration_sec": 5.0,
            "requested_sample_rate": 16000,
            "allow_short": False,
        }
        yielded += 1
        if max_samples and yielded >= max_samples:
            return


def load_image(path: Path, max_image_side: int = 512) -> dict:
    image = Image.open(path).convert("RGB")
    original_size = image.size
    if max_image_side > 0 and max(image.size) > max_image_side:
        factor = max_image_side / float(max(image.size))
        image = image.resize(
            (
                max(1, int(round(image.size[0] * factor))),
                max(1, int(round(image.size[1] * factor))),
            ),
            Image.Resampling.LANCZOS,
        )
    array = np.asarray(image, dtype=np.float32) / 255.0
    height, width, channels = array.shape
    cols = torch.linspace(0.0, 1.0, width)
    rows = torch.linspace(0.0, 1.0, height)
    grid_v, grid_u = torch.meshgrid(rows, cols, indexing="ij")
    grid = torch.stack((grid_u.reshape(-1), grid_v.reshape(-1)), dim=1)
    return {
        "grid": grid.float(),
        "target": torch.from_numpy(array).reshape(-1, channels).float(),
        "height": height,
        "width": width,
        "channels": channels,
        "original_width": original_size[0],
        "original_height": original_size[1],
    }


def iter_images(
    root: Path,
    max_samples: int = 100,
    seed: int = 123,
) -> Iterable[dict]:
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    np.random.default_rng(seed).shuffle(files)
    if max_samples:
        files = files[:max_samples]
    for path in files:
        yield {
            **load_image(path),
            "sample_id": safe_id(path.relative_to(root).as_posix()),
            "source_path": str(path.resolve()),
            "source_rel_path": path.relative_to(root).as_posix(),
            "dataset": "div2k",
            "grayscale": False,
            "max_image_side": 512,
        }


def resolve_source_path(
    sample: dict,
    saved_root: str | None = None,
    data_root: Path | None = None,
) -> Path:
    candidates = [Path(sample["source_path"])]
    rel_path = sample.get("source_rel_path")
    if data_root is not None:
        if rel_path:
            candidates.extend([
                data_root / rel_path,
                data_root / "LibriSpeech" / rel_path,
            ])
        candidates.append(data_root / sample["sample_id"])
    if saved_root:
        root = Path(saved_root)
        if rel_path:
            candidates.extend([
                root / rel_path,
                root / "LibriSpeech" / rel_path,
            ])
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    raise FileNotFoundError(
        "Could not locate source sample. Checked: "
        + ", ".join(str(path) for path in candidates)
    )
