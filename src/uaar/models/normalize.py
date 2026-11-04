"""Per-channel z-score normalization with statistics from the training subjects of a fold."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def fit_stats(X_train: np.ndarray) -> dict:
    """Channel means and standard deviations of ``X_train`` float32 ``(N, C, L)``.

    Returns ``{"mean": [C], "std": [C]}`` (std floored at 1e-6).
    """
    mean = X_train.mean(axis=(0, 2), dtype=np.float64)
    std = X_train.std(axis=(0, 2), dtype=np.float64)
    return {"mean": mean.tolist(), "std": np.maximum(std, 1e-6).tolist()}


def apply(X: np.ndarray, stats: dict) -> np.ndarray:
    """``(X - mean) / std`` per channel, returned as float32 ``(N, C, L)``."""
    mean = np.asarray(stats["mean"], dtype=np.float32)[None, :, None]
    std = np.asarray(stats["std"], dtype=np.float32)[None, :, None]
    return ((X - mean) / std).astype(np.float32)


def save(stats: dict, path: Path, **extra) -> None:
    """Write ``preprocess.json`` (stats plus window length, rate, channel names ...)."""
    Path(path).write_text(json.dumps({**stats, **extra}, indent=2))


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())
