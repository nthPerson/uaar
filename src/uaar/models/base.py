"""The interface every method implements (development-spec §4.3)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass
class Prediction:
    probs: np.ndarray  # (N, K) predictive probabilities
    label: np.ndarray  # (N,) argmax over the K known classes
    scores: dict[str, np.ndarray]  # name -> (N,), higher = more uncertain


class Method(Protocol):
    name: str
    score_names: list[str]

    def fit(self, Xtr, ytr, Xval, yval, cfg, run_dir: Path) -> None: ...
    def predict(self, X, cfg) -> Prediction: ...
    def save(self, run_dir: Path) -> None: ...
    @classmethod
    def load(cls, run_dir: Path) -> Method: ...


#: ``background.mode`` values (configs, docs/methods.md "Background windows in training").
BACKGROUND_MODES = ("none", "uncertain", "other")


def with_background(
    Xtr: np.ndarray, ytr: np.ndarray, Xbg: np.ndarray, n_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Append background windows to the training set with the marker label ``n_classes``.

    The marker is not a class of its own unless the method makes it one (``mode: other``); the
    taught losses only use it to tell background rows from known-class rows.
    """
    X = np.concatenate([np.asarray(Xtr, dtype=np.float32), np.asarray(Xbg, dtype=np.float32)])
    y = np.concatenate([np.asarray(ytr, dtype=np.int64), np.full(len(Xbg), n_classes, np.int64)])
    return X, y


def entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Shannon entropy per row, in nats."""
    p = np.clip(probs, eps, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def class_weights(y: np.ndarray, n_classes: int, kind: str = "inv_sqrt") -> np.ndarray:
    """Per-class loss weights normalized to mean 1: ``inv_sqrt`` (default), ``inv`` or ``none``."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    if kind == "none":
        w = np.ones(n_classes)
    elif kind == "inv":
        w = 1.0 / counts
    elif kind == "inv_sqrt":
        w = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(kind)
    return (w / w.mean()).astype(np.float32)
