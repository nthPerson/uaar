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


def open_macro_f1(
    y: np.ndarray, label: np.ndarray, score: np.ndarray, n_classes: int, accept: float = 0.95
) -> float:
    """Macro-F1 over the K known classes plus "rejected", for windows that include background.

    The threshold is the ``accept`` quantile of the scores of the known rows (the rule tau is
    chosen by); a row is predicted "rejected" when its score is above it, else ``label``.

    Args:
        y: ``(N,)`` true labels; ``n_classes`` marks a background row (true class "rejected").
        label: ``(N,)`` predicted known class.
        score: ``(N,)`` uncertainty score, higher = more uncertain.

    This is what a taught model's checkpoint is selected on. Macro-F1 over the known classes
    alone reaches 1.0 in the first epochs of a four-class problem and says nothing about
    whether the model has learned to be uncertain on the background windows.
    """
    from sklearn.metrics import f1_score

    y = np.asarray(y)
    score = np.asarray(score, dtype=np.float64)
    known = y < n_classes
    if not known.any():
        return float("nan")
    tau = np.quantile(score[known], accept)
    pred = np.where(score > tau, n_classes, np.asarray(label))
    labels = np.arange(n_classes + 1)
    return float(f1_score(y, pred, labels=labels, average="macro", zero_division=0))


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
