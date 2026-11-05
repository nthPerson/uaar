"""Calibration metrics on known-class windows: ECE, NLL, Brier, reliability bins (evaluation.md §4).

Owner: Sirena. AI assistance recorded in docs/ai-use-log.md.

All functions take ``probs`` of shape ``(N, K)`` (rows sum to 1) and integer labels ``y`` of shape
``(N,)``. Calibration is only defined for windows whose true class is one of the K known classes,
so rows with ``y < 0`` (unknown, from ``splits.encode_labels``) are dropped first. Empty input
(after that drop) returns ``nan`` rather than raising.

Confidence is the maximum predicted probability and a window counts as correct when its argmax
equals ``y``. Bins are ``n_bins`` equal-width intervals on [0, 1], right-closed
(``(lo, hi]``), with 0 placed in the first bin.
"""

from __future__ import annotations

import numpy as np

NAN = float("nan")


def _known_rows(probs: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y).ravel().astype(np.int64)
    if probs.ndim != 2 or len(probs) != len(y):
        raise ValueError(f"probs must be (N, K) aligned with y (N,), got {probs.shape}, {y.shape}")
    keep = y >= 0
    return probs[keep], y[keep]


def reliability_bins(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> dict[str, np.ndarray]:
    """Per-bin confidence, accuracy and counts: everything a reliability diagram needs.

    Args:
        probs: ``(N, K)`` predicted probabilities.
        y: ``(N,)`` true class index; ``y < 0`` rows are dropped.
        n_bins: Number of equal-width confidence bins on [0, 1].

    Returns:
        Dict with ``edges`` ``(n_bins + 1,)``; ``confidence``, ``accuracy`` ``(n_bins,)`` (mean
        max-probability and fraction correct per bin, ``nan`` for empty bins); ``count``
        ``(n_bins,)`` int; ``n`` (total windows, 0-d int array).
    """
    probs, y = _known_rows(probs, y)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    conf_sum = np.zeros(n_bins)
    acc_sum = np.zeros(n_bins)
    count = np.zeros(n_bins, dtype=np.int64)
    if len(y):
        conf = probs.max(axis=1)
        correct = (probs.argmax(axis=1) == y).astype(np.float64)
        # right-closed bins: (lo, hi]; conf == 0 goes to bin 0, conf == 1 to the last bin
        idx = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, n_bins - 1)
        count = np.bincount(idx, minlength=n_bins)
        conf_sum = np.bincount(idx, weights=conf, minlength=n_bins)
        acc_sum = np.bincount(idx, weights=correct, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        confidence = np.where(count > 0, conf_sum / count, np.nan)
        accuracy = np.where(count > 0, acc_sum / count, np.nan)
    return {
        "edges": edges,
        "confidence": confidence,
        "accuracy": accuracy,
        "count": count,
        "n": np.asarray(int(count.sum())),
    }


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error: ``sum_b (n_b / N) |acc_b - conf_b|`` over equal-width bins.

    Args:
        probs: ``(N, K)`` predicted probabilities.
        y: ``(N,)`` true class index; ``y < 0`` rows are dropped.
        n_bins: Number of bins (15 per evaluation.md §4).

    Returns:
        ECE in [0, 1]; ``nan`` when no known rows remain.
    """
    b = reliability_bins(probs, y, n_bins)
    n = int(b["n"])
    if n == 0:
        return NAN
    used = b["count"] > 0
    gap = np.abs(b["accuracy"][used] - b["confidence"][used])
    return float(np.sum(b["count"][used] / n * gap))


def nll(probs: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Mean negative log-likelihood of the true class, ``-mean log p_y`` (nats).

    ``p_y`` is clipped at ``eps`` so a zero probability gives a large finite value. ``nan`` when
    no known rows remain.
    """
    probs, y = _known_rows(probs, y)
    if len(y) == 0:
        return NAN
    p = np.clip(probs[np.arange(len(y)), y], eps, 1.0)
    return float(-np.mean(np.log(p)))


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    """Multi-class Brier score, ``mean_i sum_k (p_ik - 1[y_i = k])^2``, in [0, 2].

    ``nan`` when no known rows remain.
    """
    probs, y = _known_rows(probs, y)
    if len(y) == 0:
        return NAN
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
