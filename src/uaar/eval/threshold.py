"""Unknown-detection threshold tau from validation known-class scores (evaluation.md).

tau is the score at the ``accept`` quantile (default 0.95) of the validation known-class
windows: the model is allowed to reject at most 5 % of known data it was tuned on. Scores
above tau are called unknown.
"""

from __future__ import annotations

import numpy as np


def select_tau(score_val_known: np.ndarray, accept: float = 0.95) -> float:
    """The ``accept`` quantile of the known-class validation scores (higher score = less sure)."""
    s = np.asarray(score_val_known, dtype=np.float64)
    if s.size == 0:
        raise ValueError("no validation scores")
    return float(np.quantile(s, accept, method="linear"))


def apply(score: np.ndarray, tau: float) -> np.ndarray:
    """Boolean mask: True where the window is called unknown (score > tau)."""
    return np.asarray(score) > tau
