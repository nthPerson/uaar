"""Predictor interface for the live demo and a placeholder used until the models exist.

Every predictor maps one window ``X`` float32 ``(6, L)`` in g and deg/s to a dict with
``label``, ``p`` (probability of that label), ``score`` (uncertainty, higher = less sure),
``tau`` (the threshold above which the window is called unknown) and ``probs`` (label -> p).
The real methods (logistic regression, softmax, EDL, MC Dropout) wrap trained run directories
and expose the same interface; see uaar.models once it exists.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Predictor(Protocol):
    name: str
    tau: float

    def predict(self, X: np.ndarray) -> dict: ...


class MotionEnergyPredictor:
    """Placeholder: 'moving' vs 'still' from the variance of |a|, with a made-up uncertainty.

    Exists so the page, the WebSocket path and the windowing can be exercised end to end before
    any trained model is available. ``score`` is highest when the energy sits near the boundary.
    """

    name = "energy"
    tau = 0.8

    def __init__(self, threshold_g2: float = 0.01):
        self.threshold = threshold_g2

    def predict(self, X: np.ndarray) -> dict:
        amag = np.linalg.norm(X[:3], axis=0)
        energy = float(amag.var())
        # logistic in log-energy around the threshold: p(moving)
        z = np.log(energy + 1e-6) - np.log(self.threshold)
        p_moving = float(1.0 / (1.0 + np.exp(-2.0 * z)))
        probs = {"moving": p_moving, "still": 1.0 - p_moving}
        label = max(probs, key=probs.get)
        return {
            "label": label,
            "p": probs[label],
            "score": float(1.0 - abs(2 * p_moving - 1)),  # 1 at the boundary, 0 when sure
            "tau": self.tau,
            "probs": probs,
            "energy": energy,
        }
