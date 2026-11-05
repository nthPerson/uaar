"""Predictors for the live demo: trained run directories, and a placeholder without models.

Every predictor maps one window ``X`` float32 ``(6, L)`` in g and deg/s to a dict with
``label``, ``p`` (probability of that label), ``score`` (uncertainty, higher = less sure),
``tau`` (the threshold above which the window is called unknown) and ``probs`` (label -> p).
:class:`RunPredictor` wraps any trained run directory (logistic regression, softmax, EDL,
MC Dropout); :class:`MotionEnergyPredictor` needs no model and is what the server falls back to.
"""

from __future__ import annotations

import json
from pathlib import Path
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


class RunPredictor:
    """One trained run directory behind the live interface.

    Loads the method with ``uaar.models.factory.load`` and the threshold of one of its scores
    from ``threshold.json`` (chosen on the validation subjects when the run was trained or,
    for MC Dropout, built).

    Args:
        run_dir: A run directory with ``config.json``, the model file and ``threshold.json``.
        score: Which of the method's scores to show; defaults to its primary score
            (``uaar.models.factory.PRIMARY_SCORE``).
        name: Card title on the page; defaults to the method name (``softmax``, ``edl``, ...).
        top: How many classes to include in ``probs`` (the most probable ones).
    """

    def __init__(self, run_dir: Path, score: str | None = None, name: str | None = None, top=5):
        from uaar.models import factory

        run_dir = Path(run_dir)
        self.method = factory.load(run_dir)
        self.name = name or self.method.name
        self.score_name = score or factory.PRIMARY_SCORE[self.method.name]
        taus = json.loads((run_dir / "threshold.json").read_text())["tau"]
        if self.score_name not in taus:
            raise KeyError(f"{run_dir}: no threshold for {self.score_name!r}; has {list(taus)}")
        self.tau = float(taus[self.score_name])
        self.classes = list(self.method.classes)
        self.top = top

    def predict(self, X: np.ndarray) -> dict:
        pred = self.method.predict(np.asarray(X, dtype=np.float32)[None])
        probs = pred.probs[0]
        k = int(probs.argmax())
        order = np.argsort(probs)[::-1][: self.top]
        return {
            "label": self.classes[k],
            "p": float(probs[k]),
            "score": float(pred.scores[self.score_name][0]),
            "tau": self.tau,
            "probs": {self.classes[i]: float(probs[i]) for i in order},
            "score_name": self.score_name,
        }
