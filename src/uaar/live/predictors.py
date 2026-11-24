"""Predictors for the live demo: trained run directories, and a placeholder without models.

Every predictor maps one window ``X`` float32 ``(6, L)`` in g and deg/s to a dict with
``label``, ``p`` (probability of that label), ``score`` (uncertainty, higher = less sure),
``tau`` (the threshold above which the window is called unknown) and ``probs`` (label -> p).
:class:`RunPredictor` wraps any trained run directory (logistic regression, softmax, EDL,
MC Dropout); :class:`MotionEnergyPredictor` needs no model and is what the server falls back to.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Protocol

import numpy as np


def score_scale(score_name: str, n_classes: int) -> float:
    """The largest value an uncertainty score can take, so scores can share a 0-1 scale.

    ``1 - max p`` (``msp``, ``one_minus_max``) and the variation ratio reach ``1 - 1/K``,
    entropy and mutual information reach ``ln K``, vacuity reaches 1. Unknown names give 1.
    """
    if score_name in ("msp", "one_minus_max", "varratio"):
        return 1.0 - 1.0 / n_classes
    if score_name in ("entropy", "mi"):
        return math.log(n_classes)
    return 1.0


def score_info(method: str, score_name: str, passes: int | None = None) -> tuple[str, str]:
    """``(name, plain-words formula)`` of an uncertainty score, for the audience.

    The three demo models measure uncertainty differently; the page shows this line under each
    bar so "uncertainty" is not read as one common quantity.
    """
    t = passes or 30
    table = {
        "msp": ("Least confidence", "1 − highest class probability"),
        "one_minus_max": ("Least confidence", "1 − highest class probability"),
        "vacuity": ("Vacuity", "number of classes ÷ total evidence"),
        "varratio": ("Variation ratio", f"share of {t} dropout passes outvoted by the majority"),
        "mi": ("Mutual information", f"disagreement between {t} dropout passes"),
    }
    if score_name == "entropy":
        if method == "mcdropout":
            return ("Predictive entropy", f"entropy of the average of {t} dropout passes")
        return ("Entropy", "spread of the class probabilities")
    return table.get(score_name, (score_name, ""))


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
        name: Key of the card in the messages; defaults to the method name (``softmax``, ...).
        top: How many classes to include in ``probs`` (the most probable ones).
        title: Card title shown on the page; defaults to ``name``.
        subtitle: One line under the title (how the model was trained, for the audience).
        labels: Optional class name -> short display name (``"Shoulder Press (dumbbell)"`` ->
            ``"Shoulder press"``); classes not listed keep their name.
        measure, formula: Name and plain-words formula of the score, shown under the bar;
            default :func:`score_info`.

    ``predict`` also returns ``u`` and ``tau_u``: the score and the threshold divided by the
    score's largest possible value (:func:`score_scale`) and clipped to [0, 1], so the page can
    draw every method's uncertainty on one 0-100 % bar.
    """

    def __init__(
        self,
        run_dir: Path,
        score: str | None = None,
        name: str | None = None,
        top=5,
        title: str | None = None,
        subtitle: str = "",
        labels: dict[str, str] | None = None,
        measure: str | None = None,
        formula: str | None = None,
    ):
        from uaar.models import factory

        run_dir = Path(run_dir)
        self.method = factory.load(run_dir)
        self.name = name or self.method.name
        self.title = title or self.name
        self.subtitle = subtitle
        self.score_name = score or factory.PRIMARY_SCORE[self.method.name]
        taus = json.loads((run_dir / "threshold.json").read_text())["tau"]
        if self.score_name not in taus:
            raise KeyError(f"{run_dir}: no threshold for {self.score_name!r}; has {list(taus)}")
        self.tau = float(taus[self.score_name])
        labels = labels or {}
        self.classes = [labels.get(c, c) for c in self.method.classes]
        self.scale = score_scale(self.score_name, len(self.classes))
        info = score_info(self.method.name, self.score_name, getattr(self.method, "T", None))
        self.measure = measure or info[0]
        self.formula = formula if formula is not None else info[1]
        self.top = top

    def predict(self, X: np.ndarray) -> dict:
        pred = self.method.predict(np.asarray(X, dtype=np.float32)[None])
        probs = pred.probs[0]
        k = int(probs.argmax())
        order = np.argsort(probs)[::-1][: self.top]
        score = float(pred.scores[self.score_name][0])
        return {
            "label": self.classes[k],
            "p": float(probs[k]),
            "score": score,
            "tau": self.tau,
            "u": min(1.0, max(0.0, score / self.scale)),
            "tau_u": min(1.0, max(0.0, self.tau / self.scale)),
            "probs": {self.classes[i]: float(probs[i]) for i in order},
            "score_name": self.score_name,
        }
