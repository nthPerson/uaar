"""M2: softmax neural network (backbone + linear + cross-entropy). Owner: both.

Scores: ``msp`` = 1 - max softmax probability (Hendrycks & Gimpel 2017) and predictive entropy.
Implemented from the design document; AI assistance recorded in docs/ai-use-log.md.

Background windows (``background`` argument, config ``background.mode``; decisions.md
2026-11-23). ``fit`` may be given windows that belong to none of the K known classes:

``"none"``       they are not used (the plain model).
``"uncertain"``  the "taught" model. The network still has K outputs. A background window adds
                 ``weight * CE(uniform, softmax(z))``, the cross-entropy to the uniform
                 distribution, so the network is trained to spread its probability evenly on
                 them (outlier exposure). The reject decision stays ``msp > tau``.
``"other"``      the reference model. The network has K + 1 outputs and the background windows
                 are the extra class "Other", trained with ordinary cross-entropy. ``predict``
                 returns the K known-class probabilities as they are (they sum to
                 1 - p(Other)), the label is the most probable known class, ``msp`` is
                 1 - the largest known-class probability and ``p_other`` is p(Other).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from uaar.models import normalize
from uaar.models.backbone import ConvBackbone
from uaar.models.base import (
    BACKGROUND_MODES,
    Prediction,
    class_weights,
    entropy,
    open_macro_f1,
    with_background,
)
from uaar.models.train_loop import History, device_for, fit, predict_batches


class SoftmaxNet(ConvBackbone):
    """The backbone as-is: outputs are logits."""


def uniform_cross_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Per-row cross-entropy between the uniform distribution and ``softmax(logits)``:
    ``-(1/K) sum_k log softmax(z)_k``. Its minimum, ``log K``, is reached when all logits are
    equal, that is when the predicted distribution is uniform."""
    return -F.log_softmax(logits, dim=1).mean(dim=1)


def taught_cross_entropy(
    logits: torch.Tensor,
    y: torch.Tensor,
    n_classes: int,
    class_weight: torch.Tensor | None = None,
    background_weight: float = 0.5,
) -> torch.Tensor:
    """Cross-entropy on the known rows plus the uniform-target term on the background rows.

    Args:
        logits: ``(N, K)``.
        y: ``(N,)`` labels; ``n_classes`` marks a background row.
        class_weight: optional ``(K,)`` weights for the known-class term.
        background_weight: weight of the background term.

    Each term is a mean over its own rows, so the balance does not depend on how many
    background rows a batch holds.
    """
    known = y < n_classes
    loss = logits.sum() * 0.0
    if known.any():
        loss = loss + F.cross_entropy(logits[known], y[known], weight=class_weight)
    if (~known).any():
        loss = loss + background_weight * uniform_cross_entropy(logits[~known]).mean()
    return loss


class SoftmaxMethod:
    name = "softmax"
    score_names = ["msp", "entropy"]

    def __init__(
        self,
        n_classes: int,
        classes: list[str],
        dropout: float = 0.3,
        seed: int = 0,
        background: str = "none",
        background_weight: float = 0.5,
    ):
        if background not in BACKGROUND_MODES:
            raise ValueError(f"background must be one of {BACKGROUND_MODES}, got {background!r}")
        self.classes = classes
        self.seed = seed
        self.background = background
        self.background_weight = float(background_weight)
        self.model = SoftmaxNet(n_classes + (background == "other"), dropout=dropout)
        if background == "other":
            self.score_names = ["msp", "p_other"]
        self.stats: dict | None = None
        self.history: History | None = None

    def fit(self, Xtr, ytr, Xval, yval, cfg, run_dir: Path, background=None) -> None:
        """Train. ``background`` is ``(X_train, X_val)`` background windows, required unless
        the mode is ``"none"`` (then it is ignored).

        Early stopping and the choice of the best weights: macro-F1 on the known validation
        windows in mode ``"none"``. In the two background modes the validation background
        windows are added, so that the criterion also measures how they are handled: mode
        ``"other"`` uses macro-F1 over all K + 1 outputs; mode ``"uncertain"`` uses
        :func:`uaar.models.base.open_macro_f1` with ``msp`` as the score (the K classes plus
        "rejected", threshold at ``cfg.eval.accept_rate`` of the known validation scores).
        """
        K = len(self.classes)
        val_metric = None
        if self.background != "none":
            if background is None:
                raise ValueError(f"background mode {self.background!r} needs background windows")
            Xtr, ytr = with_background(Xtr, ytr, background[0], K)
            Xval, yval = with_background(Xval, yval, background[1], K)
        if self.background == "uncertain":
            accept = float(cfg.get("eval.accept_rate", 0.95))

            def val_metric(out, y):
                p = F.softmax(out, dim=1).numpy()
                return open_macro_f1(y, p.argmax(axis=1), 1.0 - p.max(axis=1), K, accept)

        self.stats = normalize.fit_stats(Xtr)
        Xtr_n, Xval_n = normalize.apply(Xtr, self.stats), normalize.apply(Xval, self.stats)
        device = device_for()
        n_out = K + (self.background == "other")
        counted = ytr if self.background == "other" else ytr[ytr < K]
        w = torch.from_numpy(class_weights(counted, n_out, cfg.train.class_weighting)).to(device)

        def loss_fn(out, y, epoch):
            if self.background == "uncertain":
                return taught_cross_entropy(out, y, K, w, self.background_weight)
            return F.cross_entropy(out, y, weight=w)

        self.history = fit(
            self.model,
            loss_fn,
            lambda out: F.softmax(out, dim=1),
            Xtr_n,
            ytr,
            Xval_n,
            yval,
            epochs=cfg.train.epochs,
            batch_size=cfg.train.batch_size,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            patience=cfg.train.patience,
            num_workers=cfg.train.num_workers,
            device=device,
            val_metric=val_metric,
        )
        self.history.write(run_dir / "history.csv")

    def predict(self, X, cfg=None) -> Prediction:
        assert self.stats is not None, "fit or load first"
        logits = predict_batches(self.model, normalize.apply(X, self.stats))
        probs = F.softmax(logits, dim=1).numpy()
        if self.background == "other":
            known = probs[:, :-1]
            return Prediction(
                probs=known,
                label=known.argmax(axis=1),
                scores={"msp": 1.0 - known.max(axis=1), "p_other": probs[:, -1]},
            )
        return Prediction(
            probs=probs,
            label=probs.argmax(axis=1),
            scores={"msp": 1.0 - probs.max(axis=1), "entropy": entropy(probs)},
        )

    def save(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), run_dir / "model.pt")
        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    **self.model.config(),
                    "classes": self.classes,
                    "seed": self.seed,
                    "method": self.name,
                    "background": self.background,
                    "background_weight": self.background_weight,
                },
                indent=2,
            )
        )
        normalize.save(
            self.stats or {},
            run_dir / "preprocess.json",
            window_samples=100,
            sample_rate_hz=50,
            channels=["ax", "ay", "az", "gx", "gy", "gz"],
        )

    @classmethod
    def load(cls, run_dir: Path) -> SoftmaxMethod:
        conf = json.loads((run_dir / "config.json").read_text())
        # n_classes in config.json is the number of network outputs (K + 1 in mode "other")
        m = cls(
            len(conf["classes"]),
            conf["classes"],
            dropout=conf["dropout"],
            seed=conf.get("seed", 0),
            background=conf.get("background", "none"),
            background_weight=conf.get("background_weight", 0.5),
        )
        m.model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
        m.model.eval()
        m.stats = normalize.load(run_dir / "preprocess.json")
        return m


__all__ = ["SoftmaxNet", "SoftmaxMethod", "np"]
