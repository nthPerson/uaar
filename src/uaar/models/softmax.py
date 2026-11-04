"""M2: softmax neural network (backbone + linear + cross-entropy). Owner: both.

Scores: ``msp`` = 1 - max softmax probability (Hendrycks & Gimpel 2017) and predictive entropy.
Implemented from the design document; AI assistance recorded in docs/ai-use-log.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from uaar.models import normalize
from uaar.models.backbone import ConvBackbone
from uaar.models.base import Prediction, class_weights, entropy
from uaar.models.train_loop import History, device_for, fit, predict_batches


class SoftmaxNet(ConvBackbone):
    """The backbone as-is: outputs are logits."""


class SoftmaxMethod:
    name = "softmax"
    score_names = ["msp", "entropy"]

    def __init__(self, n_classes: int, classes: list[str], dropout: float = 0.3, seed: int = 0):
        self.classes = classes
        self.seed = seed
        self.model = SoftmaxNet(n_classes, dropout=dropout)
        self.stats: dict | None = None
        self.history: History | None = None

    def fit(self, Xtr, ytr, Xval, yval, cfg, run_dir: Path) -> None:
        self.stats = normalize.fit_stats(Xtr)
        Xtr_n, Xval_n = normalize.apply(Xtr, self.stats), normalize.apply(Xval, self.stats)
        device = device_for()
        w = torch.from_numpy(class_weights(ytr, len(self.classes), cfg.train.class_weighting)).to(
            device
        )

        def loss_fn(out, y, epoch):
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
        )
        self.history.write(run_dir / "history.csv")

    def predict(self, X, cfg=None) -> Prediction:
        assert self.stats is not None, "fit or load first"
        logits = predict_batches(self.model, normalize.apply(X, self.stats))
        probs = F.softmax(logits, dim=1).numpy()
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
        m = cls(
            conf["n_classes"], conf["classes"], dropout=conf["dropout"], seed=conf.get("seed", 0)
        )
        m.model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
        m.model.eval()
        m.stats = normalize.load(run_dir / "preprocess.json")
        return m


__all__ = ["SoftmaxNet", "SoftmaxMethod", "np"]
