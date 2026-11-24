"""M3: evidential deep learning (Sensoy, Kaplan & Kandemir, NeurIPS 2018).

Owner: Robert. The backbone's final linear layer outputs one value per class; a non-negative
evidence activation turns it into evidence e_k, and alpha_k = e_k + 1 parameterizes a Dirichlet
over the class probabilities. Predictions are p_k = alpha_k / S with S = sum_k alpha_k, and the
single-pass uncertainty is the vacuity u = K / S. Implemented from the paper by Robert; AI
assistance recorded in docs/ai-use-log.md.

Equations referenced below use the paper's numbering: Eq. 3 (negative log marginal
likelihood), Eq. 4 (expected cross-entropy, the digamma form), Eq. 5 (expected squared error),
Eq. 7 (KL to the uniform Dirichlet in closed form), Eq. 8 (annealed total loss).

The default loss here is Eq. 4. The paper prefers Eq. 5, but on the 28-class RecoFit problem
the squared-error loss leaves more than half of the windows with zero evidence for every class
and the network does not train (docs/decisions.md, 2026-11-05).

Background windows (``background="uncertain"``, config ``background.mode``; decisions.md
2026-11-23): ``fit`` may be given windows that belong to none of the known classes. Each adds
``weight * KL(Dir(alpha) || Dir(1, ..., 1))``, the paper's own regularizer applied to all K
outputs, so the network is trained to give no evidence for any class on them (vacuity 1). The
term is not annealed. An explicit "Other" class is not offered for EDL.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from uaar.models.backbone import ConvBackbone

Evidence = Literal["softplus", "relu", "exp"]


def evidence_from_logits(z: torch.Tensor, kind: Evidence = "softplus") -> torch.Tensor:
    """Non-negative evidence from raw network outputs."""
    if kind == "softplus":
        return F.softplus(z)
    if kind == "relu":
        return F.relu(z)
    if kind == "exp":
        return torch.exp(torch.clamp(z, max=10.0))
    raise ValueError(kind)


def dirichlet_summary(alpha: torch.Tensor) -> dict[str, torch.Tensor]:
    """``probs`` (N, K) = alpha / S, ``vacuity`` (N,) = K / S, ``one_minus_max`` (N,)."""
    S = alpha.sum(dim=1, keepdim=True)
    probs = alpha / S
    K = alpha.shape[1]
    return {
        "probs": probs,
        "vacuity": (K / S).squeeze(1),
        "one_minus_max": 1.0 - probs.max(dim=1).values,
        "strength": S.squeeze(1),
    }


def kl_dirichlet_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """KL( Dir(alpha) || Dir(1, ..., 1) ) per row, Eq. 7 of Sensoy et al. 2018, closed form."""
    K = alpha.shape[1]
    S = alpha.sum(dim=1)
    return (
        torch.lgamma(S)
        - torch.lgamma(torch.tensor(float(K), dtype=alpha.dtype, device=alpha.device))
        - torch.lgamma(alpha).sum(dim=1)
        + ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(S).unsqueeze(1))).sum(dim=1)
    )


def _misleading_alpha(alpha: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
    """alpha~ = y + (1 - y) * alpha: the true class's evidence is removed, so the KL term only
    penalizes evidence for the wrong classes."""
    return y_onehot + (1.0 - y_onehot) * alpha


def edl_loss(
    alpha: torch.Tensor,
    y_onehot: torch.Tensor,
    lam: float,
    class_weights: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Eq. 5 + lam * Eq. 7 (the paper's Eq. 8 with the squared-error Bayes risk).

    Args:
        alpha: (N, K) Dirichlet parameters, alpha = evidence + 1, all > 1 - epsilon.
        y_onehot: (N, K) one-hot targets.
        lam: KL weight lambda_t, annealed by the training loop (min(1, epoch / 10)).
        class_weights: optional (K,) weights applied per sample by its true class.
    """
    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S
    bayes_risk = ((y_onehot - p) ** 2 + p * (1.0 - p) / (S + 1.0)).sum(dim=1)
    kl = kl_dirichlet_uniform(_misleading_alpha(alpha, y_onehot))
    per_sample = bayes_risk + lam * kl
    return _reduce(per_sample, y_onehot, class_weights, reduction)


def edl_loss_digamma(
    alpha: torch.Tensor,
    y_onehot: torch.Tensor,
    lam: float,
    class_weights: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Eq. 4 (expected cross-entropy, sum_k y_k (psi(S) - psi(alpha_k))) + lam * Eq. 7."""
    S = alpha.sum(dim=1, keepdim=True)
    ce = (y_onehot * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=1)
    kl = kl_dirichlet_uniform(_misleading_alpha(alpha, y_onehot))
    return _reduce(ce + lam * kl, y_onehot, class_weights, reduction)


def edl_loss_log(
    alpha: torch.Tensor,
    y_onehot: torch.Tensor,
    lam: float,
    class_weights: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Eq. 3 (negative log marginal likelihood, sum_k y_k (log S - log alpha_k)) + lam * Eq. 7."""
    S = alpha.sum(dim=1, keepdim=True)
    nll = (y_onehot * (torch.log(S) - torch.log(alpha))).sum(dim=1)
    kl = kl_dirichlet_uniform(_misleading_alpha(alpha, y_onehot))
    return _reduce(nll + lam * kl, y_onehot, class_weights, reduction)


LOSSES = {"mse": edl_loss, "digamma": edl_loss_digamma, "log": edl_loss_log}


def taught_edl_loss(
    alpha: torch.Tensor,
    y: torch.Tensor,
    n_classes: int,
    lam: float,
    loss: str = "digamma",
    class_weights: torch.Tensor | None = None,
    background_weight: float = 1.0,
) -> torch.Tensor:
    """The EDL loss on the known rows plus ``KL(Dir(alpha) || Dir(1))`` on the background rows.

    Args:
        alpha: ``(N, K)`` Dirichlet parameters.
        y: ``(N,)`` labels; ``n_classes`` marks a background row.
        lam: annealed KL weight of the known-class loss (the background term is not annealed).
        loss: key of ``LOSSES`` for the known rows.
        background_weight: weight of the background term.

    The background term is zero exactly when the evidence is zero for every class (alpha = 1),
    and positive otherwise. Each term is a mean over its own rows.
    """
    known = y < n_classes
    out = alpha.sum() * 0.0
    if known.any():
        y1 = F.one_hot(y[known], n_classes).to(alpha.dtype)
        out = out + LOSSES[loss](alpha[known], y1, lam=lam, class_weights=class_weights)
    if (~known).any():
        out = out + background_weight * kl_dirichlet_uniform(alpha[~known]).mean()
    return out


def _reduce(per_sample, y_onehot, class_weights, reduction):
    if class_weights is not None:
        per_sample = per_sample * (y_onehot * class_weights.unsqueeze(0)).sum(dim=1)
    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    return per_sample.mean()


class EDLNet(ConvBackbone):
    """Backbone whose output is evidence; ``forward`` returns alpha = evidence + 1."""

    def __init__(
        self,
        n_classes: int,
        in_channels: int = 6,
        dropout: float = 0.3,
        evidence: Evidence = "softplus",
    ):
        super().__init__(n_classes, in_channels, dropout)
        self.evidence_kind: Evidence = evidence

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # alpha
        return evidence_from_logits(super().forward(x), self.evidence_kind) + 1.0

    def config(self) -> dict:
        return {**super().config(), "arch": "EDLNet", "evidence": self.evidence_kind}


def kl_anneal(epoch: int, anneal_epochs: int = 10) -> float:
    """lambda_t = min(1, epoch / anneal_epochs); epoch is 0-based."""
    return min(1.0, epoch / float(anneal_epochs)) if anneal_epochs > 0 else 1.0


__all__ = [
    "EDLMethod",
    "EDLNet",
    "edl_loss",
    "edl_loss_digamma",
    "edl_loss_log",
    "taught_edl_loss",
    "kl_dirichlet_uniform",
    "dirichlet_summary",
    "evidence_from_logits",
    "kl_anneal",
]


class EDLMethod:
    """The Method interface for EDL (fit / predict / save / load), mirroring SoftmaxMethod.

    Args:
        loss: ``"digamma"`` (Eq. 4, default), ``"mse"`` (Eq. 5) or ``"log"`` (Eq. 3).
        kl_anneal_epochs: epochs over which the KL weight ramps linearly from 0.
        kl_weight: the value the KL weight ramps to, lambda_t = kl_weight * min(1, epoch /
            kl_anneal_epochs). 1.0 is the paper's schedule; smaller values regularize less.
        background: ``"none"`` or ``"uncertain"`` (see the module docstring).
        background_weight: weight of the background term in mode ``"uncertain"``.
    """

    name = "edl"
    score_names = ["vacuity", "one_minus_max"]

    def __init__(
        self,
        n_classes: int,
        classes: list[str],
        dropout: float = 0.3,
        evidence: Evidence = "softplus",
        loss: str = "digamma",
        kl_anneal_epochs: int = 10,
        kl_weight: float = 1.0,
        seed: int = 0,
        background: str = "none",
        background_weight: float = 1.0,
    ):
        if loss not in LOSSES:
            raise ValueError(f"unknown EDL loss {loss!r}; choose from {sorted(LOSSES)}")
        if background not in ("none", "uncertain"):
            raise ValueError(f"EDL background must be 'none' or 'uncertain', got {background!r}")
        self.background = background
        self.background_weight = float(background_weight)
        self.classes = classes
        self.seed = seed
        self.loss_kind = loss
        self.kl_anneal_epochs = kl_anneal_epochs
        self.kl_weight = float(kl_weight)
        self.model = EDLNet(n_classes, dropout=dropout, evidence=evidence)
        self.stats: dict | None = None
        self.history = None

    def fit(self, Xtr, ytr, Xval, yval, cfg, run_dir, background=None) -> None:
        """Train. ``background`` is ``(X_train, X_val)`` background windows, required in mode
        ``"uncertain"``. In that mode early stopping and the choice of the best weights use
        :func:`uaar.models.base.open_macro_f1` on the known plus background validation windows
        with vacuity as the score, instead of macro-F1 on the known windows alone."""
        import numpy as np

        from uaar.models import normalize
        from uaar.models.base import class_weights, open_macro_f1, with_background
        from uaar.models.train_loop import device_for, fit

        K = len(self.classes)
        w_np = class_weights(ytr, K, cfg.train.class_weighting)
        val_metric = None
        if self.background == "uncertain":
            if background is None:
                raise ValueError("background mode 'uncertain' needs background windows")
            Xtr, ytr = with_background(Xtr, ytr, background[0], K)
            Xval, yval = with_background(Xval, yval, background[1], K)
            accept = float(cfg.get("eval.accept_rate", 0.95))

            def val_metric(alpha, y):
                a = alpha.numpy()
                return open_macro_f1(y, a.argmax(axis=1), K / a.sum(axis=1), K, accept)

        self.stats = normalize.fit_stats(Xtr)
        Xtr_n, Xval_n = normalize.apply(Xtr, self.stats), normalize.apply(Xval, self.stats)
        device = device_for()
        w = torch.from_numpy(w_np).to(device)

        def lam(epoch):
            return self.kl_weight * kl_anneal(epoch, self.kl_anneal_epochs)

        def loss_fn(alpha, y, epoch):
            return taught_edl_loss(
                alpha, y, K, lam(epoch), self.loss_kind, w, self.background_weight
            )

        self.history = fit(
            self.model,
            loss_fn,
            lambda a: a / a.sum(dim=1, keepdim=True),
            Xtr_n,
            ytr,
            Xval_n,
            np.asarray(yval),
            epochs=cfg.train.epochs,
            batch_size=cfg.train.batch_size,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            patience=cfg.train.patience,
            num_workers=cfg.train.num_workers,
            device=device,
            extra_columns=lambda e: {"lambda_t": lam(e)},
            val_metric=val_metric,
        )
        self.history.write(run_dir / "history.csv")

    def predict(self, X, cfg=None):
        from uaar.models import normalize
        from uaar.models.base import Prediction
        from uaar.models.train_loop import predict_batches

        assert self.stats is not None, "fit or load first"
        alpha = predict_batches(self.model, normalize.apply(X, self.stats))
        s = dirichlet_summary(alpha)
        probs = s["probs"].numpy()
        return Prediction(
            probs=probs,
            label=probs.argmax(axis=1),
            scores={
                "vacuity": s["vacuity"].numpy(),
                "one_minus_max": s["one_minus_max"].numpy(),
            },
        )

    def save(self, run_dir) -> None:
        import json

        from uaar.models import normalize

        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), run_dir / "model.pt")
        conf = {
            **self.model.config(),
            "classes": self.classes,
            "seed": self.seed,
            "method": self.name,
            "loss": self.loss_kind,
            "kl_anneal_epochs": self.kl_anneal_epochs,
            "kl_weight": self.kl_weight,
            "background": self.background,
            "background_weight": self.background_weight,
        }
        (run_dir / "config.json").write_text(json.dumps(conf, indent=2))
        normalize.save(
            self.stats or {},
            run_dir / "preprocess.json",
            window_samples=100,
            sample_rate_hz=50,
            channels=["ax", "ay", "az", "gx", "gy", "gz"],
        )

    @classmethod
    def load(cls, run_dir):
        import json

        from uaar.models import normalize

        conf = json.loads((run_dir / "config.json").read_text())
        m = cls(
            conf["n_classes"],
            conf["classes"],
            dropout=conf["dropout"],
            evidence=conf["evidence"],
            loss=conf.get("loss", "mse"),
            kl_anneal_epochs=conf.get("kl_anneal_epochs", 10),
            kl_weight=conf.get("kl_weight", 1.0),
            seed=conf.get("seed", 0),
            background=conf.get("background", "none"),
            background_weight=conf.get("background_weight", 1.0),
        )
        m.model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
        m.model.eval()
        m.stats = normalize.load(run_dir / "preprocess.json")
        return m
