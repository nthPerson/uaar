"""M3: evidential deep learning (Sensoy, Kaplan & Kandemir, NeurIPS 2018).

Owner: Robert. The backbone's final linear layer outputs one value per class; a non-negative
evidence activation turns it into evidence e_k, and alpha_k = e_k + 1 parameterizes a Dirichlet
over the class probabilities. Predictions are p_k = alpha_k / S with S = sum_k alpha_k, and the
single-pass uncertainty is the vacuity u = K / S. Implemented from the paper by Robert; AI
assistance recorded in docs/ai-use-log.md.

Equations referenced below use the paper's numbering: Eq. 3 (negative log marginal
likelihood), Eq. 4 (expected cross-entropy, the digamma form), Eq. 5 (expected squared error,
the default), Eq. 7 (KL to the uniform Dirichlet in closed form), Eq. 8 (annealed total loss).
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
    """Ablation: Eq. 4 (expected cross-entropy, sum_k y_k (psi(S) - psi(alpha_k))) + lam * Eq. 7."""
    S = alpha.sum(dim=1, keepdim=True)
    ce = (y_onehot * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=1)
    kl = kl_dirichlet_uniform(_misleading_alpha(alpha, y_onehot))
    return _reduce(ce + lam * kl, y_onehot, class_weights, reduction)


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
    "EDLNet",
    "edl_loss",
    "edl_loss_digamma",
    "kl_dirichlet_uniform",
    "dirichlet_summary",
    "evidence_from_logits",
    "kl_anneal",
]
