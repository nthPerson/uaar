"""Fine-tuning a RecoFit-trained network on our own wearable recordings (protocol P4-ft).

Recipe (decided 2026-11-23, docs/decisions.md): the convolutional trunk is frozen; the input
normalization is re-fitted on the wearable training windows; the two output layers are trained
on the wearable known-class windows. A taught model (``background: uncertain``) keeps its
"be uncertain" term, now on the wearable background movements. The rejection threshold is set
again, from wearable validation windows.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from uaar.models import normalize
from uaar.models.base import class_weights, with_background
from uaar.models.train_loop import History, device_for, fit
from uaar.utils import seed_everything


@contextmanager
def frozen_trunk(net: torch.nn.Module):
    """Inside the block the convolutional trunk (``net.features``) does not change.

    Its weights get no gradient, and it stays in eval mode even when the training loop calls
    ``net.train()``, so the BatchNorm running statistics are not updated either.
    """
    params = list(net.features.parameters())
    required = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(False)
    class_train = type(net).train

    def train(mode: bool = True):
        class_train(net, mode)
        net.features.eval()
        return net

    net.train = train  # instance attribute shadows the method for the duration
    try:
        yield net
    finally:
        del net.train
        for p, r in zip(params, required, strict=True):
            p.requires_grad_(r)


def finetune(
    method,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xval: np.ndarray,
    yval: np.ndarray,
    Xbg: np.ndarray | None = None,
    *,
    epochs: int = 40,
    lr: float = 1e-3,
    batch_size: int = 32,
    weight_decay: float = 1e-4,
    patience: int = 10,
    class_weighting: str = "inv_sqrt",
    freeze: bool = True,
    seed: int = 0,
    log=None,
) -> History:
    """Fine-tune a loaded ``SoftmaxMethod`` or ``EDLMethod`` in place.

    Args:
        method: a loaded softmax or EDL method (plain or taught; not the "Other"-class model).
        Xtr, ytr: wearable known-class training windows ``(N, 6, L)`` and integer labels.
        Xval, yval: wearable known-class validation windows (early stopping on macro-F1).
        Xbg: wearable background windows. Used only by a taught model, as the windows to be
            uncertain on; a plain model is fine-tuned on the known classes alone.
        epochs, lr, batch_size, weight_decay, patience: training settings for the two output
            layers.
        freeze: keep the convolutional trunk fixed (the recipe). ``False`` trains every layer;
            a fallback for when the frozen features do not carry over to the wearable.

    EDL differs in two ways. A taught EDL network answers anything unfamiliar with outputs
    near -8 before the softplus, where the softplus is flat (slope about 0.0002), so on a new
    device its last layer barely moves and the model stays at "no evidence for anything"
    (measured on the fold-0 model, decisions.md 2026-11-23). So (1) the last layer (``head``)
    is re-initialized, and (2) the KL weight ramps from 0 to the model's ``kl_weight`` over the
    model's ``kl_anneal_epochs`` again, exactly as in the first training; with the full weight
    from the first step the new layer collapses to no evidence (decisions.md 2026-11-05).
    Softmax needs neither.

    Returns:
        The training history. ``method.stats`` is replaced by statistics of the wearable
        training windows, and ``method.model`` ends with its best-validation weights.
    """
    from uaar.models.edl import EDLMethod, kl_anneal, taught_edl_loss
    from uaar.models.softmax import taught_cross_entropy

    background = getattr(method, "background", "none")
    if background == "other":
        raise ValueError("fine-tuning is not defined for the 'Other'-class model")
    if len(Xtr) == 0 or len(Xval) == 0:
        raise ValueError("fine-tuning needs known-class training and validation windows")
    K = len(method.classes)
    w = torch.from_numpy(class_weights(np.asarray(ytr), K, class_weighting))
    X, y = np.asarray(Xtr, dtype=np.float32), np.asarray(ytr, dtype=np.int64)
    if background == "uncertain" and Xbg is not None and len(Xbg):
        X, y = with_background(X, y, Xbg, K)
    method.stats = normalize.fit_stats(X)
    device = device_for()
    w = w.to(device)

    is_edl = isinstance(method, EDLMethod)
    if is_edl:

        def loss_fn(alpha, yb, epoch):
            lam = method.kl_weight * kl_anneal(epoch, method.kl_anneal_epochs)
            return taught_edl_loss(alpha, yb, K, lam, method.loss_kind, w, method.background_weight)

        def probs_fn(alpha):
            return alpha / alpha.sum(dim=1, keepdim=True)

    else:

        def loss_fn(logits, yb, epoch):
            return taught_cross_entropy(logits, yb, K, w, method.background_weight)

        def probs_fn(logits):
            return F.softmax(logits, dim=1)

    seed_everything(seed)
    Xn = normalize.apply(X, method.stats)
    Xval_n = normalize.apply(np.asarray(Xval, dtype=np.float32), method.stats)
    yval = np.asarray(yval, dtype=np.int64)
    settings = {
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "device": device,
        "log": log,
    }
    with frozen_trunk(method.model) if freeze else nullcontext():
        if is_edl:
            method.model.head.reset_parameters()
        history = fit(
            method.model, loss_fn, probs_fn, Xn, y, Xval_n, yval, epochs=epochs, patience=patience,
            **settings,
        )  # fmt: skip
    method.model.eval()
    return history
