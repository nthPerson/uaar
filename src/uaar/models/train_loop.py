"""Shared training loop for the neural methods (models.md §2).

AdamW with cosine decay, class-weighted loss, early stopping on validation macro-F1 with
patience, ``history.csv`` per epoch, best weights restored at the end. The loss function is
passed in so softmax (cross-entropy) and EDL (Bayes risk + annealed KL) share everything else.
"""

from __future__ import annotations

import csv
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

LossFn = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
"""(model_output, y_int64, epoch) -> scalar loss; epoch is 0-based (for annealing)."""


@dataclass
class History:
    rows: list[dict] = field(default_factory=list)
    best_epoch: int = -1
    best_val_f1: float = -1.0

    def write(self, path: Path) -> None:
        if not self.rows:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            w.writeheader()
            w.writerows(self.rows)


def device_for(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, num_workers: int = 0
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
        torch.from_numpy(y.astype(np.int64)),
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False
    )


@torch.no_grad()
def predict_batches(
    model: nn.Module, X: np.ndarray, batch_size: int = 1024, device: torch.device | None = None
) -> torch.Tensor:
    """Concatenated raw model outputs for ``X`` (N, C, L) in eval mode, on CPU."""
    device = device or next(model.parameters()).device
    model.eval()
    outs = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(X[i : i + batch_size], dtype=np.float32)).to(
            device
        )
        outs.append(model(xb).cpu())
    return torch.cat(outs) if outs else torch.zeros(0)


def fit(
    model: nn.Module,
    loss_fn: LossFn,
    probs_fn: Callable[[torch.Tensor], torch.Tensor],
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xval: np.ndarray,
    yval: np.ndarray,
    *,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 8,
    num_workers: int = 0,
    device: torch.device | None = None,
    log: Callable[[str], None] | None = print,
    extra_columns: Callable[[int], dict] | None = None,
) -> History:
    """Train ``model`` and return the History; the model ends with its best-validation weights.

    ``probs_fn`` maps raw outputs to class probabilities (softmax or alpha / S) for macro-F1.
    ``extra_columns(epoch)`` may add per-epoch values (for example lambda_t) to history.csv.
    """
    device = device or device_for()
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    train_loader = make_loader(Xtr, ytr, batch_size, shuffle=True, num_workers=num_workers)
    hist = History()
    best_state = None
    bad_epochs = 0
    for epoch in range(epochs):
        t0 = time.perf_counter()
        model.train()
        total, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb, epoch)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
            n += len(xb)
        sched.step()
        out_val = predict_batches(model, Xval, device=device)
        val_probs = probs_fn(out_val)
        with torch.no_grad():
            val_loss = float(
                loss_fn(
                    out_val.to(device), torch.from_numpy(yval.astype(np.int64)).to(device), epoch
                ).item()
            )
        val_f1 = (
            float(f1_score(yval, val_probs.argmax(dim=1).numpy(), average="macro"))
            if len(yval)
            else float("nan")
        )
        row = {
            "epoch": epoch,
            "train_loss": total / max(n, 1),
            "val_loss": val_loss,
            "val_macro_f1": val_f1,
            "lr": sched.get_last_lr()[0],
            "seconds": time.perf_counter() - t0,
        }
        if extra_columns:
            row.update(extra_columns(epoch))
        hist.rows.append(row)
        if log:
            log(
                f"epoch {epoch:3d} train {row['train_loss']:.4f} val {val_loss:.4f} "
                f"f1 {val_f1:.4f} ({row['seconds']:.1f} s)"
            )
        if val_f1 > hist.best_val_f1:
            hist.best_val_f1, hist.best_epoch = val_f1, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if log:
                    log(
                        f"early stop at epoch {epoch}; best epoch {hist.best_epoch} "
                        f"f1 {hist.best_val_f1:.4f}"
                    )
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return hist
