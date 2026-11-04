"""Shared 1D-CNN backbone for M2 (softmax), M3 (EDL) and M4 (MC Dropout).

Owner: Robert. Architecture from docs/design/models.md §1; the head is left to the subclass so
the only difference between the three neural methods is how uncertainty is produced.
Implemented from the design document by Robert; AI assistance recorded in docs/ai-use-log.md.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBackbone(nn.Module):
    """``(N, C, L)`` window -> ``(N, K)`` logits (or evidence, depending on the head).

    Layers: three Conv1d + BatchNorm + ReLU blocks (the first two followed by MaxPool(2)),
    global average pooling, Dropout(p) -> Linear(64, 64) + ReLU -> Dropout(p) -> ``head``.
    ``head`` defaults to ``Linear(64, K)``; subclasses may replace it (EDL keeps a Linear and
    applies the evidence activation in ``forward``). About 30 k parameters for K = 12.

    Args:
        n_classes: K, the number of known classes.
        in_channels: 6 for ax ay az gx gy gz.
        dropout: p for both Dropout layers (0.3 by default, tuned for MC Dropout).
    """

    def __init__(self, n_classes: int, in_channels: int = 6, dropout: float = 0.3):
        super().__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.dropout_p = dropout
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(64, n_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate 64-d features (used by the Mahalanobis stretch method)."""
        return self.fc(self.pool(self.features(x)).squeeze(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))

    def count_params(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def count_macs(self, length: int = 100) -> int:
        """Multiply-accumulates for one window of ``length`` samples, counted analytically by layer.

        Conv: out_channels * out_length * (in_channels * kernel); Linear: in * out. BatchNorm,
        ReLU, pooling and dropout are not counted (they are negligible and not multiplies).
        """
        macs = 0
        cur_len = length
        for m in self.features:
            if isinstance(m, nn.Conv1d):
                macs += m.out_channels * cur_len * m.in_channels * m.kernel_size[0]
            elif isinstance(m, nn.MaxPool1d):
                cur_len //= 2
        for m in list(self.fc) + [self.head]:
            if isinstance(m, nn.Linear):
                macs += m.in_features * m.out_features
        return macs

    def config(self) -> dict:
        return {
            "arch": "ConvBackbone",
            "n_classes": self.n_classes,
            "in_channels": self.in_channels,
            "dropout": self.dropout_p,
        }
