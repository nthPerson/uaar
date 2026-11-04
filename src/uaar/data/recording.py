"""The in-memory form every loader produces: one continuous IMU recording with labels.

Units are physical (g and degrees per second), time is seconds from the start of the
recording, and ``labels`` holds one string per sample (``""`` for unlabelled samples).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Recording:
    subject: str
    recording_id: str
    source: str
    t: np.ndarray  # (n,) float32 seconds
    acc: np.ndarray  # (n, 3) float32 g
    gyro: np.ndarray  # (n, 3) float32 deg/s
    labels: np.ndarray  # (n,) str
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.t)
        if self.acc.shape != (n, 3) or self.gyro.shape != (n, 3) or self.labels.shape != (n,):
            raise ValueError(
                f"inconsistent lengths: t {self.t.shape}, acc {self.acc.shape}, "
                f"gyro {self.gyro.shape}, labels {self.labels.shape}"
            )

    @property
    def n_samples(self) -> int:
        return len(self.t)

    @property
    def duration_s(self) -> float:
        return float(self.t[-1] - self.t[0]) if self.n_samples > 1 else 0.0

    def to_frame(self) -> pd.DataFrame:
        """Long-format frame with the columns of the recordings contract (spec §3.1)."""
        return pd.DataFrame(
            {
                "subject": self.subject,
                "recording_id": self.recording_id,
                "t": self.t.astype(np.float32),
                "ax": self.acc[:, 0],
                "ay": self.acc[:, 1],
                "az": self.acc[:, 2],
                "gx": self.gyro[:, 0],
                "gy": self.gyro[:, 1],
                "gz": self.gyro[:, 2],
                "label": self.labels,
                "source": self.source,
            }
        )
