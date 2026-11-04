"""Sliding-window buffer for a live 50 Hz stream (2 s windows, 1 s hop by default)."""

from __future__ import annotations

from collections import deque

import numpy as np

from uaar.data import wearable as w


class LiveWindower:
    """Accumulates raw samples and yields fixed-length windows in physical units.

    ``push`` takes rows ``[t_ms, ax, ay, az, gx, gy, gz, flags, seq]`` (raw counts) and returns the
    list of windows completed by them, each as ``(t_start_s, t_end_s, X, flags_any)`` where ``X``
    is float32 ``(6, L)`` in g and deg/s (channel order ax ay az gx gy gz) and ``flags_any`` is the
    OR of the samples' flags (so the caller knows whether a session was active).
    """

    def __init__(self, window_s: float = 2.0, hop_s: float = 1.0, fs: float = w.SAMPLE_RATE_HZ):
        self.length = int(round(window_s * fs))
        self.hop = int(round(hop_s * fs))
        self.buf: deque[tuple[float, np.ndarray, int]] = deque(maxlen=self.length)
        self.since_emit = 0
        self.n_dropped = 0
        self._last_seq: int | None = None

    def reset(self) -> None:
        self.buf.clear()
        self.since_emit = 0
        self._last_seq = None

    def push(self, rows: list[list[int]]) -> list[tuple[float, float, np.ndarray, int]]:
        out = []
        for r in rows:
            t_ms, ax, ay, az, gx, gy, gz, flags, seq = (int(v) for v in r[:9])
            if self._last_seq is not None:
                step = (seq - self._last_seq) % 65536
                if step > 1:
                    self.n_dropped += step - 1
            self._last_seq = seq
            x = np.array(
                [
                    ax / w.ACC_LSB_PER_G,
                    ay / w.ACC_LSB_PER_G,
                    az / w.ACC_LSB_PER_G,
                    gx / w.GYR_LSB_PER_DPS,
                    gy / w.GYR_LSB_PER_DPS,
                    gz / w.GYR_LSB_PER_DPS,
                ],
                dtype=np.float32,
            )
            self.buf.append((t_ms / 1000.0, x, flags))
            self.since_emit += 1
            if len(self.buf) == self.length and self.since_emit >= self.hop:
                self.since_emit = 0
                X = np.stack([b[1] for b in self.buf], axis=1)  # (6, L)
                flags_any = 0
                for b in self.buf:
                    flags_any |= b[2]
                out.append((self.buf[0][0], self.buf[-1][0], X, flags_any))
        return out
