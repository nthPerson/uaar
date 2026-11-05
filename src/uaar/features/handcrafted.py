"""Handcrafted per-window features for M1, logistic regression (models.md §3, spec §3.4, §4.2).

Owner: Sirena (writes), Robert (reviews). AI assistance recorded in docs/ai-use-log.md.

For each of eight signals (the six IMU channels plus the acceleration and angular-rate
magnitudes) twelve statistics are computed, plus the six within-sensor axis correlations:
8 x 12 + 6 = 102 float32 columns named ``{signal}_{stat}`` and ``corr_{a}_{b}``. Everything is
vectorized over windows and computed chunk by chunk, so a memory-mapped ``X`` of several hundred
thousand windows never has to be held in memory as float64.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

#: Raw channel order of ``X`` (spec §3.2).
CHANNELS: list[str] = ["ax", "ay", "az", "gx", "gy", "gz"]
#: Signals the statistics are computed on: the six channels, then ``|a|`` and ``|g|``.
SIGNALS: list[str] = [*CHANNELS, "amag", "gmag"]
#: Per-signal statistics, in column order.
STATS: list[str] = [
    "mean",
    "std",
    "min",
    "max",
    "p10",
    "p50",
    "p90",
    "rms",
    "zcr",
    "domfreq",
    "dompow",
    "specent",
]
#: Axis pairs correlated within each sensor (never across accelerometer and gyroscope).
CORR_PAIRS: list[tuple[str, str]] = [
    *combinations(["ax", "ay", "az"], 2),
    *combinations(["gx", "gy", "gz"], 2),
]

# A signal whose standard deviation is below FLAT_RTOL * (1 + |mean|) (in its own units, g or
# deg/s) is treated as constant: far below any real IMU's noise floor, but well above the
# floating-point residue that mean removal leaves on an exactly constant window.
FLAT_RTOL = 1e-6


def feature_names() -> list[str]:
    """Column names in output order: ``{signal}_{stat}`` for SIGNALS x STATS, then correlations.

    Returns:
        A list of ``len(SIGNALS) * len(STATS) + len(CORR_PAIRS)`` (= 102) names.
    """
    names = [f"{s}_{st}" for s in SIGNALS for st in STATS]
    names += [f"corr_{a}_{b}" for a, b in CORR_PAIRS]
    return names


def compute(X: np.ndarray, fs: float = 50.0, chunk: int = 8192) -> pd.DataFrame:
    """Handcrafted features of every window in ``X``.

    Args:
        X: ``(N, 6, L)`` windows, channel order ``ax ay az gx gy gz`` in g and deg/s,
            unnormalized (spec §3.2). Any array-like that supports slicing along axis 0 works,
            including a ``np.load(..., mmap_mode="r")`` memmap; only ``chunk`` windows are
            converted to float64 at a time.
        fs: Sample rate in Hz (50 for RecoFit and MM-Fit, 20 for RecGym); sets the unit of
            ``domfreq``.
        chunk: Windows processed per step. Peak working memory besides the output is about
            ``chunk * L * 350 bytes`` (about 280 MB at the default with L = 100, measured).

    Returns:
        DataFrame of shape ``(N, 102)``, float32 columns named by :func:`feature_names`, with a
        default ``RangeIndex`` (row ``i`` is window ``i`` of ``X``). Per signal, with
        ``d = x - mean(x)`` and the power spectrum ``P_k = |rfft(d)_k|^2`` for ``k >= 1``
        (DC excluded):

        - ``mean``, ``std`` (population), ``min``, ``max``, ``p10``/``p50``/``p90`` (linear
          interpolation), ``rms`` = sqrt(mean(x^2)): in the signal's units.
        - ``zcr``: fraction of the ``L - 1`` consecutive sample pairs at which ``d`` changes
          sign (``d >= 0`` vs ``d < 0``); dimensionless in [0, 1], about ``2 f / fs`` for a
          sinusoid of frequency ``f``.
        - ``domfreq``: frequency in Hz of the largest ``P_k`` (``k * fs / L``).
        - ``dompow``: that bin's share of the total non-DC power, in (0, 1].
        - ``specent``: Shannon entropy of ``P_k / sum(P)``, divided by ``log(n_bins)`` so it
          lies in [0, 1] (0 = a single spectral line, 1 = flat spectrum).

        Correlations ``corr_{a}_{b}`` are Pearson coefficients over the window's samples, in
        [-1, 1].

    Zero-variance convention: a constant (flat) signal has no defined spectrum or correlation.
    For such a signal ``std`` (otherwise ~1e-16 of rounding residue), ``zcr``, ``domfreq``,
    ``dompow`` and ``specent`` are 0, and every correlation involving it is 0. "Flat" means
    ``std <= 1e-6 * (1 + |mean|)``. With this convention the output is finite for every finite
    input, including all-zero windows.

    Raises:
        ValueError: if ``X`` is not 3-D with 6 channels, or a chunk contains NaN or inf.
    """
    if X.ndim != 3 or X.shape[1] != len(CHANNELS):
        raise ValueError(f"X must be (N, {len(CHANNELS)}, L), got {X.shape}")
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk}")
    n = X.shape[0]
    out = np.empty((n, len(feature_names())), dtype=np.float32)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        out[start:stop] = _compute_block(np.asarray(X[start:stop], dtype=np.float64), fs)
    return pd.DataFrame(out, columns=feature_names())


def _compute_block(x6: np.ndarray, fs: float) -> np.ndarray:
    """Features of one chunk: float64 ``(n, 6, L)`` -> float32 ``(n, 102)``."""
    if not np.isfinite(x6).all():
        raise ValueError("X contains NaN or inf")
    n, _, L = x6.shape
    amag = np.sqrt((x6[:, 0:3] ** 2).sum(axis=1, keepdims=True))
    gmag = np.sqrt((x6[:, 3:6] ** 2).sum(axis=1, keepdims=True))
    x = np.concatenate([x6, amag, gmag], axis=1)  # (n, 8, L)

    mean = x.mean(axis=2)
    d = x - mean[..., None]
    std = np.sqrt((d**2).mean(axis=2))
    flat = std <= FLAT_RTOL * (1.0 + np.abs(mean))  # (n, 8)
    p10, p50, p90 = np.percentile(x, [10, 50, 90], axis=2)
    rms = np.sqrt((x**2).mean(axis=2))

    pos = d >= 0
    zcr = (pos[..., 1:] != pos[..., :-1]).sum(axis=2) / max(L - 1, 1)

    power = np.abs(np.fft.rfft(d, axis=2)[..., 1:]) ** 2  # (n, 8, n_bins), DC dropped
    n_bins = power.shape[2]
    total = power.sum(axis=2)
    safe_total = np.where(flat | (total <= 0), 1.0, total)
    k = power.argmax(axis=2)
    domfreq = (k + 1) * fs / L
    dompow = np.take_along_axis(power, k[..., None], axis=2)[..., 0] / safe_total
    p = power / safe_total[..., None]
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)
    specent = -plogp.sum(axis=2) / np.log(n_bins) if n_bins > 1 else np.zeros_like(total)
    for arr in (std, zcr, domfreq, dompow, specent):
        arr[flat] = 0.0

    stats = {
        "mean": mean,
        "std": std,
        "min": x.min(axis=2),
        "max": x.max(axis=2),
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "rms": rms,
        "zcr": zcr,
        "domfreq": domfreq,
        "dompow": dompow,
        "specent": specent,
    }
    per_signal = np.stack([stats[s] for s in STATS], axis=2)  # (n, 8, 12)

    idx = {s: i for i, s in enumerate(SIGNALS)}
    corrs = np.empty((n, len(CORR_PAIRS)))
    for j, (a, b) in enumerate(CORR_PAIRS):
        ia, ib = idx[a], idx[b]
        cov = (d[:, ia] * d[:, ib]).mean(axis=1)
        denom = std[:, ia] * std[:, ib]
        ok = ~(flat[:, ia] | flat[:, ib]) & (denom > 0)
        corrs[:, j] = np.where(ok, cov / np.where(ok, denom, 1.0), 0.0)
    np.clip(corrs, -1.0, 1.0, out=corrs)

    return np.concatenate([per_signal.reshape(n, -1), corrs], axis=1).astype(np.float32)
