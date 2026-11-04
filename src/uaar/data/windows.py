"""Fixed-length windows from long-format recordings (spec §3.2, datasets.md §5 step 4).

Each recording is cut into ``window_s`` second windows every ``hop_s`` seconds. A window keeps
the majority label of its samples when that label covers at least ``purity`` of the window;
otherwise it is dropped. Unlabelled majorities (``""``) and the trailing partial window are
dropped as well. Windows are built by sample index under the assumption of uniform sampling,
so the same code serves 50 Hz (``L = 100``) and RecGym's 20 Hz (``L = 40``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numpy.lib.stride_tricks import sliding_window_view

CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]
META_COLUMNS = [
    "window_id",
    "subject",
    "recording_id",
    "source",
    "label",
    "start_s",
    "label_purity",
    "n_gaps",
]
DEFAULT_FS = 50.0


def window_length(window_s: float, fs: float) -> int:
    """Samples per window: ``round(window_s * fs)`` (100 at 50 Hz, 40 at 20 Hz)."""
    return int(round(window_s * fs))


def _empty_meta() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": pd.Series(dtype=np.int64),
            "subject": pd.Series(dtype=str),
            "recording_id": pd.Series(dtype=str),
            "source": pd.Series(dtype=str),
            "label": pd.Series(dtype=str),
            "start_s": pd.Series(dtype=np.float32),
            "label_purity": pd.Series(dtype=np.float32),
            "n_gaps": pd.Series(dtype=np.int64),
        }
    )


def _window_gaps(seq: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    """Number of ``seq`` gaps (steps > 1 mod 65536) strictly inside each window."""
    seq = np.asarray(seq, dtype=np.int64)
    if len(seq) < 2:
        return np.zeros(len(starts), dtype=np.int64)
    gap = (np.diff(seq) % 65536) > 1  # gap[i]: a gap between sample i and i+1
    cum = np.concatenate([[0], np.cumsum(gap)])  # cum[i] = gaps before sample i
    return (cum[starts + length - 1] - cum[starts]).astype(np.int64)


def window_recording(
    df: pd.DataFrame,
    window_s: float = 2.0,
    hop_s: float = 1.0,
    purity: float = 0.8,
    fs: float = DEFAULT_FS,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Cut one recording's long-format frame into labelled windows.

    Args:
        df: One recording in the §3.1 contract: columns ``t`` (s), ``ax ay az`` (g),
            ``gx gy gz`` (deg/s), ``label`` (str, ``""`` = unlabelled), plus ``subject``,
            ``recording_id`` and ``source`` (constant within the frame). An optional ``seq``
            column (wearable sample counter) enables gap counting. Rows must be in time order
            and uniformly sampled at ``fs``.
        window_s: Window length in seconds.
        hop_s: Hop between window starts in seconds.
        purity: Minimum fraction of samples carrying the majority label; windows below it are
            dropped. A window at exactly ``purity`` is kept.
        fs: Sample rate in Hz; ``L = round(window_s * fs)`` samples per window.

    Returns:
        ``(X, meta)``. ``X`` is float32 of shape ``(N, 6, L)`` in channel order
        ``ax ay az gx gy gz`` and unnormalized physical units. ``meta`` has one row per window
        with columns ``window_id`` (0..N-1, local to this recording), ``subject``,
        ``recording_id``, ``source``, ``label``, ``start_s`` (s, the ``t`` of the first sample),
        ``label_purity`` (fraction in [purity, 1]) and ``n_gaps`` (``seq`` gaps inside the
        window; 0 without a ``seq`` column).
    """
    length = window_length(window_s, fs)
    hop = max(1, int(round(hop_s * fs)))
    n = len(df)
    n_cand = (n - length) // hop + 1 if n >= length else 0
    if n_cand <= 0:
        return np.zeros((0, 6, length), dtype=np.float32), _empty_meta()

    starts = np.arange(n_cand, dtype=np.int64) * hop

    # Signal windows: (n, 6) -> (n - L + 1, 6, L) view, then every hop-th start.
    data = np.ascontiguousarray(df[CHANNELS].to_numpy(dtype=np.float32))
    views = sliding_window_view(data, length, axis=0)[::hop][:n_cand]

    # Majority label per window from cumulative one-hot counts: counts[w, k] is the number of
    # samples with label code k in window w, computed without a per-window loop.
    codes, uniques = pd.factorize(df["label"].astype(str), sort=True)
    k = len(uniques)
    onehot = np.zeros((n + 1, k), dtype=np.int32)
    onehot[np.arange(1, n + 1), codes] = 1
    cum = np.cumsum(onehot, axis=0)
    counts = cum[starts + length] - cum[starts]  # (n_cand, k)
    major = counts.argmax(axis=1)
    major_count = counts[np.arange(n_cand), major]
    label_purity = major_count / float(length)
    labels = np.asarray(uniques, dtype=object)[major].astype(str)

    keep = (major_count >= purity * length - 1e-9) & (labels != "")
    if not keep.any():
        return np.zeros((0, 6, length), dtype=np.float32), _empty_meta()

    t = df["t"].to_numpy(dtype=np.float32)
    if "seq" in df.columns:
        n_gaps = _window_gaps(df["seq"].to_numpy(), starts, length)
    else:
        n_gaps = np.zeros(n_cand, dtype=np.int64)

    X = np.ascontiguousarray(views[keep]).astype(np.float32, copy=False)
    first = df.iloc[0]
    meta = pd.DataFrame(
        {
            "window_id": np.arange(int(keep.sum()), dtype=np.int64),
            "subject": str(first["subject"]),
            "recording_id": str(first["recording_id"]),
            "source": str(first["source"]),
            "label": labels[keep],
            "start_s": t[starts[keep]],
            "label_purity": label_purity[keep].astype(np.float32),
            "n_gaps": n_gaps[keep],
        }
    )
    return X, meta


def _candidate_count(n: int, window_s: float, hop_s: float, fs: float) -> int:
    length = window_length(window_s, fs)
    hop = max(1, int(round(hop_s * fs)))
    return (n - length) // hop + 1 if n >= length else 0


def make_windows(
    recordings: Path,
    out_dir: Path,
    window_s: float = 2.0,
    hop_s: float = 1.0,
    purity: float = 0.8,
) -> None:
    """Window every recording in ``recordings.parquet`` and write ``X.npy`` + ``meta.parquet``.

    Args:
        recordings: Path to the long-format ``recordings.parquet`` (spec §3.1). A companion
            ``recordings.meta.json`` next to it supplies ``sample_rate_hz`` (default 50).
        out_dir: Directory that receives ``X.npy`` (float32 ``(N, 6, L)``) and ``meta.parquet``
            (spec §3.2) with a global, contiguous ``window_id`` equal to the row index.
        window_s: Window length in seconds.
        hop_s: Hop in seconds.
        purity: Majority-label threshold, see :func:`window_recording`.

    Returns:
        None. Prints a one-line summary with kept/dropped window counts and per-label counts.
    """
    recordings = Path(recordings)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = recordings.with_name("recordings.meta.json")
    fs = DEFAULT_FS
    if meta_path.exists():
        fs = float(json.loads(meta_path.read_text()).get("sample_rate_hz", DEFAULT_FS))

    frame = pq.read_table(recordings).to_pandas()
    xs: list[np.ndarray] = []
    metas: list[pd.DataFrame] = []
    n_candidates = 0
    for _rid, rec in frame.groupby("recording_id", sort=True):
        n_candidates += _candidate_count(len(rec), window_s, hop_s, fs)
        x, m = window_recording(rec, window_s=window_s, hop_s=hop_s, purity=purity, fs=fs)
        if len(m):
            xs.append(x)
            metas.append(m)

    length = window_length(window_s, fs)
    if xs:
        X = np.concatenate(xs, axis=0)
        meta = pd.concat(metas, ignore_index=True)
    else:
        X = np.zeros((0, 6, length), dtype=np.float32)
        meta = _empty_meta()
    meta["window_id"] = np.arange(len(meta), dtype=np.int64)
    meta = meta[META_COLUMNS]

    np.save(out_dir / "X.npy", X)
    meta.to_parquet(out_dir / "meta.parquet", index=False)

    per_label = ", ".join(f"{k}={v}" for k, v in meta["label"].value_counts().sort_index().items())
    print(
        f"windows: kept {len(meta)} / dropped {n_candidates - len(meta)} "
        f"(L={length}, fs={fs:g} Hz, hop={hop_s:g} s, purity>={purity:g}); labels: {per_label}"
    )
