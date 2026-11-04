"""Windowing invariants (spec §7, test_windows row) on synthetic recordings."""

import json

import numpy as np
import pandas as pd
import pytest

from uaar.data.windows import CHANNELS, make_windows, window_recording

FS = 50.0
L = 100  # 2 s at 50 Hz
HOP = 50


def _frame(n: int, labels=None, fs: float = FS, seq=None, **const) -> pd.DataFrame:
    """One recording with a distinct constant per channel unless overridden in ``const``."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "subject": "S1",
            "recording_id": "rec0",
            "source": "single",
            "t": (np.arange(n) / fs).astype(np.float32),
        }
    )
    for c, value in zip(CHANNELS, [1.0, 2.0, 3.0, 10.0, 20.0, 30.0], strict=True):
        df[c] = np.full(n, const.get(c, value), dtype=np.float32) + rng.normal(0, 1e-3, n).astype(
            np.float32
        )
    df["label"] = (
        np.full(n, "A", dtype=object) if labels is None else np.asarray(labels, dtype=object)
    )
    if seq is not None:
        df["seq"] = np.asarray(seq)
    return df


def test_window_count_for_known_length():
    # 10 s at 50 Hz -> starts at 0,1,...,8 s = 9 windows of 2 s with 1 s hop.
    X, meta = window_recording(_frame(500), fs=FS)
    assert X.shape == (9, 6, L)
    assert X.dtype == np.float32
    assert len(meta) == 9
    assert meta["window_id"].tolist() == list(range(9))
    np.testing.assert_allclose(meta["start_s"].to_numpy(), np.arange(9, dtype=np.float32))
    assert (meta["label"] == "A").all()
    assert (meta["label_purity"] == 1.0).all()
    assert (meta["n_gaps"] == 0).all()
    assert set(meta.columns) == {
        "window_id",
        "subject",
        "recording_id",
        "source",
        "label",
        "start_s",
        "label_purity",
        "n_gaps",
    }


def test_partial_last_window_dropped():
    # 149 samples: a full window at 0 s, a partial one at 1 s (only 99 samples) -> 1 window.
    X, meta = window_recording(_frame(149), fs=FS)
    assert X.shape[0] == 1 and len(meta) == 1
    # Exactly 150 samples -> the window at 1 s is complete.
    X, _ = window_recording(_frame(150), fs=FS)
    assert X.shape[0] == 2
    # Shorter than one window -> nothing, but shapes are still well-formed.
    X, meta = window_recording(_frame(99), fs=FS)
    assert X.shape == (0, 6, L) and len(meta) == 0


def test_majority_rule_at_exactly_80_percent_passes():
    labels = ["A"] * 80 + ["B"] * 20
    X, meta = window_recording(_frame(100, labels=labels), fs=FS)
    assert len(meta) == 1
    assert meta.loc[0, "label"] == "A"
    assert meta.loc[0, "label_purity"] == pytest.approx(0.8)


def test_majority_rule_just_below_80_percent_drops():
    labels = ["A"] * 79 + ["B"] * 21
    X, meta = window_recording(_frame(100, labels=labels), fs=FS)
    assert len(meta) == 0 and X.shape == (0, 6, L)


def test_mixed_recording_keeps_only_pure_windows():
    # 4 s: 2 s of A then 2 s of B. Windows at 0 s (A), 1 s (50/50, dropped), 2 s (B).
    labels = ["A"] * 100 + ["B"] * 100
    X, meta = window_recording(_frame(200, labels=labels), fs=FS)
    assert meta["label"].tolist() == ["A", "B"]
    np.testing.assert_allclose(meta["start_s"].to_numpy(), [0.0, 2.0])
    assert meta["window_id"].tolist() == [0, 1]


def test_unlabelled_majority_dropped():
    labels = [""] * 100 + ["A"] * 100
    _, meta = window_recording(_frame(200, labels=labels), fs=FS)
    assert meta["label"].tolist() == ["A"]
    assert meta["start_s"].tolist() == [2.0]


def test_channel_order():
    X, _ = window_recording(_frame(300), fs=FS)
    for c, value in enumerate([1.0, 2.0, 3.0, 10.0, 20.0, 30.0]):
        np.testing.assert_allclose(X[:, c, :], value, atol=1e-2)


def test_window_content_matches_signal():
    df = _frame(300)
    df["ax"] = np.arange(300, dtype=np.float32)  # ramp so position inside the window is visible
    X, meta = window_recording(df, fs=FS)
    for i, start in enumerate(meta["start_s"]):
        s = int(round(start * FS))
        np.testing.assert_array_equal(X[i, 0, :], np.arange(s, s + L, dtype=np.float32))


def test_recgym_rate_gives_40_samples():
    # 20 Hz: 2 s = 40 samples, hop 20 samples; 100 samples -> starts 0..60 = 4 windows.
    X, meta = window_recording(_frame(100, fs=20.0), fs=20.0)
    assert X.shape == (4, 6, 40)
    np.testing.assert_allclose(meta["start_s"].to_numpy(), [0.0, 1.0, 2.0, 3.0])


def test_n_gaps_counted_from_seq():
    seq = np.arange(300)
    seq[125:] += 3  # one gap (3 lost samples) between sample 124 and 125
    _, meta = window_recording(_frame(300, seq=seq), fs=FS)
    # Windows start at 0,1,2,3,4 s = samples 0,50,...,200; the gap lies inside [50,150) and
    # [100,200) only.
    assert meta["n_gaps"].tolist() == [0, 1, 1, 0, 0]
    # Without a seq column every window has zero gaps.
    _, meta = window_recording(_frame(300), fs=FS)
    assert (meta["n_gaps"] == 0).all()


def test_make_windows_round_trip(tmp_path):
    rec_a = _frame(300)
    rec_b = _frame(200, labels=["B"] * 200)
    rec_b["subject"] = "S2"
    rec_b["recording_id"] = "rec1"
    rec_b["source"] = "multi"
    frame = pd.concat([rec_a, rec_b], ignore_index=True)
    frame.to_parquet(tmp_path / "recordings.parquet", index=False)
    (tmp_path / "recordings.meta.json").write_text(json.dumps({"sample_rate_hz": 50}))

    out = tmp_path / "windows"
    make_windows(tmp_path / "recordings.parquet", out)
    X = np.load(out / "X.npy", mmap_mode="r")
    meta = pd.read_parquet(out / "meta.parquet")
    assert X.shape == (5 + 3, 6, L)
    assert len(meta) == 8
    assert meta["window_id"].tolist() == list(range(8))  # global, contiguous
    assert meta["label"].tolist() == ["A"] * 5 + ["B"] * 3
    assert meta["subject"].tolist() == ["S1"] * 5 + ["S2"] * 3
    assert meta["source"].tolist() == ["single"] * 5 + ["multi"] * 3
    np.testing.assert_allclose(X[:5, 0, :], 1.0, atol=1e-2)
