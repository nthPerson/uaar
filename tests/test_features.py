"""Handcrafted features (spec §7 test_features): count, names, finiteness, known signals."""

import numpy as np
import pytest

from uaar.features import handcrafted as hc

FS = 50.0
L = 100
T = np.arange(L) / FS


def _random(n=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6, L))
    X[:, :3] += np.array([0.0, 0.0, 1.0])[None, :, None]  # gravity on az
    X[:, 3:] *= 50.0  # deg/s scale
    return X.astype(np.float32)


def _window(**channels):
    """One (1, 6, L) window; unnamed channels are 0."""
    X = np.zeros((1, 6, L), np.float32)
    for name, sig in channels.items():
        X[0, hc.CHANNELS.index(name)] = sig
    return X


def test_feature_count_and_names():
    names = hc.feature_names()
    assert len(hc.SIGNALS) == 8 and len(hc.STATS) == 12 and len(hc.CORR_PAIRS) == 6
    assert len(names) == 8 * 12 + 6 == 102
    assert len(set(names)) == len(names)
    assert names[0] == "ax_mean" and "amag_specent" in names and "gmag_zcr" in names
    assert names[-6:] == [
        "corr_ax_ay",
        "corr_ax_az",
        "corr_ay_az",
        "corr_gx_gy",
        "corr_gx_gz",
        "corr_gy_gz",
    ]
    df = hc.compute(_random(3))
    assert list(df.columns) == names and df.shape == (3, 102)


@pytest.mark.parametrize(
    "X",
    [
        _random(),
        np.full((4, 6, L), 0.7, np.float32),  # constant
        np.zeros((4, 6, L), np.float32),  # all zero
        np.concatenate([_random(2), np.full((2, 6, L), -3.0, np.float32)]),  # mixed
    ],
    ids=["random", "constant", "zeros", "mixed"],
)
def test_finite_and_float32(X):
    df = hc.compute(X)
    assert (df.dtypes == np.float32).all()
    assert np.isfinite(df.to_numpy()).all()


def test_constant_window_conventions():
    df = hc.compute(np.full((1, 6, L), 0.7, np.float32))
    for s in hc.SIGNALS:
        for st in ("zcr", "domfreq", "dompow", "specent", "std"):
            assert df[f"{s}_{st}"].iloc[0] == 0.0, f"{s}_{st}"
    corr = [c for c in df.columns if c.startswith("corr_")]
    assert (df[corr].to_numpy() == 0).all()
    assert df["ax_mean"].iloc[0] == pytest.approx(0.7)
    assert df["amag_mean"].iloc[0] == pytest.approx(0.7 * np.sqrt(3), rel=1e-6)


def test_basic_stats_match_numpy():
    X = _random(5)
    df = hc.compute(X)
    x = X[:, 1].astype(np.float64)
    np.testing.assert_allclose(df["ay_mean"], x.mean(1), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(df["ay_std"], x.std(1), rtol=1e-5)
    np.testing.assert_allclose(df["ay_p90"], np.percentile(x, 90, axis=1), rtol=1e-5)
    np.testing.assert_allclose(df["ay_rms"], np.sqrt((x**2).mean(1)), rtol=1e-5)
    g = np.linalg.norm(X[:, 3:].astype(np.float64), axis=1)
    np.testing.assert_allclose(df["gmag_max"], g.max(1), rtol=1e-5)


@pytest.mark.parametrize("f", [1.0, 2.5, 5.0, 12.0])
def test_sine_dominant_frequency(f):
    sig = 0.5 + np.sin(2 * np.pi * f * T + 0.3)  # integer number of cycles in 2 s
    df = hc.compute(_window(ax=sig), fs=FS)
    assert df["ax_domfreq"].iloc[0] == pytest.approx(f)
    assert df["ax_dompow"].iloc[0] > 0.99
    assert df["ax_specent"].iloc[0] < 0.05


def test_white_noise_spectrum_is_spread():
    df = hc.compute(_random(50, seed=3))
    assert df["gx_specent"].mean() > 0.8 and df["gx_dompow"].mean() < 0.2


def test_domfreq_respects_fs():
    f, fs = 4.0, 20.0
    t = np.arange(40) / fs
    X = np.zeros((1, 6, 40), np.float32)
    X[0, 0] = np.sin(2 * np.pi * f * t + 0.3)
    assert hc.compute(X, fs=fs)["ax_domfreq"].iloc[0] == pytest.approx(f)


def test_zcr_of_sine():
    f = 5.0  # 10 cycles in the window -> ~20 crossings over 99 sample pairs
    df = hc.compute(_window(gy=30 * np.sin(2 * np.pi * f * T + 0.3)))
    # Zeros at t = (k*pi - 0.3) / (2*pi*f) for k = 1..19 fall inside [0, 1.98 s].
    assert df["gy_zcr"].iloc[0] == pytest.approx(19 / 99, abs=1e-6)
    assert df["gy_zcr"].iloc[0] == pytest.approx(2 * f / FS, abs=0.01)


def test_correlations_identical_and_opposite():
    sig = np.sin(2 * np.pi * 2 * T) + 0.2 * np.cos(2 * np.pi * 7 * T)
    df = hc.compute(_window(ax=sig, ay=3 * sig + 1, az=-sig, gx=sig, gy=-2 * sig, gz=sig))
    assert df["corr_ax_ay"].iloc[0] == pytest.approx(1.0, abs=1e-6)
    assert df["corr_ax_az"].iloc[0] == pytest.approx(-1.0, abs=1e-6)
    assert df["corr_ay_az"].iloc[0] == pytest.approx(-1.0, abs=1e-6)
    assert df["corr_gx_gy"].iloc[0] == pytest.approx(-1.0, abs=1e-6)
    assert df["corr_gx_gz"].iloc[0] == pytest.approx(1.0, abs=1e-6)
    corr = df[[c for c in df.columns if c.startswith("corr_")]].to_numpy()
    assert (np.abs(corr) <= 1).all()


def test_chunked_equals_unchunked_and_memmap(tmp_path):
    X = _random(37, seed=1)
    full = hc.compute(X, chunk=1000)
    for chunk in (1, 5, 36):
        np.testing.assert_array_equal(hc.compute(X, chunk=chunk).to_numpy(), full.to_numpy())
    np.save(tmp_path / "X.npy", X)
    Xm = np.load(tmp_path / "X.npy", mmap_mode="r")
    np.testing.assert_array_equal(hc.compute(Xm, chunk=8).to_numpy(), full.to_numpy())


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        hc.compute(np.zeros((2, 5, L), np.float32))
    X = _random(2)
    X[1, 2, 7] = np.nan
    with pytest.raises(ValueError):
        hc.compute(X)
