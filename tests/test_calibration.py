"""uaar.eval.calibration: ECE of a calibrated predictor, known values, degenerate inputs."""

import math

import numpy as np
import pytest

from uaar.eval import calibration


def _calibrated(n=200_000, K=3, seed=0):
    """Predictor whose confidence c is exactly its probability of being right."""
    rng = np.random.default_rng(seed)
    c = rng.uniform(1 / K, 1.0, n)
    pred = rng.integers(0, K, n)
    probs = np.repeat(((1 - c) / (K - 1))[:, None], K, axis=1)
    probs[np.arange(n), pred] = c
    right = rng.uniform(size=n) < c
    other = (pred + rng.integers(1, K, n)) % K
    y = np.where(right, pred, other)
    return probs, y


def test_ece_of_perfectly_calibrated_predictor_is_near_zero():
    probs, y = _calibrated()
    assert calibration.ece(probs, y) < 0.01


def test_ece_of_overconfident_predictor():
    n = 1000
    probs = np.tile([0.9, 0.1], (n, 1))
    y = np.zeros(n, dtype=int)
    y[: int(0.4 * n)] = 1  # 60 % right at 90 % confidence
    assert calibration.ece(probs, y) == pytest.approx(0.3)


def test_reliability_bins_layout():
    probs, y = _calibrated(n=20_000)
    b = calibration.reliability_bins(probs, y, n_bins=10)
    assert b["edges"].shape == (11,) and b["count"].shape == (10,)
    assert int(b["n"]) == 20_000 == b["count"].sum()
    used = b["count"] > 0
    assert np.all(b["confidence"][used] > b["edges"][:-1][used] - 1e-12)
    assert np.all(b["confidence"][used] <= b["edges"][1:][used] + 1e-12)
    assert np.isnan(b["accuracy"][~used]).all()  # bins below 1/K are empty


def test_confidence_one_goes_to_last_bin():
    b = calibration.reliability_bins(np.array([[1.0, 0.0], [0.5, 0.5]]), np.array([0, 0]), 4)
    assert b["count"].tolist() == [0, 1, 0, 1]


def test_nll_and_brier_known_values():
    K = 4
    uniform = np.full((10, K), 1 / K)
    y = np.arange(10) % K
    assert calibration.nll(uniform, y) == pytest.approx(math.log(K))
    onehot = np.eye(K)[y]
    assert calibration.brier(onehot, y) == pytest.approx(0.0)
    assert calibration.nll(onehot, y) == pytest.approx(0.0, abs=1e-9)
    assert calibration.brier(np.full((3, 2), 0.5), np.array([0, 1, 0])) == pytest.approx(0.5)


def test_unknown_rows_dropped_and_empty_is_nan():
    probs = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert calibration.nll(probs, np.array([0, -1])) == pytest.approx(-math.log(0.9))
    e = np.zeros((0, 3))
    for fn in (calibration.ece, calibration.nll, calibration.brier):
        assert math.isnan(fn(e, np.array([], dtype=int)))
        assert math.isnan(fn(probs, np.array([-1, -1])))
    assert int(calibration.reliability_bins(e, np.array([], dtype=int))["n"]) == 0
