"""M1 logistic regression (spec §4.3): fit, predict, scores, save/load, both C-selection paths."""

import json

import numpy as np
import pandas as pd
import pytest

from uaar.config import Config
from uaar.models.logreg import LogRegMethod

K = 4
CLASSES = [f"class{k}" for k in range(K)]
FREQS = [1.0, 3.0, 6.0, 10.0]  # Hz, one oscillation frequency per class


def _data(n_subjects=6, per=10, seed=0):
    """Windows whose class sets the oscillation frequency and the active gyro axis."""
    rng = np.random.default_rng(seed)
    t = np.arange(100) / 50
    X, y, g = [], [], []
    for s in range(n_subjects):
        for k in range(K):
            for _ in range(per):
                ph = rng.uniform(0, 2 * np.pi)
                amp = rng.uniform(0.5, 1.5)
                w = rng.normal(0, 0.1, (6, 100))
                w[2] += 1.0  # gravity
                w[0] += amp * np.sin(2 * np.pi * FREQS[k] * t + ph)
                w[3 + k % 3] += 80 * amp * np.cos(2 * np.pi * FREQS[k] * t + ph)
                X.append(w)
                y.append(k)
                g.append(f"S{s}")
    return np.asarray(X, np.float32), np.asarray(y), np.asarray(g)


def _cfg(**logreg):
    return Config(
        {
            "data": {"window_s": 2.0},
            "train": {"class_weighting": "inv_sqrt"},
            "model": {"logreg": {"C_grid": [0.01, 1.0, 10.0], "inner_folds": 3, **logreg}},
        }
    )


@pytest.fixture(scope="module")
def data():
    Xtr, ytr, gtr = _data(seed=0)
    Xva, yva, _ = _data(n_subjects=2, seed=1)
    return Xtr, ytr, gtr, Xva, yva


@pytest.fixture(scope="module")
def fitted(data, tmp_path_factory):
    Xtr, ytr, _, Xva, yva = data
    run_dir = tmp_path_factory.mktemp("run")
    m = LogRegMethod(K, CLASSES, seed=0)
    m.fit(Xtr, ytr, Xva, yva, _cfg(), run_dir)
    return m, run_dir


def test_fit_predict_accuracy_and_scores(fitted, data):
    m, _ = fitted
    Xva, yva = data[3], data[4]
    pred = m.predict(Xva)
    assert pred.probs.shape == (len(Xva), K) and pred.label.shape == (len(Xva),)
    assert (pred.label == yva).mean() > 0.95
    np.testing.assert_allclose(pred.probs.sum(axis=1), 1.0, atol=1e-5)
    assert set(pred.scores) == set(m.score_names) == {"msp", "entropy"}
    msp, ent = pred.scores["msp"], pred.scores["entropy"]
    assert msp.shape == ent.shape == (len(Xva),)
    assert (msp >= 0).all() and (msp <= 1 - 1 / K + 1e-6).all()
    assert (ent >= -1e-6).all() and (ent <= np.log(K) + 1e-6).all()
    np.testing.assert_allclose(msp, 1 - pred.probs.max(axis=1))


def test_single_window_predict(fitted, data):
    m, _ = fitted
    one = m.predict(data[3][0])
    assert one.probs.shape == (1, K)


def test_validation_path_history(fitted):
    m, run_dir = fitted
    hist = pd.read_csv(run_dir / "history.csv")
    assert list(hist["C"]) == [0.01, 1.0, 10.0] and (hist["mode"] == "validation").all()
    assert hist["chosen"].sum() == 1 and hist.loc[hist["chosen"], "C"].item() == m.C
    assert hist["macro_f1"].between(0, 1).all()
    assert m.C in (0.01, 1.0, 10.0) and m.selection["mode"] == "validation"


def test_grouped_path(data, tmp_path):
    Xtr, ytr, gtr, Xva, yva = data
    m = LogRegMethod(K, CLASSES, seed=0)
    m.fit(Xtr, ytr, Xva, yva, _cfg(), tmp_path, groups=gtr)
    hist = pd.read_csv(tmp_path / "history.csv")
    assert (hist["mode"] == "group_kfold3").all() and len(hist) == 3
    assert {"fold0_macro_f1", "fold1_macro_f1", "fold2_macro_f1"} <= set(hist.columns)
    assert hist.loc[hist["chosen"], "C"].item() == m.C
    best = hist["macro_f1"].max()
    assert hist.loc[hist["chosen"], "macro_f1"].item() == pytest.approx(best)
    assert (m.predict(Xva).label == yva).mean() > 0.95
    with pytest.raises(ValueError):
        m.fit(Xtr, ytr, Xva, yva, _cfg(), tmp_path, groups=gtr[:-1])
    par = LogRegMethod(K, CLASSES, seed=0)
    par.fit(Xtr, ytr, Xva, yva, _cfg(n_jobs=2), tmp_path / "par", groups=gtr)
    # n_jobs only changes OpenMP summation order (see the module docstring), not the result
    par_hist = pd.read_csv(tmp_path / "par" / "history.csv")
    pd.testing.assert_frame_equal(par_hist, hist, check_exact=False, rtol=1e-4)
    assert par.C == m.C
    np.testing.assert_allclose(par.predict(Xva).probs, m.predict(Xva).probs, atol=1e-5)


def test_save_load_round_trip(fitted, data):
    m, run_dir = fitted
    m.save(run_dir)
    for f in ("model.joblib", "config.json", "preprocess.json", "history.csv"):
        assert (run_dir / f).exists(), f
    conf = json.loads((run_dir / "config.json").read_text())
    assert conf["method"] == "logreg" and conf["classes"] == CLASSES and conf["n_classes"] == K
    assert conf["C"] == m.C and conf["seed"] == 0 and len(conf["feature_names"]) == 102
    pre = json.loads((run_dir / "preprocess.json").read_text())
    assert pre["window_samples"] == 100 and pre["sample_rate_hz"] == 50.0
    assert len(pre["feature_mean"]) == 102

    m2 = LogRegMethod.load(run_dir)
    a, b = m.predict(data[3]), m2.predict(data[3])
    np.testing.assert_array_equal(a.probs, b.probs)
    np.testing.assert_array_equal(a.label, b.label)
    for k in a.scores:
        np.testing.assert_array_equal(a.scores[k], b.scores[k])


def test_deterministic(data, tmp_path):
    Xtr, ytr, _, Xva, yva = data
    preds = []
    for i in range(2):
        m = LogRegMethod(K, CLASSES, seed=3)
        m.fit(Xtr, ytr, Xva, yva, _cfg(), tmp_path / str(i))
        preds.append(m.predict(Xva).probs)
    np.testing.assert_array_equal(preds[0], preds[1])


def test_missing_class_in_train(data, tmp_path):
    Xtr, ytr, _, Xva, yva = data
    keep = ytr != 2
    m = LogRegMethod(K, CLASSES, seed=0)
    m.fit(Xtr[keep], ytr[keep], Xva, yva, _cfg(C_grid=[1.0]), tmp_path)
    pred = m.predict(Xva)
    assert pred.probs.shape == (len(Xva), K) and (pred.probs[:, 2] == 0).all()
    np.testing.assert_allclose(pred.probs.sum(axis=1), 1.0, atol=1e-5)


def test_wrong_window_length(fitted):
    m, _ = fitted
    with pytest.raises(ValueError):
        m.predict(np.zeros((2, 6, 40), np.float32))
