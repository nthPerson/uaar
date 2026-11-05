"""uaar.eval.metrics: spec §7 invariants, hand-built cases and degenerate inputs."""

import math

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from uaar.eval import metrics, threshold


def _random_scores(seed, n_known=700, n_unknown=300, decimals=None):
    rng = np.random.default_rng(seed)
    sk = rng.normal(0.0, 1.0, n_known)
    su = rng.normal(0.8, 1.2, n_unknown)
    if decimals is not None:  # force many ties
        sk, su = np.round(sk, decimals), np.round(su, decimals)
    return sk, su


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("decimals", [None, 1])
def test_auroc_aupr_agree_with_sklearn(seed, decimals):
    sk, su = _random_scores(seed, decimals=decimals)
    y = np.r_[np.zeros(len(sk)), np.ones(len(su))]
    s = np.r_[sk, su]
    m = metrics.unknown_detection(sk, su)
    assert m["auroc"] == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert m["aupr"] == pytest.approx(average_precision_score(y, s), abs=1e-12)
    assert m["n_known"] == len(sk) and m["n_unknown"] == len(su)
    assert m["ratio"] == pytest.approx(len(su) / len(sk))


def test_roc_points_area_matches_auroc():
    sk, su = _random_scores(3, decimals=1)
    fpr, tpr = metrics.roc_points(sk, su)
    assert (fpr[0], tpr[0]) == (0.0, 0.0) and (fpr[-1], tpr[-1]) == (1.0, 1.0)
    assert np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0)
    area = np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)
    assert area == pytest.approx(metrics.unknown_detection(sk, su)["auroc"], abs=1e-12)


def test_auroc_hand_built():
    assert metrics.unknown_detection([0, 1], [2, 3])["auroc"] == 1.0
    assert metrics.unknown_detection([2, 3], [0, 1])["auroc"] == 0.0
    assert metrics.unknown_detection([1, 1], [1, 1])["auroc"] == 0.5


def test_fpr95_hand_built():
    known = np.arange(1, 21, dtype=float)  # 95th percentile (linear) = 19.05
    unknown = np.array([5.0, 19.0, 19.5, 30.0])  # 5 and 19 are accepted (<= 19.05)
    m = metrics.unknown_detection(known, unknown)
    assert m["fpr95"] == pytest.approx(0.5)
    # perfectly separated -> nothing unknown accepted; fully overlapping and higher -> 1
    assert metrics.fpr_at_accept(known, known + 100) == 0.0
    assert metrics.fpr_at_accept(known, np.zeros(3)) == 1.0


def test_select_tau_accepts_95_percent_of_known():
    s = np.random.default_rng(0).uniform(size=20_000)
    tau = threshold.select_tau(s, 0.95)
    assert (s <= tau).mean() == pytest.approx(0.95, abs=1e-3)


def test_closed_set_hand_built_and_ignores_unknown_rows():
    y = np.array([0, 0, 1, 1, -1])
    p = np.array([0, 1, 1, 1, 0])
    m = metrics.closed_set(y, p)
    assert m["n"] == 4 and m["accuracy"] == pytest.approx(0.75)
    # class 0: P=1, R=.5 -> 2/3; class 1: P=2/3, R=1 -> .8
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)


def test_open_set_hand_built():
    # four known windows (K = 2) and two unknown, tau = 0.5
    y_true = np.array([0, 0, 1, 1, -1, -1])
    y_pred = np.array([0, 1, 1, 1, 0, 1])
    score = np.array([0.1, 0.2, 0.9, 0.3, 0.8, 0.1])
    m = metrics.open_set(y_true, y_pred, score, tau=0.5)
    # decisions [0, 1, unk, 1, unk, 1]: F1 class0 = 2/3, class1 = 0.4, unknown = 0.5
    assert m["open_macro_f1"] == pytest.approx((2 / 3 + 0.4 + 0.5) / 3)
    assert m["unknown_recall"] == pytest.approx(0.5)
    assert m["known_accept_rate"] == pytest.approx(0.75)
    assert (m["n_known"], m["n_unknown"]) == (4, 2)


def test_misclassification_auroc():
    score = np.array([0.1, 0.2, 0.3, 0.9, 0.8])
    correct = np.array([True, True, True, False, False])
    assert metrics.misclassification_auroc(score, correct) == 1.0
    assert metrics.misclassification_auroc(-score, correct) == 0.0


def _aurc_by_loop(score, correct):
    order = np.argsort(score, kind="mergesort")
    wrong = ~correct[order]
    return np.mean(np.cumsum(wrong) / np.arange(1, len(wrong) + 1))


def test_risk_coverage_hand_built():
    cov, risk, aurc = metrics.risk_coverage(
        np.array([0.1, 0.2, 0.3, 0.4]), np.array([True, True, False, True])
    )
    assert np.allclose(cov, [0.25, 0.5, 0.75, 1.0])
    assert np.allclose(risk, [0, 0, 1 / 3, 1 / 4])
    assert aurc == pytest.approx((1 / 3 + 1 / 4) / 4)


def test_aurc_of_perfect_ranking():
    rng = np.random.default_rng(0)
    # every prediction right: AURC = 0 whatever the scores
    assert metrics.risk_coverage(rng.uniform(size=50), np.ones(50, bool))[2] == 0.0
    # errors ranked strictly last: AURC equals the oracle value and E-AURC = 0
    correct = rng.uniform(size=500) > 0.2
    score = np.where(correct, rng.uniform(0, 1, 500), rng.uniform(2, 3, 500))
    _, _, aurc = metrics.risk_coverage(score, correct)
    assert aurc == pytest.approx(_aurc_by_loop(score, correct))
    assert metrics.excess_aurc(score, correct) == pytest.approx(0.0, abs=1e-12)
    # a random ranking is worse
    assert metrics.excess_aurc(rng.uniform(size=500), correct) > 0.01


def test_aurc_is_tie_order_invariant():
    rng = np.random.default_rng(1)
    score = rng.integers(0, 5, 300).astype(float)
    correct = rng.uniform(size=300) > 0.3
    a = metrics.risk_coverage(score, correct)[2]
    perm = rng.permutation(300)
    b = metrics.risk_coverage(score[perm], correct[perm])[2]
    assert a == pytest.approx(b, abs=1e-12)


def test_degenerate_inputs_return_nan():
    e = np.array([])
    m = metrics.unknown_detection(e, e)
    assert all(math.isnan(m[k]) for k in ("auroc", "aupr", "fpr95", "ratio"))
    m = metrics.unknown_detection([0.1, 0.2], e)  # no unknown windows
    assert math.isnan(m["auroc"]) and math.isnan(m["aupr"]) and math.isnan(m["fpr95"])
    assert m["ratio"] == 0.0
    assert math.isnan(metrics.unknown_detection(e, [0.3])["auroc"])

    cs = metrics.closed_set(e, e)
    assert math.isnan(cs["accuracy"]) and math.isnan(cs["macro_f1"]) and cs["n"] == 0
    assert math.isnan(metrics.closed_set([-1, -1], [0, 1])["accuracy"])

    os_ = metrics.open_set(e, e, e, 0.5)
    assert all(math.isnan(os_[k]) for k in ("open_macro_f1", "unknown_recall", "known_accept_rate"))
    only_known = metrics.open_set([0, 1], [0, 1], [0.1, 0.2], 0.5)
    assert math.isnan(only_known["unknown_recall"]) and only_known["known_accept_rate"] == 1.0

    # one class only for the misclassification AUROC
    assert math.isnan(metrics.misclassification_auroc([0.1, 0.2], [True, True]))
    assert math.isnan(metrics.misclassification_auroc([0.1, 0.2], [False, False]))
    assert math.isnan(metrics.misclassification_auroc(e, e))

    cov, risk, aurc = metrics.risk_coverage(e, e)
    assert cov.size == 0 and risk.size == 0 and math.isnan(aurc)
    assert math.isnan(metrics.excess_aurc(e, e))
    fpr, tpr = metrics.roc_points(e, [1.0])
    assert fpr.size == 0 and tpr.size == 0
