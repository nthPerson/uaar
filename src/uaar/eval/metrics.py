"""Closed-set, unknown-detection, open-set and selective-prediction metrics (evaluation.md §4).

Owner: Sirena. AI assistance recorded in docs/ai-use-log.md.

Conventions used throughout:

* A score is "higher = more uncertain" (the ``Prediction.scores`` convention, spec §4.3).
* For unknown detection the **unknown windows are the positive class**.
* A window is **accepted** (called known) when ``score <= tau`` and called unknown when
  ``score > tau``, matching :func:`uaar.eval.threshold.apply`.
* Degenerate inputs never raise: a metric that is undefined for the input (an empty array, or
  only one of the two classes an AUROC needs) is returned as ``nan``. Each function's docstring
  says which metrics can be ``nan`` and when.

AUROC and average precision are computed here from ranks and cumulative counts (NumPy/SciPy) and
checked against scikit-learn in ``tests/test_metrics.py``.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import f1_score

from uaar.eval.threshold import select_tau

NAN = float("nan")


# ---- Ranking primitives ----------------------------------------------------------------------
def _auroc(score_neg: np.ndarray, score_pos: np.ndarray) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie), via the Mann-Whitney U statistic.

    Ties get average ranks, which gives the same value as the trapezoidal ROC area. ``nan`` when
    either group is empty.
    """
    n_neg, n_pos = len(score_neg), len(score_pos)
    if n_neg == 0 or n_pos == 0:
        return NAN
    ranks = rankdata(np.concatenate([score_neg, score_pos]))  # average ranks for ties
    u = ranks[n_neg:].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_neg * n_pos))


def _average_precision(score_neg: np.ndarray, score_pos: np.ndarray) -> float:
    """Area under the precision-recall curve as the step sum ``sum_n (R_n - R_{n-1}) P_n``.

    One operating point per distinct score value (tied scores enter together), the same
    definition as ``sklearn.metrics.average_precision_score``. ``nan`` when there are no
    positives; 0 cannot happen with at least one positive.
    """
    if len(score_pos) == 0:
        return NAN
    s = np.concatenate([score_neg, score_pos])
    y = np.concatenate([np.zeros(len(score_neg)), np.ones(len(score_pos))])
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    last_of_group = np.r_[np.diff(s) != 0, True]  # keep the last index of each tied block
    tp, fp = tp[last_of_group], fp[last_of_group]
    precision = tp / (tp + fp)
    recall = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def roc_points(score_known: np.ndarray, score_unknown: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ROC curve for unknown detection (unknown = positive), one point per distinct score.

    Args:
        score_known: ``(n_known,)`` scores of known-class windows.
        score_unknown: ``(n_unknown,)`` scores of unknown windows.

    Returns:
        ``(fpr, tpr)``, both ``(n_points,)`` starting at ``(0, 0)`` and ending at ``(1, 1)``:
        ``fpr`` is the fraction of known windows called unknown, ``tpr`` the fraction of unknown
        windows called unknown, as the threshold sweeps from high to low. Both arrays are empty
        when either input is empty.
    """
    sk = np.asarray(score_known, dtype=np.float64).ravel()
    su = np.asarray(score_unknown, dtype=np.float64).ravel()
    if sk.size == 0 or su.size == 0:
        return np.array([]), np.array([])
    s = np.concatenate([sk, su])
    y = np.concatenate([np.zeros(sk.size), np.ones(su.size)])
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    last_of_group = np.r_[np.diff(s) != 0, True]
    tp = np.cumsum(y)[last_of_group]
    fp = np.cumsum(1.0 - y)[last_of_group]
    return np.r_[0.0, fp / sk.size], np.r_[0.0, tp / su.size]


# ---- Closed set --------------------------------------------------------------------------------
def closed_set(
    y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray | None = None
) -> dict[str, float]:
    """Accuracy and macro-F1 on known-class windows.

    Args:
        y_true: ``(N,)`` int labels; rows with ``y_true < 0`` (unknown) are dropped first.
        y_pred: ``(N,)`` int predicted known class.
        labels: Classes to average F1 over. Default: the classes present in ``y_true``, so a
            known class absent from the test subjects does not count as F1 = 0.

    Returns:
        ``{"accuracy", "macro_f1", "n"}``; both metrics are ``nan`` when no known rows remain.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    keep = y_true >= 0
    y_true, y_pred = y_true[keep], y_pred[keep]
    if y_true.size == 0:
        return {"accuracy": NAN, "macro_f1": NAN, "n": 0}
    labels = np.unique(y_true) if labels is None else np.asarray(labels)
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "n": int(y_true.size),
    }


# ---- Unknown detection -------------------------------------------------------------------------
def fpr_at_accept(
    score_known: np.ndarray, score_unknown: np.ndarray, accept: float = 0.95
) -> float:
    """Fraction of unknown windows accepted as known at the threshold accepting ``accept`` of
    the known windows.

    The threshold is the ``accept`` quantile of ``score_known`` (the same rule as
    :func:`uaar.eval.threshold.select_tau`); a window is accepted when ``score <= threshold``.
    ``nan`` when either input is empty.
    """
    sk = np.asarray(score_known, dtype=np.float64).ravel()
    su = np.asarray(score_unknown, dtype=np.float64).ravel()
    if sk.size == 0 or su.size == 0:
        return NAN
    t = select_tau(sk, accept)
    return float(np.mean(su <= t))


def unknown_detection(
    score_known: np.ndarray, score_unknown: np.ndarray, accept: float = 0.95
) -> dict[str, float]:
    """Threshold-free unknown-detection metrics with unknown windows as the positive class.

    Args:
        score_known: ``(n_known,)`` scores of known-class test windows.
        score_unknown: ``(n_unknown,)`` scores of unknown test windows.
        accept: Known-acceptance rate for the FPR metric (0.95 gives FPR@95).

    Returns:
        ``auroc``; ``aupr`` (average precision, unknown positive; its chance level is the
        unknown fraction, so ``ratio`` is reported with it); ``fpr95`` (fraction of unknown
        accepted when ``accept`` of known are accepted; lower is better); ``ratio`` =
        ``n_unknown / n_known``; ``n_known``; ``n_unknown``. ``auroc``, ``aupr`` and ``fpr95``
        are ``nan`` when either group is empty; ``ratio`` is ``nan`` when ``n_known == 0``.
    """
    sk = np.asarray(score_known, dtype=np.float64).ravel()
    su = np.asarray(score_unknown, dtype=np.float64).ravel()
    return {
        "auroc": _auroc(sk, su),
        "aupr": _average_precision(sk, su) if sk.size else NAN,
        "fpr95": fpr_at_accept(sk, su, accept),
        "ratio": float(su.size / sk.size) if sk.size else NAN,
        "n_known": int(sk.size),
        "n_unknown": int(su.size),
    }


# ---- Open set ----------------------------------------------------------------------------------
def open_set(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score: np.ndarray,
    tau: float,
    labels: np.ndarray | None = None,
) -> dict[str, float]:
    """Open-set classification with the decision ``unknown if score > tau else y_pred``.

    Args:
        y_true: ``(N,)`` int labels, ``-1`` for unknown windows (``splits.encode_labels``).
        y_pred: ``(N,)`` closed-set predicted class (argmax over the K known classes).
        score: ``(N,)`` uncertainty score, higher = more uncertain.
        tau: Threshold from validation known data.
        labels: Known classes to include. Default: the known classes present in ``y_true``;
            ``-1`` (unknown) is always added, giving the K + 1 classes.

    Returns:
        ``open_macro_f1`` (macro-F1 over the known classes plus unknown, with unknown as its own
        class); ``unknown_recall`` (fraction of ``y_true == -1`` with ``score > tau``);
        ``known_accept_rate`` (fraction of known windows with ``score <= tau``); ``n_known``;
        ``n_unknown``. ``open_macro_f1`` is ``nan`` for empty input; ``unknown_recall`` is
        ``nan`` without unknown rows and ``known_accept_rate`` without known rows.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    score = np.asarray(score, dtype=np.float64).ravel()
    is_unk = y_true < 0
    rejected = score > tau
    n_unk, n_known = int(is_unk.sum()), int((~is_unk).sum())
    out = {
        "open_macro_f1": NAN,
        "unknown_recall": float(rejected[is_unk].mean()) if n_unk else NAN,
        "known_accept_rate": float((~rejected[~is_unk]).mean()) if n_known else NAN,
        "n_known": n_known,
        "n_unknown": n_unk,
    }
    if y_true.size == 0:
        return out
    yt = np.where(is_unk, -1, y_true)
    yp = np.where(rejected, -1, y_pred)
    known = np.unique(yt[~is_unk]) if labels is None else np.asarray(labels)
    all_labels = np.union1d(known, [-1])
    out["open_macro_f1"] = float(
        f1_score(yt, yp, labels=all_labels, average="macro", zero_division=0)
    )
    return out


# ---- Misclassification detection and selective prediction --------------------------------------
def misclassification_auroc(score: np.ndarray, correct: np.ndarray) -> float:
    """AUROC of ``score`` for flagging the model's own mistakes (wrong predictions = positive).

    Args:
        score: ``(N,)`` uncertainty score on known-class windows.
        correct: ``(N,)`` bool, True where the closed-set prediction is right.

    Returns:
        AUROC in [0, 1]; ``nan`` when every prediction is right, every one is wrong, or N = 0.
    """
    score = np.asarray(score, dtype=np.float64).ravel()
    correct = np.asarray(correct, dtype=bool).ravel()
    return _auroc(score[correct], score[~correct])


def _tie_aware_error_cumsum(score: np.ndarray, wrong: np.ndarray) -> tuple:
    """Sort by score ascending and group ties; helper for the risk-coverage functions."""
    order = np.argsort(score, kind="mergesort")
    s, w = score[order], wrong[order].astype(np.float64)
    _, start, count = np.unique(s, return_index=True, return_counts=True)
    group_err = np.add.reduceat(w, start)
    return start, count, group_err


def risk_coverage(score: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Risk-coverage curve and its area (AURC) for selective prediction.

    Windows are retained from the most confident (lowest score) upward. Tied scores enter
    together, so the curve has one point per distinct score value. AURC is the mean risk over
    the N per-window coverage levels ``1/N, 2/N, ..., 1``, where inside a block of tied scores
    the errors are spread evenly (the expectation over orderings of the ties), so the result
    does not depend on how ties happen to be sorted.

    Args:
        score: ``(N,)`` uncertainty score, higher = less confident.
        correct: ``(N,)`` bool, True where the prediction is right.

    Returns:
        ``(coverage, risk, aurc)``: ``coverage`` and ``risk`` are ``(n_distinct,)`` (fraction
        retained, error rate among the retained); ``aurc`` is a float in [0, 1]. AURC is 0 when
        every prediction is right; with errors present even a perfect ranking has AURC > 0 (see
        :func:`excess_aurc`). Empty input gives empty arrays and ``nan``.
    """
    score = np.asarray(score, dtype=np.float64).ravel()
    wrong = ~np.asarray(correct, dtype=bool).ravel()
    n = score.size
    if n == 0:
        return np.array([]), np.array([]), NAN
    start, count, group_err = _tie_aware_error_cumsum(score, wrong)
    n_cum = np.cumsum(count)
    err_cum = np.cumsum(group_err)
    coverage = n_cum / n
    risk = err_cum / n_cum

    # Per-window risk with errors spread evenly inside each tie block.
    group = np.repeat(np.arange(len(count)), count)
    k_in_group = np.arange(n) - start[group] + 1  # 1..count within the block
    err_before = (err_cum - group_err)[group]
    err_i = err_before + k_in_group / count[group] * group_err[group]
    aurc = float(np.mean(err_i / np.arange(1, n + 1)))
    return coverage, risk, aurc


def excess_aurc(score: np.ndarray, correct: np.ndarray) -> float:
    """AURC minus the AURC of a perfect ranking with the same error count (E-AURC).

    0 for a perfect ranking (every error ranked less confident than every correct prediction),
    which isolates ranking quality from accuracy. ``nan`` for empty input.
    """
    correct = np.asarray(correct, dtype=bool).ravel()
    if correct.size == 0:
        return NAN
    _, _, aurc = risk_coverage(score, correct)
    oracle = np.where(correct, 0.0, 1.0)  # errors get the highest score
    _, _, best = risk_coverage(oracle, correct)
    return float(aurc - best)
