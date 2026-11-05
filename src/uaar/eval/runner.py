"""Evaluate one fitted method on the P1 test sets: predict, then compute every metric (spec §3.5).

Owner: Sirena. AI assistance recorded in docs/ai-use-log.md.

This is the reusable core behind ``scripts/evaluate.py``. It has no CLI and does not care where
the method came from: anything with ``predict(X, cfg) -> Prediction`` (and ideally ``name`` and
``score_names``) works, including wrappers such as MC Dropout that have no run directory of
their own. :func:`compute_metrics` needs only the prediction frames, so metrics can also be
recomputed from saved ``preds_*.parquet`` files without a model.

Test sets
---------
``test_known``    test subjects, known classes (``splits.select(..., "test_known")``)
``test_unknown``  test subjects, held-out classes (``splits.select(..., "test_unknown")``)
``test_nonex``    test subjects, ``source == "multi"``, label in ``cfg.split.nonexercise_labels``
                  (``splits.select_nonexercise``; P1-b)

Unknown-detection and open-set metrics for ``test_unknown`` and ``test_nonex`` are each computed
against ``test_known`` of the same run, so ``test_known`` is always predicted (and returned) when
either of them is requested.

``metrics.json`` layout
-----------------------
``NaN`` (an undefined metric, see :mod:`uaar.eval.metrics`) is written as ``null``. Keys marked
(script) are added by ``scripts/evaluate.py``; everything else comes from :func:`evaluate`::

    {
      "run_id": str, "method": str, "dataset": str, "protocol": str,   # run_id, protocol (script)
      "split_id": "fold0_rand0", "seed": int,                         # seed (script)
      "K": int,                       # number of known classes
      "git_sha": str,                 # commit the run was trained at (script, from config.yaml)
      "eval_git_sha": str,            # commit evaluate.py ran at (script)
      "eval_seconds": float,          # wall time of model/data load + predict + metrics (script)
      "accept_rate": 0.95,            # cfg.eval.accept_rate, used for tau and FPR@95
      "n_bins": 15,                   # ECE bins
      "score_names": ["msp", "entropy"],
      "tau": {"msp": float, ...},     # from threshold.json
      "test_known": {
        "n": int,
        "closed_set":  {"accuracy", "macro_f1", "n"},
        "calibration": {"ece", "nll", "brier"},
        "scores": {
          "<score>": {"mean", "known_accept_rate", "misclassification_auroc", "aurc", "eaurc"}
        }
      },
      "test_unknown": {               # same layout for "test_nonex"
        "n": int,
        "scores": {
          "<score>": {
            "mean",                                                 # mean score on this set
            "auroc", "aupr", "fpr95", "ratio", "n_known", "n_unknown",   # metrics.unknown_detection
            "open_macro_f1", "unknown_recall", "known_accept_rate"      # metrics.open_set at tau
          }
        }
      }
    }

``ratio`` is n_unknown / n_known (AUPR's chance level depends on it). ``known_accept_rate`` is the
fraction of ``test_known`` windows with score <= tau; it should be close to ``accept_rate`` when
the validation and test subjects behave alike.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from uaar.data import splits
from uaar.eval import calibration, metrics, threshold
from uaar.eval.preds import preds_frame, probs_matrix
from uaar.models.base import Prediction

KNOWN_SET = "test_known"
UNKNOWN_SETS: tuple[str, ...] = ("test_unknown", "test_nonex")
SUPPORTED_SETS: tuple[str, ...] = (KNOWN_SET, *UNKNOWN_SETS)


# ---- Selecting and predicting ---------------------------------------------------------------
def window_ids_for(meta: pd.DataFrame, split: splits.Split, set_name: str, cfg) -> np.ndarray:
    """``window_id`` values of one test set (see the module docstring for the definitions)."""
    if set_name in ("test_known", "test_unknown"):
        return splits.select(meta, split, set_name)
    if set_name == "test_nonex":
        labels = list(cfg.get("split.nonexercise_labels", []) or [])
        return splits.select_nonexercise(meta, split, labels)
    raise ValueError(f"unsupported set {set_name!r}; supported: {SUPPORTED_SETS}")


def _empty_prediction(n_classes: int, score_names: Sequence[str]) -> Prediction:
    return Prediction(
        probs=np.zeros((0, n_classes), dtype=np.float32),
        label=np.zeros(0, dtype=np.int64),
        scores={s: np.zeros(0, dtype=np.float64) for s in score_names},
    )


def predict_chunked(
    method,
    X: np.ndarray,
    positions: np.ndarray,
    cfg,
    n_classes: int,
    score_names: Sequence[str],
    chunk_size: int = 8192,
) -> Prediction:
    """Predict ``X[positions]`` in chunks so a memory-mapped ``X`` is never fully loaded.

    Args:
        method: Object with ``predict(X, cfg) -> Prediction``.
        X: ``(N_all, C, L)`` windows (a ``np.load(..., mmap_mode="r")`` array is fine).
        positions: ``(n,)`` row positions into ``X``.
        cfg: Passed through to ``method.predict``.
        n_classes: K, used only to shape an empty result.
        score_names: Score keys, used only to shape an empty result.
        chunk_size: Windows per ``predict`` call.

    Returns:
        One :class:`Prediction` for the ``n`` windows, in ``positions`` order.
    """
    positions = np.asarray(positions, dtype=np.int64)
    if positions.size == 0:
        return _empty_prediction(n_classes, score_names)
    probs, labels, scores = [], [], {}
    for i in range(0, positions.size, chunk_size):
        xb = np.asarray(X[positions[i : i + chunk_size]], dtype=np.float32)
        p = method.predict(xb, cfg)
        probs.append(np.asarray(p.probs, dtype=np.float32))
        labels.append(np.asarray(p.label))
        for k, v in p.scores.items():
            scores.setdefault(k, []).append(np.asarray(v, dtype=np.float64))
    return Prediction(
        probs=np.concatenate(probs),
        label=np.concatenate(labels),
        scores={k: np.concatenate(v) for k, v in scores.items()},
    )


# ---- Metrics from prediction frames ---------------------------------------------------------
def _known_set_metrics(
    df: pd.DataFrame, taus: Mapping[str, float], score_names: Sequence[str], k: int, n_bins: int
) -> dict:
    y = df["y_true"].to_numpy(dtype=np.int64)
    yp = df["y_pred"].to_numpy(dtype=np.int64)
    probs = probs_matrix(df, k)
    correct = y == yp
    out = {
        "n": len(df),
        "closed_set": metrics.closed_set(y, yp),
        "calibration": {
            "ece": calibration.ece(probs, y, n_bins),
            "nll": calibration.nll(probs, y),
            "brier": calibration.brier(probs, y),
        },
        "scores": {},
    }
    for name in score_names:
        s = df[name].to_numpy(dtype=np.float64)
        _, _, aurc = metrics.risk_coverage(s, correct)
        out["scores"][name] = {
            "mean": float(s.mean()) if s.size else math.nan,
            "known_accept_rate": float(np.mean(s <= taus[name])) if s.size else math.nan,
            "misclassification_auroc": metrics.misclassification_auroc(s, correct),
            "aurc": aurc,
            "eaurc": metrics.excess_aurc(s, correct),
        }
    return out


def _unknown_set_metrics(
    df_known: pd.DataFrame,
    df_unk: pd.DataFrame,
    taus: Mapping[str, float],
    score_names: Sequence[str],
    accept: float,
) -> dict:
    y_all = np.concatenate(
        [df_known["y_true"].to_numpy(dtype=np.int64), np.full(len(df_unk), -1, dtype=np.int64)]
    )
    yp_all = np.concatenate(
        [df_known["y_pred"].to_numpy(dtype=np.int64), df_unk["y_pred"].to_numpy(dtype=np.int64)]
    )
    out = {"n": len(df_unk), "scores": {}}
    for name in score_names:
        sk = df_known[name].to_numpy(dtype=np.float64)
        su = df_unk[name].to_numpy(dtype=np.float64)
        row = {"mean": float(su.mean()) if su.size else math.nan}
        row |= metrics.unknown_detection(sk, su, accept)
        row |= {
            k: v
            for k, v in metrics.open_set(
                y_all, yp_all, np.concatenate([sk, su]), taus[name]
            ).items()
            if k not in ("n_known", "n_unknown")
        }
        out["scores"][name] = row
    return out


def compute_metrics(
    frames: Mapping[str, pd.DataFrame],
    taus: Mapping[str, float],
    score_names: Sequence[str],
    n_classes: int,
    accept_rate: float = 0.95,
    n_bins: int = 15,
) -> dict:
    """Per-set metrics from prediction frames alone (no model needed).

    Args:
        frames: ``{set_name: preds frame}`` (columns as in :func:`uaar.eval.preds.preds_frame`).
            ``test_known`` must be present whenever an unknown set is.
        taus: ``{score_name: tau}``; every name in ``score_names`` needs one.
        score_names: Scores to evaluate (columns of the frames).
        n_classes: K.
        accept_rate: Known acceptance rate for FPR@95 (and the rate tau was chosen at).
        n_bins: ECE bins.

    Returns:
        ``{set_name: {...}}`` with the per-set layout in the module docstring.
    """
    missing = [s for s in score_names if s not in taus]
    if missing:
        raise KeyError(f"no tau for score(s) {missing}; taus has {sorted(taus)}")
    out: dict = {}
    if KNOWN_SET in frames:
        out[KNOWN_SET] = _known_set_metrics(frames[KNOWN_SET], taus, score_names, n_classes, n_bins)
    for name, df in frames.items():
        if name == KNOWN_SET:
            continue
        if name not in UNKNOWN_SETS:
            raise ValueError(f"unsupported set {name!r}; supported: {SUPPORTED_SETS}")
        if KNOWN_SET not in frames:
            raise ValueError(f"{name} metrics need the {KNOWN_SET} frame as reference")
        out[name] = _unknown_set_metrics(frames[KNOWN_SET], df, taus, score_names, accept_rate)
    return out


def jsonable(obj):
    """Recursively convert NumPy scalars to Python and NaN/inf to ``None`` for strict JSON."""
    if isinstance(obj, Mapping):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.integer | bool | np.bool_):
        return obj.item() if hasattr(obj, "item") else obj
    if isinstance(obj, float | np.floating):
        f = float(obj)
        return f if math.isfinite(f) else None
    return obj


# ---- The whole evaluation -------------------------------------------------------------------
def window_positions(meta: pd.DataFrame, window_ids: np.ndarray) -> np.ndarray:
    """Row positions in ``meta`` (and in the aligned ``X``) of the given ``window_id`` values."""
    if not meta["window_id"].is_unique:
        raise ValueError("meta.window_id must be unique")
    pos = pd.Index(meta["window_id"].to_numpy()).get_indexer(np.asarray(window_ids))
    if (pos < 0).any():
        raise ValueError("window ids missing from meta")
    return pos


def validation_thresholds(
    method,
    X: np.ndarray,
    meta: pd.DataFrame,
    split: splits.Split,
    cfg,
    chunk_size: int = 8192,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Predict on the validation known-class windows and choose tau per score.

    ``train.py`` does this for the methods it trains; this is the same step for a method that
    does not train (MC Dropout on top of a softmax run).

    Returns:
        ``(preds_val frame, {score_name: tau})`` with tau at ``cfg.eval.accept_rate``.
    """
    ids = splits.select(meta, split, "val")
    pos = window_positions(meta, ids)
    score_names = list(method.score_names)
    pred = predict_chunked(method, X, pos, cfg, len(split.known_classes), score_names, chunk_size)
    y = splits.encode_labels(meta["label"].astype(str).to_numpy()[pos], split)
    accept = float(cfg.get("eval.accept_rate", 0.95))
    taus = {s: threshold.select_tau(pred.scores[s], accept) for s in score_names}
    return preds_frame(ids, y, pred), taus


def evaluate(
    method,
    taus: Mapping[str, float],
    X: np.ndarray,
    meta: pd.DataFrame,
    split: splits.Split,
    sets: Iterable[str],
    cfg,
    chunk_size: int = 8192,
) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Predict on each requested test set and compute every metric.

    Args:
        method: Any fitted object with ``predict(X, cfg) -> Prediction``; ``name`` and
            ``score_names`` attributes are used when present (else ``"unknown"`` and the keys of
            ``taus``).
        taus: ``{score_name: tau}`` chosen on validation known windows (``threshold.json``).
        X: ``(N_all, 6, L)`` windows aligned with ``meta`` (memory-mapped is fine).
        meta: Window metadata with ``window_id``, ``subject``, ``source``, ``label``.
        split: The run's split (its ``test_subjects`` and class lists define the sets).
        sets: Subset of ``SUPPORTED_SETS``. ``test_known`` is added when an unknown set is
            requested, because it is their reference.
        cfg: Run config; ``cfg.split.nonexercise_labels`` and ``cfg.eval.accept_rate`` are read,
            and it is passed to ``method.predict``.
        chunk_size: Windows per ``predict`` call.

    Returns:
        ``(metrics, frames)``: ``metrics`` is the ``metrics.json`` content minus the
        script-level keys (see module docstring); ``frames`` maps each evaluated set to its
        predictions frame (``preds_frame`` layout).
    """
    sets = list(dict.fromkeys(sets))
    bad = [s for s in sets if s not in SUPPORTED_SETS]
    if bad:
        raise ValueError(f"unsupported set(s) {bad}; supported: {SUPPORTED_SETS}")
    if any(s in UNKNOWN_SETS for s in sets) and KNOWN_SET not in sets:
        sets.insert(0, KNOWN_SET)
    sets.sort(key=SUPPORTED_SETS.index)

    k = len(split.known_classes)
    score_names = list(getattr(method, "score_names", None) or taus)
    accept_rate = float(cfg.get("eval.accept_rate", 0.95))
    n_bins = int(cfg.get("eval.n_bins", 15))

    if not meta["window_id"].is_unique:
        raise ValueError("meta.window_id must be unique")
    id_index = pd.Index(meta["window_id"].to_numpy())
    labels = meta["label"].astype(str).to_numpy()

    frames: dict[str, pd.DataFrame] = {}
    for name in sets:
        ids = window_ids_for(meta, split, name, cfg)
        pos = id_index.get_indexer(ids)
        if (pos < 0).any():
            raise ValueError(f"{name}: window ids missing from meta")
        pred = predict_chunked(method, X, pos, cfg, k, score_names, chunk_size)
        absent = [s for s in score_names if s not in pred.scores]
        if absent:
            raise KeyError(f"method did not return score(s) {absent}")
        y_true = (
            splits.encode_labels(labels[pos], split)
            if name == KNOWN_SET
            else np.full(len(pos), -1, dtype=np.int64)
        )
        frames[name] = preds_frame(ids, y_true, pred)

    out = {
        "method": getattr(method, "name", "unknown"),
        "dataset": split.dataset,
        "split_id": split.split_id,
        "K": k,
        "accept_rate": accept_rate,
        "n_bins": n_bins,
        "score_names": score_names,
        "tau": {s: float(taus[s]) for s in score_names if s in taus},
    }
    out |= compute_metrics(frames, taus, score_names, k, accept_rate, n_bins)
    return jsonable(out), frames
