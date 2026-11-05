"""M1: multinomial logistic regression on handcrafted features (models.md §3, spec §4.3).

Owner: both. AI assistance recorded in docs/ai-use-log.md.

The method takes raw windows ``(N, 6, L)`` like the neural methods and computes the 102
handcrafted features (``uaar.features.handcrafted``) internally in both ``fit`` and ``predict``,
so the live demo can call ``predict`` on raw windows. The model is
``Pipeline(StandardScaler, LogisticRegression)``: L2 penalty, multinomial (softmax) loss,
class weights from :func:`uaar.models.base.class_weights` so the imbalance handling matches M2-M4.

Choosing C (``cfg.model.logreg.C_grid``), scored by macro-F1:

- with ``groups`` (subject id per training window): grouped inner cross-validation,
  ``GroupKFold(cfg.model.logreg.inner_folds)`` over the training subjects, then a refit on all
  training windows (the spec);
- without ``groups``: macro-F1 on the validation windows; the model is trained on train only.

The grid is fitted in ascending C with warm starts (each fit starts from the previous C's
solution), the same idea as ``LogisticRegressionCV``. Every fit still runs ``lbfgs`` to its own
tolerance, so the result is the optimum for that C; the warm start only saves iterations (about
2.4x on RecoFit fold 0). The scaler is refitted on each inner training part, so no inner
validation statistics leak into the standardization.

Determinism: ``lbfgs`` is deterministic for a fixed thread count. scikit-learn's loss uses
OpenMP reductions, so the inner-fold scores can differ in the 4th decimal between ``n_jobs``
settings or machines (each parallel worker gets fewer threads); on RecoFit fold 0 the chosen C
and the final model (fitted in the main process) were bit-identical for ``n_jobs`` 1 and 3.

The per-C scores and the chosen C go to ``history.csv`` and ``config.json``.
Scores: ``msp`` = 1 - max probability, and predictive entropy (nats).
"""

from __future__ import annotations

import copy
import json
import time
import warnings
from collections.abc import Iterator
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from uaar.features import handcrafted
from uaar.models.base import Prediction, class_weights, entropy

#: Used for any ``model.logreg.*`` key the config does not set.
DEFAULTS = {
    "C_grid": [0.01, 0.1, 1.0, 10.0],
    "inner_folds": 3,
    "solver": "lbfgs",
    "max_iter": 2000,  # lbfgs needs ~1000 iterations at C = 10 on ~140k RecoFit windows
    "tol": 1e-4,
    # Inner folds run in parallel (joblib/loky caps each worker's OpenMP threads). 3 workers cut
    # the grouped selection on RecoFit fold 0 from 517 s to 142 s on 8 cores.
    "n_jobs": 3,
}


def _cfg_get(cfg, dotted: str, default):
    """``cfg.get`` for a :class:`uaar.config.Config` (or ``None`` -> default)."""
    if cfg is None:
        return default
    v = cfg.get(dotted, default)
    return default if v is None else v


def _c_path(
    F: np.ndarray, y: np.ndarray, grid: list[float], clf_params: dict
) -> Iterator[tuple[float, StandardScaler, LogisticRegression]]:
    """Fit scaler + logistic regression at each C of the ascending ``grid``, warm-started.

    Yields ``(C, scaler, clf)``; ``clf`` is the same object refitted at each step, so copy it
    (:func:`_freeze`) to keep one.
    """
    scaler = StandardScaler().fit(F)
    Z = scaler.transform(F)
    clf = LogisticRegression(**clf_params, warm_start=True)
    for c in grid:
        clf.set_params(C=c)
        clf.fit(Z, y)
        yield c, scaler, clf


def _freeze(scaler: StandardScaler, clf: LogisticRegression) -> Pipeline:
    """An independent fitted ``Pipeline(scaler, clf)`` with warm start switched off."""
    clf = copy.deepcopy(clf)
    clf.set_params(warm_start=False)
    return Pipeline([("scaler", copy.deepcopy(scaler)), ("clf", clf)])


def _fold_scores(
    F: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
    grid: list[float],
    clf_params: dict,
) -> list[tuple[float, int]]:
    """``(macro-F1 on eval_idx, lbfgs iterations)`` per C for one inner fold."""
    out = []
    for _, scaler, clf in _c_path(F[fit_idx], y[fit_idx], grid, clf_params):
        pred = clf.predict(scaler.transform(F[eval_idx]))
        out.append((float(f1_score(y[eval_idx], pred, average="macro")), int(clf.n_iter_.max())))
    return out


class LogRegMethod:
    """M1 behind the common Method interface (spec §4.3).

    Args:
        n_classes: K, the number of known classes; labels are ints in ``[0, K)``.
        classes: The K class names in label order.
        seed: Passed to ``LogisticRegression(random_state=...)`` (only stochastic solvers use
            it; ``lbfgs`` is deterministic either way).
    """

    name = "logreg"
    score_names = ["msp", "entropy"]

    def __init__(self, n_classes: int, classes: list[str], seed: int = 0):
        if len(classes) != n_classes:
            raise ValueError(f"{n_classes=} but {len(classes)} class names")
        self.n_classes = n_classes
        self.classes = list(classes)
        self.seed = seed
        self.pipeline: Pipeline | None = None
        self.C: float | None = None
        self.fs: float = 50.0
        self.window_samples: int = 100
        self.selection: dict = {}
        self.settings: dict = {}

    # ---- features ---------------------------------------------------------------------------
    def features(self, X: np.ndarray) -> np.ndarray:
        """Handcrafted features of raw windows ``X`` ``(N, 6, L)`` -> float32 ``(N, 102)``."""
        if not isinstance(X, np.ndarray):  # a memmap is an ndarray and is left as is
            X = np.asarray(X)
        if X.ndim == 2:  # one window (6, L)
            X = X[None]
        return handcrafted.compute(X, fs=self.fs).to_numpy()

    def _clf_params(self, weights: np.ndarray) -> dict:
        s = self.settings
        return {
            "l1_ratio": 0.0,  # pure L2 (sklearn >= 1.8 spelling of penalty="l2")
            "solver": s["solver"],
            "max_iter": s["max_iter"],
            "tol": s["tol"],
            "class_weight": {k: float(w) for k, w in enumerate(weights)},
            "random_state": self.seed,
        }

    # ---- Method interface -------------------------------------------------------------------
    def fit(
        self,
        Xtr: np.ndarray,
        ytr: np.ndarray,
        Xval: np.ndarray,
        yval: np.ndarray,
        cfg,
        run_dir: Path,
        groups: np.ndarray | None = None,
    ) -> None:
        """Choose C, fit the final pipeline and write ``run_dir/history.csv``.

        Args:
            Xtr: Training windows ``(N, 6, L)``, raw units (g, deg/s).
            ytr: Training labels ``(N,)`` int in ``[0, K)``.
            Xval, yval: Validation windows and labels; used to choose C when ``groups`` is None.
            cfg: Resolved :class:`uaar.config.Config`. Reads ``model.logreg.{C_grid,
                inner_folds, solver, max_iter, tol, n_jobs}`` (defaults in ``DEFAULTS``),
                ``train.class_weighting`` and ``data.window_s`` (sample rate = L / window_s).
            run_dir: Run directory; ``history.csv`` (one row per C) is written there.
            groups: Optional ``(N,)`` subject id per training window. When given, C is chosen
                by ``GroupKFold(inner_folds)`` cross-validation on the training subjects
                (mean macro-F1 over folds) and the final model is refit on all training
                windows. Ties go to the smaller C.
        """
        lr = dict(DEFAULTS)
        lr.update(_cfg_get(cfg, "model.logreg", {}) or {})
        grid = sorted(float(c) for c in lr["C_grid"])
        if not grid:
            raise ValueError("model.logreg.C_grid is empty")
        self.settings = {
            "solver": str(lr["solver"]),
            "max_iter": int(lr["max_iter"]),
            "tol": float(lr["tol"]),
            "class_weighting": str(_cfg_get(cfg, "train.class_weighting", "inv_sqrt")),
        }
        max_iter = self.settings["max_iter"]
        ytr = np.asarray(ytr, dtype=np.int64)
        yval = np.asarray(yval, dtype=np.int64)
        self.window_samples = int(Xtr.shape[-1])
        self.fs = self.window_samples / float(_cfg_get(cfg, "data.window_s", 2.0))
        params = self._clf_params(
            class_weights(ytr, self.n_classes, self.settings["class_weighting"])
        )

        t0 = time.perf_counter()
        Ftr = self.features(Xtr)
        feat_s = time.perf_counter() - t0
        rows: list[dict] = []
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            if groups is not None:
                groups = np.asarray(groups)
                if len(groups) != len(ytr):
                    raise ValueError("groups must have one entry per training window")
                n_folds = int(lr["inner_folds"])
                folds = list(GroupKFold(n_splits=n_folds).split(Ftr, ytr, groups))
                per_fold = joblib.Parallel(n_jobs=int(lr["n_jobs"]))(
                    joblib.delayed(_fold_scores)(Ftr, ytr, a, b, grid, params) for a, b in folds
                )
                f1 = np.array([[s for s, _ in fold] for fold in per_fold])  # (folds, C)
                iters = np.array([[n for _, n in fold] for fold in per_fold])
                for i, c in enumerate(grid):
                    rows.append(
                        {
                            "C": c,
                            "macro_f1": float(f1[:, i].mean()),
                            "macro_f1_std": float(f1[:, i].std()),
                            **{f"fold{f}_macro_f1": float(f1[f, i]) for f in range(n_folds)},
                            "n_iter": int(iters[:, i].max()),
                            "converged": bool(iters[:, i].max() < max_iter),
                        }
                    )
                self.C = grid[int(np.argmax(f1.mean(axis=0)))]  # first max = smallest C
                select_s = time.perf_counter() - t0
                t0 = time.perf_counter()
                for c, scaler, clf in _c_path(Ftr, ytr, grid, params):
                    if c == self.C:
                        self.pipeline = _freeze(scaler, clf)
                        break
                mode = f"group_kfold{n_folds}"
            else:
                Fval = self.features(Xval)
                best = -np.inf
                for c, scaler, clf in _c_path(Ftr, ytr, grid, params):
                    score = float(
                        f1_score(yval, clf.predict(scaler.transform(Fval)), average="macro")
                    )
                    n_iter = int(clf.n_iter_.max())
                    rows.append(
                        {
                            "C": c,
                            "macro_f1": score,
                            "n_iter": n_iter,
                            "converged": n_iter < max_iter,
                        }
                    )
                    if score > best:  # ties keep the smaller C
                        best, self.C, self.pipeline = score, c, _freeze(scaler, clf)
                select_s = time.perf_counter() - t0
                t0 = time.perf_counter()
                mode = "validation"
            refit_s = time.perf_counter() - t0
        n_conv_warnings = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
        for w in caught:  # re-emit anything that was not a convergence warning
            if not issubclass(w.category, ConvergenceWarning):
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)

        final_iter = int(self.pipeline[-1].n_iter_.max())
        self.selection = {
            "mode": mode,
            "metric": "macro_f1",
            "warm_start_path": True,
            "C_grid": grid,
            "chosen_C": self.C,
            "scores": rows,
            "final_n_iter": final_iter,
            "final_converged": final_iter < max_iter,
            "convergence_warnings": n_conv_warnings,
            "feature_seconds": feat_s,
            "selection_seconds": select_s,
            "refit_seconds": refit_s,
        }
        hist = pd.DataFrame(rows)
        hist["chosen"] = hist["C"] == self.C
        hist.insert(0, "mode", mode)
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        hist.to_csv(run_dir / "history.csv", index=False)

    def predict(self, X: np.ndarray, cfg=None) -> Prediction:
        """Probabilities, labels and uncertainty scores for raw windows.

        Args:
            X: ``(N, 6, L)`` raw windows (or one ``(6, L)`` window), same L as in training.
            cfg: Unused; kept for the common interface.

        Returns:
            :class:`Prediction` with ``probs`` float32 ``(N, K)`` (rows sum to 1; a class absent
            from training gets probability 0), ``label`` ``(N,)`` and scores ``msp`` =
            1 - max p in [0, 1 - 1/K] and ``entropy`` in [0, log K] nats.
        """
        if self.pipeline is None:
            raise RuntimeError("fit or load first")
        if X.shape[-1] != self.window_samples:
            raise ValueError(
                f"windows have {X.shape[-1]} samples, trained on {self.window_samples}"
            )
        F = self.features(X)
        p_seen = self.pipeline.predict_proba(F)
        probs = np.zeros((len(F), self.n_classes), dtype=np.float64)
        probs[:, self.pipeline[-1].classes_.astype(np.int64)] = p_seen
        probs = probs.astype(np.float32)
        return Prediction(
            probs=probs,
            label=probs.argmax(axis=1),
            scores={"msp": 1.0 - probs.max(axis=1), "entropy": entropy(probs)},
        )

    def save(self, run_dir: Path) -> None:
        """Write ``model.joblib``, ``config.json`` and ``preprocess.json`` to ``run_dir``."""
        if self.pipeline is None:
            raise RuntimeError("nothing to save: fit first")
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, run_dir / "model.joblib")
        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    "method": self.name,
                    "n_classes": self.n_classes,
                    "classes": self.classes,
                    "seed": self.seed,
                    "C": self.C,
                    **self.settings,
                    "n_features": len(handcrafted.feature_names()),
                    "feature_names": handcrafted.feature_names(),
                    "C_selection": self.selection,
                },
                indent=2,
            )
        )
        scaler: StandardScaler = self.pipeline[0]
        (run_dir / "preprocess.json").write_text(
            json.dumps(
                {
                    "window_samples": self.window_samples,
                    "sample_rate_hz": self.fs,
                    "channels": handcrafted.CHANNELS,
                    "features": "uaar.features.handcrafted",
                    "feature_mean": scaler.mean_.tolist(),
                    "feature_std": scaler.scale_.tolist(),
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, run_dir: Path) -> LogRegMethod:
        """Restore a saved method; ``predict`` gives the same output as before ``save``."""
        run_dir = Path(run_dir)
        conf = json.loads((run_dir / "config.json").read_text())
        pre = json.loads((run_dir / "preprocess.json").read_text())
        m = cls(conf["n_classes"], conf["classes"], seed=conf.get("seed", 0))
        m.pipeline = joblib.load(run_dir / "model.joblib")
        m.C = conf.get("C")
        m.selection = conf.get("C_selection", {})
        m.settings = {
            k: conf[k] for k in ("solver", "max_iter", "tol", "class_weighting") if k in conf
        }
        m.fs = float(pre["sample_rate_hz"])
        m.window_samples = int(pre["window_samples"])
        return m
