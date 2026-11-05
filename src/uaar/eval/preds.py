"""Write and read the per-window prediction files ``preds_{set}.parquet`` (spec §3.5).

Owner: Sirena. AI assistance recorded in docs/ai-use-log.md.

One row per window with the columns ``window_id``, ``y_true`` (−1 for unknown), ``y_pred``,
``p_max``, one float column per score (``msp``, ``entropy``, ``vacuity``, ...) and ``probs`` as a
fixed-length list column of float32. Because these files are saved, tables and figures can be
rebuilt without models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from uaar.models.base import Prediction

#: Columns that are never scores.
BASE_COLUMNS: tuple[str, ...] = ("window_id", "y_true", "y_pred", "p_max", "probs")


def preds_frame(window_ids: np.ndarray, y_true: np.ndarray, pred: Prediction) -> pd.DataFrame:
    """One row per window, in the layout of spec §3.5 (identical to what train.py writes).

    Args:
        window_ids: ``(N,)`` ``window_id`` values.
        y_true: ``(N,)`` int labels, −1 for unknown.
        pred: The method's :class:`~uaar.models.base.Prediction` for the same N windows.

    Returns:
        DataFrame with ``window_id``, ``y_true``, ``y_pred``, ``p_max``, one column per score in
        ``pred.scores`` order, and ``probs`` (list of K float32 per row).
    """
    df = pd.DataFrame(
        {
            "window_id": window_ids,
            "y_true": y_true,
            "y_pred": pred.label,
            "p_max": pred.probs.max(axis=1),
        }
    )
    for name, s in pred.scores.items():
        df[name] = s
    df["probs"] = list(pred.probs.astype(np.float32))
    return df


def preds_path(run_dir: Path, set_name: str) -> Path:
    """``<run_dir>/preds_{set_name}.parquet``."""
    return Path(run_dir) / f"preds_{set_name}.parquet"


def write_preds(df: pd.DataFrame, run_dir: Path, set_name: str) -> Path:
    """Write one predictions frame to ``preds_{set_name}.parquet``; returns the path."""
    path = preds_path(run_dir, set_name)
    df.to_parquet(path, index=False)
    return path


def read_preds(run_dir: Path, set_name: str) -> pd.DataFrame:
    """Read ``preds_{set_name}.parquet`` from a run directory."""
    return pd.read_parquet(preds_path(run_dir, set_name))


def available_sets(run_dir: Path) -> list[str]:
    """Set names with a ``preds_*.parquet`` file in ``run_dir``, sorted."""
    return sorted(p.stem.removeprefix("preds_") for p in Path(run_dir).glob("preds_*.parquet"))


def score_columns(df: pd.DataFrame) -> list[str]:
    """Names of the score columns of a predictions frame (everything not in BASE_COLUMNS)."""
    return [c for c in df.columns if c not in BASE_COLUMNS]


def probs_matrix(df: pd.DataFrame, n_classes: int | None = None) -> np.ndarray:
    """The ``probs`` list column as a float32 ``(N, K)`` array.

    ``n_classes`` is needed only for an empty frame (otherwise K is read from the first row);
    an empty frame without it gives shape ``(0, 0)``.
    """
    if len(df) == 0:
        return np.zeros((0, n_classes or 0), dtype=np.float32)
    return np.stack(df["probs"].to_numpy()).astype(np.float32)


def to_prediction(df: pd.DataFrame, n_classes: int | None = None) -> Prediction:
    """Rebuild a :class:`Prediction` from a predictions frame (inverse of :func:`preds_frame`)."""
    return Prediction(
        probs=probs_matrix(df, n_classes),
        label=df["y_pred"].to_numpy(),
        scores={c: df[c].to_numpy(dtype=np.float64) for c in score_columns(df)},
    )
