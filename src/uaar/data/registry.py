"""The single place that knows the processed-data directory layout (spec §3, §4.1).

Layout under ``<data_root>/processed/<dataset>/``::

    recordings.parquet        long format, one row per sample (§3.1)
    recordings.meta.json      rate, units, label counts, provenance
    windows/X.npy             float32 (N, 6, L)                     (§3.2)
    windows/meta.parquet      one row per window
    splits/<split_id>.json    fold{k}_{classsplit}                   (§3.3)
    features/handcrafted.parquet                                     (§3.4)

``<data_root>`` is ``$UAAR_DATA_DIR`` when set, else ``<repo>/data`` where ``<repo>`` is the
first ancestor of this package that contains ``pyproject.toml``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ENV_VAR = "UAAR_DATA_DIR"


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def data_root() -> Path:
    """``$UAAR_DATA_DIR`` if set, else ``<repo>/data``."""
    env = os.environ.get(ENV_VAR)
    return Path(env).expanduser() if env else _repo_root() / "data"


def paths(dataset: str) -> dict[str, Path]:
    """Every path of one dataset's processed layout.

    Args:
        dataset: ``recofit``, ``mmfit``, ``recgym`` or ``wearable``.

    Returns:
        Dict with keys ``root``, ``external``, ``recordings``, ``recordings_meta``,
        ``windows_dir``, ``windows_x``, ``windows_meta``, ``splits_dir``, ``features_dir``
        and ``features``; nothing is created.
    """
    root = data_root() / "processed" / dataset
    return {
        "root": root,
        "external": data_root() / "external" / dataset,
        "recordings": root / "recordings.parquet",
        "recordings_meta": root / "recordings.meta.json",
        "windows_dir": root / "windows",
        "windows_x": root / "windows" / "X.npy",
        "windows_meta": root / "windows" / "meta.parquet",
        "splits_dir": root / "splits",
        "features_dir": root / "features",
        "features": root / "features" / "handcrafted.parquet",
    }


def split_path(dataset: str, split_id: str) -> Path:
    """``splits/<split_id>.json`` for ``split_id = "fold{k}_{classsplit}"``."""
    return paths(dataset)["splits_dir"] / f"{split_id}.json"


def load_windows(dataset: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Memory-mapped ``X`` (float32 ``(N, 6, L)``) and the aligned window ``meta`` frame."""
    p = paths(dataset)
    X = np.load(p["windows_x"], mmap_mode="r")
    meta = pd.read_parquet(p["windows_meta"])
    if len(meta) != len(X):
        raise ValueError(f"{dataset}: X has {len(X)} windows but meta has {len(meta)} rows")
    return X, meta


def load_features(dataset: str) -> pd.DataFrame:
    """``features/handcrafted.parquet``: one row per ``window_id`` (spec §3.4)."""
    return pd.read_parquet(paths(dataset)["features"])
