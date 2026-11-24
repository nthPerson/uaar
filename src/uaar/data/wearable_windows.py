"""Windows of our own wearable recordings, with the role each recorded label plays.

A session folder (``raw.csv``, ``events.csv``, ``meta.json``; ``uaar.data.wearable``) is cut
into the same 2 s / 1 s-hop windows as RecoFit. ``configs/wearable.yaml`` says what each label
typed at the recorder is for: a known class, a background movement (the taught models are
fine-tuned to be uncertain on these), an unknown movement kept for testing, or a known exercise
recorded with the kit worn differently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from uaar.data import wearable as w
from uaar.data.windows import window_recording

ROLES = ("known", "background", "unknown", "shifted")


@dataclass
class Roles:
    """The label roles of ``configs/wearable.yaml``.

    ``known`` and ``shifted`` map a recorded label to a model class; a class name used as a
    label maps to itself, so sessions labelled with the RecoFit names also work.
    """

    known: dict[str, str] = field(default_factory=dict)
    background: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    shifted: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str) -> Roles:
        d = yaml.safe_load(Path(path).read_text()) or {}
        return cls(
            known=dict(d.get("known") or {}),
            background=list(d.get("background") or []),
            unknown=list(d.get("unknown") or []),
            shifted=dict(d.get("shifted") or {}),
        )

    def of(self, label: str) -> tuple[str, str]:
        """``(role, class name)`` of a recorded label; role ``""`` if the label is not listed.

        The class name is ``""`` for background and unknown labels.
        """
        if label in self.known:
            return "known", self.known[label]
        if label in self.known.values():
            return "known", label
        if label in self.shifted:
            return "shifted", self.shifted[label]
        if label in self.background:
            return "background", ""
        if label in self.unknown:
            return "unknown", ""
        return "", ""


def session_windows(
    session_dir: Path | str, window_s: float = 2.0, hop_s: float = 1.0, purity: float = 0.8
) -> tuple[np.ndarray, pd.DataFrame]:
    """Labelled windows of one session.

    Returns:
        ``(X, table)``: ``X`` float32 ``(N, 6, L)`` in g and deg/s; ``table`` with one row per
        window: ``subject``, ``session``, ``label`` (as recorded), ``set_index`` (the row of
        ``events.csv`` the window starts in, -1 if none), ``start_s`` and ``n_gaps``. Windows
        outside the labelled sets are dropped.
    """
    session_dir = Path(session_dir)
    rec = w.load_session(session_dir)
    X, meta = window_recording(rec.to_frame(), window_s, hop_s, purity, fs=w.SAMPLE_RATE_HZ)
    keep = (meta["label"].astype(str) != "").to_numpy()
    X, meta = X[keep], meta[keep].reset_index(drop=True)
    set_index = np.full(len(meta), -1, dtype=np.int64)
    events_path = session_dir / "events.csv"
    if events_path.exists():
        start = meta["start_s"].to_numpy(dtype=np.float64)
        for ev in pd.read_csv(events_path).itertuples():
            inside = (start >= ev.t_start_ms / 1000.0 - 1e-3) & (start <= ev.t_end_ms / 1000.0)
            set_index[inside] = int(ev.set_index)
    table = pd.DataFrame(
        {
            "subject": rec.subject,
            "session": session_dir.name,
            "label": meta["label"].astype(str),
            "set_index": set_index,
            "start_s": meta["start_s"].to_numpy(),
            "n_gaps": meta["n_gaps"].to_numpy(),
        }
    )
    return X, table


def subject_windows(
    subject_dirs: list[Path], roles: Roles, **window_kwargs
) -> tuple[np.ndarray, pd.DataFrame]:
    """All sessions under the given subject folders (``data/wearable/W01`` ...), with roles.

    Adds ``role`` (``known`` / ``background`` / ``unknown`` / ``shifted``) and ``class_name``
    to the table of :func:`session_windows`. Windows whose label has no role are dropped and
    their labels returned in ``table.attrs["ignored"]`` (label -> window count).
    """
    Xs, tables = [], []
    for d in subject_dirs:
        sessions = sorted(p.parent for p in Path(d).glob("*/raw.csv"))
        if not sessions:
            raise FileNotFoundError(f"no sessions (…/<session>/raw.csv) under {d}")
        for s in sessions:
            X, t = session_windows(s, **window_kwargs)
            Xs.append(X)
            tables.append(t)
    X = np.concatenate(Xs) if Xs else np.zeros((0, 6, 100), dtype=np.float32)
    table = pd.concat(tables, ignore_index=True)
    role_class = [roles.of(label) for label in table["label"]]
    table["role"] = [r for r, _ in role_class]
    table["class_name"] = [c for _, c in role_class]
    listed = (table["role"] != "").to_numpy()
    ignored = table.loc[~listed, "label"].value_counts().to_dict()
    table = table[listed].reset_index(drop=True)
    table.attrs["ignored"] = ignored
    return X[listed], table


def validation_split(table: pd.DataFrame, frac: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """``(train, val)`` boolean masks: validation is the last set of every label and subject.

    Splitting by set (not by window) keeps overlapping windows of one set on the same side. A
    label recorded as a single set gives up its last ``frac`` of windows instead; the window
    that overlaps both parts is left out of both.
    """
    val = np.zeros(len(table), dtype=bool)
    skip = np.zeros(len(table), dtype=bool)
    for _, g in table.groupby(["subject", "label"], sort=False):
        sets = g[["session", "set_index"]].drop_duplicates()
        if len(sets) >= 2:
            last = sets.iloc[-1]
            in_last = (g["session"] == last["session"]) & (g["set_index"] == last["set_index"])
            val[g.index[in_last.to_numpy()]] = True
        else:
            n_val = int(round(frac * len(g)))
            if n_val >= 2:
                val[g.index[-(n_val - 1) :]] = True
                skip[g.index[-n_val]] = True
    return ~val & ~skip, val
