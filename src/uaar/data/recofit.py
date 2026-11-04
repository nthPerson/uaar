"""RecoFit loader (Morris et al., CHI 2014): the two MATLAB v5 files -> recordings.parquet.

File facts (see docs/design/datasets.md §1): ``subject_data`` is a cell matrix with one row per
subject; in the single-activity file one column per activity in ``exerciseConstants.activities``,
each cell a struct array with one element per time that subject did that exercise; in the
multi-activity file one column, each cell a struct array with one element per lab visit and an
``activityStartMatrix`` (name, start_s, end_s, notes, reps, start_s, sequence struct) giving the
labelled segments. Every recording holds ``data.accelDataMatrix`` ``[t_s, x, y, z]`` in g and
``data.gyroDataMatrix`` ``[t_s, x, y, z]`` in deg/s at exactly 50 Hz.

Memory: ``loadmat`` on one 1.5 GB file needs about 3 GB; recordings are streamed to Parquet
one subject at a time and only one file is open at once.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.io import loadmat
from scipy.io.matlab import mat_struct

from uaar.data.recording import Recording
from uaar.utils import git_sha

LFS_BASE = "https://media.githubusercontent.com/media/microsoft/Exercise-Recognition-from-Wearable-Sensors/main/"
FILES = {
    "exercise_data.50.0000_singleonly.mat": 1_562_300_284,
    "exercise_data.50.0000_multionly.mat": 1_571_881_721,
}
SAMPLE_RATE_HZ = 50.0
MASTER_DEVICE = "rightArm"

# Labels that are not movements of interest (device taps, band adjustment, notes). Matched as
# case-insensitive substrings of the activity name. Finalized after the label list is printed
# (development-spec §12); until then this list is a placeholder.
# Names seen in the multi-activity file on 2026-11-03: "Tap Left Device", "Tap Right Device",
# "Device on Table", "<Initial Activity>". "Non-Exercise" (idle) and "Unlisted Exercise" are kept.
JUNK_LABEL_PATTERNS: tuple[str, ...] = ("tap ", "device on table", "initial activity", "adjust")


def is_junk(label: str) -> bool:
    low = label.lower()
    return any(p in low for p in JUNK_LABEL_PATTERNS)


# ---- download ------------------------------------------------------------------------------
def download(dest: Path, force: bool = False) -> None:
    """Fetch the two LFS objects into ``dest`` (skips files already present with the right size)."""
    import urllib.request

    from tqdm import tqdm

    dest.mkdir(parents=True, exist_ok=True)
    for name, size in FILES.items():
        path = dest / name
        if path.exists() and path.stat().st_size == size and not force:
            print(f"{name}: present ({size} B)")
            continue
        with (
            urllib.request.urlopen(LFS_BASE + name) as r,
            open(path, "wb") as f,
            tqdm(total=size, unit="B", unit_scale=True, desc=name) as bar,
        ):
            while chunk := r.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
        if path.stat().st_size != size:
            raise OSError(f"{name}: got {path.stat().st_size} B, expected {size}")


# ---- MATLAB structure helpers --------------------------------------------------------------
def _mat(path: Path) -> dict:
    return loadmat(str(path), struct_as_record=False, squeeze_me=True)


def _structs(cell) -> list[mat_struct]:
    """A cell's content as a list of structs (handles empty cells and 1-element struct arrays)."""
    if isinstance(cell, mat_struct):
        return [cell]
    if isinstance(cell, np.ndarray):
        return [x for x in cell.ravel() if isinstance(x, mat_struct)]
    return []


def _str(x) -> str:
    if isinstance(x, np.ndarray):
        return "" if x.size == 0 else str(x.ravel()[0])
    return "" if x is None else str(x)


def _subject_table(mat: dict) -> np.ndarray:
    """``subject_data`` as a 2-D (subjects x columns) object array.

    ``squeeze_me`` turns the multi file's (94, 1) cell matrix into a 1-D array of 94 cells, so a
    1-D array means one column per subject, never one row.
    """
    sd = np.asarray(mat["subject_data"], dtype=object)
    return sd.reshape(-1, 1) if sd.ndim == 1 else sd


def activity_names(mat: dict) -> list[str]:
    """The 75 activity labels in column order of the single-activity ``subject_data``."""
    acts = mat["exerciseConstants"].activities
    return [_str(a) for a in np.asarray(acts, dtype=object).ravel()]


def _subject_code(rec: mat_struct, row: int) -> str:
    for name in ("subjectID", "subjectId", "subject"):
        if hasattr(rec, name):
            return str(int(getattr(rec, name)))
    return f"row{row}"


def _arrays(rec: mat_struct) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t, acc, gyro) from a recording, aligned on the accelerometer clock."""
    a = np.atleast_2d(np.asarray(rec.data.accelDataMatrix, dtype=np.float64))
    g = np.atleast_2d(np.asarray(rec.data.gyroDataMatrix, dtype=np.float64))
    n = min(len(a), len(g))
    a, g = a[:n], g[:n]
    if n and not np.allclose(a[:, 0], g[:, 0], atol=1e-3):
        raise ValueError("accelerometer and gyroscope timestamps differ")
    t = (a[:, 0] - a[0, 0]).astype(np.float32) if n else np.zeros(0, np.float32)
    return t, a[:, 1:4].astype(np.float32), g[:, 1:4].astype(np.float32)


# ---- recordings ----------------------------------------------------------------------------
def load_single(path: Path, subjects: slice | None = None) -> Iterator[Recording]:
    """One Recording per (subject, activity, instance) from the single-activity file."""
    mat = _mat(path)
    names = activity_names(mat)
    table = _subject_table(mat)
    rows = range(table.shape[0])[subjects] if subjects else range(table.shape[0])
    for i in rows:
        for j, cell in enumerate(table[i]):
            for k, rec in enumerate(_structs(cell)):
                label = _str(getattr(rec, "activityName", names[j])) or names[j]
                t, acc, gyro = _arrays(rec)
                if len(t) == 0:
                    continue
                reps = getattr(rec, "activityReps", None)
                yield Recording(
                    subject=_subject_code(rec, i),
                    recording_id=f"single/{_subject_code(rec, i)}/{j}/{k}",
                    source="single",
                    t=t,
                    acc=acc,
                    gyro=gyro,
                    labels=np.full(len(t), label, dtype=object).astype(str),
                    meta={"activity_index": j, "reps": None if reps is None else int(reps)},
                )


def _segments(rec: mat_struct) -> list[tuple[str, float, float]]:
    m = getattr(rec, "activityStartMatrix", None)
    if m is None:
        return []
    m = np.asarray(m, dtype=object)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    out = []
    for row in m:
        name = _str(row[0])
        try:
            start, end = float(np.asarray(row[1]).ravel()[0]), float(np.asarray(row[2]).ravel()[0])
        except (IndexError, ValueError, TypeError):
            continue
        if name and end > start:
            out.append((name, start, end))
    return out


def load_multi(path: Path, subjects: slice | None = None) -> Iterator[Recording]:
    """One Recording per lab visit from the multi-activity file, labelled from activityStartMatrix.

    Samples outside every segment get ``""``. Segment times are in the file's own clock
    (``accelDataMatrix[:, 0]``), so they are compared against that column, not the re-zeroed ``t``.
    """
    mat = _mat(path)
    table = _subject_table(mat)
    rows = range(table.shape[0])[subjects] if subjects else range(table.shape[0])
    for i in rows:
        for v, rec in enumerate(_structs(table[i, 0])):
            t, acc, gyro = _arrays(rec)
            if len(t) == 0:
                continue
            t_file = np.asarray(rec.data.accelDataMatrix, dtype=np.float64)[: len(t), 0]
            labels = np.full(len(t), "", dtype=object)
            for name, start, end in _segments(rec):
                labels[(t_file >= start) & (t_file < end)] = name
            yield Recording(
                subject=_subject_code(rec, i),
                recording_id=f"multi/{_subject_code(rec, i)}/{v}",
                source="multi",
                t=t,
                acc=acc,
                gyro=gyro,
                labels=labels.astype(str),
                meta={"segments": _segments(rec)},
            )


# ---- to_recordings -------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def to_recordings(
    out_dir: Path,
    single_path: Path | None,
    multi_path: Path | None,
    drop_junk: bool = True,
    hash_sources: bool = True,
) -> Path:
    """Stream both files into ``out_dir/recordings.parquet`` (+ ``recordings.meta.json``).

    Junk-labelled single recordings are skipped; junk segments in multi visits become ``""``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "recordings.parquet"
    writer: pq.ParquetWriter | None = None
    label_samples: dict[str, int] = {}
    label_subjects: dict[str, set[str]] = {}
    n_recordings = n_samples = 0

    def emit(rec: Recording) -> None:
        nonlocal writer, n_recordings, n_samples
        if drop_junk:
            if rec.source == "single" and is_junk(rec.labels[0]):
                return
            junk = np.array([is_junk(x) for x in np.unique(rec.labels)])
            if junk.any():
                bad = set(np.unique(rec.labels)[junk])
                rec.labels = np.array(["" if x in bad else x for x in rec.labels], dtype=str)
        df = rec.to_frame()
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out, table.schema, compression="zstd")
        writer.write_table(table)
        n_recordings += 1
        n_samples += len(df)
        for lab, cnt in df["label"].value_counts().items():
            label_samples[lab] = label_samples.get(lab, 0) + int(cnt)
            label_subjects.setdefault(lab, set()).add(rec.subject)

    sources = []
    for path, loader in ((single_path, load_single), (multi_path, load_multi)):
        if path is None:
            continue
        print(f"loading {path.name} ...", file=sys.stderr)
        for rec in loader(path):
            emit(rec)
        sources.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path) if hash_sources else None,
            }
        )
        print(f"  {n_recordings} recordings, {n_samples} samples so far", file=sys.stderr)
    if writer is not None:
        writer.close()
    meta = {
        "dataset": "recofit",
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "units": {"acc": "g", "gyro": "deg/s", "t": "s"},
        "device": MASTER_DEVICE,
        "n_recordings": n_recordings,
        "n_samples": n_samples,
        "labels": {
            lab: {"samples": label_samples[lab], "subjects": len(label_subjects[lab])}
            for lab in sorted(label_samples)
        },
        "junk_patterns": list(JUNK_LABEL_PATTERNS) if drop_junk else [],
        "source_files": sources,
        "created_by": git_sha(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out_dir / "recordings.meta.json").write_text(json.dumps(meta, indent=2))
    return out


def print_label_table(meta_path: Path) -> pd.DataFrame:
    """Per-label sample minutes and subject counts from ``recordings.meta.json`` (for T1)."""
    meta = json.loads(Path(meta_path).read_text())
    rows = [
        {
            "label": lab,
            "minutes": v["samples"] / meta["sample_rate_hz"] / 60,
            "subjects": v["subjects"],
        }
        for lab, v in meta["labels"].items()
    ]
    df = pd.DataFrame(rows).sort_values("minutes", ascending=False).reset_index(drop=True)
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(df.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    return df
