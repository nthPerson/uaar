"""RecoFit loader on a tiny synthetic .mat with the documented structure (spec §7)."""

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from scipy.io import savemat

from uaar.data import recofit

FS = 50.0


def _rec(label: str, seconds: float, subject_id: int, ax: float = 0.1):
    n = int(seconds * FS)
    t = np.arange(n) / FS + 100.0  # the file clock does not start at 0
    acc = np.c_[t, np.full(n, ax), np.zeros(n), np.ones(n)]
    gyro = np.c_[t, np.zeros(n), np.full(n, 5.0), np.zeros(n)]
    return {
        "activityName": label,
        "subjectID": subject_id,
        "activityReps": 10,
        "data": {"accelDataMatrix": acc, "gyroDataMatrix": gyro},
    }


def _struct_array(recs: list[dict]) -> np.ndarray:
    """A MATLAB struct array (one element per recording) for savemat."""
    arr = np.zeros(
        len(recs),
        dtype=[("activityName", "O"), ("subjectID", "O"), ("activityReps", "O"), ("data", "O")],
    )
    for i, r in enumerate(recs):
        for k in arr.dtype.names:
            arr[i][k] = r[k]
    return arr


@pytest.fixture
def synthetic(tmp_path):
    activities = np.array(["Curl", "Squat", "Device tap"], dtype=object)
    single = np.empty((2, 3), dtype=object)
    single[0, 0] = _struct_array([_rec("Curl", 4.0, 526), _rec("Curl", 3.0, 526)])
    single[0, 1] = _struct_array([_rec("Squat", 2.0, 526)])
    single[0, 2] = _struct_array([_rec("Device tap", 2.0, 526)])  # junk
    single[1, 0] = _struct_array([_rec("Curl", 5.0, 731)])
    single[1, 1] = np.zeros((0, 0))  # empty cell
    single[1, 2] = np.zeros((0, 0))
    single_path = tmp_path / "single.mat"
    savemat(single_path, {"exerciseConstants": {"activities": activities}, "subject_data": single})

    visit = _rec("", 10.0, 526)
    # columns: name, start_s, end_s, notes, reps, start_s, sequence struct
    visit["activityStartMatrix"] = np.array(
        [
            ["Curl", 101.0, 103.0, "", 8, 101.0, {"seq": 1}],
            ["Non-exercise", 103.0, 105.0, "", -1, 103.0, {"seq": 2}],
            ["Device tap", 105.0, 106.0, "junk", -1, 105.0, {"seq": 3}],
        ],
        dtype=object,
    )
    multi = np.empty((2, 1), dtype=object)
    v = np.zeros(
        1,
        dtype=[
            ("activityName", "O"),
            ("subjectID", "O"),
            ("activityReps", "O"),
            ("data", "O"),
            ("activityStartMatrix", "O"),
        ],
    )
    for k in v.dtype.names:
        v[0][k] = visit[k]
    multi[0, 0] = v
    multi[1, 0] = np.zeros((0, 0))
    multi_path = tmp_path / "multi.mat"
    savemat(multi_path, {"subject_data": multi})
    return single_path, multi_path


def test_load_single(synthetic):
    single_path, _ = synthetic
    recs = list(recofit.load_single(single_path))
    assert [r.recording_id for r in recs] == [
        "single/526/0/0",
        "single/526/0/1",
        "single/526/1/0",
        "single/526/2/0",
        "single/731/0/0",
    ]
    r = recs[0]
    assert r.subject == "526" and r.source == "single" and r.n_samples == 200
    assert set(r.labels) == {"Curl"}
    assert r.t[0] == 0 and np.allclose(np.diff(r.t), 0.02, atol=1e-6)  # re-zeroed, 50 Hz
    assert np.allclose(r.acc[:, 0], 0.1) and np.allclose(r.gyro[:, 1], 5.0)  # g and deg/s unchanged
    assert r.meta["reps"] == 10


def test_load_multi_labels_segments(synthetic):
    _, multi_path = synthetic
    recs = list(recofit.load_multi(multi_path))
    assert len(recs) == 1
    r = recs[0]
    assert r.recording_id == "multi/526/0" and r.source == "multi" and r.n_samples == 500
    counts = pd.Series(r.labels).value_counts()
    assert counts["Curl"] == 100 and counts["Non-exercise"] == 100 and counts["Device tap"] == 50
    assert counts[""] == 250


def test_to_recordings_drops_junk_and_writes_meta(synthetic, tmp_path):
    single_path, multi_path = synthetic
    out = recofit.to_recordings(tmp_path / "proc", single_path, multi_path, hash_sources=False)
    df = pq.read_table(out).to_pandas()
    assert list(df.columns) == [
        "subject",
        "recording_id",
        "t",
        "ax",
        "ay",
        "az",
        "gx",
        "gy",
        "gz",
        "label",
        "source",
    ]
    assert "single/526/2/0" not in set(df["recording_id"])  # junk single recording skipped
    assert df["t"].dtype == np.float32 and df["ax"].dtype == np.float32
    multi = df[df["source"] == "multi"]
    assert "Device tap" not in set(multi["label"]) and (multi["label"] == "").sum() == 300
    meta = json.loads((tmp_path / "proc" / "recordings.meta.json").read_text())
    assert meta["dataset"] == "recofit" and meta["sample_rate_hz"] == 50.0
    assert meta["labels"]["Curl"] == {"samples": 200 + 150 + 250 + 100, "subjects": 2}
    assert meta["n_recordings"] == 5 and [s["name"] for s in meta["source_files"]] == [
        "single.mat",
        "multi.mat",
    ]


def test_is_junk():
    assert recofit.is_junk("Device tap") and recofit.is_junk("Arm band adjustment")
    assert not recofit.is_junk("Bicep Curl")
