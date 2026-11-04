"""Wearable protocol and session parsing (spec §7: test_loaders wearable rows)."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uaar.data import wearable as w

PROTOCOL_H = Path(__file__).resolve().parents[1] / "firmware" / "include" / "protocol.h"


def _define(name: str) -> str:
    m = re.search(rf"#define {name}\s+(\S+)", PROTOCOL_H.read_text())
    assert m, name
    return m.group(1).strip('"')


def test_constants_match_protocol_h():
    assert _define("UAAR_SERVICE_UUID") == w.SERVICE_UUID
    assert _define("UAAR_IMU_CHAR_UUID") == w.IMU_CHAR_UUID
    assert _define("UAAR_CONTROL_CHAR_UUID") == w.CONTROL_CHAR_UUID
    assert _define("UAAR_STATUS_CHAR_UUID") == w.STATUS_CHAR_UUID
    assert int(_define("UAAR_REQUESTED_MTU")) == w.REQUESTED_MTU
    assert int(_define("UAAR_SAMPLE_RATE_HZ")) == w.SAMPLE_RATE_HZ
    assert int(_define("UAAR_BATCH_SAMPLES")) == w.BATCH_SAMPLES
    assert int(_define("UAAR_ACC_LSB_PER_G")) == w.ACC_LSB_PER_G
    assert float(_define("UAAR_GYR_LSB_PER_DPS").rstrip("f")) == pytest.approx(w.GYR_LSB_PER_DPS)
    assert int(_define("UAAR_SAMPLE_SIZE")) == w.SAMPLE_SIZE
    assert int(_define("UAAR_STATUS_SIZE")) == w.STATUS_SIZE
    assert _define("UAAR_SERIAL_CSV_HEADER") == w.SERIAL_CSV_HEADER
    src = PROTOCOL_H.read_text()
    for name, code in [
        ("START_SESSION", w.CMD_START_SESSION),
        ("STOP_SESSION", w.CMD_STOP_SESSION),
        ("CALIBRATE", w.CMD_CALIBRATE),
        ("PING", w.CMD_PING),
        ("IDENTIFY", w.CMD_IDENTIFY),
        ("LED_KNOWN", w.CMD_LED_KNOWN),
        ("LED_UNKNOWN", w.CMD_LED_UNKNOWN),
    ]:
        assert re.search(rf"UAAR_CMD_{name} = 0x{code:02X}", src), name


def test_batch_round_trip():
    samples = [
        w.Sample(1000 + 20 * i, 1408, 235, 3800, -3, 7, -2, w.FLAG_SESSION_ACTIVE, 65531 + i)
        for i in range(5)
    ]
    payload = w.encode_batch(7, samples)
    assert len(payload) == w.MAX_BATCH_SIZE
    sid, out = w.decode_batch(payload)
    assert sid == 7 and out == samples
    with pytest.raises(ValueError):
        w.decode_batch(payload[:30])


def test_status_decode():
    payload = w.STATUS_STRUCT.pack(0, 1, 0, 3, 0, 4242, 1, 1)
    st = w.decode_status(payload)
    assert st == {
        "firmware": "0.1.0",
        "session_id": 3,
        "uptime_s": 4242,
        "calibrated": True,
        "session_active": True,
    }


def test_parse_csv_line():
    assert w.parse_csv_line("# UAAR fw 0.1.0") is None
    assert w.parse_csv_line(w.SERIAL_CSV_HEADER) is None
    assert w.parse_csv_line("12,1,2,3,4,5,6,8,9") == w.Sample(12, 1, 2, 3, 4, 5, 6, 8, 9)
    assert w.parse_csv_line("12,1,2,3") is None
    assert w.parse_csv_line("12,1,2,3,4,5,x,8,9") is None


def test_seq_gaps_counts_drops_and_wraps():
    seq = np.array([65533, 65534, 65535, 0, 1, 4, 5, 9])  # wrap ok; drops: 2 and 3 lost
    assert w.seq_gaps(seq) == (2 + 3, 2)
    assert w.seq_gaps(np.array([5])) == (0, 0)
    assert w.seq_gaps(np.array([10, 11, 12])) == (0, 0)


def test_raw_to_physical_units():
    raw = pd.DataFrame(
        {
            "t_ms": [0, 20],
            "ax": [4096, 0],
            "ay": [0, 2048],
            "az": [0, 0],
            "gx": [328, 0],
            "gy": [0, -328],
            "gz": [0, 0],
            "flags": [0, 2],
            "seq": [0, 1],
        }
    )
    phys = w.raw_to_physical(raw)
    assert phys["t"].tolist() == pytest.approx([0.0, 0.02])
    assert phys["ax"].tolist() == pytest.approx([1.0, 0.0])
    assert phys["ay"].tolist() == pytest.approx([0.0, 0.5])
    assert phys["gx"].tolist() == pytest.approx([10.0, 0.0])
    assert phys["gy"].tolist() == pytest.approx([0.0, -10.0])


def _write_session(tmp_path: Path, drop_at: int | None = None) -> Path:
    d = tmp_path / "W01" / "s01"
    d.mkdir(parents=True)
    lines = ["# UAAR fw 0.1.0 name=UAAR-TEST", w.SERIAL_CSV_HEADER]
    for i in range(300):  # 6 s at 50 Hz
        if drop_at is not None and i == drop_at:
            continue
        lines.append(w.Sample(20 * i, 4000, 0, 0, 0, 0, 0, 0, i).as_csv())
    (d / "raw.csv").write_text("\n".join(lines) + "\n")
    pd.DataFrame(
        [
            {
                "set_index": 0,
                "label": "curl",
                "session_id": 1,
                "t_start_ms": 1000,
                "t_end_ms": 2000,
                "seq_start": 50,
                "seq_end": 100,
                "n_samples": 51,
                "n_dropped": 0,
            },
            {
                "set_index": 1,
                "label": "squat",
                "session_id": 2,
                "t_start_ms": 4000,
                "t_end_ms": 5000,
                "seq_start": 200,
                "seq_end": 250,
                "n_samples": 51,
                "n_dropped": 0,
            },
        ]
    ).to_csv(d / "events.csv", index=False)
    (d / "meta.json").write_text(
        json.dumps({"subject": "W01", "session": "s01", "firmware": "0.1.0"})
    )
    return d


def test_load_session_labels_and_gaps(tmp_path):
    rec = w.load_session(_write_session(tmp_path, drop_at=225))
    assert rec.subject == "W01" and rec.recording_id == "s01"
    assert rec.n_samples == 299
    assert rec.meta["n_dropped"] == 1 and rec.meta["n_gaps"] == 1
    assert rec.acc[0, 0] == pytest.approx(4000 / 4096)
    labels = pd.Series(rec.labels)
    assert set(labels.unique()) == {"", "curl", "squat"}
    assert (labels == "curl").sum() == 51
    assert (labels == "squat").sum() == 50  # one sample dropped inside this set
    assert rec.meta["banner"][0].startswith("# UAAR fw")


def test_to_app_csv(tmp_path):
    rec = w.load_session(_write_session(tmp_path))
    out = tmp_path / "app.csv"
    w.to_app_csv(rec, out)
    df = pd.read_csv(out)
    assert list(df.columns) == ["t", "ax", "ay", "az", "gx", "gy", "gz"]
    assert len(df) == 300


def test_to_recordings_concatenates_sessions(tmp_path):
    _write_session(tmp_path)
    d2 = tmp_path / "W02" / "s01"
    d2.mkdir(parents=True)
    (d2 / "raw.csv").write_text(
        "\n".join(
            [w.SERIAL_CSV_HEADER]
            + [w.Sample(20 * i, 4096, 0, 0, 0, 0, 0, 0, i).as_csv() for i in range(100)]
        )
        + "\n"
    )
    out = w.to_recordings(tmp_path, tmp_path / "proc")
    df = pd.read_parquet(out)
    assert set(df["subject"]) == {"W01", "W02"} and len(df) == 400
    assert set(df["recording_id"]) == {"W01/s01", "W02/s01"}
    meta = json.loads((tmp_path / "proc" / "recordings.meta.json").read_text())
    assert (
        meta["dataset"] == "wearable"
        and len(meta["sessions"]) == 2
        and meta["labels"]["curl"]["subjects"] == 1
    )
