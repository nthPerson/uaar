"""The recording page's server side: sessions written to disk as they are recorded."""

import json

import pytest
from fastapi.testclient import TestClient

from uaar.data import wearable as w
from uaar.data.wearable_windows import Roles, session_windows
from uaar.live.recorder import SessionRecorder
from uaar.live.server import make_app

CONFIG = "configs/wearable.yaml"


def rows(n, seq0=0, t0=0, flags=0):
    return [[t0 + 20 * i, 10, -20, 4096, 1, 2, 3, flags, (seq0 + i) % 65536] for i in range(n)]


def test_recorder_writes_a_session_the_loader_reads(tmp_path):
    rec = SessionRecorder(tmp_path, CONFIG)
    assert [p["label"] for p in rec.protocol][:4] == [
        "curl",
        "shoulder_press",
        "lateral_raise",
        "jumping_jacks",
    ]
    assert rec.protocol[0]["role"] == "known" and rec.protocol[4]["role"] == "background"
    assert not rec.recording and rec.state()["sets"] == []
    rec.add_samples(rows(5))  # ignored while idle

    st = rec.start("W09", placement="left wrist", device="UAAR-TEST", firmware="0.1.0")
    assert st["recording"] and st["subject"] == "W09"
    rec.add_samples(rows(50))  # before the set: recorded, not labelled
    rec.set_start("curl", session_id=7)
    rec.add_samples(rows(500, seq0=50, t0=1000))
    kept = rec.set_end()
    assert kept["label"] == "curl" and kept["n_samples"] == 500 and kept["seconds"] == 10.0
    assert kept["t_start_ms"] == 1000 and kept["t_end_ms"] == 10980 and kept["lost_share"] == 0

    rec.set_start("waving")
    rec.add_samples(rows(100, seq0=550, t0=11000))
    rec.add_samples(rows(100, seq0=660, t0=13200))  # ten samples lost inside the set
    lossy = rec.set_end()
    assert lossy["n_dropped"] == 10 and lossy["lost_share"] == round(10 / 210, 4)

    rec.set_start("curl")
    rec.add_samples(rows(200, seq0=760, t0=15200))
    third = rec.set_end()
    assert rec.discard(third["set_index"]) and not rec.discard(99)
    rec.set_start("walking")
    rec.set_cancel()
    assert rec.set_end() is None  # nothing open

    folder = rec.dir
    assert len((folder / "raw.csv").read_text().splitlines()) == 3 + 950  # written as it goes
    summary = rec.stop()
    assert summary["saved"] and summary["n_dropped"] == 10
    by_label = {r["label"]: r for r in summary["labels"]}
    assert by_label["curl"] == {
        "label": "curl",
        "sets": 1,
        "seconds": 10.0,
        "role": "known",
        "needed": 3,
    }
    assert {"label": "curl", "missing": 2} in summary["missing"]
    assert not rec.recording

    session = w.load_session(folder)
    assert session.subject == "W09" and session.n_samples == 950
    assert sorted(set(session.labels)) == ["", "curl", "waving"]
    meta = json.loads((folder / "meta.json").read_text())
    assert meta["device_name"] == "UAAR-TEST" and [s["label"] for s in meta["sets"]] == [
        "curl",
        "waving",
    ]
    X, t = session_windows(folder)
    assert set(t["label"]) == {"curl", "waving"} and set(t["set_index"]) == {0, 1}
    assert Roles.load(CONFIG).of("waving")[0] == "unknown"


def test_recorder_rejects_bad_input_and_drops_empty_sessions(tmp_path):
    rec = SessionRecorder(tmp_path, CONFIG)
    with pytest.raises(ValueError, match="subject"):
        rec.start("../etc")
    with pytest.raises(ValueError, match="start a session"):
        rec.set_start("curl")
    with pytest.raises(ValueError, match="no session"):
        rec.stop()
    rec.start("W09")
    with pytest.raises(ValueError, match="already"):
        rec.start("W09")
    with pytest.raises(ValueError, match="label"):
        rec.set_start("a,b")
    folder = rec.dir
    rec.add_samples(rows(20))
    assert rec.stop() == {"type": "rec_summary", "saved": False, "folder": None, "labels": []}
    assert not folder.exists()  # a session without a kept set leaves nothing behind


def test_recording_over_the_websocket(tmp_path):
    rec = SessionRecorder(tmp_path, CONFIG)
    client = TestClient(make_app(recorder=rec))

    def until(ws, kind):
        while True:
            m = json.loads(ws.receive_text())
            if m["type"] == kind:
                return m

    with client.websocket_connect("/ws") as ws:
        state = until(ws, "rec_state")
        assert state["recording"] is False and len(state["protocol"]) == 12
        ws.send_text(json.dumps({"type": "rec_start", "subject": "bad subject"}))
        assert "subject" in until(ws, "rec_error")["text"]
        assert until(ws, "rec_state")["recording"] is False  # the state follows every error
        ws.send_text(json.dumps({"type": "rec_start", "subject": "W09", "device": "UAAR-TEST"}))
        assert until(ws, "rec_state")["recording"] is True
        ws.send_text(json.dumps({"type": "set_start", "label": "curl", "session_id": 3}))
        assert until(ws, "rec_state")["open_label"] == "curl"
        ws.send_text(json.dumps({"type": "samples", "session_id": 3, "samples": rows(150)}))
        ws.send_text(json.dumps({"type": "set_end"}))
        kept = until(ws, "set")
        assert kept["label"] == "curl" and kept["n_samples"] == 150 and kept["session_id"] == 3

    # a reload in the middle of a session: the new socket is told where things stand
    with client.websocket_connect("/ws") as ws:
        state = until(ws, "rec_state")
        assert state["recording"] and [s["label"] for s in state["sets"]] == ["curl"]
        ws.send_text(json.dumps({"type": "rec_stop"}))
        summary = until(ws, "rec_summary")
        assert summary["saved"] and summary["labels"][0]["sets"] == 1
        assert until(ws, "rec_state")["recording"] is False
    assert w.load_session(summary["folder"]).n_samples == 150
