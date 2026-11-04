"""Live backend: windowing of a raw stream and the WebSocket round trip."""

import json

import numpy as np
from fastapi.testclient import TestClient

from uaar.live.server import make_app
from uaar.live.windower import LiveWindower


def _stream(n: int, moving: bool, seq0: int = 0, t0_ms: int = 0) -> list[list[int]]:
    rows = []
    for i in range(n):
        t = t0_ms + 20 * i
        wobble = int(800 * np.sin(2 * np.pi * i / 50)) if moving else 0
        rows.append([t, 0, 0, 4096 + wobble, 0, 0, 0, 2, (seq0 + i) % 65536])
    return rows


def test_windower_emits_every_hop_with_physical_units():
    lw = LiveWindower(window_s=2.0, hop_s=1.0, fs=50)
    out = lw.push(_stream(100, moving=False))
    assert len(out) == 1
    t0, t1, X, flags = out[0]
    assert X.shape == (6, 100) and X.dtype == np.float32
    assert np.allclose(X[2], 1.0) and np.allclose(X[:2], 0) and (t0, t1) == (0.0, 1.98)
    assert flags & 2
    out = lw.push(_stream(49, moving=False, seq0=100, t0_ms=2000))
    assert out == []
    out = lw.push(_stream(1, moving=False, seq0=149, t0_ms=2980))
    assert len(out) == 1 and out[0][0] == 1.0
    assert lw.n_dropped == 0


def test_windower_counts_dropped_samples():
    lw = LiveWindower()
    lw.push(_stream(10, moving=False))
    lw.push(_stream(10, moving=False, seq0=13, t0_ms=260))  # seq 10, 11, 12 missing
    assert lw.n_dropped == 3


def test_websocket_round_trip():
    client = TestClient(make_app())
    assert client.get("/health").json()["methods"] == ["energy"]
    with client.websocket_connect("/ws") as ws:
        status = json.loads(ws.receive_text())
        assert status["type"] == "status" and status["methods"] == ["energy"]
        ws.send_text(
            json.dumps({"type": "samples", "session_id": 1, "samples": _stream(100, moving=True)})
        )
        pred = json.loads(ws.receive_text())
        assert pred["type"] == "prediction" and pred["session_active"] is True
        m = pred["methods"]["energy"]
        assert m["label"] == "moving" and 0 <= m["score"] <= 1 and isinstance(m["unknown"], bool)
        ws.send_text(
            json.dumps(
                {
                    "type": "samples",
                    "session_id": 1,
                    "samples": _stream(50, moving=False, seq0=100, t0_ms=2000),
                }
            )
        )
        pred2 = json.loads(ws.receive_text())
        assert pred2["t_start"] == 1.0
