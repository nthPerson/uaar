"""Live backend: windowing of a raw stream and the WebSocket round trip."""

import json

import numpy as np
from fastapi.testclient import TestClient

from uaar.live.predictors import RunPredictor
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


def _softmax_run(tmp_path, classes):
    import torch

    from uaar.models.softmax import SoftmaxMethod

    torch.manual_seed(0)
    sm = SoftmaxMethod(len(classes), classes)
    sm.model.eval()
    sm.stats = {"mean": [0.0] * 6, "std": [1.0] * 6}
    run = tmp_path / "p1_x_softmax_fold0_rand0_seed0"
    sm.save(run)
    (run / "threshold.json").write_text(json.dumps({"tau": {"msp": 0.4, "entropy": 0.9}}))
    return run


def test_run_predictor_wraps_a_run_directory(tmp_path):
    from uaar.models.mcdropout import MCDropoutMethod

    classes = ["Curl", "Press", "Raise", "Squat", "Jacks", "Row", "Dip"]
    run = _softmax_run(tmp_path, classes)
    X = np.random.default_rng(0).normal(0, 1, (6, 100)).astype(np.float32)

    p = RunPredictor(run)
    assert p.name == "softmax" and p.score_name == "msp" and p.tau == 0.4
    r = p.predict(X)
    assert r["label"] in classes and r["label"] == max(r["probs"], key=r["probs"].get)
    assert len(r["probs"]) == 5 and r["p"] == max(r["probs"].values())
    assert abs(r["score"] - (1 - r["p"])) < 1e-6
    assert RunPredictor(run, score="entropy", name="softmax (entropy)").tau == 0.9

    mc = tmp_path / "mc"
    MCDropoutMethod.from_softmax_run(run, T=5).save(mc)
    (mc / "threshold.json").write_text(json.dumps({"tau": {"entropy": 1.0, "mi": 0.1}}))
    q = RunPredictor(mc)
    assert q.name == "mcdropout" and q.score_name == "mi" and q.predict(X)["score"] >= 0

    client = TestClient(make_app([p, q]))
    assert client.get("/health").json()["methods"] == ["softmax", "mcdropout"]
    with client.websocket_connect("/ws") as ws:
        ws.receive_text()
        ws.send_text(
            json.dumps({"type": "samples", "session_id": 1, "samples": _stream(100, moving=True)})
        )
        m = json.loads(ws.receive_text())["methods"]
        assert set(m) == {"softmax", "mcdropout"}
        for v in m.values():
            assert v["label"] in classes and v["unknown"] == (v["score"] > v["tau"])
