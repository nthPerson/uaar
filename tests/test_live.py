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


def test_scores_share_a_zero_to_one_scale(tmp_path):
    import math

    from uaar.live.predictors import score_scale

    assert score_scale("msp", 4) == 0.75 and score_scale("varratio", 4) == 0.75
    assert score_scale("entropy", 4) == math.log(4) == score_scale("mi", 4)
    assert score_scale("vacuity", 4) == 1.0 and score_scale("something else", 4) == 1.0

    classes = ["Two-arm Dumbbell Curl (both arms, not alternating)", "Press", "Raise", "Jacks"]
    run = _softmax_run(tmp_path, classes)
    X = np.random.default_rng(0).normal(0, 1, (6, 100)).astype(np.float32)
    p = RunPredictor(
        run,
        name="plain",
        title="Standard network",
        subtitle="four exercises only",
        labels={classes[0]: "Curl"},
    )
    assert p.classes == ["Curl", "Press", "Raise", "Jacks"] and p.title == "Standard network"
    r = p.predict(X)
    assert r["label"] in p.classes and set(r["probs"]) <= set(p.classes)
    assert r["u"] == min(1.0, r["score"] / 0.75) and r["tau_u"] == 0.4 / 0.75
    e = RunPredictor(run, score="entropy", measure="Spread", formula="")
    q = e.predict(X)
    assert 0.0 <= q["u"] <= 1.0 and abs(q["tau_u"] - 0.9 / math.log(4)) < 1e-9
    assert (p.measure, e.measure, e.formula) == ("Least confidence", "Spread", "")


def test_score_info_names_each_models_measure():
    from uaar.live.predictors import score_info

    assert score_info("edl", "vacuity")[0] == "Vacuity"
    assert score_info("softmax", "entropy")[0] == "Entropy"
    name, formula = score_info("mcdropout", "entropy", passes=30)
    assert name == "Predictive entropy" and "30 dropout passes" in formula
    assert score_info("mcdropout", "mi", passes=10)[1] == "disagreement between 10 dropout passes"
    assert score_info("x", "new_score") == ("new_score", "")


def test_one_app_serves_page_replay_and_cards(tmp_path):
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<!doctype html><title>demo</title>")
    (web / "demo.js").write_text("export {};")
    rec = tmp_path / "raw.csv"
    rec.write_text("# banner\nt_ms,ax,ay,az,gx,gy,gz,flags,seq\n0,0,0,4096,0,0,0,0,0\n")
    run = _softmax_run(tmp_path, ["Curl", "Press", "Raise", "Jacks"])
    p = RunPredictor(run, name="plain", title="Standard network", subtitle="s")

    client = TestClient(make_app([p], web_dir=web, replay=rec))
    assert "demo" in client.get("/").text
    assert client.get("/demo.js").status_code == 200
    assert client.get("/replay.csv").text == rec.read_text()
    assert client.get("/health").json()["methods"] == ["plain"]
    with client.websocket_connect("/ws") as ws:
        status = json.loads(ws.receive_text())
        assert status["replay"] is True and status["classes"] == p.classes
        assert status["cards"] == [
            {
                "name": "plain",
                "title": "Standard network",
                "subtitle": "s",
                "score_name": "msp",
                "measure": "Least confidence",
                "formula": "1 \u2212 highest class probability",
            }
        ]
        ws.send_text(
            json.dumps({"type": "samples", "session_id": 0, "samples": _stream(100, moving=True)})
        )
        m = json.loads(ws.receive_text())["methods"]["plain"]
        assert 0.0 <= m["u"] <= 1.0 and m["tau_u"] == round(0.4 / 0.75, 4)

    api_only = TestClient(make_app([p]))
    assert api_only.get("/replay.csv").status_code == 404
    with api_only.websocket_connect("/ws") as ws:
        assert json.loads(ws.receive_text())["replay"] is False


def test_demo_script_builds_cards_from_its_config(tmp_path):
    import importlib.util

    import pytest
    import yaml

    spec = importlib.util.spec_from_file_location("demo", "scripts/demo.py")
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)

    run = _softmax_run(tmp_path, ["A long class name", "Press", "Raise", "Jacks"])
    conf = tmp_path / "demo.yaml"
    card = {"name": "plain", "title": "Standard network", "run": str(run), "score": "entropy"}
    conf.write_text(
        yaml.safe_dump(
            {"labels": {"A long class name": "Curl"}, "cards": [card, {"run": str(run)}]}
        )
    )
    a, b = demo.load_predictors(conf)
    assert (a.name, a.title, a.score_name, a.classes[0]) == (
        "plain",
        "Standard network",
        "entropy",
        "Curl",
    )
    assert (b.name, b.title, b.score_name) == ("card1", "card1", "msp")

    conf.write_text(yaml.safe_dump({"cards": [{"run": str(tmp_path / "missing")}]}))
    with pytest.raises(FileNotFoundError, match="not a trained run directory"):
        demo.load_predictors(conf)
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "raw.csv").write_text("x")
    assert demo.replay_file(tmp_path / "s") == tmp_path / "s" / "raw.csv"
    with pytest.raises(FileNotFoundError):
        demo.replay_file(tmp_path / "nope.csv")


def test_export_model_copies_what_is_needed_to_load(tmp_path):
    import importlib.util

    import pytest

    spec = importlib.util.spec_from_file_location("export_model", "scripts/export_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    run = _softmax_run(tmp_path, ["Curl", "Press", "Raise", "Jacks"])
    (run / "preds_val.parquet").write_text("big")
    (run / "metrics.json").write_text("{}")
    out = mod.export(run, tmp_path / "models")
    names = {p.name for p in out.iterdir()}
    assert {"model.pt", "config.json", "preprocess.json", "threshold.json", "metrics.json"} == names
    X = np.zeros((6, 100), dtype=np.float32)
    assert RunPredictor(out).predict(X)["label"] in ["Curl", "Press", "Raise", "Jacks"]
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="not a finished run"):
        mod.export(tmp_path / "empty", tmp_path / "models")
