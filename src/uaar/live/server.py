"""FastAPI WebSocket server for the live demo.

Protocol (JSON text frames; mirrored in web/README.md):

client -> server
    {"type": "samples", "session_id": 3,
     "samples": [[t_ms, ax, ay, az, gx, gy, gz, flags, seq], ...]}
    {"type": "reset"}                      # clear the window buffer (new session / reconnect)
server -> client
    {"type": "status", "text": "...", "methods": ["edl", ...], "window_s": 2.0, "hop_s": 1.0,
     "cards": [{"name": "edl", "title": "EDL", "subtitle": "...", "score_name": "vacuity",
                "measure": "Vacuity", "formula": "number of classes / total evidence"}],
     "classes": ["Curl", ...], "replay": false}
    {"type": "prediction", "t_start": 12.34, "t_end": 14.32, "session_id": 3,
     "session_active": true, "n_dropped": 0,
     "methods": {"edl": {"label": "Curl", "p": 0.93, "score": 0.12, "tau": 0.35,
                         "u": 0.12, "tau_u": 0.35, "unknown": false, "probs": {...}}, ...}}

``u`` and ``tau_u`` are the score and its threshold on a common 0-1 scale (see
``uaar.live.predictors.score_scale``). ``classes`` is the class list of the first predictor, in
class order, so the page can give every class a fixed colour. With ``web_dir`` the same app also
serves the page (``/``), and with ``replay`` it serves that recording at ``/replay.csv`` and
tells the page to play it (``"replay": true``).

With ``recorder`` (``uaar.live.recorder.SessionRecorder``) the socket also drives a recording
session; the samples a client sends are written to disk while a session is open:

client -> server
    {"type": "rec_start", "subject": "W01", "placement": "...", "device": "UAAR-8428",
     "firmware": "0.1.0"}
    {"type": "set_start", "label": "curl"}   # the samples that follow belong to this set
    {"type": "set_end"}                      # keep it;  {"type": "set_cancel"} drops it
    {"type": "set_discard", "set_index": 3}
    {"type": "rec_stop"}
server -> client
    {"type": "rec_state", "recording": true, "subject": ..., "session": ..., "sets": [...],
     "protocol": [...], ...}                 # on connect and after every change
    {"type": "set", ...}                     # the set just kept: seconds, samples, lost_share
    {"type": "rec_summary", "saved": true, "folder": ..., "labels": [...], "missing": [...]}
    {"type": "rec_error", "text": "..."}

Run: ``uv run scripts/demo.py`` (page and predictions on port 8000).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from uaar.data import wearable as w
from uaar.live.predictors import MotionEnergyPredictor, Predictor
from uaar.live.recorder import SessionRecorder
from uaar.live.windower import LiveWindower

RECORDER_MESSAGES = (
    "rec_start",
    "rec_stop",
    "rec_state",
    "set_start",
    "set_end",
    "set_cancel",
    "set_discard",
)


def record_message(recorder: SessionRecorder, kind: str, msg: dict) -> list[dict]:
    """Apply one recording message and return the replies (the new state, or an error)."""
    try:
        if kind == "rec_start":
            recorder.start(
                str(msg.get("subject", "")),
                placement=str(msg.get("placement", "")),
                device=str(msg.get("device", "")),
                firmware=str(msg.get("firmware", "")),
                transport=str(msg.get("transport", "ble")),
            )
        elif kind == "rec_stop":
            return [recorder.stop(), recorder.state()]
        elif kind == "set_start":
            recorder.set_start(str(msg.get("label", "")), int(msg.get("session_id", 0) or 0))
        elif kind == "set_end":
            kept = recorder.set_end()
            return ([{"type": "set", **kept}] if kept else []) + [recorder.state()]
        elif kind == "set_cancel":
            recorder.set_cancel()
        elif kind == "set_discard":
            recorder.discard(int(msg.get("set_index", -1)))
    except ValueError as e:
        return [{"type": "rec_error", "text": str(e)}, recorder.state()]
    return [recorder.state()]


def make_app(
    predictors: Sequence[Predictor] | None = None,
    window_s: float = 2.0,
    hop_s: float = 1.0,
    web_dir: Path | None = None,
    replay: Path | None = None,
    recorder: SessionRecorder | None = None,
) -> FastAPI:
    """The live app: ``/ws`` and ``/health``, plus the page and a replay file when given.

    Args:
        predictors: one per card on the page; the motion-energy placeholder when ``None``.
        web_dir: directory with ``index.html`` to serve at ``/`` (``None``: API only).
        replay: a recorded ``raw.csv`` the page plays on load, served at ``/replay.csv``.
        recorder: writes recording sessions for ``/record.html`` (``None``: no recording).
    """
    preds: list[Predictor] = (
        list(predictors) if predictors is not None else [MotionEnergyPredictor()]
    )
    app = FastAPI(title="uaar live")
    app.state.predictors = preds
    cards = [
        {
            "name": p.name,
            "title": getattr(p, "title", p.name),
            "subtitle": getattr(p, "subtitle", ""),
            "score_name": getattr(p, "score_name", ""),
            "measure": getattr(p, "measure", ""),
            "formula": getattr(p, "formula", ""),
        }
        for p in preds
    ]
    classes = list(getattr(preds[0], "classes", [])) if preds else []

    if replay is not None:

        @app.get("/replay.csv")
        def replay_csv() -> FileResponse:
            return FileResponse(replay, media_type="text/csv", filename=Path(replay).name)

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "methods": [p.name for p in preds]}

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        windower = LiveWindower(window_s=window_s, hop_s=hop_s)
        await socket.send_text(
            json.dumps(
                {
                    "type": "status",
                    "text": f"connected; methods: {', '.join(p.name for p in preds)}",
                    "methods": [p.name for p in preds],
                    "window_s": window_s,
                    "hop_s": hop_s,
                    "cards": cards,
                    "classes": classes,
                    "replay": replay is not None,
                }
            )
        )
        if recorder is not None:
            await socket.send_text(json.dumps(recorder.state()))
        try:
            while True:
                msg = json.loads(await socket.receive_text())
                kind = msg.get("type")
                if kind == "reset":
                    windower.reset()
                    continue
                if recorder is not None and kind in RECORDER_MESSAGES:
                    for reply in record_message(recorder, kind, msg):
                        await socket.send_text(json.dumps(reply))
                    continue
                if kind != "samples":
                    continue
                if recorder is not None:
                    recorder.add_samples(msg.get("samples", []))
                for t0, t1, X, flags_any in windower.push(msg.get("samples", [])):
                    methods = {}
                    for p in preds:
                        r = p.predict(X)
                        tau = float(r.get("tau", p.tau))
                        methods[p.name] = {
                            "label": r["label"],
                            "p": round(float(r["p"]), 4),
                            "score": round(float(r["score"]), 4),
                            "tau": tau,
                            "u": round(float(r.get("u", r["score"])), 4),
                            "tau_u": round(float(r.get("tau_u", tau)), 4),
                            "unknown": bool(r["score"] > tau),
                            "probs": {k: round(float(v), 4) for k, v in r.get("probs", {}).items()},
                        }
                    await socket.send_text(
                        json.dumps(
                            {
                                "type": "prediction",
                                "t_start": round(t0, 3),
                                "t_end": round(t1, 3),
                                "session_id": msg.get("session_id"),
                                "session_active": bool(flags_any & w.FLAG_SESSION_ACTIVE),
                                "n_dropped": windower.n_dropped,
                                "methods": methods,
                            }
                        )
                    )
        except WebSocketDisconnect:
            return

    if web_dir is not None:  # last, so it never shadows the routes above
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return app


app = make_app()
