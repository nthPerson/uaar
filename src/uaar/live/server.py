"""FastAPI WebSocket server for the live demo.

Protocol (JSON text frames; mirrored in web/README.md):

client -> server
    {"type": "samples", "session_id": 3,
     "samples": [[t_ms, ax, ay, az, gx, gy, gz, flags, seq], ...]}
    {"type": "reset"}                      # clear the window buffer (new session / reconnect)
server -> client
    {"type": "status", "text": "...", "methods": ["edl", ...], "window_s": 2.0, "hop_s": 1.0}
    {"type": "prediction", "t_start": 12.34, "t_end": 14.32, "session_id": 3,
     "session_active": true, "n_dropped": 0,
     "methods": {"edl": {"label": "Bicep Curl", "p": 0.93, "score": 0.12, "tau": 0.35,
                         "unknown": false, "probs": {...}}, ...}}

Run: ``uv run scripts/live_server.py`` (port 8765). The page in ``web/`` is the client.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from uaar.data import wearable as w
from uaar.live.predictors import MotionEnergyPredictor, Predictor
from uaar.live.windower import LiveWindower


def make_app(
    predictors: Sequence[Predictor] | None = None, window_s: float = 2.0, hop_s: float = 1.0
) -> FastAPI:
    preds: list[Predictor] = (
        list(predictors) if predictors is not None else [MotionEnergyPredictor()]
    )
    app = FastAPI(title="uaar live")
    app.state.predictors = preds

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
                }
            )
        )
        try:
            while True:
                msg = json.loads(await socket.receive_text())
                kind = msg.get("type")
                if kind == "reset":
                    windower.reset()
                    continue
                if kind != "samples":
                    continue
                for t0, t1, X, flags_any in windower.push(msg.get("samples", [])):
                    methods = {}
                    for p in preds:
                        r = p.predict(X)
                        methods[p.name] = {
                            "label": r["label"],
                            "p": round(float(r["p"]), 4),
                            "score": round(float(r["score"]), 4),
                            "tau": float(r.get("tau", p.tau)),
                            "unknown": bool(r["score"] > r.get("tau", p.tau)),
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

    return app


app = make_app()
