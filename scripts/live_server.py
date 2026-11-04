#!/usr/bin/env python3
"""Run the live-demo WebSocket backend (ws://localhost:8765/ws).

    uv run scripts/live_server.py [--port 8765] [--host 0.0.0.0]

With no trained models yet it serves the motion-energy placeholder so the page can be tested.
"""

import argparse

import uvicorn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    uvicorn.run("uaar.live.server:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
