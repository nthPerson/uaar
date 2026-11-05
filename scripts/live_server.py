#!/usr/bin/env python3
"""Run the live-demo WebSocket backend (ws://localhost:8765/ws).

    uv run scripts/live_server.py [--port 8765] [--host 0.0.0.0] [--runs RUN_DIR ...]

``--runs`` takes trained run directories, one card per run on the page, in the order given:

    uv run scripts/live_server.py --runs \
        reports/runs/p1_recofit_{logreg,softmax,mcdropout,edl}_fold0_near_seed0

Without ``--runs`` it serves the motion-energy placeholder so the page can be tested without
models. ``--score method=name`` shows another of a method's scores (default: its primary one).
"""

import argparse
from pathlib import Path

import uvicorn

from uaar.live.predictors import RunPredictor
from uaar.live.server import make_app


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--runs", type=Path, nargs="*", default=[], help="trained run directories")
    ap.add_argument("--score", nargs="*", default=[], metavar="METHOD=SCORE")
    args = ap.parse_args()
    scores = dict(s.split("=", 1) for s in args.score)
    predictors = None
    if args.runs:
        predictors = []
        for run in args.runs:
            p = RunPredictor(run)
            if p.method.name in scores:
                p = RunPredictor(run, score=scores[p.method.name])
            print(f"{p.name}: {run} ({len(p.classes)} classes, {p.score_name}, tau {p.tau:.4g})")
            predictors.append(p)
    uvicorn.run(make_app(predictors), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
