#!/usr/bin/env python3
"""Run the live demo: the page and the predictions from one server on one port.

    uv run scripts/demo.py                                   # http://localhost:8000
    uv run scripts/demo.py --replay data/wearable/W01/test-60s
    uv run scripts/demo.py --no-models                       # pages only, for recording

Open the address in Edge or Chrome, click "Connect wearable" and choose the kit; or "Upload a
recording" and pick a ``raw.csv``. ``--replay`` plays a recorded session (a session folder or
its ``raw.csv``) through the page as soon as it opens: for testing without a kit and as the
fallback on stage if Bluetooth fails. ``/record.html`` is the recording page: it saves
sessions straight into ``data/wearable/<subject>/<session>/``. With ``--no-models`` the pages
are served without trained models (a motion placeholder answers).

The cards on the page come from ``configs/demo.yaml`` (``--config``): one trained run directory
per card, with the title shown to the audience and the uncertainty score to draw.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn
import yaml

from uaar.live.predictors import RunPredictor
from uaar.live.recorder import SessionRecorder
from uaar.live.server import make_app

ROOT = Path(__file__).resolve().parents[1]


def load_predictors(config: Path) -> list[RunPredictor]:
    """One :class:`RunPredictor` per entry of ``cards`` in the demo config."""
    conf = yaml.safe_load(config.read_text())
    labels = conf.get("labels") or {}
    out = []
    for i, card in enumerate(conf["cards"]):
        run = Path(card["run"])
        run = run if run.is_absolute() else ROOT / run
        if not (run / "threshold.json").exists():
            raise FileNotFoundError(
                f"{config}: card {card.get('title', i)!r} points at {run}, which is not a "
                "trained run directory (train and evaluate it first, or edit the config)"
            )
        out.append(
            RunPredictor(
                run,
                score=card.get("score"),
                name=card.get("name", f"card{i}"),
                title=card.get("title"),
                subtitle=card.get("subtitle", ""),
                labels=labels,
                measure=card.get("measure"),
                formula=card.get("formula"),
            )
        )
    return out


def replay_file(path: Path) -> Path:
    """A session folder or a ``raw.csv`` -> the ``raw.csv`` path (must exist)."""
    path = path / "raw.csv" if path.is_dir() else path
    if not path.exists():
        raise FileNotFoundError(f"no recording at {path}")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "demo.yaml")
    ap.add_argument("--replay", type=Path, default=None, help="session folder or raw.csv")
    ap.add_argument("--no-models", action="store_true", help="serve the pages without models")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    try:
        predictors = None if args.no_models else load_predictors(args.config)
        replay = replay_file(args.replay) if args.replay else None
    except (FileNotFoundError, KeyError) as e:
        print(f"demo: {e}", file=sys.stderr)
        return 1
    if predictors is None:
        print("no models loaded (--no-models): the cards show a motion placeholder", flush=True)
    for p in predictors or []:
        print(f"card {p.title!r}: {p.method.name}, score {p.score_name}, threshold {p.tau:.3f}")
    print(f"\nopen http://localhost:{args.port}/ in Edge or Chrome  (Ctrl-C to stop)\n", flush=True)
    recorder = SessionRecorder(ROOT / "data" / "wearable", ROOT / "configs" / "wearable.yaml")
    app = make_app(predictors, web_dir=ROOT / "web", replay=replay, recorder=recorder)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
