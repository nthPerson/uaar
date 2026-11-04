#!/usr/bin/env python3
"""Replay a recorded session into the live backend as if a kit were streaming.

    uv run scripts/live_replay.py data/wearable/W01/test-60s [--url ws://host:8765/ws] [--speed 1]

Sends 5-sample batches at the recorded pace (``--speed 10`` for ten times faster) and prints
every prediction the backend returns. This is the on-stage fallback and the way to test the
backend without hardware.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import websockets

from uaar.data import wearable as w


async def replay(session_dir: Path, url: str, speed: float, limit_s: float | None) -> int:
    raw, _ = w.read_raw_csv(session_dir / "raw.csv")
    rows = raw[w.RAW_COLUMNS].to_numpy().tolist()
    t0 = rows[0][0]
    if limit_s is not None:
        rows = [r for r in rows if (r[0] - t0) / 1000 <= limit_s]
    n_pred = 0
    async with websockets.connect(url) as ws:
        print(json.loads(await ws.recv()).get("text"))
        await ws.send(json.dumps({"type": "reset"}))

        async def reader() -> None:
            nonlocal n_pred
            async for msg in ws:
                m = json.loads(msg)
                if m.get("type") != "prediction":
                    continue
                n_pred += 1
                parts = [
                    f"{k}: {v['label']:<14} p={v['p']:.2f} u={v['score']:.2f}"
                    + (" UNKNOWN" if v["unknown"] else "")
                    for k, v in m["methods"].items()
                ]
                print(f"[{m['t_start']:7.2f}-{m['t_end']:7.2f} s] " + " | ".join(parts))

        task = asyncio.create_task(reader())
        loop = asyncio.get_running_loop()
        start = loop.time()
        for i in range(0, len(rows), w.BATCH_SAMPLES):
            batch = rows[i : i + w.BATCH_SAMPLES]
            due = start + ((batch[0][0] - t0) / 1000.0) / speed
            delay = due - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            session_id = 1 if batch[0][7] & w.FLAG_SESSION_ACTIVE else 0
            await ws.send(
                json.dumps({"type": "samples", "session_id": session_id, "samples": batch})
            )
        await asyncio.sleep(0.5)
        task.cancel()
    print(f"sent {len(rows)} samples, received {n_pred} predictions")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("session_dir", type=Path)
    ap.add_argument("--url", default="ws://localhost:8765/ws")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--limit", type=float, default=None, help="replay only the first N seconds")
    args = ap.parse_args()
    return asyncio.run(replay(args.session_dir, args.url, args.speed, args.limit))


if __name__ == "__main__":
    sys.exit(main())
