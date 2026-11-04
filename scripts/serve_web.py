#!/usr/bin/env python3
"""Serve ``web/`` (the UAAR wearable console) on http://localhost:8000 with the stdlib.

Web Bluetooth needs a secure context; ``http://localhost`` qualifies, so no TLS is needed.

    uv run scripts/serve_web.py            # http://localhost:8000/
    uv run scripts/serve_web.py --port 8080
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parents[1] / "web"


class Handler(SimpleHTTPRequestHandler):
    """Static files with correct MIME types for ES modules and no caching while developing."""

    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".css": "text/css",
        ".json": "application/json",
        ".html": "text/html; charset=utf-8",
        ".csv": "text/csv",
        ".md": "text/markdown; charset=utf-8",
    }

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:  # quieter than the default
        sys.stderr.write(f"{self.address_string()} {fmt % args}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--dir", type=Path, default=WEB_DIR, help="directory to serve")
    args = ap.parse_args()
    if not (args.dir / "index.html").exists():
        print(f"no index.html in {args.dir}", file=sys.stderr)
        return 1
    server = ThreadingHTTPServer((args.host, args.port), partial(Handler, directory=str(args.dir)))
    print(f"serving {args.dir} at http://{args.host}:{args.port}/  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
