"""Server side of the recording page: writes a session to ``data/wearable`` as it is recorded.

The page (``web/record.html``) streams the kit's samples to the server and tells it when a
labelled set starts and ends. The server appends every sample to ``raw.csv`` at once, so a
closed tab or a reload loses nothing, and keeps ``events.csv`` and ``meta.json`` in the format
of ``scripts/record_wearable.py``; ``uaar.data.wearable.load_session`` reads the folder.

    <root>/<subject>/<session>/raw.csv      every sample received while the session was open
    <root>/<subject>/<session>/events.csv   one row per kept set
    <root>/<subject>/<session>/meta.json    subject, device, counts, the sets
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import yaml

from uaar.data import wearable as w
from uaar.data.wearable_windows import Roles

SUBJECT_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
LABEL_RE = re.compile(r"^[^,\n\r]{1,80}$")


class SessionRecorder:
    """One recording session at a time, driven by the WebSocket messages of the page.

    Args:
        root: the recordings folder (``data/wearable``).
        config: ``configs/wearable.yaml``: the label roles and the ``protocol`` checklist.
    """

    def __init__(self, root: Path, config: Path | None = None):
        self.root = Path(root)
        conf = yaml.safe_load(Path(config).read_text()) if config else {}
        self.roles = Roles.load(config) if config else Roles()
        self.protocol = [
            {
                "label": str(p["label"]),
                "role": self.roles.of(str(p["label"]))[0],
                "sets": int(p.get("sets", 1)),
                "seconds": float(p.get("seconds", 30)),
                "hint": str(p.get("hint", "")),
            }
            for p in (conf or {}).get("protocol", [])
        ]
        self.dir: Path | None = None
        self._raw = None
        self.meta: dict = {}
        self.sets: list[dict] = []
        self._open: dict | None = None
        self._next_index = 0
        self.n_samples = 0
        self.n_dropped = 0
        self.n_gaps = 0
        self._last: list[int] | None = None

    # ---- state ---------------------------------------------------------------------------
    @property
    def recording(self) -> bool:
        return self.dir is not None

    def state(self) -> dict:
        """What the page needs to draw itself, also after a reload in the middle of a session."""
        return {
            "type": "rec_state",
            "recording": self.recording,
            "subject": self.meta.get("subject"),
            "session": self.meta.get("session"),
            "folder": str(self.dir) if self.dir else None,
            "n_samples": self.n_samples,
            "n_dropped": self.n_dropped,
            "open_label": self._open["label"] if self._open else None,
            "sets": [self.describe(s) for s in self.sets],
            "protocol": self.protocol,
            "subjects": sorted(p.name for p in self.root.glob("*") if p.is_dir()),
        }

    # ---- session -------------------------------------------------------------------------
    def start(self, subject: str, placement: str = "", device: str = "", firmware: str = "",
              transport: str = "ble") -> dict:  # fmt: skip
        """Open a new session folder for ``subject`` and start writing ``raw.csv``."""
        if self.recording:
            raise ValueError("a session is already being recorded; finish it first")
        if not SUBJECT_RE.match(subject or ""):
            raise ValueError("the subject code must be letters, digits, '-' or '_' (e.g. W01)")
        now = datetime.now()
        session = now.strftime("%Y%m%d-%H%M%S")
        self.dir = self.root / subject / session
        self.dir.mkdir(parents=True, exist_ok=False)
        self.meta = {
            "subject": subject,
            "session": session,
            "started_at": now.isoformat(timespec="seconds"),
            "device_name": device,
            "firmware": firmware,
            "transport": transport,
            "placement": placement,
            "sample_rate_hz": w.SAMPLE_RATE_HZ,
            "acc_lsb_per_g": w.ACC_LSB_PER_G,
            "gyr_lsb_per_dps": w.GYR_LSB_PER_DPS,
        }
        self._raw = open(self.dir / "raw.csv", "w", encoding="utf-8")  # noqa: SIM115
        self._raw.write(f"# UAAR fw {firmware or '?'} name={device or '?'} via={transport}\n")
        self._raw.write(f"# recorded {self.meta['started_at']}\n")
        self._raw.write(",".join(w.RAW_COLUMNS) + "\n")
        self.sets, self._open, self._next_index = [], None, 0
        self.n_samples = self.n_dropped = self.n_gaps = 0
        self._last = None
        self._write_meta()
        return self.state()

    def add_samples(self, rows: list[list[int]]) -> None:
        """Append raw rows ``[t_ms, ax, ay, az, gx, gy, gz, flags, seq]`` (ignored when idle)."""
        if not self.recording:
            return
        for r in rows:
            row = [int(v) for v in r[:9]]
            dropped = 0
            if self._last is not None:
                step = (row[8] - self._last[8]) % 65536
                if step > 1:
                    dropped = step - 1
                    self.n_dropped += dropped
                    self.n_gaps += 1
            self._last = row
            self.n_samples += 1
            self._raw.write(",".join(str(v) for v in row) + "\n")
            if self._open is not None:
                s = self._open
                if s["n_samples"] == 0:
                    s["t_start_ms"], s["seq_start"] = row[0], row[8]
                else:
                    s["n_dropped"] += dropped
                s["t_end_ms"], s["seq_end"] = row[0], row[8]
                s["n_samples"] += 1
        self._raw.flush()

    # ---- sets ----------------------------------------------------------------------------
    def set_start(self, label: str, session_id: int = 0) -> None:
        """The samples from now on belong to a set with this label, until :meth:`set_end`."""
        if not self.recording:
            raise ValueError("start a session first")
        if not LABEL_RE.match(label or ""):
            raise ValueError("the label must be 1-80 characters without commas")
        self._open = {
            "set_index": self._next_index,
            "label": label,
            "session_id": int(session_id),
            "t_start_ms": 0,
            "t_end_ms": 0,
            "seq_start": 0,
            "seq_end": 0,
            "n_samples": 0,
            "n_dropped": 0,
        }

    def set_end(self) -> dict | None:
        """Close the open set and keep it; ``None`` if there was none or it has no samples."""
        s, self._open = self._open, None
        if s is None or s["n_samples"] == 0:
            return None
        self._next_index += 1
        self.sets.append(s)
        self._write_events()
        return self.describe(s)

    def set_cancel(self) -> None:
        """Drop the open set without keeping it."""
        self._open = None

    def discard(self, set_index: int) -> bool:
        """Remove a kept set (its samples stay in ``raw.csv``, unlabelled)."""
        before = len(self.sets)
        self.sets = [s for s in self.sets if s["set_index"] != int(set_index)]
        if len(self.sets) != before:
            self._write_events()
        return len(self.sets) != before

    @staticmethod
    def describe(s: dict) -> dict:
        """A set as the page shows it: seconds, samples and the share of samples lost."""
        expected = s["n_samples"] + s["n_dropped"]
        return {
            **s,
            "seconds": round((s["t_end_ms"] - s["t_start_ms"]) / 1000.0 + 1 / w.SAMPLE_RATE_HZ, 2),
            "lost_share": round(s["n_dropped"] / expected, 4) if expected else 0.0,
        }

    # ---- finish --------------------------------------------------------------------------
    def stop(self) -> dict:
        """Close the session. A session without a kept set is deleted. Returns the summary."""
        if not self.recording:
            raise ValueError("no session is being recorded")
        self._open = None
        self._raw.close()
        self._raw = None
        folder, self.dir = self.dir, None
        if not self.sets:
            shutil.rmtree(folder)
            return {"type": "rec_summary", "saved": False, "folder": None, "labels": []}
        self._write_events(folder)
        self._write_meta(folder)
        return {"type": "rec_summary", "saved": True, "folder": str(folder), **self.summary()}

    def summary(self) -> dict:
        """Per label of this session: sets and seconds, against what the checklist asks for."""
        need = {p["label"]: p["sets"] for p in self.protocol}
        rows = {}
        for s in self.sets:
            r = rows.setdefault(s["label"], {"label": s["label"], "sets": 0, "seconds": 0.0})
            r["sets"] += 1
            r["seconds"] = round(r["seconds"] + self.describe(s)["seconds"], 1)
        for label, r in rows.items():
            r["role"] = self.roles.of(label)[0]
            r["needed"] = need.get(label, 0)
        missing = [
            {"label": p["label"], "missing": p["sets"] - rows.get(p["label"], {"sets": 0})["sets"]}
            for p in self.protocol
            if rows.get(p["label"], {"sets": 0})["sets"] < p["sets"]
        ]
        return {
            "subject": self.meta.get("subject"),
            "session": self.meta.get("session"),
            "n_samples": self.n_samples,
            "n_dropped": self.n_dropped,
            "labels": list(rows.values()),
            "missing": missing,
        }

    # ---- files ---------------------------------------------------------------------------
    def _write_events(self, folder: Path | None = None) -> None:
        lines = [",".join(w.EVENTS_COLUMNS)]
        lines += [",".join(str(s[c]) for c in w.EVENTS_COLUMNS) for s in self.sets]
        ((folder or self.dir) / "events.csv").write_text("\n".join(lines) + "\n")

    def _write_meta(self, folder: Path | None = None) -> None:
        meta = {
            **self.meta,
            "n_samples": self.n_samples,
            "n_dropped": self.n_dropped,
            "n_gaps": self.n_gaps,
            "sets": [
                {
                    "set_index": s["set_index"],
                    "label": s["label"],
                    "seconds": self.describe(s)["seconds"],
                }
                for s in self.sets
            ],
        }
        ((folder or self.dir) / "meta.json").write_text(json.dumps(meta, indent=2))
