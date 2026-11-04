"""Our wearable (UAAR firmware, ``firmware/``): wire protocol, parsing and unit conversion.

The firmware streams raw BMI160 counts at 50 Hz as CSV lines over USB serial and as packed
5-sample batches over BLE. This module is the Python mirror of ``firmware/include/protocol.h``
(``tests/test_wearable.py`` pins the two together) and turns a recording session on disk into
a :class:`~uaar.data.recording.Recording` with physical units and per-set labels.

Session directory layout, written by ``scripts/record_wearable.py``::

    <session_dir>/raw.csv      every sample the device sent (t_ms,ax,...,flags,seq)
    <session_dir>/events.csv   one row per labelled set (see EVENTS_COLUMNS)
    <session_dir>/meta.json    subject, device, firmware, calibration, notes
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from uaar.data.recording import Recording

# ---- Constants mirrored from firmware/include/protocol.h -------------------------------------
DEVICE_NAME_PREFIX = "UAAR-"
SERVICE_UUID = "7a0c0001-4e1e-4b9a-9a1c-0f0c0c0c0001"
IMU_CHAR_UUID = "7a0c0002-4e1e-4b9a-9a1c-0f0c0c0c0001"
CONTROL_CHAR_UUID = "7a0c0003-4e1e-4b9a-9a1c-0f0c0c0c0001"
STATUS_CHAR_UUID = "7a0c0004-4e1e-4b9a-9a1c-0f0c0c0c0001"
REQUESTED_MTU = 185

SAMPLE_RATE_HZ = 50
BATCH_SAMPLES = 5
ACC_LSB_PER_G = 4096.0
GYR_LSB_PER_DPS = 32.8

CMD_LED_KNOWN = 0x01
CMD_LED_UNKNOWN = 0x02
CMD_START_SESSION = 0x03
CMD_STOP_SESSION = 0x04
CMD_CALIBRATE = 0x05
CMD_PING = 0x10
CMD_IDENTIFY = 0x11
SERIAL_COMMANDS: dict[int, bytes] = {
    CMD_LED_KNOWN: b"K",
    CMD_LED_UNKNOWN: b"U",
    CMD_START_SESSION: b"S",
    CMD_STOP_SESSION: b"X",
    CMD_CALIBRATE: b"C",
    CMD_PING: b"P",
    CMD_IDENTIFY: b"I",
}

FLAG_RESERVED = 1 << 0
FLAG_SESSION_ACTIVE = 1 << 1
FLAG_BUTTON_PRESSED = 1 << 2
FLAG_CALIBRATED = 1 << 3

SERIAL_CSV_HEADER = "t_ms,ax,ay,az,gx,gy,gz,flags,seq"
RAW_COLUMNS = SERIAL_CSV_HEADER.split(",")

SAMPLE_STRUCT = struct.Struct("<I6hBH")  # t_ms, ax..gz, flags, seq -> 19 bytes
BATCH_HEADER_STRUCT = struct.Struct("<HBB")  # session_id, n, reserved -> 4 bytes
STATUS_STRUCT = struct.Struct("<BBBHBIBB")  # fw x3, session_id, reserved, uptime, cal, active
SAMPLE_SIZE = SAMPLE_STRUCT.size
BATCH_HEADER_SIZE = BATCH_HEADER_STRUCT.size
MAX_BATCH_SIZE = BATCH_HEADER_SIZE + BATCH_SAMPLES * SAMPLE_SIZE
STATUS_SIZE = STATUS_STRUCT.size
assert SAMPLE_SIZE == 19 and BATCH_HEADER_SIZE == 4 and MAX_BATCH_SIZE == 99 and STATUS_SIZE == 12

EVENTS_COLUMNS = [
    "set_index",
    "label",
    "session_id",
    "t_start_ms",
    "t_end_ms",
    "seq_start",
    "seq_end",
    "n_samples",
    "n_dropped",
]


# ---- Wire-level decoding -------------------------------------------------------------------
@dataclass(frozen=True)
class Sample:
    t_ms: int
    ax: int
    ay: int
    az: int
    gx: int
    gy: int
    gz: int
    flags: int
    seq: int

    def as_csv(self) -> str:
        fields = (
            self.t_ms,
            self.ax,
            self.ay,
            self.az,
            self.gx,
            self.gy,
            self.gz,
            self.flags,
            self.seq,
        )
        return ",".join(str(v) for v in fields)


def parse_csv_line(line: str) -> Sample | None:
    """One serial line -> Sample, or None for comments, the header and malformed lines."""
    line = line.strip()
    if not line or line.startswith("#") or line == SERIAL_CSV_HEADER:
        return None
    parts = line.split(",")
    if len(parts) != 9:
        return None
    try:
        values = [int(p) for p in parts]
    except ValueError:
        return None
    return Sample(*values)


def decode_batch(payload: bytes) -> tuple[int, list[Sample]]:
    """One IMU notification -> (session_id, samples). Raises ValueError on a bad length."""
    if len(payload) < BATCH_HEADER_SIZE:
        raise ValueError(f"batch too short: {len(payload)} bytes")
    session_id, n, _reserved = BATCH_HEADER_STRUCT.unpack_from(payload, 0)
    expected = BATCH_HEADER_SIZE + n * SAMPLE_SIZE
    if n > BATCH_SAMPLES or len(payload) < expected:
        raise ValueError(f"batch declares {n} samples but has {len(payload)} bytes")
    samples = [
        Sample(*SAMPLE_STRUCT.unpack_from(payload, BATCH_HEADER_SIZE + i * SAMPLE_SIZE))
        for i in range(n)
    ]
    return session_id, samples


def encode_batch(session_id: int, samples: list[Sample]) -> bytes:
    """Inverse of decode_batch; used by tests and the simulator."""
    if len(samples) > BATCH_SAMPLES:
        raise ValueError("too many samples for one batch")
    out = bytearray(BATCH_HEADER_STRUCT.pack(session_id, len(samples), 0))
    for s in samples:
        out += SAMPLE_STRUCT.pack(s.t_ms, s.ax, s.ay, s.az, s.gx, s.gy, s.gz, s.flags, s.seq)
    return bytes(out)


def decode_status(payload: bytes) -> dict:
    major, minor, patch, session_id, _res, uptime_s, calibrated, active = STATUS_STRUCT.unpack(
        payload[:STATUS_SIZE]
    )
    return {
        "firmware": f"{major}.{minor}.{patch}",
        "session_id": session_id,
        "uptime_s": uptime_s,
        "calibrated": bool(calibrated),
        "session_active": bool(active),
    }


# ---- Streams and sessions ------------------------------------------------------------------
def seq_gaps(seq: np.ndarray) -> tuple[int, int]:
    """(dropped samples, number of gaps) from the rolling 16-bit ``seq`` counter.

    A step of exactly 1 (mod 65536) is normal; a step of k > 1 means k-1 samples were lost.
    Steps <= 0 (device reset) are ignored rather than counted as drops.
    """
    seq = np.asarray(seq, dtype=np.int64)
    if len(seq) < 2:
        return 0, 0
    step = np.diff(seq) % 65536
    missing = step[step > 1] - 1
    return int(missing.sum()), int(len(missing))


def raw_to_physical(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw counts -> ``t`` (s), ``ax..az`` (g), ``gx..gz`` (deg/s); keeps ``flags`` and ``seq``."""
    out = pd.DataFrame(
        {
            "t": (raw["t_ms"].to_numpy(dtype=np.float64) / 1000.0).astype(np.float32),
            "ax": (raw["ax"] / ACC_LSB_PER_G).astype(np.float32),
            "ay": (raw["ay"] / ACC_LSB_PER_G).astype(np.float32),
            "az": (raw["az"] / ACC_LSB_PER_G).astype(np.float32),
            "gx": (raw["gx"] / GYR_LSB_PER_DPS).astype(np.float32),
            "gy": (raw["gy"] / GYR_LSB_PER_DPS).astype(np.float32),
            "gz": (raw["gz"] / GYR_LSB_PER_DPS).astype(np.float32),
            "flags": raw["flags"].astype(np.uint8),
            "seq": raw["seq"].astype(np.uint16),
        }
    )
    return out


def read_raw_csv(path: Path | str) -> tuple[pd.DataFrame, list[str]]:
    """Read a device CSV (as written by the recorder or copied from a serial monitor).

    Returns the sample rows with RAW_COLUMNS as integers, in file order, plus the ``#`` comment
    lines so the caller can recover the banner (firmware version, calibration, session marks).
    """
    rows: list[Sample] = []
    comments: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                comments.append(line.rstrip("\n"))
                continue
            s = parse_csv_line(line)
            if s is not None:
                rows.append(s)
    raw = pd.DataFrame([s.__dict__ for s in rows], columns=RAW_COLUMNS).astype(np.int64)
    return raw, comments


def labels_from_events(raw: pd.DataFrame, events: pd.DataFrame) -> np.ndarray:
    """Per-sample label strings from the set table (by ``t_ms`` interval, inclusive)."""
    labels = np.full(len(raw), "", dtype=object)
    t = raw["t_ms"].to_numpy()
    for ev in events.itertuples(index=False):
        mask = (t >= ev.t_start_ms) & (t <= ev.t_end_ms)
        labels[mask] = str(ev.label)
    return labels.astype(str)


def load_session(session_dir: Path | str) -> Recording:
    """A recorded session directory -> Recording in physical units with per-set labels."""
    session_dir = Path(session_dir)
    raw, comments = read_raw_csv(session_dir / "raw.csv")
    meta = (
        json.loads((session_dir / "meta.json").read_text())
        if (session_dir / "meta.json").exists()
        else {}
    )
    events_path = session_dir / "events.csv"
    events = (
        pd.read_csv(events_path) if events_path.exists() else pd.DataFrame(columns=EVENTS_COLUMNS)
    )
    phys = raw_to_physical(raw)
    n_dropped, n_gaps = seq_gaps(raw["seq"].to_numpy())
    meta = {**meta, "n_dropped": n_dropped, "n_gaps": n_gaps, "banner": comments[:12]}
    return Recording(
        subject=str(meta.get("subject", session_dir.parent.name)),
        recording_id=str(meta.get("session", session_dir.name)),
        source=str(meta.get("session", session_dir.name)),
        t=phys["t"].to_numpy(),
        acc=phys[["ax", "ay", "az"]].to_numpy(dtype=np.float32),
        gyro=phys[["gx", "gy", "gz"]].to_numpy(dtype=np.float32),
        labels=labels_from_events(raw, events),
        meta=meta,
    )


def to_app_csv(rec: Recording, path: Path | str) -> None:
    """Write the demo app's input format: ``t,ax,ay,az,gx,gy,gz`` (s, g, deg/s)."""
    df = rec.to_frame()[["t", "ax", "ay", "az", "gx", "gy", "gz"]]
    df.to_csv(path, index=False, float_format="%.5f")
