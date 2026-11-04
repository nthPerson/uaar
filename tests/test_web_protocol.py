"""web/protocol.js must mirror uaar.data.wearable (which tests/test_wearable.py pins to protocol.h).

The constants are regex-parsed out of the JS file. When ``node`` (>= 18) is available, the
node:test suite in ``web/test/protocol.test.mjs`` is also run against a golden packet produced
here by ``encode_batch``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from uaar.data import wearable as w

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_JS = ROOT / "web" / "protocol.js"
NODE_TEST = ROOT / "web" / "test" / "protocol.test.mjs"


def _js_consts() -> dict[str, str | int | float]:
    """``export const NAME = <literal>;`` lines -> {NAME: python value}."""
    out: dict[str, str | int | float] = {}
    for m in re.finditer(r"^export const (\w+) = ([^;]+);", PROTOCOL_JS.read_text(), re.M):
        name, raw = m.group(1), m.group(2).strip()
        raw = re.sub(r"\s*//.*$", "", raw)
        if raw.startswith(("'", '"')):
            out[name] = raw[1:-1]
        elif re.fullmatch(r"0x[0-9a-fA-F]+", raw):
            out[name] = int(raw, 16)
        elif re.fullmatch(r"-?\d+(\.\d+)?", raw):
            out[name] = float(raw) if "." in raw else int(raw)
        elif re.fullmatch(r"1 << \d+", raw):
            out[name] = 1 << int(raw.split("<<")[1])
        # anything else (expressions) is checked by the node tests, not here
    return out


JS = _js_consts()


def test_identity_constants_match():
    assert JS["DEVICE_NAME_PREFIX"] == w.DEVICE_NAME_PREFIX
    assert JS["SERVICE_UUID"] == w.SERVICE_UUID
    assert JS["IMU_CHAR_UUID"] == w.IMU_CHAR_UUID
    assert JS["CONTROL_CHAR_UUID"] == w.CONTROL_CHAR_UUID
    assert JS["STATUS_CHAR_UUID"] == w.STATUS_CHAR_UUID
    assert JS["REQUESTED_MTU"] == w.REQUESTED_MTU


def test_sampling_constants_match():
    assert JS["SAMPLE_RATE_HZ"] == w.SAMPLE_RATE_HZ
    assert JS["BATCH_SAMPLES"] == w.BATCH_SAMPLES
    assert JS["ACC_LSB_PER_G"] == w.ACC_LSB_PER_G
    assert JS["GYR_LSB_PER_DPS"] == pytest.approx(w.GYR_LSB_PER_DPS)
    assert JS["SAMPLE_SIZE"] == w.SAMPLE_SIZE
    assert JS["BATCH_HEADER_SIZE"] == w.BATCH_HEADER_SIZE
    assert JS["STATUS_SIZE"] == w.STATUS_SIZE


def test_command_and_flag_codes_match():
    for name in [
        "LED_KNOWN",
        "LED_UNKNOWN",
        "START_SESSION",
        "STOP_SESSION",
        "CALIBRATE",
        "PING",
        "IDENTIFY",
    ]:
        assert JS[f"CMD_{name}"] == getattr(w, f"CMD_{name}"), name
    for name in ["RESERVED", "SESSION_ACTIVE", "BUTTON_PRESSED", "CALIBRATED"]:
        assert JS[f"FLAG_{name}"] == getattr(w, f"FLAG_{name}"), name


def test_csv_headers_match():
    assert JS["SERIAL_CSV_HEADER"] == w.SERIAL_CSV_HEADER
    assert JS["EVENTS_HEADER"] == ",".join(w.EVENTS_COLUMNS)


def test_page_download_format_loads(tmp_path: Path):
    """The three files the page's Download button writes, as load_session must read them."""
    d = tmp_path / "W01" / "20261103-101500"
    d.mkdir(parents=True)
    samples = [
        w.Sample(20 * i, 4000, 0, 0, 0, 0, 0, 2 if 50 <= i <= 100 else 0, i) for i in range(200)
    ]
    raw = [
        "# UAAR fw 0.1.0 name=UAAR-ABCD via=ble",
        "# recorded 2026-11-03T10:15:00",
        w.SERIAL_CSV_HEADER,
    ]
    raw += [s.as_csv() for s in samples]
    (d / "raw.csv").write_text("\n".join(raw) + "\n")
    (d / "events.csv").write_text(
        ",".join(w.EVENTS_COLUMNS) + "\n" + '0,"curl, slow",1,1000,2000,50,100,51,0\n'
    )
    meta = {
        "subject": "W01",
        "session": "20261103-101500",
        "started_at": "2026-11-03T10:15:00",
        "device_name": "UAAR-ABCD",
        "firmware": "0.1.0",
        "transport": "ble",
        "sample_rate_hz": 50,
        "acc_lsb_per_g": 4096,
        "gyr_lsb_per_dps": 32.8,
        "placement": "left wrist, case arrow toward the elbow",
        "note": "",
        "n_samples": 200,
        "n_dropped": 0,
        "n_gaps": 0,
        "sets": [{"set_index": 0, "label": "curl, slow", "seconds": 1.0}],
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    rec = w.load_session(d)
    assert rec.subject == "W01" and rec.recording_id == "20261103-101500"
    assert rec.n_samples == 200
    assert (pd.Series(rec.labels) == "curl, slow").sum() == 51
    assert rec.meta["banner"][0] == "# UAAR fw 0.1.0 name=UAAR-ABCD via=ble"


def _node_version() -> tuple[int, ...] | None:
    node = shutil.which("node")
    if node is None:
        return None
    try:
        out = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.match(r"v(\d+)\.(\d+)\.(\d+)", out.stdout.strip())
    return tuple(int(x) for x in m.groups()) if m else None


@pytest.mark.skipif((_node_version() or (0,)) < (18,), reason="node >= 18 not available")
def test_node_decodes_python_golden_packet(tmp_path: Path):
    samples = [
        w.Sample(
            5000 + 20 * i,
            -4096,
            123,
            2048,
            -32768,
            32767,
            0,
            w.FLAG_SESSION_ACTIVE,
            (65534 + i) % 65536,
        )
        for i in range(5)
    ]
    golden = {
        "batch": {
            "hex": w.encode_batch(65535, samples).hex(),
            "session_id": 65535,
            "samples": [list(s.__dict__.values()) for s in samples],
        },
        "status": {
            "hex": w.STATUS_STRUCT.pack(1, 2, 3, 65535, 0, 123456, 0, 1).hex(),
            "expected": {
                "firmware": "1.2.3",
                "sessionId": 65535,
                "uptimeS": 123456,
                "calibrated": False,
                "sessionActive": True,
            },
        },
    }
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(json.dumps(golden))
    proc = subprocess.run(
        [shutil.which("node"), "--test", str(NODE_TEST)],
        cwd=ROOT,
        env={**os.environ, "UAAR_GOLDEN_JSON": str(golden_path)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert re.search(r"fail 0\b", proc.stdout), proc.stdout
    assert re.search(r"skipped 0\b", proc.stdout), proc.stdout  # the golden test must have run
