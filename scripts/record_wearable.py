#!/usr/bin/env python3
"""Record labelled sets from the wearable over USB serial.

The wearable streams one CSV line per sample (``uaar.data.wearable``). This script writes every
line to ``raw.csv``, watches the session flag that the wearable's button toggles, and turns each
button-delimited session into one labelled set in ``events.csv``. ``meta.json`` records the
device, firmware, calibration and placement so the loader can trust the units.

Interactive use (default): type the label for the next set, then press the wearable's button
to start and again to stop; repeat; ``q`` to finish.

    uv run scripts/record_wearable.py --port /dev/ttyACM0 --subject W01

Unattended test (no button, one set):

    uv run scripts/record_wearable.py --port /dev/ttyACM0 --subject W01 --duration 60 --label test

Output: ``<out>/<subject>/<session>/{raw.csv,events.csv,meta.json,app.csv,quicklook.png}``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
import serial

from uaar.data import wearable as w


@dataclass
class StreamState:
    """What the reader thread knows; the main thread only reads it (under ``lock``)."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    banner: list[str] = field(default_factory=list)
    n_samples: int = 0
    n_dropped: int = 0
    n_gaps: int = 0
    last_seq: int | None = None
    last_t_ms: int = 0
    first_t_ms: int | None = None
    session_active: bool = False
    session_id: int = 0
    # Per-set bookkeeping (t/seq of the first and last sample with the session flag set).
    set_t_start: int | None = None
    set_seq_start: int | None = None
    set_t_end: int | None = None
    set_seq_end: int | None = None
    set_samples: int = 0
    set_dropped: int = 0
    completed_sets: list[dict] = field(default_factory=list)
    pending_label: str = ""
    stop: bool = False


class Reader(threading.Thread):
    """Reads the serial port, appends every line to raw.csv, updates StreamState."""

    def __init__(self, port: serial.Serial, raw_path: Path, state: StreamState):
        super().__init__(daemon=True)
        self.port = port
        self.raw = open(raw_path, "a", encoding="utf-8", buffering=1)
        self.state = state

    def run(self) -> None:
        st = self.state
        while not st.stop:
            try:
                line = self.port.readline()
            except serial.SerialException as e:
                print(f"\nserial error: {e}", file=sys.stderr)
                st.stop = True
                break
            if not line:
                continue
            text = line.decode("utf-8", errors="replace").rstrip("\r\n")
            self.raw.write(text + "\n")
            if text.startswith("#"):
                with st.lock:
                    if len(st.banner) < 12:
                        st.banner.append(text)
                    m = re.match(r"# session start id=(\d+)", text)
                    if m:
                        st.session_id = int(m.group(1))
                continue
            s = w.parse_csv_line(text)
            if s is None:
                continue
            with st.lock:
                self._on_sample(s)
        self.raw.close()

    def _on_sample(self, s: w.Sample) -> None:
        st = self.state
        st.n_samples += 1
        st.last_t_ms = s.t_ms
        if st.first_t_ms is None:
            st.first_t_ms = s.t_ms
        dropped = 0
        if st.last_seq is not None:
            step = (s.seq - st.last_seq) % 65536
            if step > 1:
                dropped = step - 1
                st.n_dropped += dropped
                st.n_gaps += 1
        st.last_seq = s.seq
        active = bool(s.flags & w.FLAG_SESSION_ACTIVE)
        if active and not st.session_active:  # rising edge: a set starts
            st.set_t_start, st.set_seq_start = s.t_ms, s.seq
            st.set_samples, st.set_dropped = 0, 0
        if active:
            st.set_t_end, st.set_seq_end = s.t_ms, s.seq
            st.set_samples += 1
            st.set_dropped += dropped
        if not active and st.session_active:  # falling edge: the set is complete
            st.completed_sets.append(
                {
                    "set_index": len(st.completed_sets),
                    "label": st.pending_label,
                    "session_id": st.session_id,
                    "t_start_ms": st.set_t_start,
                    "t_end_ms": st.set_t_end,
                    "seq_start": st.set_seq_start,
                    "seq_end": st.set_seq_end,
                    "n_samples": st.set_samples,
                    "n_dropped": st.set_dropped,
                }
            )
        st.session_active = active


def send(port: serial.Serial, cmd: int) -> None:
    port.write(w.SERIAL_COMMANDS[cmd])
    port.flush()


def parse_banner(banner: list[str]) -> dict:
    info: dict = {}
    for line in banner:
        m = re.match(r"# UAAR fw (\S+) name=(\S+)", line)
        if m:
            info["firmware"], info["device_name"] = m.group(1), m.group(2)
        m = re.match(r"# bmi160 (\S+) (.*)", line)
        if m:
            info["imu_status"], info["imu_registers"] = m.group(1), m.group(2)
        m = re.match(r"# gyro bias (\w+) \((-?\d+),(-?\d+),(-?\d+)\)", line)
        if m:
            info["gyro_bias_applied"] = m.group(1) == "applied"
            info["gyro_bias_lsb"] = [int(m.group(i)) for i in (2, 3, 4)]
    return info


def status_line(st: StreamState, t0: float) -> str:
    with st.lock:
        dur = (st.last_t_ms - st.first_t_ms) / 1000 if st.first_t_ms is not None else 0.0
        rate = (st.n_samples - 1) / dur if dur > 0 else 0.0
        set_s = (
            (st.set_t_end - st.set_t_start) / 1000 if st.session_active and st.set_t_start else 0.0
        )
        rec = f"REC {st.pending_label!r} {set_s:5.1f} s" if st.session_active else "idle"
        return (
            f"\r[{time.time() - t0:6.1f} s] {st.n_samples:7d} samples  {rate:5.1f} Hz  "
            f"dropped {st.n_dropped:3d} in {st.n_gaps:2d} gaps  | {rec}      "
        )


def wait_for(st: StreamState, predicate, t0: float, poll: float = 0.25) -> bool:
    """Spin until predicate() is true, printing the status line; False if the stream stopped."""
    while not st.stop:
        with st.lock:
            if predicate():
                return True
        print(status_line(st, t0), end="", flush=True)
        time.sleep(poll)
    return False


def interactive(port: serial.Serial, st: StreamState, t0: float, keyboard: bool) -> None:
    print("\nFor each set: type the label, press Enter, then press the wearable button to start")
    print("and again to stop. Enter alone repeats the previous label; 'q' finishes the session.")
    if keyboard:
        print("(--keyboard: Enter starts and stops the set instead of the button.)")
    previous = ""
    while not st.stop:
        try:
            label = input(f"\nlabel for the next set [{previous or 'required'}]: ").strip()
        except EOFError:
            break
        if label.lower() == "q":
            break
        label = label or previous
        if not label:
            continue
        previous = label
        with st.lock:
            st.pending_label = label
        if keyboard:
            input("Enter to START ... ")
            send(port, w.CMD_START_SESSION)
        else:
            print("press the button to START ...")
        if not wait_for(st, lambda: st.session_active, t0):
            break
        print(
            f"\nrecording {label!r}; "
            + ("Enter to STOP" if keyboard else "press the button to STOP")
        )
        if keyboard:
            stopper = threading.Thread(
                target=lambda: (input(), send(port, w.CMD_STOP_SESSION)), daemon=True
            )
            stopper.start()
        if not wait_for(st, lambda: not st.session_active, t0):
            break
        with st.lock:
            done = st.completed_sets[-1]
        print(
            f"\nset {done['set_index']} {done['label']!r}: "
            f"{(done['t_end_ms'] - done['t_start_ms']) / 1000:.1f} s, {done['n_samples']} samples, "
            f"{done['n_dropped']} dropped"
        )


def unattended(
    port: serial.Serial, st: StreamState, t0: float, duration: float, label: str
) -> None:
    with st.lock:
        st.pending_label = label
    print(f"\nunattended: recording one set {label!r} for {duration:.0f} s")
    send(port, w.CMD_START_SESSION)
    if not wait_for(st, lambda: st.session_active, t0):
        return
    end = time.time() + duration
    wait_for(st, lambda: time.time() >= end, t0)
    send(port, w.CMD_STOP_SESSION)
    wait_for(st, lambda: not st.session_active, t0)
    print()


def quicklook(session_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rec = w.load_session(session_dir)
    amag = np.linalg.norm(rec.acc, axis=1)
    gmag = np.linalg.norm(rec.gyro, axis=1)
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    axes[0].plot(rec.t, amag, lw=0.6)
    axes[0].set_ylabel("|a| (g)")
    axes[1].plot(rec.t, gmag, lw=0.6, color="tab:orange")
    axes[1].set_ylabel("|ω| (deg/s)")
    axes[1].set_xlabel("t (s)")
    labels = pd.Series(rec.labels)
    for lab in [x for x in labels.unique() if x]:
        idx = np.flatnonzero(labels.to_numpy() == lab)
        if len(idx) == 0:
            continue
        for ax in axes:
            ax.axvspan(rec.t[idx[0]], rec.t[idx[-1]], alpha=0.15)
        axes[0].text(rec.t[idx[0]], float(amag.max()), lab, fontsize=8, va="top")
    title = f"{rec.subject} / {rec.recording_id}: {rec.n_samples} samples"
    fig.suptitle(f"{title}, {rec.meta['n_dropped']} dropped")
    fig.tight_layout()
    fig.savefig(session_dir / "quicklook.png", dpi=120)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--subject", required=True, help="subject code, e.g. W01")
    ap.add_argument("--session", default=None, help="session id (default: timestamp)")
    ap.add_argument("--out", type=Path, default=Path("data/wearable"))
    ap.add_argument("--placement", default="left wrist, case arrow toward the elbow")
    ap.add_argument("--note", default="")
    ap.add_argument(
        "--duration", type=float, default=None, help="unattended: record one set for N seconds"
    )
    ap.add_argument("--label", default="test", help="label for the unattended set")
    ap.add_argument(
        "--keyboard", action="store_true", help="start/stop sets from the keyboard, not the button"
    )
    args = ap.parse_args()

    session = args.session or datetime.now().strftime("%Y%m%d-%H%M%S")
    session_dir = args.out / args.subject / session
    session_dir.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now().isoformat(timespec="seconds")

    port = serial.Serial(args.port, args.baud, timeout=0.2)
    time.sleep(0.3)
    port.reset_input_buffer()
    st = StreamState()
    reader = Reader(port, session_dir / "raw.csv", st)
    reader.start()
    send(port, w.CMD_STOP_SESSION)  # make sure we start from idle
    send(port, w.CMD_IDENTIFY)
    time.sleep(1.0)
    with st.lock:
        info = parse_banner(st.banner)
    bias = "applied" if info.get("gyro_bias_applied") else "none"
    print(f"device: {info.get('device_name', '?')} fw {info.get('firmware', '?')}", end="  ")
    print(f"imu {info.get('imu_status', '?')}  gyro bias {bias}")
    if info.get("imu_status") != "ok":
        print("the IMU did not report ok; check the banner in raw.csv", file=sys.stderr)
    print(f"writing to {session_dir}")
    t0 = time.time()
    try:
        if args.duration:
            unattended(port, st, t0, args.duration, args.label)
        else:
            interactive(port, st, t0, args.keyboard)
    except KeyboardInterrupt:
        print("\ninterrupted; finishing the session files")
    finally:
        send(port, w.CMD_STOP_SESSION)
        time.sleep(0.5)
        st.stop = True
        reader.join(timeout=2)
        port.close()

    with st.lock:
        sets = list(st.completed_sets)
        summary = {"n_samples": st.n_samples, "n_dropped": st.n_dropped, "n_gaps": st.n_gaps}
    pd.DataFrame(sets, columns=w.EVENTS_COLUMNS).to_csv(session_dir / "events.csv", index=False)
    meta = {
        "subject": args.subject,
        "session": session,
        "started_at": started_at,
        "port": args.port,
        "placement": args.placement,
        "note": args.note,
        "sample_rate_hz": w.SAMPLE_RATE_HZ,
        "acc_lsb_per_g": w.ACC_LSB_PER_G,
        "gyr_lsb_per_dps": w.GYR_LSB_PER_DPS,
        **info,
        **summary,
        "sets": [
            {
                "set_index": s["set_index"],
                "label": s["label"],
                "seconds": (s["t_end_ms"] - s["t_start_ms"]) / 1000,
            }
            for s in sets
        ],
    }
    (session_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    rec = w.load_session(session_dir)
    w.to_app_csv(rec, session_dir / "app.csv")
    quicklook(session_dir)
    print(
        f"\nsession {session}: {summary['n_samples']} samples, {summary['n_dropped']} dropped",
        end=" ",
    )
    print(f"in {summary['n_gaps']} gaps, {len(sets)} sets")
    for s in sets:
        secs = (s["t_end_ms"] - s["t_start_ms"]) / 1000
        print(f"  set {s['set_index']:2d} {s['label']:<24} {secs:6.1f} s", end="  ")
        print(f"{s['n_samples']:5d} samples  {s['n_dropped']} dropped")
    print(f"files: {session_dir}/{{raw.csv,events.csv,meta.json,app.csv,quicklook.png}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
