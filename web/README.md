# UAAR wearable console (`web/`)

A single-page browser console for the UAAR wrist wearable. It talks to the device over Web
Bluetooth, shows the live IMU stream, records labelled sets into the same three files that
`scripts/record_wearable.py` writes, and forwards samples to a local prediction backend over a
WebSocket so the live open-set classifier can be shown side by side.

Vanilla JavaScript ES modules, no build step, no CDN or network dependencies.

| file | role |
| --- | --- |
| `index.html`, `style.css` | layout; dark, large type for a projector |
| `protocol.js` | wire protocol (constants, `decodeBatch`, `encodeBatch`, `decodeStatus`, `toPhysical`, `seqGap`, `sampleToCsvLine`); mirrors `firmware/include/protocol.h` and `src/uaar/data/wearable.py` |
| `app.js` | BLE link, charts, recorder, WebSocket client, prediction panel, simulator |
| `test/protocol.test.mjs` | `node:test` suite for `protocol.js` (run by `tests/test_web_protocol.py` when `node` exists) |

## Serve and open

Web Bluetooth needs a secure context; `http://localhost` counts, so no TLS is required.

```bash
uv run scripts/serve_web.py            # http://localhost:8000/
uv run scripts/serve_web.py --port 8080
```

Open the URL in **Chrome or Edge** (Windows, macOS, Linux, Android). Opening `index.html` as a
`file://` URL does not work: ES modules and Web Bluetooth both need an HTTP origin.

## Connect to the wearable

1. Power the wearable; it advertises as `UAAR-xxxx`.
2. Click **Connect wearable**, choose the device in the browser's chooser.
3. The page reads the status characteristic once (firmware, uptime, calibrated, session id)
   and subscribes to the IMU and status notifications. Device name, packet rate (10/s expected),
   sample rate (50 Hz expected), dropped samples (from gaps in `seq`), the button pill and a
   large SESSION ACTIVE / SESSION IDLE indicator sit in the status bar.
4. **Disconnect** drops the link; if the device drops it, the state shows "disconnected" and a
   **Reconnect** button re-opens GATT on the same device without a new chooser. Notification
   handlers are added with a fixed bound function and removed before re-adding, so a reconnect
   never stacks subscriptions.

Commands (one byte on the control characteristic): Ping, Start session, Stop session,
Calibrate (hold still 3 s), LED known, LED unknown, Identify. The wearable's button also toggles
the session.

Web Bluetooth cannot request an MTU. The page decodes every notification by its `n` header
field, so whatever size the link negotiates works.

**Simulate** runs without hardware: a synthetic 50 Hz stream (gravity plus a 1 Hz "curl" while
the session flag is on, session toggled every 10 s, one dropped `seq` every ~7 s) is encoded with
`encodeBatch` and decoded with `decodeBatch`, so every feature on the page exercises the real
wire path. The command buttons act on the simulator (Start/Stop session, Calibrate, Ping).

## Charts

Two canvases, last 10 s, redrawn at ~20 fps from a ring buffer: accelerometer `ax ay az` in g
plus `|a|`, and gyroscope `gx gy gz` in deg/s. The green band marks samples with the session flag
set. Lines break across gaps of more than 5 sample periods.

## Record mode and the files it downloads

Fill in **Label for next set**, **Subject**, **Session** (defaults to a `YYYYMMDD-HHMMSS`
timestamp, like the serial recorder), **Placement** and **Note**, click **Start recording**, then
run sets: press the wearable button (or **Start session**) to begin a set and again to end it.
Each set takes the label that is in the input at its rising edge; the label can be changed
between sets. **Stop recording**, then **Download session** produces three downloads (Chrome asks
once to allow multiple downloads). Put the three files in one folder, e.g.
`data/wearable/W01/20261103-101500/`, and `uaar.data.wearable.load_session(folder)` reads them.

`raw.csv` -- every sample received while recording, in the exact serial format:

```
# UAAR fw 0.1.0 name=UAAR-1A2B via=ble
# recorded 2026-11-03T10:15:00
t_ms,ax,ay,az,gx,gy,gz,flags,seq
184120,1408,-235,3800,-3,7,-2,2,4411
...
```

(`via=sim` and `transport: "sim"` when the data came from the simulator.)

`events.csv` -- one row per set, derived from the session flag's rising and falling edges:

```
set_index,label,session_id,t_start_ms,t_end_ms,seq_start,seq_end,n_samples,n_dropped
0,bicep_curl,3,184120,196100,4411,5010,600,0
```

`meta.json`:

```json
{
  "subject": "W01",
  "session": "20261103-101500",
  "started_at": "2026-11-03T10:15:00",
  "device_name": "UAAR-1A2B",
  "firmware": "0.1.0",
  "transport": "ble",
  "sample_rate_hz": 50,
  "acc_lsb_per_g": 4096,
  "gyr_lsb_per_dps": 32.8,
  "placement": "left wrist, case arrow toward the elbow",
  "note": "",
  "n_samples": 12000,
  "n_dropped": 0,
  "n_gaps": 0,
  "sets": [{ "set_index": 0, "label": "bicep_curl", "seconds": 11.98 }]
}
```

`subject` is omitted when left blank so the loader falls back to the folder name. A set still
open when recording stops is closed at the last recorded sample (the serial recorder drops it).

## WebSocket: live prediction panel

The page is a WebSocket **client**. Enter the backend URL (default `ws://localhost:8765`) and
click **Connect**. Until a backend accepts the connection the panel reads "backend not connected".

Page -> backend, one message per IMU notification, raw integer counts:

```json
{"type": "samples", "session_id": 3,
 "samples": [[184120, 1408, -235, 3800, -3, 7, -2, 2, 4411], ...]}
```

Each row is `[t_ms, ax, ay, az, gx, gy, gz, flags, seq]` in the order of `raw.csv`.

Backend -> page:

```json
{"type": "prediction", "t_start": 184.1, "t_end": 186.1,
 "methods": {
   "softmax":  {"label": "Bicep Curl", "p": 0.93, "score": 0.12, "tau": 0.35, "unknown": false},
   "edl":      {"label": "Bicep Curl", "p": 0.71, "score": 0.41, "tau": 0.30, "unknown": true}
 }}
{"type": "status", "text": "model loaded: edl_v3, window 2.0 s"}
```

One large card per method: method name, predicted label (or **UNKNOWN** in red), a probability
bar for `p`, and an uncertainty bar for `score` drawn against `tau` (the tick sits at the middle
of the bar; the fill turns red past it). Under each card a 30-window timeline is coloured by label,
red-hatched for unknown. `status` messages appear as a line above the cards. Any other `type` is
ignored, so the backend can add message kinds freely.

## Tests

```bash
uv run pytest tests/test_web_protocol.py   # pins protocol.js to uaar.data.wearable; runs node tests if node >= 18
node --test web/test/protocol.test.mjs      # the JS suite alone
```

## Known limits

- Web Bluetooth exists in Chrome and Edge (and Chromium derivatives). Firefox and Safari do not
  support it; the page loads there, but only **Simulate** and the WebSocket panel work.
- The page must be served from `http://localhost` or an `https://` origin; `file://` and plain
  `http://<lan-ip>` are blocked by the browser.
- On Windows, Chrome uses the system Bluetooth stack: the wearable must not be paired/connected
  by another app at the same time.
- No MTU control from the browser; throughput is fine at 10 notifications/s of 99 bytes.
- Downloads are three separate files, not a zip; the browser may rename on name collisions
  (`raw (1).csv`), so download into an empty folder or rename before loading.
- Recorded samples live in page memory until downloaded; reloading the page loses them.
