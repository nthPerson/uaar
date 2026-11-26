# Software and firmware

## 1. Repository

The repository `uaar` (GitHub, public), separate from the FormCoach (BDA 696) repository; Python package `uaar`. The layout is in [development-spec.md §2.1](development-spec.md#21-location-and-layout), the modules in §4 and the scripts in §5 of the same document.

## 2. Software stack

| Purpose | Tools |
|---|---|
| Environment | Python 3.12, `uv` |
| Data | NumPy, SciPy (`io.loadmat`), pandas, PyArrow (Parquet), PyYAML |
| Classical ML | scikit-learn (logistic regression, GroupKFold, metrics) |
| Deep learning | PyTorch 2.x (backbone, EDL loss, MC Dropout inference) |
| Plots | Matplotlib |
| Live page | FastAPI + uvicorn (`uaar.live`, `scripts/demo.py`); pages in plain JavaScript (`web/`), no build step |
| Wearable I/O | `pyserial` (USB serial, `scripts/record_wearable.py`); Bluetooth through Web Bluetooth in the browser (`web/link.js`) |
| Testing | pytest; `node:test` for `web/protocol.js` |

**Outside code:** none beyond the libraries above. EDL and MC Dropout are implemented from the papers (Sensoy et al. 2018; Gal & Ghahramani 2016), not copied from reference repositories. Dataset loaders are written from the documented file formats in [datasets.md](datasets.md). AI assistance is recorded in [ai-use-log.md](../ai-use-log.md).

**Compute:** a 16-core workstation under WSL2, with an RTX 3070 eGPU over Thunderbolt when attached. Expected training cost is minutes per run for a ~30 k-parameter network.

## 3. Demo page (the course app)

The course requirement "upload data, show prediction, explain output" is met by the live page; no separate app was built (`../decisions.md`, 2026-11-23). Full guide: [web/README.md](../../web/README.md).

```bash
uv run scripts/demo.py                                      # http://localhost:8000
uv run scripts/demo.py --replay data/wearable/W01/test-60s  # plays a recording on load
```

- **Upload:** the kit's Bluetooth stream (**Connect wearable**, Edge or Chrome) or a recorded `raw.csv` (**Upload a recording**), sent at the pace it was recorded. `--replay` plays a recording as soon as the page opens: the fallback on stage.
- **Prediction:** one card per entry in `configs/demo.yaml`: the plain softmax network (score `msp`), MC Dropout taught to be uncertain on other movements (`entropy`) and EDL taught the same way (`vacuity`). The models are in `models/`, so the page runs from a fresh clone.
- **Explanation:** each card shows the answer (an exercise, or **Unknown** when the uncertainty is over the model's limit), the uncertainty on a 0–100 % bar with the limit marked, and the last 30 answers. The bar is the score divided by its largest possible value; each limit τ comes from the model's own validation data (`threshold.json`). The presenters explain the output; each card names its uncertainty measure in plain words.
- **Recording page:** `/record.html` on the same server records labelled sets for §5.

## 4. Wearable hardware and firmware interface

> **Revised 2026-11-03** (see `../decisions.md`): the hardware is the same kit, but it runs **UAAR firmware v0.1.0** from `firmware/` in this repository, not FormCoach's. The wire format is kept compatible with FormCoach's by design so one recorder can read either device; the firmware code is written separately for this course. Units were verified on 2026-11-03 (|a| = 0.987 g at rest; 4096 LSB/g, 32.8 LSB per °/s).

The wearable hardware was built in BDA 696 (FormCoach). UAAR uses the device as a measuring instrument: the hosts read its IMU stream and send it one-byte commands (session start and stop, gyro calibration). The hardware specifications below come from FormCoach's design documents (`docs/02-system-design.md`, `docs/03-hardware.md`, in the FormCoach repository). The firmware version is recorded in each session's `meta.json`.

**Hardware**

| Part | Specification |
|---|---|
| MCU | Seeed Studio XIAO ESP32-S3: 240 MHz dual core, 8 MB PSRAM, 8 MB flash, Wi-Fi + BLE 5.0, USB-C, onboard LiPo charger |
| IMU | Bosch BMI160 (GY-BMI160 module), I2C at 400 kHz on D4/D5, address 0x69 or 0x68 |
| Battery | 3.7 V 300 mAh protected LiPo, behind a slide switch |
| Input / output | One push button (D1), built-in LED |
| Case | 3D-printed, Apple Watch band lugs; the printed arrow sets the IMU axes (X along the forearm) |

**IMU configuration:** accelerometer ±8 g (4096 LSB/g), gyroscope ±1000 °/s (32.8 LSB per °/s), 100 Hz internal output data rate averaged down to **50 Hz**.

**Wire protocol.** The source of truth is `firmware/include/protocol.h`, mirrored by `src/uaar/data/wearable.py` and `web/protocol.js` (pinned by `tests/test_wearable.py` and `tests/test_web_protocol.py`). Behaviour of the button and LED: `firmware/README.md`.

**USB serial:** 115200 baud; one CSV line per sample, `t_ms,ax,ay,az,gx,gy,gz,flags,seq`, in raw sensor counts; lines starting with `#` are comments (boot banner, session messages). `seq` is a rolling 16-bit counter for detecting dropped samples. Serial commands are single letters (below).

**BLE:**

| Item | Value |
|---|---|
| Device name | `UAAR-<last 4 hex of MAC>` |
| Service UUID | `7a0c0001-4e1e-4b9a-9a1c-0f0c0c0c0001` |
| IMU characteristic (notify) | `7a0c0002-…-0001`: up to 5 samples (100 ms) per notification, 99 bytes (4-byte header + 5 × 19-byte samples); the firmware requests MTU 185. Web Bluetooth cannot request an MTU; the page decodes each notification by its `n` field |
| Control characteristic (write) | `7a0c0003-…-0001`, one byte (serial letter in brackets): `0x01` LED "known" (`K`), `0x02` LED "unknown" (`U`), `0x03` start session (`S`), `0x04` stop session (`X`), `0x05` calibrate the gyro bias, 3 s still, stored in NVS (`C`), `0x10` ping (`P`), `0x11` reprint the boot banner (`I`) |
| Status characteristic (read, notify) | `7a0c0004-…-0001`: 12 bytes, once per second: firmware version, `session_id`, uptime, calibrated, session active |
| Batch header | `session_id` uint16, `n` uint8, reserved uint8 |
| Sample (little-endian, 19 B) | `t_ms` uint32; `ax, ay, az` int16; `gx, gy, gz` int16; `flags` uint8 (bit 0 reserved, always 0; bit 1 session active; bit 2 button pressed; bit 3 calibrated); `seq` uint16 |

**Conversion in `uaar.data.wearable`:** `raw_to_physical` gives acc_g = raw / 4096, gyro_dps = raw / 32.8, t = t_ms / 1000; `seq_gaps` counts dropped samples; `read_raw_csv` and `load_session` read a session folder (`raw.csv`, `events.csv`, `meta.json`). The recorders write those files: `scripts/record_wearable.py` (USB, which also writes `app.csv`, a plain `t, ax … gz` CSV in s, g and °/s, and `quicklook.png`) and the recording page's server (`uaar.live.recorder`).

## 5. Recording protocol

The protocol as recorded is in [wearable-recording.md](../wearable-recording.md), section "Protocol"; the checklist and the role of every label are in `configs/wearable.yaml`. In short:

- **Subjects:** Robert (`W01`) and Sirena (`W02`). Subject codes only; no video, no names in files.
- **Placement:** left wrist, case arrow toward the elbow (FormCoach convention). This differs from RecoFit's right-forearm band, and is part of the shift being measured.
- **Sets:** the four known exercises (`curl`, `shoulder_press`, `lateral_raise`, `jumping_jacks`), 3 sets each; background movements used for fine-tuning (`standing_still`, `walking`, `arm_circles`, `shadow_boxing`), 2 sets each; unknown movements for testing only (`waving`, `torso_twists`), 2 sets each; placement shift (`curl_rotated90`, `curl_loose`), 1 set each.
- **Labels:** the recording page (`/record.html`) shows the checklist; the kit's button starts a set, which by default stops by itself. `scripts/record_wearable.py` over USB is the fallback: the label is typed, the button starts and stops the set.
- **Check immediately:** the page marks a set that lost more than 1 % of its samples or is shorter than 10 s, to be discarded and recorded again.
- **Battery safety:** protected cells only; charge through USB-C while attended, on a non-flammable surface; never use a swollen or hot cell.
