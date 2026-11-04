# Software and firmware

## 1. Repository

A **new repository**, separate from the FormCoach (BDA 696) repository, owned by the CompE 510 team. Proposed package name `uaar`.

```text
uaar/
├── pyproject.toml            # uv-managed; Python 3.12
├── configs/                  # one YAML per experiment (protocol, method, split, seed)
├── src/uaar/
│   ├── data/                 # recofit.py, mmfit.py, recgym.py, wearable.py
│   │                         # windows.py (windowing, labels), splits.py (subject folds, class splits)
│   ├── features/             # handcrafted.py (M1 features)
│   ├── models/               # backbone.py, logreg.py, softmax.py, edl.py, mcdropout.py
│   ├── eval/                 # metrics.py (AUROC, FPR@95, AURC), calibration.py (ECE, reliability),
│   │                         # cost.py (latency, MACs), stats.py (Wilcoxon, bootstrap), plots.py
│   └── app/                  # main.py (Streamlit)
├── scripts/                  # prepare_data.py, train.py, evaluate.py, make_tables.py,
│                             # record_wearable.py
├── tests/                    # unit tests: loaders, windowing, splits (no leakage), metrics, EDL loss
├── reports/                  # generated tables and figures (never hand-edited)
└── data/                     # gitignored: external/ (downloads), processed/ (windows, splits)
```

## 2. Software stack

| Purpose | Tools |
|---|---|
| Environment | Python 3.12, `uv` |
| Data | NumPy, SciPy (`io.loadmat`, `signal.resample_poly`), pandas, PyArrow (Parquet) |
| Classical ML | scikit-learn (logistic regression, GroupKFold, metrics) |
| Deep learning | PyTorch 2.x (backbone, EDL loss, MC Dropout inference) |
| Plots | Matplotlib |
| App | Streamlit |
| Wearable I/O | `pyserial` (USB serial), `bleak` (BLE) |
| Testing | pytest |

**Outside code:** none beyond the libraries above. EDL and MC Dropout are implemented by the team from the papers (Sensoy et al. 2018; Gal & Ghahramani 2016), not copied from reference repositories. Dataset loaders are written by the team from the documented file formats in [datasets.md](datasets.md).

**Compute:** Robert's machines (a 16-core workstation under WSL2; an RTX 3070 eGPU over Thunderbolt when attached); Sirena's machine for evaluation and the app. Expected training cost is minutes per run for a ~30 k-parameter network on ~100 k windows.

## 3. Demo app (Streamlit)

Run with `uv run streamlit run src/uaar/app/main.py`.

**Input**

- Upload a CSV with columns `t, ax, ay, az, gx, gy, gz`: `t` in seconds, acceleration in g or m/s², rotation rate in °/s or rad/s (unit selectors in the sidebar). The sample rate is detected from `t` and resampled to 50 Hz.
- Or pick a bundled example: a RecoFit test-subject recording, an MM-Fit segment, or one of our wearable recordings.

**Controls**

- Method: logistic regression, softmax NN, EDL, MC Dropout (T slider).
- Uncertainty threshold τ (defaults to the validation value from `threshold.json`).

**Output**

1. The signal, with each 2-s window shaded by predicted exercise and windows above τ marked **Unknown**.
2. An uncertainty timeline with the threshold line.
3. A per-window table: start time, predicted exercise, confidence, uncertainty, decision.
4. A side-by-side view of two methods on the same file.
5. An "About this model" panel: test-set metrics and reliability diagram, precomputed from the P1 results.

**Stretch:** a live tab that reads the wearable over USB serial or BLE and updates the timeline in real time.

## 4. Wearable hardware and firmware interface

> **Revised 2026-11-03** (see `../decisions.md`): the hardware is the same kit, but it runs **UAAR firmware v0.1.0** from `firmware/` in this repository, not FormCoach's. The wire format below is kept identical to FormCoach's by design so one recorder can read either device; the firmware code is written separately for this course. The device name prefix is `UAAR-` and the serial command letters are documented in `firmware/README.md`. Units were verified on 2026-11-03 (|a| = 0.987 g at rest; 4096 LSB/g, 32.8 LSB per °/s).

The wearable hardware was built in BDA 696 (FormCoach). UAAR uses the device as a measuring instrument: it only reads the IMU stream. The hardware specifications below come from FormCoach's design documents (`docs/02-system-design.md`, `docs/03-hardware.md`). Record the firmware version in each session's `meta.json`.

**Hardware**

| Part | Specification |
|---|---|
| MCU | Seeed Studio XIAO ESP32-S3: 240 MHz dual core, 8 MB PSRAM, 8 MB flash, Wi-Fi + BLE 5.0, USB-C, onboard LiPo charger |
| IMU | Bosch BMI160 (GY-BMI160 module), I2C at 400 kHz on D4/D5, address 0x69 or 0x68 |
| Battery | 3.7 V 300 mAh protected LiPo, behind a slide switch |
| Input / output | One push button (D1), built-in LED |
| Case | 3D-printed, Apple Watch band lugs; the printed arrow sets the IMU axes (X along the forearm) |

**IMU configuration:** accelerometer ±8 g (4096 LSB/g), gyroscope ±1000 °/s (32.8 LSB per °/s), 100 Hz internal output data rate averaged down to **50 Hz**.

**USB serial (simplest for recording):** 115200 baud; in development mode one CSV line per sample:
`t_ms,ax,ay,az,gx,gy,gz,flags,seq`. Values are raw sensor counts **(verify against the firmware version)**; `seq` is a rolling counter for detecting dropped samples.

**BLE:**

| Item | Value |
|---|---|
| Device name | `FormCoach-<last 4 hex of MAC>` |
| Service UUID | `7a0c0001-4e1e-4b9a-9a1c-0f0c0c0c0001` |
| IMU characteristic (notify) | `7a0c0002-…-0001`: 5 samples (100 ms) per notification, 89 bytes; request MTU 185 |
| Control characteristic (write) | `7a0c0003-…-0001`: `0x03` start session, `0x04` stop session, `0x05` calibrate, `0x10` ping |
| Batch header | `session_id` uint16, `n` uint8, reserved uint8 |
| Sample (little-endian, 17 B) | `t_ms` uint32; `ax, ay, az` int16; `gx, gy, gz` int16; `flags` uint8 (bit 0 gate, bit 1 session active, bit 2 button pressed, bit 3 calibrated); `seq` uint16 |

**Conversion in `uaar.data.wearable`:** acc_g = raw / 4096; gyro_dps = raw / 32.8; t = t_ms / 1000; drop and count gaps in `seq`; write the app's CSV format plus `meta.json`.

## 5. Recording protocol (stretch goal 3)

- **Subjects:** Robert and Sirena (optionally a few volunteers who consent). Subject codes only; no video, no names in files.
- **Placement:** left wrist, case arrow toward the elbow (FormCoach convention). Note the difference from RecoFit's right-forearm band: this is part of the shift being measured.
- **Blocks (each set about 30–45 s, 3 sets per item):**
  1. **Known:** 3–4 exercises from the known set of the chosen RecoFit split (for example curls, shoulder press, squats, if they are known in that split).
  2. **New movements:** 3–4 movements not among RecoFit's 75 labels, chosen after reading the list (candidates: arm circles, shadow boxing, towel wringing, typing or phone use).
  3. **Placement shift:** one known exercise with the case rotated 90° on the wrist, and one with a loose strap.
  4. **Rest:** 60 s standing still and 60 s walking.
- **Labels:** press the wearable's button at the start and end of each set (`flags` bit 2); `record_wearable.py` prompts for the block name before each set and writes an events file.
- **Check immediately:** dropped-sample count from `seq`, duration per set, and a quick plot, before moving to the next block.
- **Battery safety:** protected cells only; charge through USB-C while attended, on a non-flammable surface; never use a swollen or hot cell.
