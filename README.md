# UAAR: uncertainty-aware activity recognition

CompE 510 (Machine Learning for Engineers), SDSU, Fall 2026. Team: Robert Ashe, Sirena Rocca.

A wrist-worn exercise recognizer is trained on a fixed list of exercises, but in use it sees
movements that were never in that list. This project compares four ways of letting the
recognizer say "I don't know" (logistic regression, a softmax network, evidential deep
learning and Monte Carlo dropout) on public IMU datasets, then runs the result live on our own
wearable (XIAO ESP32-S3 + BMI160).

Status (2026-11-03): repository bootstrapped; firmware v0.1.0 flashed and verified on kit K1; serial recorder, browser page and live backend working with a placeholder predictor; data pipeline and the softmax/EDL methods implemented and tested on synthetic data; the RecoFit run is next.

## Layout

| Path | What |
|---|---|
| `src/uaar/` | the Python package: data loaders, windows, splits, features, models, evaluation, app |
| `scripts/` | command-line entry points (prepare data, train, evaluate, record from the wearable) |
| `firmware/` | PlatformIO project for the wearable: a 50 Hz IMU streamer over USB serial and BLE |
| `web/` | browser page for live demo and recording (Web Bluetooth) |
| `configs/` | experiment configuration files |
| `docs/` | design documents and engineering docs |
| `tests/` | pytest suite (synthetic data only; no downloads needed) |
| `reports/` | generated tables and figures |
| `data/` | gitignored: `external/` downloads, `processed/` derived files, `wearable/` our recordings |

## Quickstart

```bash
uv sync --extra cpu --group dev        # or --extra cu128 on the GPU machine
uv run pytest                          # synthetic data only, no downloads needed

# data pipeline (RecoFit .mat files under data/external/recofit/)
uv run scripts/prepare_data.py recofit --steps recordings windows splits
uv run scripts/prepare_data.py recofit --steps labels     # per-label minutes and subjects

# one training run
uv run scripts/train.py --config configs/p1/edl.yaml --fold 0 --class-split rand0 --seed 0

# wearable
firmware/tools/attach-xiao.sh                                   # WSL: hand the USB port to Linux
uv tool run platformio run -d firmware -t upload --upload-port /dev/ttyACM0
uv run scripts/record_wearable.py --port /dev/ttyACM0 --subject W01
uv run scripts/serve_web.py          # browser page on http://localhost:8000 (Chrome/Edge, Web Bluetooth)
uv run scripts/live_server.py        # live-demo backend on ws://localhost:8765/ws
```

Documentation: [docs/design/](docs/design/) (what and why), [docs/decisions.md](docs/decisions.md) (what changed and when), [docs/wearable-recording.md](docs/wearable-recording.md) (devices, protocol, session log), [firmware/README.md](firmware/README.md), [web/README.md](web/README.md), [docs/ai-use-log.md](docs/ai-use-log.md).

## AI use

AI assistance (Claude Code) is used in this project with the instructor's permission and is
recorded in `docs/ai-use-log.md`.
