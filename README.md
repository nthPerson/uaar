# UAAR: uncertainty-aware activity recognition

CompE 510 (Machine Learning for Engineers), SDSU, Fall 2026. Team: Robert Ashe, Sirena Rocca.

A wrist-worn exercise recognizer is trained on a fixed list of exercises, but in use it sees
movements that were never in that list. This project compares four ways of letting the
recognizer say "I don't know" (logistic regression, a softmax network, evidential deep
learning and Monte Carlo dropout) on public IMU datasets, then runs the result live on our own
wearable (XIAO ESP32-S3 + BMI160).

Status (2026-11-03): repository bootstrapped; firmware and recorder under construction.

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
uv sync --extra cpu          # or --extra cu128 on the GPU machine
uv run pytest
```

More commands are added here as they exist.

## AI use

AI assistance (Claude Code) is used in this project with the instructor's permission and is
recorded in `docs/ai-use-log.md`.
