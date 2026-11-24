# UAAR: uncertainty-aware activity recognition

CompE 510 (Machine Learning for Engineers), SDSU, Fall 2026. Team: Robert Ashe, Sirena Rocca.

A wrist-worn exercise recognizer is trained on a fixed list of exercises, but in use it sees
movements that were never in that list. This project compares four ways of letting the
recognizer say "I don't know" (logistic regression, a softmax network, evidential deep
learning and Monte Carlo dropout) on public IMU datasets, then runs the result live on our own
wearable (XIAO ESP32-S3 + BMI160).

Status (2026-11-23): the benchmark is four known exercises (curl, shoulder press, lateral raise, jumping jacks) against 16 held-out exercises and non-exercise movement, on RecoFit, in three tiers: plain models, models taught to be uncertain on background windows, and an "Other"-class reference. All tiers are run on five subject folds and three background divisions (120 runs; results in [docs/methods.md](docs/methods.md), "Background windows in training"). The earlier 28-class benchmark was run on fold 0 only and is kept as the hard case. Firmware v0.1.0 runs on kit K1; the second kit, the recording session with both team members, statistics and the report are still to do. Earlier status (2026-11-04): firmware v0.1.0 flashed and verified on kit K1; serial recorder, browser page and live backend working with a placeholder predictor; RecoFit pipeline run on the real data; all four methods and the evaluation scripts implemented and tested, with two-epoch smoke runs only; the class-eligibility decision and the full fold-0 runs are next.

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

# one training run, its evaluation, MC Dropout on top of a softmax run, the summary table
uv run scripts/train.py --config configs/p1/edl.yaml --fold 0 --class-split rand0 --seed 0
uv run scripts/evaluate.py reports/runs/p1_recofit_edl_fold0_rand0_seed0
uv run scripts/evaluate.py --method mcdropout --from reports/runs/p1_recofit_softmax_fold0_rand0_seed0
uv run scripts/make_tables.py

# the four-exercise benchmark: splits, all three tiers on every fold (120 runs), tables
uv run scripts/prepare_data.py recofit --steps background_splits
uv run scripts/sweep.py configs/sweeps/k4.yaml
uv run scripts/make_tables.py          # reports/tables/scores_by_protocol.md

# wearable
firmware/tools/attach-xiao.sh                                   # WSL: hand the USB port to Linux
uv tool run platformio run -d firmware -t upload --upload-port /dev/ttyACM0
uv run scripts/record_wearable.py --port /dev/ttyACM0 --subject W01

# live demo and recording pages (Edge or Chrome; the models are in models/, tracked in git)
uv run scripts/demo.py               # http://localhost:8000 : upload a stream, see each model's answer
uv run scripts/demo.py --replay data/wearable/W01/<session>      # play a recording (stage fallback)
uv run scripts/demo.py --no-models   # pages only; /record.html is the recording console
uv run scripts/finetune.py --run models/<run> --train data/wearable/W01 --test data/wearable/W02
uv run scripts/export_model.py reports/runs/<run_id>             # final copy of a model into models/
```

Documentation: [docs/design/](docs/design/) (what and why), [docs/decisions.md](docs/decisions.md) (what changed and when), [docs/wearable-recording.md](docs/wearable-recording.md) (devices, protocol, session log), [firmware/README.md](firmware/README.md), [web/README.md](web/README.md), [docs/ai-use-log.md](docs/ai-use-log.md).

## AI use

AI assistance (Claude Code) is used in this project with the instructor's permission and is
recorded in `docs/ai-use-log.md`.
