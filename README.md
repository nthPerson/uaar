# UAAR: uncertainty-aware activity recognition

CompE 510 (Machine Learning for Engineers), SDSU, Fall 2026. Team: Robert Ashe, Sirena Rocca.

A wrist-worn exercise recognizer is trained on a fixed list of exercises, but in use it sees
movements that were never in that list. This project compares four ways of letting the
recognizer say "I don't know" (logistic regression, a softmax network, evidential deep
learning and Monte Carlo dropout) on a public IMU dataset (RecoFit), then runs the result live
on our own wearable (XIAO ESP32-S3 + BMI160).

Status (2026-11-25):

- **Benchmark.** Four known exercises (curl, shoulder press, lateral raise, jumping jacks)
  against 16 held-out exercises and non-exercise movement, on RecoFit, in three tiers: plain
  models, models taught to be uncertain on background windows, and an "Other"-class reference.
  5 subject folds × 3 background divisions, one seed: 120 runs, 15 per result.
- **Headline** (share of windows accepted as a known exercise at the threshold τ): plain
  models accept 60 to 77 % of unknown-exercise windows, taught models 5 to 6 %, the Other
  class 5 %; known accuracy 0.96 for every model. Details: [docs/methods.md](docs/methods.md),
  section "Background windows in training".
- **Hard case.** The earlier 28-class benchmark (protocol `p1`) was run on fold 0 only and is
  kept as a reference.
- **Course app.** The live web page (`scripts/demo.py`): live Bluetooth stream or recorded
  `raw.csv` in, one card per model out.
- **Still to do.** The recording session with both team members (kit K2 is not flashed yet),
  fine-tuning on those recordings (tested on synthetic sessions only), the video and the
  report (due Dec 8; the 7-slide presentation is on Dec 3).
- **Not done.** Several seeds, significance tests and bootstrap confidence intervals,
  inference-cost measurements and figures. MM-Fit and RecGym were dropped
  ([docs/decisions.md](docs/decisions.md), 2026-11-23).

## Layout

| Path | What |
|---|---|
| `src/uaar/` | the Python package: data loaders, windows, splits, features, models, evaluation, live server (`uaar.live`) |
| `scripts/` | command-line entry points (prepare data, train, evaluate, sweep, tables, record from the wearable, live demo, fine-tune, export models) |
| `firmware/` | PlatformIO project for the wearable: a 50 Hz IMU streamer over USB serial and BLE |
| `web/` | browser pages for the live demo and for recording (Web Bluetooth) |
| `configs/` | experiment configuration files; `demo.yaml` (the cards of the live page), `wearable.yaml` (recording labels and checklist) |
| `models/` | final models, tracked in git ([models/README.md](models/README.md)) |
| `docs/` | design documents and engineering docs |
| `report/` | outline of the final report (`outline.docx`), the slide deck (`slides.pptx`), and the scripts that build them (`build/`) |
| `tests/` | pytest suite (synthetic data only; no downloads needed); CI runs ruff, pytest and the firmware build on every push |
| `reports/` | generated tables (`scripts/make_tables.py`); `reports/runs/` is not in git |
| `data/` | `external/` downloads and `processed/` derived files (gitignored); `wearable/` our recordings (tracked) |

## Quickstart

```bash
uv sync --extra cpu --group dev        # or --extra cu128 on the GPU machine
uv run pytest                          # synthetic data only, no downloads needed

# data pipeline (RecoFit .mat files under data/external/recofit/)
uv run scripts/prepare_data.py recofit --steps recordings windows splits
uv run scripts/prepare_data.py recofit --steps labels     # per-label minutes and subjects

# the four-exercise benchmark: splits, all three tiers on every fold (120 runs), tables
uv run scripts/prepare_data.py recofit --steps background_splits
uv run scripts/sweep.py configs/sweeps/k4.yaml
uv run scripts/make_tables.py          # reports/tables/scores_by_protocol.md

# one training run, its evaluation, MC Dropout on top of a softmax run
uv run scripts/train.py --config configs/k4taught/edl.yaml --fold 0 --class-split bg0 --seed 0
uv run scripts/evaluate.py reports/runs/k4taught_recofit_edl_fold0_bg0_seed0
uv run scripts/evaluate.py --method mcdropout --from reports/runs/k4plain_recofit_softmax_fold0_bg0_seed0

# the 28-class hard case (protocol p1, fold 0 only)
uv run scripts/sweep.py configs/sweeps/p1_fold0.yaml

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

Documentation: [docs/design/](docs/design/) (what and why), [docs/decisions.md](docs/decisions.md) (what changed and when), [docs/architecture.md](docs/architecture.md) (pipeline and module map), [docs/data-pipeline.md](docs/data-pipeline.md) (data, eligibility, splits), [docs/methods.md](docs/methods.md) (methods and results), [docs/wearable-recording.md](docs/wearable-recording.md) (devices, protocol, session log), [firmware/README.md](firmware/README.md), [web/README.md](web/README.md), [models/README.md](models/README.md), [docs/contributions.md](docs/contributions.md), [docs/ai-use-log.md](docs/ai-use-log.md).

## AI use

AI assistance (Claude Code) is used in this project with the instructor's permission and is
recorded in `docs/ai-use-log.md`.
