# Architecture

What the pieces are and how data flows between them. The data contracts are in
[design/development-spec.md §3](design/development-spec.md#3-data-contracts); this page is the map.

## Pipeline

```mermaid
flowchart LR
    DL[RecoFit .mat files<br/>data/external/recofit/] -->|uaar.data.recofit| REC[recordings.parquet<br/>+ meta.json]
    REC -->|uaar.data.windows| WIN[windows/X.npy<br/>meta.parquet]
    WIN -->|uaar.data.splits<br/>prepare_data.py splits, background_splits| SPL[splits/fold{k}_{split}.json<br/>rand0-2, near, bg0-2]
    WIN & SPL -->|scripts/train.py, evaluate.py<br/>scripts/sweep.py| RUN[reports/runs/run_id/<br/>model, threshold, metrics.json]
    RUN -->|scripts/make_tables.py| TAB[reports/tables/]
    RUN -->|scripts/export_model.py| MOD[models/<br/>tracked in git]
    MOD -->|configs/demo.yaml| LIVE[scripts/demo.py<br/>uaar.live]
    KIT[wearable kit<br/>firmware/] -->|BLE| PAGE[web/ pages] -->|WebSocket| LIVE --> PAGE
    LIVE -->|/record.html<br/>uaar.live.recorder| WEAR[data/wearable/<br/>person/session/]
    WEAR -->|uaar.data.wearable_windows<br/>scripts/finetune.py| MOD
```

Logistic regression computes its handcrafted features from the windows itself
(`uaar.features.handcrafted`); the optional `features` step of `prepare_data.py` writes the same
features to `features/handcrafted.parquet`, which training does not read.

## Module map

EDL (M3) is Robert's new algorithm and MC Dropout (M4) is Sirena's; who did what else is
recorded in [contributions.md](contributions.md).

| Package / path | Role |
|---|---|
| `firmware/` | XIAO ESP32-S3 + BMI160 streamer (50 Hz, serial CSV + BLE, button sessions) |
| `uaar.data.recording`, `uaar.data.wearable`, `scripts/record_wearable.py` | recordings contract, wire protocol mirror, session parsing, serial recorder |
| `web/` | demo page (`/`: upload a stream, one card per model) and recording page (`/record.html`), Web Bluetooth |
| `uaar.live`, `scripts/demo.py` | one server for the pages and the predictions: sliding windows + per-model answers over WebSocket; the recording page's server side (`uaar.live.recorder`) |
| `uaar.data.wearable_windows`, `uaar.models.finetune`, `scripts/finetune.py` | our recordings as labelled windows; fine-tuning a RecoFit model on them |
| `models/`, `scripts/export_model.py` | final models, tracked in git |
| `uaar.data.recofit` | RecoFit loader to the recordings contract |
| `uaar.data.windows`, `splits`, `registry` | windowing, subject folds, class splits and background splits, data layout |
| `uaar.config`, `uaar.utils` | configuration loading with `--set` overrides; seeding, timers, git hash |
| `uaar.features.handcrafted` | 102 features per window for M1 |
| `uaar.models.backbone`, `softmax`, `edl`, `train_loop`, `normalize`, `base`, `factory` | shared CNN, M2, M3, training, method construction and primary scores |
| `uaar.models.logreg`, `mcdropout` | M1, M4 |
| `uaar.eval.*` | metrics, calibration, threshold, predictions, evaluation runner, plotting functions (not called by any script yet) |
| `scripts/` | prepare_data, train, evaluate, sweep, make_tables, record_wearable, demo, finetune, export_model, live_replay |

## Live demo path

1. The kit streams 5-sample BLE batches (99 B) ten times a second.
2. The page decodes them, charts the last 10 s, and forwards the raw rows to the backend.
3. The backend keeps a 2 s buffer, emits a window every second, runs every loaded method on it,
   and returns label, probability, uncertainty score and the unknown decision per method.
4. The page shows one card per model side by side: its answer, its uncertainty on a 0–100 % bar
   against its limit, and a strip of the last 30 windows.

The page can also upload a recorded `raw.csv` instead of the live stream; both go through the
same path.

The backend loads the model folders listed in `configs/demo.yaml` (one card each through
`uaar.live.predictors.RunPredictor`, with the score to show and the τ from the model's
`threshold.json`); with `--no-models` it serves a motion-energy placeholder.

Fallback on stage: `scripts/demo.py --replay <session>` plays a recording through the page.
`scripts/live_replay.py` streams a recorded session into the backend and prints the answers,
for testing from the terminal.
