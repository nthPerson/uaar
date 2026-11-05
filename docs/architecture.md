# Architecture

What the pieces are, how data flows between them, and who owns what. The data contracts are in
[design/development-spec.md §3](design/development-spec.md#3-data-contracts); this page is the map.

## Pipeline

```mermaid
flowchart LR
    DL[downloads<br/>data/external/] -->|uaar.data.recofit / mmfit / recgym| REC[recordings.parquet<br/>+ meta.json]
    W[wearable sessions<br/>data/wearable/] -->|uaar.data.wearable| REC
    REC -->|uaar.data.windows| WIN[windows/X.npy<br/>meta.parquet]
    WIN -->|uaar.data.splits| SPL[splits/fold{k}_{split}.json]
    WIN -->|uaar.features.handcrafted| FEA[features/handcrafted.parquet]
    WIN & SPL & FEA -->|scripts/train.py| RUN[reports/runs/run_id/<br/>model, preds_val, threshold]
    RUN -->|scripts/evaluate.py| MET[metrics.json, preds_*.parquet]
    MET -->|scripts/make_tables.py| TAB[reports/tables, figures]
    RUN -->|scripts/export_app_assets.py| APP[Streamlit app]
    RUN --> LIVE[uaar.live backend]
    KIT[wearable kit<br/>firmware/] -->|BLE| PAGE[web/ page] -->|WebSocket| LIVE --> PAGE
```

## Module map

| Package / path | Role | Owner |
|---|---|---|
| `firmware/` | XIAO ESP32-S3 + BMI160 streamer (50 Hz, serial CSV + BLE, button sessions) | Robert |
| `uaar.data.wearable`, `scripts/record_wearable.py` | wire protocol mirror, session parsing, serial recorder | Robert |
| `web/` | Web Bluetooth page: live charts, recording, prediction cards | Robert |
| `uaar.live` , `scripts/live_server.py` | sliding windows + per-method predictions over WebSocket | Robert |
| `uaar.data.recofit` / `mmfit` / `recgym` | dataset loaders to the recordings contract | Robert (RecGym: Sirena) |
| `uaar.data.windows`, `splits`, `registry` | windowing, subject folds and class splits, data layout | Robert |
| `uaar.features.handcrafted` | ~90 features per window for M1 | Sirena |
| `uaar.models.backbone`, `softmax`, `edl`, `train_loop`, `normalize` | shared CNN, M2, M3, training | Robert (M2 reviewed by Sirena) |
| `uaar.models.logreg`, `mcdropout` | M1, M4 | Sirena (M1 reviewed by Robert) |
| `uaar.eval.*` | metrics, calibration, threshold, cost, stats, plots | Sirena (cost, stats: Robert) |
| `uaar.app` | Streamlit upload app | Sirena |
| `scripts/` | prepare_data, train, evaluate, sweep, make_tables, export_app_assets, record_wearable, serve_web, live_server | per module above |

## Live demo path

1. The kit streams 5-sample BLE batches (99 B) ten times a second.
2. The page decodes them, charts the last 10 s, and forwards the raw rows to the backend.
3. The backend keeps a 2 s buffer, emits a window every second, runs every loaded method on it,
   and returns label, probability, uncertainty score and the unknown decision per method.
4. The page shows one card per method side by side and a timeline of the last 30 windows.
5. The page can ask the kit to flash its LED for known/unknown (control characteristic).

The backend loads trained run directories (`scripts/live_server.py --runs <run_dir> ...`, one
card per run through `uaar.live.predictors.RunPredictor`, each with its primary score and the τ
from its `threshold.json`); without `--runs` it serves a motion-energy placeholder.

Fallback on stage: `scripts/live_replay.py` streams a recorded session into the same backend.
