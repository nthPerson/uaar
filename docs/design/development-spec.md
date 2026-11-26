# Development spec and plan

This document turns the design in [README.md](README.md), [datasets.md](datasets.md), [models.md](models.md), [evaluation.md](evaluation.md) and [software-and-firmware.md](software-and-firmware.md) into a build plan: what the code looks like, what each piece must do, how pieces hand data to each other, how it is tested and documented, and what is done and still to do. Design facts are not repeated here; they are linked. Where this document decides something the design docs left open, the decision is marked **(decision)**.

Drafted 2026-10-08; schedule and scope revised 2026-11-03 and 2026-11-23 (`../decisions.md`). **Status 2026-11-25:** the four-exercise benchmark in three tiers (plain, taught, Other class) is done on all five folds (120 runs); the live page and the recording page are done; the final models are in `models/`. Still to do: the second kit, the recording session, fine-tuning on our own recordings, the rehearsal, the video and the report (§10).

---

## 0. Ground rules

1. **AI-written code is allowed** (instructor guidance, see [ai-use-log.md](../ai-use-log.md)) but every submitted artifact must "reflect the team's collective understanding," and AI use "must be clearly documented" in the final report (Project_instruction.pdf, p. 1). Two consequences:
   - **Each member must be able to explain the code they submit**, line by line if asked (§9.2).
   - **Documentation is a first-class deliverable**, written alongside the code, not after it (§8). The repository must describe the project well enough that a grader can understand what was built without reading the code.
2. **The design docs are the source of truth for *what*; this document for *how*.** When implementation forces a design change, update the design doc in the same commit and log the change in `../decisions.md`.
3. **Reproducibility over speed.** Every number in the report comes from saved runs that one command can regenerate ([evaluation.md §8](evaluation.md#8-reproducibility)).

## 1. Deliverables

| # | Deliverable | Source | Due | Status 2026-11-25 |
|---|---|---|---|---|
| D1 | `uaar` repository: installable package, scripts, tests, docs | this spec | continuous | in place; ongoing |
| D2 | RecoFit open-set benchmark: four known exercises, four methods, three tiers (`k4plain`, `k4taught`, `k4other`) on 5 folds × 3 background divisions, one seed | minimum goals 1–2 | — | **done 2026-11-23**; table `reports/tables/scores_by_protocol.md`. The 28-class benchmark (P1) was run on fold 0 only and is kept as the hard case |
| D3 | Course app: upload data, show prediction, explain output | minimum goal 3 | — | **done 2026-11-23** as the live page (D8); the Streamlit app was dropped (`../decisions.md`, 2026-11-23) |
| D4 | Wearable zero-shot (P4) and fine-tuned (P4-ft) results | stretch 3 | — | code done (`scripts/finetune.py`), tested on synthetic sessions only; waits for the recording session. P2 (MM-Fit) and P3 (RecGym) dropped 2026-11-23 |
| D5 | Final report, 5–7 pages, with individual contributions and AI-use disclosure | course | Dec 8 | outline `report/outline.docx`; report to do |
| D6 | Final slides and 1–3 min video | course | Dec 3 (slides) | slides `report/slides.pptx` (7 slides for 6–7 minutes; the handout's 10–15 was shortened by the instructor); video to do |
| D7 | Submission zip: all code, report, proposal slides, final slides | course | Dec 8 | to do |
| D8 | **Live demo**: the wearable streams over Bluetooth (or a recorded `raw.csv` is uploaded) to the browser page, which shows one card per model: plain softmax, taught MC Dropout, taught EDL | 2026-11-03 decision | Dec 3 | **done 2026-11-23** (`scripts/demo.py`, `web/`); rehearsal with both kits to do |

Minimum goals must not depend on stretch goals. The live page runs from a fresh clone without a kit: the models are in `models/`, and a recorded `raw.csv` (for example `data/wearable/W01/test-60s`) can be uploaded or replayed (`scripts/demo.py --replay`).

## 2. Repository and environment

### 2.1 Location and layout

- **GitHub repository `uaar`**, public. The development clone lives on the **WSL2 ext4 filesystem** at `~/academics/COMPE510_uaar`, **not** under `/mnt/c` or OneDrive **(decision)**. Reasons: `loadmat` on a 1.5 GB file is several times slower through the Windows mount, and OneDrive must never try to sync `data/`.
- The design folder of the proposal repository was copied here as `docs/design/` on 2026-11-03 and is maintained here; the proposal repository keeps the proposal deck and course handouts.
- Layout (the one tree of the project; [software-and-firmware.md §1](software-and-firmware.md#1-repository) points here):

```text
COMPE510_uaar/                # GitHub `uaar`
├── README.md                 # what, why, quickstart (fresh clone → live page)
├── pyproject.toml            # uv-managed; Python 3.12; ruff + pytest config
├── uv.lock
├── .github/workflows/ci.yml  # ruff, pytest -m "not slow", firmware build, on every push and PR
├── configs/
│   ├── base.yaml             # defaults shared by every experiment (§5)
│   ├── p1/                   # 28-class benchmark (hard case), one file per method
│   ├── k4plain/ k4taught/ k4other/   # the three tiers of the four-exercise benchmark
│   ├── sweeps/               # k4.yaml (120 runs), p1_fold0.yaml
│   ├── known4_recofit.yaml   # known exercises, look-alikes, background divisions bg0–bg2
│   ├── near_unknown_recofit.yaml  # near split pairs and twin labels
│   ├── demo.yaml             # the cards of the live page
│   └── wearable.yaml         # label roles and the recording checklist
├── docs/
│   ├── design/               # design docs, kept current
│   ├── architecture.md       # pipeline diagram, module map, live demo path
│   ├── data-pipeline.md      # from downloads to windows and splits; label and split tables
│   ├── methods.md            # equation-to-code map for M1–M4; background windows and results
│   ├── wearable-recording.md # devices, recording procedure, protocol, session log
│   ├── decisions.md          # dated log of design changes
│   ├── contributions.md      # individual contributions
│   └── ai-use-log.md         # one row per session that used an AI tool
├── src/uaar/                 # see §4
├── scripts/                  # see §5
├── tests/                    # see §7
├── web/                      # demo page and recording page, plain JavaScript (web/README.md)
├── firmware/                 # PlatformIO project for the wearable (firmware/README.md)
├── models/                   # final models, tracked in git (models/README.md)
├── report/                   # outline.docx, slides.pptx, their build scripts and assets
├── reports/
│   ├── tables/               # generated by make_tables.py (tracked)
│   └── runs/                 # per-run directories (not in git)
└── data/                     # gitignored, except wearable/
    ├── external/             # downloads, never modified
    ├── processed/            # recordings, windows, splits
    └── wearable/             # our recordings (tracked)
```

### 2.2 Tooling (decision)

| Concern | Choice |
|---|---|
| Package manager | `uv`; `uv sync` creates the environment; `uv run` for every command |
| Python | 3.12, pinned in `pyproject.toml` (`requires-python = ">=3.12,<3.13"`) |
| PyTorch | 2.x. Two extras: `uv sync --extra cu128` (RTX 3070 under WSL2) and `uv sync --extra cpu` (CPU machines, CI). Configured with `[tool.uv.sources]` index entries so both resolve from one lockfile |
| Lint and format | `ruff` (lint + format), line length 100 |
| Tests | `pytest`; `pytest -m "not slow"` in CI (no test is marked `slow` at present) |
| Type hints | Required on public functions; `mypy` is **not** run (time budget) |
| Pre-commit | Optional; CI is the gate |
| Random seeds | `uaar.utils.seed_everything(seed)` sets `random`, NumPy, PyTorch (CPU and CUDA) and `torch.backends.cudnn.deterministic` |

### 2.3 Bootstrap (done 2026-11-03)

The repository, the `uv` environment, CI and the first docs were set up on 2026-11-03 (§10, item 4.1). CI runs `uv sync --extra cpu --group dev`, `ruff check`, `ruff format --check` and `pytest -m "not slow"`, and builds the firmware.

## 3. Data contracts

Every stage reads and writes files with the schemas below, so stages can be developed and tested independently. Paths are relative to `data/processed/<dataset>/` (`recofit` or `wearable`).

### 3.1 Recordings (`recordings.parquet`, long format, one row per sample)

| Column | Type | Note |
|---|---|---|
| `subject` | string | Dataset's subject code (RecoFit numeric id as string; wearable `W01`, `W02`) |
| `recording_id` | string | Unique within dataset (RecoFit `single/<subject>/<activity-index>` or `multi/<visit>`; wearable `<subject>/<session>`) |
| `t` | float32 | Seconds from recording start, uniform 0.02 s |
| `ax, ay, az` | float32 | **g** |
| `gx, gy, gz` | float32 | **°/s** |
| `label` | string | Dataset's original label string; `""` for unlabelled samples; RecoFit junk labels already dropped |
| `source` | string | RecoFit: `single` or `multi`; wearable: session id |

Companion `recordings.meta.json`: `dataset`, `sample_rate_hz`, `units`, `labels` (per-label sample counts and subject counts), `source_files` (name, byte size, SHA-256), `created_by` (git commit), `created_at`.

### 3.2 Windows (`windows/X.npy` + `windows/meta.parquet`)

- `X.npy`: float32, shape `(N, 6, 100)`, channel order `ax ay az gx gy gz`, **unnormalized** physical units. Normalization is applied at training time from the fold's statistics (so the same windows serve every fold).
- `meta.parquet`, one row per window, index aligned with `X`: `window_id` (int), `subject`, `recording_id`, `source`, `label`, `start_s`, `label_purity` (fraction of samples with the majority label, ≥ 0.8 by construction), `n_gaps` (wearable only).
- Windowing rule from [datasets.md §5](datasets.md#5-preprocessing-pipeline): 2 s, 1 s hop, majority label ≥ 80 %, else dropped.

### 3.3 Splits (`splits/<split_id>.json`)

`split_id` = `fold{k}_{classsplit}` with `classsplit ∈ {rand0, rand1, rand2, near}` (28-class benchmark) or `{bg0, bg1, bg2}` (four-exercise benchmark, from `configs/known4_recofit.yaml`; [data-pipeline.md](../data-pipeline.md), section "Background splits").

```json
{
  "dataset": "recofit",
  "fold": 0, "n_folds": 5,
  "class_split": "bg0", "class_split_seed": 0,
  "train_subjects": ["..."], "val_subjects": ["..."], "test_subjects": ["..."],
  "known_classes": ["Jumping Jacks", "..."],
  "unknown_classes": ["..."],
  "eligibility": {"min_subjects": 25, "min_minutes": 10.0},
  "class_source": "single",
  "fold_strategy": "class_stratified",
  "background_classes": ["..."],
  "lookalike_classes": ["Bicep Curl", "Squat Rack Shoulder Press"],
  "created_by": "<git sha>"
}
```

`background_classes` and `lookalike_classes` exist only in the `bg` files. Known classes are stored in sorted order; the integer label of a known class is its index in this list. Unknown-class windows are labelled `-1` by `splits.encode_labels`. Non-exercise windows (P1-b) are selected by `source == "multi"` and a label in `split.nonexercise_labels` (`Non-Exercise`, `Rest`, `Walk`), never by class split. In the `bg` files, background windows (`select_background`) and look-alike windows are further sets.

### 3.4 Features (`features/handcrafted.parquet`)

102 float32 columns per window named `{signal}_{stat}` (for example `ax_mean`, `amag_specent`) plus `corr_{a}_{b}`. `uaar.models.logreg.LogRegMethod` computes them from the windows itself and standardizes them at fit time; `prepare_data.py --steps features` can write the parquet, but training does not read it.

### 3.5 Model artifacts (`reports/runs/<run_id>/`)

`run_id` = `{protocol}_{dataset}_{method}_{split_id}_seed{seed}` (example `k4taught_recofit_edl_fold0_bg0_seed0`). The final models are copied to `models/` with the same layout (`scripts/export_model.py`).

| File | Content |
|---|---|
| `config.yaml` | Fully resolved config (base + method file + CLI overrides) |
| `split.json` | Copy of the split used |
| `model.pt` / `model.joblib` | `state_dict` (M2–M4) or fitted sklearn pipeline (M1) |
| `config.json`, `preprocess.json`, `threshold.json` | As in [models.md §8](models.md#8-model-artifacts); `threshold.json` maps score name → τ |
| `history.csv` | Per-epoch train/val loss, val metric, learning rate, λ_t (EDL); one row per C for M1 |
| `train_summary.json` | Summary of the training run |
| `preds_{set}.parquet` | `set ∈ {val, test_known, test_unknown, test_nonex, test_background, test_lookalike}` (the last two for `bg` splits): `window_id`, `y_true` (−1 for unknown), `y_pred`, `p_max`, one column per score (`msp`, `entropy`, `vacuity`, `one_minus_max`, `mi`, `varratio`, `p_other`) and `probs` as a fixed-length list column |
| `metrics.json` | Metrics of [evaluation.md §4](evaluation.md#4-metrics): closed-set and calibration metrics once under `test_known`, per-score metrics under `scores` (layout documented in `uaar/eval/runner.py`) |

A fine-tuned model (`scripts/finetune.py`) also has `finetune.json`, `report.json` and `report.md`. Because `preds_*.parquet` are saved, `make_tables.py` runs without models or GPUs.

## 4. Module specifications

Signatures are the public interface; private helpers are free. Owners are not recorded here: EDL is Robert's new algorithm and MC Dropout Sirena's (the course rule of one new algorithm per member); actual contributions are in [`../contributions.md`](../contributions.md).

### 4.1 `uaar.data`

| Module | Public interface | Notes |
|---|---|---|
| `recofit.py` | `download(dest, force=False)`; `load_single(path, subjects=None) -> Iterator[Recording]`; `load_multi(path, subjects=None) -> Iterator[Recording]`; `to_recordings(out_dir, single_path, multi_path, drop_junk=True, hash_sources=True) -> Path` | LFS media URL from [datasets.md §1](datasets.md#1-recofit-primary). `loadmat` with `struct_as_record=False, squeeze_me=True`; one subject at a time, appended to Parquet. Junk labels in a module constant. Only the master `rightArm` device. |
| `recording.py` | `Recording` dataclass (`subject`, `recording_id`, `source`, `t`, `acc`, `gyro`, `labels`) | Used inside the loaders |
| `wearable.py` | `parse_csv_line`; `decode_batch`, `encode_batch`, `decode_status`; `seq_gaps`; `raw_to_physical`; `read_raw_csv`; `labels_from_events`; `load_session(session_dir) -> Recording`; `to_app_csv`; `to_recordings(sessions_root, out_dir)` | Python mirror of `firmware/include/protocol.h`. Raw counts → g and °/s; `seq` gap detection; labels from `events.csv`, written by `scripts/record_wearable.py` or by the recording page's server (`uaar.live.recorder`) |
| `wearable_windows.py` | `Roles` (label roles of `configs/wearable.yaml`); `session_windows`; `subject_windows`; `validation_split` | Our sessions as labelled windows with roles (known, background, unknown, shifted), for fine-tuning |
| `windows.py` | `make_windows(recordings, out_dir, window_s=2.0, hop_s=1.0, purity=0.8)`; `window_recording(df, ...) -> tuple[np.ndarray, pd.DataFrame]` | Pure NumPy; the per-recording function is what the tests exercise |
| `splits.py` | `eligible_classes`; `label_table`; `make_class_split`; `near_unknown_split`; `stratified_assign`; `make_subject_folds`; `write_splits`; `make_background_split`; `write_background_splits`; `load_split(path) -> Split`; `select(meta, split, part)` (parts `train`, `val`, `test_known`, `test_unknown`, `train_background`, `val_background`, `test_background`, `test_lookalike`); `select_nonexercise`; `select_background`; `encode_labels(labels, split)` | Subject folds by iterative stratification on the subject × exercise table (`stratified_assign`, decided 2026-11-05). `near` pairs and twin labels in `configs/near_unknown_recofit.yaml`; known and look-alike classes of the `bg` splits in `configs/known4_recofit.yaml` |
| `registry.py` | `load_windows(dataset)`; `load_features(dataset)`; `paths(dataset)`; `split_path(dataset, split_id)` | Single place that knows the directory layout |

MM-Fit and RecGym loaders were not written (both datasets dropped 2026-11-23, `../decisions.md`).

### 4.2 `uaar.features.handcrafted`

`compute(X: np.ndarray, fs: float = 50.0, chunk: int = 8192) -> pd.DataFrame` implementing [models.md §3](models.md#3-m1--logistic-regression-in-class-baseline): 102 features (12 statistics for each of the six channels and the two magnitudes, plus 6 correlations). Vectorized over windows; FFT with `np.fft.rfft`. `feature_names()` gives the column names, so the test can check the count and the absence of NaN.

### 4.3 `uaar.models`

Common interface in `uaar.models.base` **(decision)**:

```python
@dataclass
class Prediction:
    probs: np.ndarray  # (N, K) predictive probabilities (p̂ for EDL, p̄ for MC Dropout)
    label: np.ndarray  # (N,) argmax
    scores: dict[str, np.ndarray]  # higher = more uncertain; keys listed per method below


class Method(Protocol):
    name: str
    score_names: list[str]

    def fit(self, Xtr, ytr, Xval, yval, cfg: Config, run_dir: Path) -> None: ...
    def predict(self, X, cfg: Config) -> Prediction: ...
    def save(self, run_dir: Path) -> None: ...
    @classmethod
    def load(cls, run_dir: Path) -> "Method": ...
```

| Module | Contents | Scores |
|---|---|---|
| `base.py` | `Prediction`, `Method`; `with_background` (adds background windows to a training set); `open_macro_f1` (checkpoint criterion of taught and Other-class models); `class_weights` (`none`, `inv`, `inv_sqrt`); `entropy` | — |
| `factory.py` | `build(cfg, classes, seed)`; `load(run_dir)`; `primary_score(method, background)` | — |
| `backbone.py` | `ConvBackbone(nn.Module)` as [models.md §1](models.md#1-shared-backbone-m2m4), `head` left to the subclass; `count_params()`, `count_macs()` (analytic, by layer) | — |
| `logreg.py` | `LogRegMethod`: `StandardScaler` + multinomial L2 `LogisticRegression`; C chosen by an explicit `GroupKFold(3)` loop over the training subjects on a warm-started C path {0.01, 0.1, 1, 10} (decided 2026-11-04) | `msp`, `entropy` |
| `softmax.py` | `SoftmaxNet` = backbone + linear; class-weighted cross-entropy; taught mode `taught_cross_entropy` (uniform target on background windows, weight 0.5); Other-class mode (background windows as an extra class) | `msp`, `entropy`; Other class: `msp`, `p_other` |
| `edl.py` | `EDLNet` = backbone + evidence head (`softplus` default); `edl_loss_digamma` (Eq. 4, the default since 2026-11-05), `edl_loss` (Bayes-risk squared error, Eq. 5), `edl_loss_log` (Eq. 3), KL term `kl_dirichlet_uniform` with `kl_anneal` over 10 epochs and `kl_weight` 1; taught mode `taught_edl_loss` (KL(Dir(α) ‖ Dir(1)) on background windows, weight 1) | `vacuity`, `one_minus_max` |
| `mcdropout.py` | `MCDropoutMethod.from_softmax_run(run_dir, T)`: loads a `SoftmaxNet`, `enable_dropout_only()` (Dropout in train mode, BatchNorm in eval mode, asserted in a test), T = 30 passes, `share_trunk=True` by default (decided 2026-11-04); `mc_scores` | `entropy`, `mi`, `varratio` |
| `train_loop.py` | `fit(model, loss_fn, probs_fn, Xtr, ytr, Xval, yval, *, epochs, batch_size, lr, weight_decay, patience, ..., val_metric) -> History`; AdamW, cosine schedule, early stopping on validation macro-F1 or on the method's `val_metric` | — |
| `normalize.py` | `fit_stats(X_train) -> dict`; `apply(X, stats)`; `save`, `load` (`preprocess.json`) | — |
| `finetune.py` | `frozen_trunk(net)`; `finetune(method, Xtr, ytr, Xval, yval, Xbg=None, ...)`: recipe of [models.md §10](models.md#10-fine-tuning-on-our-own-recordings) and `../decisions.md` (2026-11-23) | — |

Primary scores (`factory.primary_score`): `msp` for softmax and logistic regression, `vacuity` for EDL, `mi` for MC Dropout on a plain network and `entropy` on a taught one. Taught and Other-class models are selected (early stopping) on open-set validation macro-F1 (accepted 2026-11-24); the tiers are described in [models.md §9](models.md#9-background-windows-plain-taught-and-other).

MC Dropout **does not train**: `fit` raises; `evaluate.py --method mcdropout --from <softmax run>` is the entry point.

### 4.4 `uaar.eval`

| Module | Public interface |
|---|---|
| `metrics.py` | `closed_set(y_true, y_pred) -> dict` (accuracy, macro-F1); `unknown_detection(score_known, score_unknown) -> dict` (AUROC, AUPR, FPR@95); `fpr_at_accept`; `roc_points`; `open_set(y_true, y_pred, score, tau) -> dict` (open macro-F1 over K+1, unknown recall); `misclassification_auroc(score, correct)`; `risk_coverage(score, correct) -> (coverage, risk, aurc)`; `excess_aurc` |
| `calibration.py` | `ece(probs, y, n_bins=15)`, `nll`, `brier`, `reliability_bins(...)` |
| `threshold.py` | `select_tau(score_val_known, accept=0.95) -> float` (the 95th percentile of validation known-class scores); `apply(score, tau) -> bool mask` |
| `preds.py` | `preds_frame`, `write_preds`, `read_preds`, `available_sets`, `to_prediction` (the `preds_*.parquet` files of §3.5) |
| `runner.py` | `default_sets(split)`; `validation_thresholds(...)`; `evaluate(method, taus, X, meta, split, sets, cfg)` (predictions, `metrics.json`) |
| `plots.py` | `score_histograms`, `roc_curves`, `reliability_diagrams`, `risk_coverage_curves`, each writing one figure from saved predictions; the style is a dict in the module. No script calls them yet and no figure has been generated |

Not written: `cost.py` (inference-cost measurements) and `stats.py` (significance tests, bootstrap confidence intervals). Results are mean ± sd over runs, one seed.

### 4.5 `uaar.live`

The course app is the live page (`../decisions.md`, 2026-11-23; [software-and-firmware.md §3](software-and-firmware.md#3-demo-page-the-course-app)). `scripts/demo.py` serves the pages in `web/` and the predictions from one server.

| Module | Contents |
|---|---|
| `server.py` | `make_app(...)`: FastAPI app with the WebSocket (`/ws`) for samples, predictions and recording messages, and the static pages of `web/`; `--replay` serves a recording at `/replay.csv` |
| `windower.py` | `LiveWindower`: 2 s windows at a 1 s hop from the raw sample stream |
| `predictors.py` | `RunPredictor` (one per card in `configs/demo.yaml`: loads a run directory, applies its τ); `MotionEnergyPredictor` (placeholder for `--no-models`); `score_scale` (score divided by its largest possible value, for the 0–100 % bar); `score_info` |
| `recorder.py` | `SessionRecorder`: the recording page's sessions, written to `data/wearable/<person>/<session>/` (`raw.csv` as it arrives, `events.csv` and `meta.json` at the end) |

### 4.6 `uaar.config` and `uaar.utils`

`uaar.config.Config`: loaded from YAML with dotted-key CLI overrides, `to_yaml()`. `uaar.utils`: `repo_root`, `git_sha()`, `seed_everything`, `timer`.

## 5. Scripts and configs

| Script | Does | Example |
|---|---|---|
| `prepare_data.py` | Steps `download`, `recordings`, `windows`, `splits`, `background_splits`, `features` for `recofit` or `wearable`; idempotent (skips existing outputs unless `--force`); `--steps labels` prints the per-label minutes and subject counts | `uv run scripts/prepare_data.py recofit --steps all` |
| `train.py` | One run: resolves config, loads split, normalizes, fits, writes the run directory including `preds_val.parquet` and `threshold.json` | `uv run scripts/train.py --config configs/k4taught/edl.yaml --fold 0 --class-split bg0 --seed 0` |
| `evaluate.py` | Loads a run, predicts on the test sets of its split (§3.5), writes `preds_*.parquet` and `metrics.json`; `--method mcdropout --from <softmax run>` builds the MC Dropout run first | `uv run scripts/evaluate.py reports/runs/k4taught_recofit_edl_fold0_bg0_seed0` |
| `sweep.py` | Runs `train` + `evaluate` over a sweep file (folds × class splits × seeds × methods, or a `protocols:` map for the tiers), one run after the other; skips completed runs; `--dry-run` | `uv run scripts/sweep.py configs/sweeps/k4.yaml` |
| `make_tables.py` | Aggregates every `metrics.json` into `reports/tables/`: `metrics_long.csv`, `t2_headline.md/.csv`, `scores_by_protocol.md/.csv` (mean ± sd; no significance tests) | `uv run scripts/make_tables.py` |
| `export_model.py` | Copies a run's model and its small records into `models/` (tracked in git) | `uv run scripts/export_model.py reports/runs/k4taught_recofit_edl_fold0_bg0_seed0` |
| `demo.py` | Serves the demo page (`/`), the recording page (`/record.html`) and the predictions on one port; `--replay` plays a recording on load; `--no-models` serves the pages only | `uv run scripts/demo.py` |
| `finetune.py` | Fine-tunes a RecoFit model on one person's recordings and reports before/after on the other person; writes to `models/` | `uv run scripts/finetune.py --run models/k4taught_recofit_edl_fold0_bg0_seed0 --train data/wearable/W01 --test data/wearable/W02` |
| `record_wearable.py` | USB serial recorder: a label typed per set, the kit's button starts and stops it; writes `raw.csv`, `events.csv`, `meta.json`, `app.csv` and `quicklook.png`, and reports dropped samples per set | `uv run scripts/record_wearable.py --port /dev/ttyACM0 --subject W01` |
| `live_replay.py` | Streams a recorded session into the backend and prints the predictions (terminal test of the backend) | `uv run scripts/live_replay.py data/wearable/W01/test-60s` |

Config resolution: `configs/base.yaml` ← `configs/<protocol>/<method>.yaml` (protocols `p1`, `k4plain`, `k4taught`, `k4other`) ← CLI `--set key.path=value` overrides; the resolved result is what gets saved. `base.yaml`:

```yaml
dataset: recofit
protocol: p1
method: softmax
data: {window_s: 2.0, hop_s: 1.0, purity: 0.8, channels: [ax, ay, az, gx, gy, gz], augment: false}
split: {n_folds: 5, val_frac: 0.15, frac_unknown: 0.25, min_subjects: 25, min_minutes: 10.0, nonexercise_labels: [Non-Exercise, Rest, Walk], class_source: single, stratify_folds: true}
train: {epochs: 60, batch_size: 256, lr: 1.0e-3, weight_decay: 1.0e-4, patience: 8, class_weighting: inv_sqrt, num_workers: 0}
model: {dropout: 0.3}
eval: {accept_rate: 0.95}
background: {mode: none, weight: 0.5, n_exercise: 8000, n_nonex: 4000, n_val_exercise: 1200, n_val_nonex: 600, seed: 0}
```

Method files add only what differs, for example `configs/k4taught/edl.yaml`: `method: edl`, `model.edl: {evidence: softplus, loss: digamma, kl_anneal_epochs: 10}`, `background: {mode: uncertain, weight: 1.0}` (`loss: mse` until 2026-11-05; see `../decisions.md`).

## 6. Experiment plan and compute budget

| Sweep | Runs | Notes |
|---|---|---|
| `configs/sweeps/k4.yaml` (main) | 5 folds × 3 background divisions (`bg0`–`bg2`) × 1 seed × 8 = **120** | `k4plain`: softmax, MC Dropout, EDL, logistic regression; `k4taught`: softmax, MC Dropout, EDL; `k4other`: softmax. MC Dropout is evaluation only, on the softmax run of the same cell and tier. Done 2026-11-23; the taught tier was re-run after the checkpoint criterion was changed (`../decisions.md`, 2026-11-23) |
| `configs/sweeps/p1_fold0.yaml` (hard case) | fold 0 × 4 class splits (`near`, `rand0`–`rand2`) × 1 seed × 4 methods | 28 known classes; done 2026-11-05; not extended to the other folds |

`uv run scripts/sweep.py configs/sweeps/k4.yaml`, then `uv run scripts/make_tables.py`. Results are mean ± sd over the 15 runs of a row, one seed.

Not run: several seeds per cell, inference-cost measurements, P2 (MM-Fit) and P3 (RecGym), both dropped 2026-11-23.

## 7. Testing

Unit tests run on synthetic data, never on downloads, so CI needs no data. The `slow` marker (tests that need `data/processed`) is declared in `pyproject.toml`; no test uses it at present.

| Test file | Invariants checked |
|---|---|
| `test_windows.py` | Window count for a known-length signal; majority-label rule at exactly 80 %; dropped windows when purity < 80 %; last partial window dropped; channel order; `n_gaps` from `seq` |
| `test_splits.py` | **No subject in two of train/val/test**; **no unknown-class window in train or val**; test windows come only from test subjects; eligibility thresholds; known-class encoding is stable across folds; the `near` split uses only eligible classes; twin labels stay on one side; stratified folds spread every class; background splits are disjoint and keep tied groups together |
| `test_recofit.py` | The RecoFit loader on a tiny synthetic `.mat` (written with `scipy.io.savemat`): single and multi files, junk labels dropped, meta written |
| `test_wearable.py` | Constants match `protocol.h`; batch and status decoding; CSV parsing; `seq` gap count including wrap-around; unit conversion; session loading with labels and gaps; `app.csv`; `to_recordings` |
| `test_features.py` | Feature count and names, no NaN/inf, a sine window has the expected dominant frequency |
| `test_backbone.py` | Output shape `(N, K)`; parameter budget; MAC count positive and stable |
| `test_edl.py` | KL(Dir(1)‖Dir(1)) = 0; loss is finite for extreme evidence; `gradcheck` on a tiny batch in float64; the `mse` and `digamma` variants agree in ordering on a fixed batch; vacuity ∈ (0, 1] and decreases as total evidence increases |
| `test_mcdropout.py` | After `enable_dropout_only()`, every `Dropout` is in train mode and every `BatchNorm` in eval mode; with T = 1 and dropout p = 0 the output equals the softmax net's; shared trunk equals full passes; score ranges |
| `test_logreg.py` | Fit, predict, scores, save/load, both C-selection paths |
| `test_metrics.py` | AUROC/AUPR agree with scikit-learn on random data; FPR@95 on a hand-built case; excess AURC of a perfect ranking = 0; `select_tau` accepts 95 % of the known validation scores |
| `test_calibration.py` | ECE of a calibrated predictor ≈ 0; known values; degenerate inputs |
| `test_threshold.py`, `test_normalize.py`, `test_config.py` | τ accepts 95 %; normalization round trip; config merge, overrides, and every repo config resolves |
| `test_background.py` | Taught losses, the Other class, and the three tiers end to end on a tiny synthetic dataset |
| `test_train_smoke.py`, `test_train_script.py` | Softmax and EDL learn on synthetic windows and round-trip; `train.py` end to end writes the files of §3.5 |
| `test_evaluate_script.py` | `evaluate.py` and `make_tables.py` end to end on synthetic data |
| `test_plots.py` | Each figure function writes a non-empty file |
| `test_finetune.py` | Wearable windows with label roles, the frozen-trunk fine-tune, `scripts/finetune.py` |
| `test_live.py`, `test_recorder.py` | Live windowing and the WebSocket round trip; recording sessions written to disk |
| `test_web_protocol.py` | `web/protocol.js` mirrors `uaar.data.wearable`; runs `web/test/protocol.test.mjs` when `node` is available |

Definition of done for a module: public functions documented, tests above passing, review by the second team member (§9.2).

## 8. Documentation plan

The instructor's requirement is that the submission reflects the team's understanding and documents the project clearly. The repository is the primary evidence, and the final report is distilled from the repository docs.

### 8.1 Documentation set

| Document | Audience | Content | Status 2026-11-25 |
|---|---|---|---|
| `README.md` (repo root) | Grader | Problem statement; the four methods; quickstart (`uv sync`, `prepare_data`, `train`, the sweep, `scripts/demo.py`) that works from a fresh clone; link map; AI-use pointer | exists |
| `docs/design/*` | Both | The design docs, edited as decisions change, with the change logged in `decisions.md` | brought in line 2026-11-25 |
| `docs/architecture.md` | Grader | Pipeline diagram (downloads → recordings → windows → splits → train → evaluate → tables; models → live page); module map; live demo path | exists |
| `docs/data-pipeline.md` | Anyone reproducing | Commands; label tables; eligibility; splits; class-stratified folds; background splits | exists |
| `docs/methods.md` | Grader (report §4) | For each of M1–M4: the equations of [models.md](models.md) next to the code that implements them; background windows in training (plain, taught, Other) with the results | exists; the section text in the authors' own words (§8.2) is still to be written for M1, M3 and M4 |
| `web/README.md` | Grader (report §7), users | The demo page and the recording page: how to run them, what each part shows, how the limit τ is set | exists (no separate `docs/app.md`) |
| `docs/wearable-recording.md` | Anyone repeating the session | Devices, recording procedure (recording page and USB), protocol, fine-tuning commands, session log | exists; session log has the 60-s test only |
| `models/README.md`, `firmware/README.md` | Anyone running the page or flashing a kit | What each model folder is; how to build and flash the firmware | exist |
| `docs/decisions.md` | Both | Dated entries: what changed, why, which design doc was updated | continuous |
| `docs/contributions.md` | Report §11 | Per member: what was built, reviewed, written; each member writes their own entries | to complete before Dec 8 |
| `docs/ai-use-log.md` | Report disclosure | One row per session that used an AI tool (date, tool, what it produced, what the human did with it) | continuous |
| Docstrings | Readers of the code | Every public function and class: one-line summary, arguments with units and shapes, return value, reference to the paper equation where applicable | with the code |
| Module headers | Grader | Each file in `src/uaar/models/` names the paper or design section it implements and points to `docs/ai-use-log.md` for AI assistance | with the code |

No separate `docs/experiments.md` was written: the commands that regenerate the results are in the root README quickstart and §6, the tables in `reports/tables/`, and their interpretation in `docs/methods.md`.

### 8.2 Standards

- **Plain language first.** Each doc opens with what the thing is and why it exists before any command or equation.
- **Own words.** Each member writes the `docs/methods.md` subsection for their own new algorithm themselves, without AI, after the code works. This is the artifact that demonstrates understanding and it is the thing we would be asked about in the presentation.
- **Commands are copy-pasteable** and are re-run before each milestone. A doc that lists a command that does not run is a bug.
- **Generated, not edited:** `reports/runs/` is written by `train.py`, `evaluate.py` and `sweep.py`, `reports/tables/` by `make_tables.py`. Docs link to those files rather than pasting numbers that can go stale.
- **Verification marks:** the design docs' **(verify)** items are resolved in `decisions.md` with the measured value and the date, then the mark is removed from the design doc.

### 8.3 Traceability to the final report

| Report section (Project_instruction.pdf §7) | Points | Source documents |
|---|---|---|
| 1 Introduction | 5 | `README.md`, [README.md §1](README.md#1-problem) |
| 2 Related work | 7 | [references.md](references.md) |
| 3 Problem formulation | 10 | [evaluation.md §1](evaluation.md#1-formal-problem-definition) |
| 4 Methods | 10 | `docs/methods.md`, [models.md](models.md) |
| 5 Data description and experimental setup | 10 | `docs/data-pipeline.md`, §6 of this document, resolved `config.yaml` files |
| 6 Results and analyses | 20 | `reports/tables/scores_by_protocol.md`, `reports/tables/t2_headline.md`, the results in `docs/methods.md` |
| 7 Application | 10 | `web/README.md`, [software-and-firmware.md §3](software-and-firmware.md#3-demo-page-the-course-app), `docs/architecture.md` |
| 8 Conclusion | 3 | written last |
| 9 Video | 10 | link (video to do) |
| 10 References | — | [references.md](references.md) |
| 11 Individual contributions | required | `docs/contributions.md` |
| AI-use disclosure | required | `docs/ai-use-log.md`, summarized into a paragraph |

The report outline is `report/outline.docx`; the report is due Dec 8 2026.

### 8.4 AI-use disclosure process

1. Each working session that uses an AI tool adds a row to `docs/ai-use-log.md`: date, tool, what it produced, what the human did with it.
2. Before Dec 8 the log is summarized into the report's disclosure paragraph, in the style the instructor gave ("Used X to help Y").

## 9. Working agreements

### 9.1 Git

- Commits go directly on `main`. CI (`.github/workflows/ci.yml`) runs `ruff check`, `ruff format --check`, `pytest -m "not slow"` and a firmware build on every push.
- The aim is that `main` is green and runs from a fresh clone (the models are in `models/`).
- Commit messages: imperative summary line; body explains why when not obvious.
- No pull requests and no tags so far. A `final` tag marks the submitted commit (§10, item 8.4).

### 9.2 Review rule (the "explain it" rule)

Each member must be able to explain any code they submit (§0), in their own words. Review of the code by the second team member is not recorded in the repository; it is pending. AI assistance is recorded per session in `docs/ai-use-log.md`.

### 9.3 Communication

Contributions are recorded per member in `docs/contributions.md`; each member writes their own entries. Run results are shared as run directories or as files in `reports/tables/`, not as screenshots of terminals.

## 10. Schedule

**Status 2026-11-25.** No commits were made between Nov 5 and Nov 23, so the W5 and W6 rows did not run as dated. On Nov 23 (`../decisions.md`) the benchmark was cut to four known exercises in three tiers (120 runs, done Nov 23), MM-Fit and RecGym were dropped, and the course app became the live page. The presentation is on Dec 3 (6–7 minutes, 7 slides) and the report is due Dec 8.

Still to do before Dec 3: flash and calibrate the second kit (4.7); the recording session (5.5); fine-tuning on the recordings for the demo models (6.2); the rehearsal with both kits (6.5, 8.1); review of the deck and a demo script with the fallback (7.3).
Still to do before Dec 8: the P4 and P4-ft table (6.2); figures, if any are used (6.3); the methods text in own words (5.7); the video (7.2); the report and the contributions (7.1, 8.3); the submission (8.4). Pending throughout: review of the code by the second team member.
Not done and not planned unless time allows: several seeds, significance tests and bootstrap confidence intervals, inference-cost measurements, LED feedback from the page to the kit.

Revised 2026-11-03. The original W0–W3 plan (Oct 8 – Nov 1) did not run: the wearable hardware came first. Dates below run from Nov 3 to the Dec 8 submission. The original item numbers are kept.

**Critical path (remaining):** second kit → recording session with both people → fine-tuning and the P4/P4-ft table → rehearsal → presentation Dec 3 → report Dec 8.

### W4 · Nov 3–8 · Bootstrap, firmware, data pipeline (Exam 2 is Nov 5)

| # | Item | Status 2026-11-25 |
|---|---|---|
| 4.1 | Repository bootstrap (§2.3); firmware v0.1.0 flashed and verified; `wearable.py` + tests; `record_wearable.py`; 60-s test recording | **done Nov 3** |
| 4.1b | `windows.py`, `splits.py`, `registry.py`, `Config`, `train_loop.py`, `backbone.py`, `softmax.py`, `edl.py`, `normalize.py`, `threshold.py`, `prepare_data.py`, `train.py` on synthetic data; `uaar.live` backend, `live_replay.py` and the `web/` page with a placeholder predictor | **done Nov 3**; real-data runs Nov 5 (P1) and Nov 23 (k4); the placeholder is replaced by the models in `models/`; the page was replaced on Nov 23 by the demo page and the recording page |
| 4.2 | GitHub remote `uaar` (public), CI green, second clone passes `pytest` | remote and CI exist (ruff, pytest, firmware build on every push); a second clone is not recorded in the repo |
| 4.3 | `recofit.py` single- and multi-activity loaders → `recordings.parquet`; the 75 labels printed to `docs/data-pipeline.md` | **done Nov 3** |
| 4.4 | `windows.py`, `splits.py` + leakage tests; eligibility thresholds; `rand0–2` and `near` splits; T1 | **done Nov 5**: 25 subjects / 10 min, 38 classes (`../decisions.md`); background splits `bg0`–`bg2` added Nov 23; review of the `near` pairs pending |
| 4.5 | `metrics.py`, `calibration.py`, `threshold.py` + tests (§7); read Gal & Ghahramani 2016 | written Nov 4 with AI assistance, tests pass, used in every run; review pending |
| 4.6 | Browser page `web/`: Web Bluetooth connect, live charts, record mode | **done**: replaced Nov 23 by the demo page (`/`) and the recording page (`/record.html`); used with kit K1 over Bluetooth on Nov 23, with a sample loss not yet understood (`../decisions.md`, 2026-11-23) |
| 4.7 | Second kit assembled and flashed; both kits gyro-calibrated; exercise set confirmed against the label list | **to do before Dec 3**: kit K2 not flashed yet; exercise set confirmed Nov 5 and revised Nov 23 (`configs/wearable.yaml`) |

**Milestone M1 (Nov 8):** windows, splits and T1 done Nov 5; "both wearables streaming" not met (K2 not flashed). No tag.

### W5 · Nov 9–15 · All four methods on fold 0; recording session

| # | Item | Status 2026-11-25 |
|---|---|---|
| 5.1 | `handcrafted.py`, `logreg.py`, `evaluate.py` with P1 known/unknown sets, `metrics.json` | written Nov 4 with AI assistance (also `sweep.py`, `make_tables.py`), used in every run; review pending |
| 5.2 | `backbone.py`, `softmax.py`, `train_loop.py`, `normalize.py`, `Config`, `train.py`, run-directory writer | **done**: fold 0 Nov 5 (P1), all folds Nov 23 (k4); review pending |
| 5.3 | `edl.py` + `test_edl.py`; fold-0 run; vacuity histograms | **done Nov 5** with the digamma loss; the target "val macro-F1 within 2 points of M2" was not met on 28 classes (`../decisions.md`, 2026-11-05); on four exercises known accuracy is 0.96 for every model |
| 5.4 | `mcdropout.py` + `test_mcdropout.py`; fold-0 evaluation at T = 30 | written Nov 4 with AI assistance; evaluated at T = 30 in the P1 fold-0 runs and the k4 sweep; review pending |
| 5.5 | **Recording session**: both subjects, about 20 min each, protocol in [software-and-firmware.md §5](software-and-firmware.md#5-recording-protocol) | **to do before Dec 3**: not held yet; the recording page is ready; protocol in `docs/wearable-recording.md` |
| 5.6 | Live backend: WebSocket, 2-s windowing at 1-s hop, inference from exported run directories, side-by-side panel | **done Nov 23**: one server (`scripts/demo.py`, `uaar.live`), three cards from `models/` (plain softmax, taught MC Dropout, taught EDL), used with kit K1 |
| 5.7 | `docs/methods.md` for M1–M4 in own words (§8.2); `docs/architecture.md` | **to do before Dec 8**: the equation-to-code tables and the background-windows section exist; the own-words text for M1, M3 and M4 is not written; `architecture.md` exists |

**Milestone M2 (Nov 15):** four methods on fold 0, all splits (Nov 5); live demo end to end (Nov 23); wearable data on disk: only the 60-s test. No tag.

### W6 · Nov 16–22 · Full sweep, fine-tuning, app

| # | Item | Status 2026-11-25 |
|---|---|---|
| 6.1 | Full sweep; `stats.py`; `make_tables.py` → T2 | **done Nov 23** as `configs/sweeps/k4.yaml` (120 runs), `reports/tables/scores_by_protocol.md`, mean ± sd over 15 runs; `stats.py` not written, so no confidence intervals |
| 6.2 | `finetune.py`: freeze backbone, adapt head and normalization on wearable sets; leave-one-subject-out; **P4 zero-shot vs P4-ft** table | code done Nov 23 (`scripts/finetune.py`, `uaar.models.finetune`, `configs/wearable.yaml`), tested on synthetic sessions only; **to do after the recording session**: demo models before Dec 3, the table before Dec 8 |
| 6.3 | P1-b and P2 evaluation on all runs; F2–F6 | P1-b (`test_nonex`) on every run: done; P2 dropped 2026-11-23; no figure generated yet (the functions in `uaar.eval.plots` exist, no script calls them) |
| 6.4 | Streamlit app | **dropped 2026-11-23**; replaced by the live page (done Nov 23, `web/README.md`) |
| 6.5 | Live demo polish: τ from `threshold.json`, "unknown" flagging, LED feedback commands to the kit, method side-by-side, replay mode for a recorded session (the fallback if Bluetooth misbehaves on stage) | τ, unknown flag, side-by-side cards and `scripts/demo.py --replay` done; LED feedback from the page not done (the firmware has the commands); **rehearsal with both kits to do before Dec 3** |
| 6.6 | Decision point Nov 18 | superseded: the decision of Nov 23 (four exercises, three tiers) took its place |
| 6.7 | Freeze and tag | not done; no tags so far |

### W7 · Nov 23–29 · Report draft, video, rehearsal (Thanksgiving Nov 26)

| # | Item | Status 2026-11-25 |
|---|---|---|
| 7.1 | Report draft assembled from §8.3 sources; AI-use paragraph checked against the log; `contributions.md` | outline `report/outline.docx` exists; **draft to do before Dec 8**; `contributions.md` to complete |
| 7.2 | Video 1–3 min: goal, live demo, key result | **to do** |
| 7.3 | Final slides; demo script with the fallback | `report/slides.pptx` drafted (7 slides for 6–7 minutes on Dec 3); review and demo script **to do before Dec 3** |
| 7.4 | RecGym P3 replication | **dropped 2026-11-23** |

### W8 · Nov 30 – Dec 8 · Final

| # | Item | Status 2026-11-25 |
|---|---|---|
| 8.1 | Rehearse twice with both kits charged; verify Bluetooth in the presentation room if possible | to do, Dec 2 |
| 8.2 | Present Dec 3 | — |
| 8.3 | Report final pass: 5–7 pages; every number traced to a table | to do, Dec 7 |
| 8.4 | Submission zip: `git archive` of the commit tagged `final`, report PDF, proposal deck and final deck | to do, Dec 8 |

## 11. Implementation risks

Beyond [README.md §8](README.md#8-risks):

| Risk | Mitigation |
|---|---|
| `loadmat` on the 1.5 GB files needs ~3 GB RAM and 14 s per file; the multi-activity file's nested structs are awkward | Process per subject, append to Parquet; the loader was written against a tiny synthetic `.mat` first; the raw `.mat` stays in `data/external/` |
| GPU not available under WSL2 when the eGPU is detached | `train.py` falls back to CPU |
| Too few subjects per class in a fold | Class-stratified subject folds (decided 2026-11-05) |
| Two copies of the design docs drift apart | The proposal-era copy is frozen; `docs/design/` here is the maintained one |
| Bluetooth loses samples while the kit is worn and moved (`../decisions.md`, 2026-11-23; cause not known) | The recording page shows the loss per set and marks sets to discard; USB recording (`record_wearable.py`) is the fallback; `scripts/demo.py --replay` is the stage fallback |
| Second kit not ready, recording session not held; fine-tuning tested on synthetic sessions only | Kit K1 and the RecoFit models in `models/` run the demo on their own |
| Report over 7 pages | Supporting tables to an appendix if the instructor allows, else cut |

## 12. Open decisions

Logged in `decisions.md` when settled.

| Decision | Status 2026-11-25 |
|---|---|
| Roles | Only the new algorithms are assigned: EDL (Robert), MC Dropout (Sirena). Actual contributions are recorded in `docs/contributions.md` |
| ~~Known-class eligibility thresholds~~ | settled Nov 5 |
| `near` split pairs | settled Nov 5; review pending |
| ~~Class weighting vs balanced sampling~~ | `inv_sqrt` class weights (`configs/base.yaml`); no balanced sampler was written |
| Report format and who holds the master file | open; the outline is `report/outline.docx` |
| ~~Which model the app shows~~ | settled Nov 23: three cards (`configs/demo.yaml`) |
| ~~Seeds per cell~~ | one seed (`configs/sweeps/k4.yaml`) |
| ~~Exercise set for our recordings~~ | settled Nov 5, revised Nov 23: curl, shoulder press, lateral raise, jumping jacks; background and unknown movements in `configs/wearable.yaml` |
| ~~Fine-tuning recipe~~ | settled Nov 23 (`../decisions.md`) |
| ~~Headline metric~~ | settled Nov 24: share of windows accepted at τ, with known accuracy |
| EDL KL weight | stays 1 (the paper's schedule) |

Settled decisions are logged with their reasons in [`../decisions.md`](../decisions.md).
