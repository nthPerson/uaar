# Development spec and plan

This document turns the design in [README.md](README.md), [datasets.md](datasets.md), [models.md](models.md), [evaluation.md](evaluation.md) and [software-and-firmware.md](software-and-firmware.md) into a build plan: what the code looks like, what each piece must do, how pieces hand data to each other, how it is tested, how it is documented, and who does what by when. Design facts are not repeated here; they are linked. Where this document decides something the design docs left open, the decision is marked **(decision)**.

Drafted 2026-10-08; **schedule and scope revised 2026-11-03** (§10, §12, `../decisions.md`). Status on 2026-11-03: repository bootstrapped at `~/academics/COMPE510_uaar`; wearable firmware v0.1.0 flashed and verified; serial recorder working; datasets copied from the FormCoach machine; no ML code yet. The live wearable demo (D8) and fine-tuning on our own recordings (D4) were added on that date.

---

## 0. Ground rules

1. **AI-written code is allowed** (instructor guidance, see [ai-use-log.md](ai-use-log.md)) but every submitted artifact must "reflect the team's collective understanding," and AI use "must be clearly documented" in the final report (Project_instruction.pdf, p. 1). Two consequences shape this plan:
   - **Each member must be able to explain every module they own**, line by line if asked. The review rule in §9.2 enforces this.
   - **Documentation is a first-class deliverable**, written alongside the code, not after it (§8). The repository must describe the project well enough that a grader can understand what was built without reading the code.
2. **The design docs are the source of truth for *what*; this document for *how*.** When implementation forces a design change, update the design doc in the same pull request and note it in §12.
3. **Reproducibility over speed.** Every number in the report comes from a saved run that one command can regenerate ([evaluation.md §8](evaluation.md#8-reproducibility)).

## 1. Deliverables

| # | Deliverable | Source | Due | Status |
|---|---|---|---|---|
| D1 | `uaar` repository: installable package, scripts, tests, docs | this spec | continuous | not started |
| D2 | RecoFit open-set benchmark: four methods, protocol P1 and P1-b, headline table T2 | minimum goals 1–2 | Nov 8 | not started |
| D3 | Streamlit demo app that runs from a fresh clone | minimum goal 3 | Nov 29 | not started |
| D4 | Device-shift (P2), wearable zero-shot (P4) and **fine-tuned (P4-ft)** results; RecGym (P3) only if time | stretch 1–3 | Nov 22 | not started |
| D5 | Final report, 5–7 pages, with individual contributions and AI-use disclosure | course | Dec 8 | not started |
| D6 | Final slides (10–15) and 1–3 min video | course | Dec 3 or 8 | not started |
| D7 | Submission zip: all code, report, proposal slides, final slides | course | Dec 8 | not started |
| D8 | **Live demo**: wearable streaming over BLE to a browser page, per-window predictions and uncertainty from all four methods side by side, during the final presentation | 2026-11-03 decision | Dec 3 or 8 | firmware and recorder done |

Minimum goals must not depend on stretch goals. In particular, the app (D3) must work with RecoFit example files alone.

## 2. Repository and environment

### 2.1 Location and layout

- **GitHub repository `uaar`**, private, both members as admins. The clone used for development lives on the **WSL2 ext4 filesystem** (for example `~/src/uaar`), **not** under `/mnt/c` or OneDrive **(decision)**. Reasons: `loadmat` on a 1.5 GB file is several times slower through the Windows mount, and OneDrive must never try to sync `data/`.
- This design folder (`docs/uncertainty-aware-activity-recognition/`) is copied into the new repository as `docs/design/` at bootstrap and maintained there from then on **(decision)**. The present repository keeps the proposal deck and course handouts only.
- Layout as in [software-and-firmware.md §1](software-and-firmware.md#1-repository), with these additions:

```text
uaar/
├── README.md                 # what, why, quickstart (fresh clone → app running)
├── pyproject.toml            # uv-managed; Python 3.12; ruff + pytest config
├── uv.lock
├── .github/workflows/ci.yml  # ruff + pytest on every push and PR
├── configs/
│   ├── base.yaml             # defaults shared by every experiment
│   ├── p1/                   # one file per (method); fold/split/seed are CLI overrides
│   ├── p2/ p3/ p4/
│   └── sweeps/               # lists of (fold, split, seed) for scripts/sweep.py
├── docs/
│   ├── design/               # the files in this folder, moved
│   ├── architecture.md       # pipeline diagram, data contracts (§3), module map (§4)
│   ├── data-pipeline.md      # how to get from downloads to windows and splits
│   ├── methods.md            # equation-to-code map for M1–M4
│   ├── experiments.md        # how to reproduce every table and figure
│   ├── app.md                # user guide for the Streamlit app
│   ├── wearable-recording.md # recording session procedure and session log
│   ├── decisions.md          # dated log of design changes
│   ├── contributions.md      # running individual-contributions log
│   └── ai-use-log.md         # moved from design/
├── src/uaar/                 # see §4
├── scripts/                  # see §5
├── tests/                    # see §7
├── notebooks/                # exploratory only; nothing in the report depends on them
├── reports/
│   ├── tables/               # generated .md and .csv (tracked)
│   ├── figures/              # generated .png and .pdf (tracked)
│   └── runs/                 # per-run artifacts (gitignored except metrics.json)
└── data/                     # gitignored
    ├── external/             # downloads, never modified
    └── processed/            # recordings, windows, splits, features
```

### 2.2 Tooling (decision)

| Concern | Choice |
|---|---|
| Package manager | `uv`; `uv sync` creates the environment; `uv run` for every command |
| Python | 3.12, pinned in `pyproject.toml` (`requires-python = ">=3.12,<3.13"`) |
| PyTorch | 2.x. Two extras: `uv sync --extra cu128` (Robert, RTX 3070 under WSL2) and `uv sync --extra cpu` (Sirena, CI). Configured with `[tool.uv.sources]` index entries so both resolve from one lockfile |
| Lint and format | `ruff` (lint + format), line length 100 |
| Tests | `pytest`; `pytest -m "not slow"` in CI, full suite locally |
| Type hints | Required on public functions; `mypy` is **not** run (time budget) |
| Pre-commit | Optional; CI is the gate |
| Random seeds | `uaar.utils.seed_everything(seed)` sets `random`, NumPy, PyTorch (CPU and CUDA) and `torch.backends.cudnn.deterministic` |

### 2.3 Bootstrap checklist (W0, Robert)

1. `uv init --package uaar`, pin Python, add dependencies from [software-and-firmware.md §2](software-and-firmware.md#2-software-stack).
2. `.gitignore`: `data/`, `reports/runs/**` except `**/metrics.json` and `**/config.yaml`, `*.pt`, `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`.
3. CI workflow: `uv sync --extra cpu`, `ruff check`, `ruff format --check`, `pytest -m "not slow"`.
4. `README.md` with the quickstart skeleton (filled in as commands exist).
5. Copy this folder to `docs/design/`; create empty `docs/*.md` files from §8.1 with their section headings.
6. Invite Sirena; confirm `uv sync --extra cpu` and `pytest` pass on her machine. This is the first CI-green commit.

## 3. Data contracts

Every stage reads and writes files with the schemas below, so stages can be developed and tested independently. Paths are relative to `data/processed/<dataset>/`.

### 3.1 Recordings (`recordings.parquet`, long format, one row per sample)

| Column | Type | Note |
|---|---|---|
| `subject` | string | Dataset's subject code (RecoFit numeric id as string; MM-Fit `P0`…`P9`; RecGym `S1`…; wearable `W01`…) |
| `recording_id` | string | Unique within dataset (RecoFit `single/<subject>/<activity-index>` or `multi/<visit>`) |
| `t` | float32 | Seconds from recording start, uniform 0.02 s after resampling |
| `ax, ay, az` | float32 | **g** (RecGym: normalized, unchanged) |
| `gx, gy, gz` | float32 | **°/s** (RecGym: normalized, unchanged) |
| `label` | string | Dataset's original label string; `""` for unlabelled samples; RecoFit junk labels already dropped |
| `source` | string | RecoFit: `single` or `multi`; MM-Fit: `sw_r` or `sw_l`; wearable: session id |

Companion `recordings.meta.json`: `dataset`, `sample_rate_hz`, `units`, `labels` (sorted list with per-label sample counts and subject counts), `source_files` (name, byte size, SHA-256), `created_by` (git commit), `created_at`.

### 3.2 Windows (`windows/X.npy` + `windows/meta.parquet`)

- `X.npy`: float32, shape `(N, 6, 100)` (RecGym: `(N, 6, 40)` at 20 Hz), channel order `ax ay az gx gy gz`, **unnormalized** physical units. Normalization is applied at training time from the fold's statistics (so the same windows serve every fold).
- `meta.parquet`, one row per window, index aligned with `X`: `window_id` (int), `subject`, `recording_id`, `source`, `label`, `start_s`, `label_purity` (fraction of samples with the majority label, ≥ 0.8 by construction), `n_gaps` (wearable only).
- Windowing rule from [datasets.md §5](datasets.md#5-preprocessing-pipeline-all-datasets): 2 s, 1 s hop, majority label ≥ 80 %, else dropped.

### 3.3 Splits (`splits/<split_id>.json`)

`split_id` = `fold{k}_{classsplit}` with `classsplit ∈ {rand0, rand1, rand2, near}`.

```json
{
  "dataset": "recofit",
  "fold": 0, "n_folds": 5,
  "class_split": "rand0", "class_split_seed": 0,
  "train_subjects": ["..."], "val_subjects": ["..."], "test_subjects": ["..."],
  "known_classes": ["Bicep Curl", "..."],
  "unknown_classes": ["..."],
  "eligibility": {"min_subjects": 20, "min_minutes": 5.0},
  "created_by": "<git sha>"
}
```

Known classes are stored in sorted order; the integer label of a known class is its index in this list. Unknown-class windows are labelled `-1` by `splits.encode_labels`. Non-exercise windows (P1-b) are a separate set selected by `source == "multi"` and label in the idle/transition list, never by class split.

### 3.4 Features (`features/handcrafted.parquet`)

One row per `window_id`, about 90 float32 columns named `{channel}_{stat}` (for example `ax_mean`, `amag_specent`) plus `corr_{a}_{b}`. Computed once for all windows; standardized per fold at fit time.

### 3.5 Model artifacts (`reports/runs/<run_id>/`)

`run_id` = `{protocol}_{dataset}_{method}_{split_id}_seed{seed}` (example `p1_recofit_edl_fold0_rand0_seed0`).

| File | Content |
|---|---|
| `config.yaml` | Fully resolved config (base + method file + CLI overrides) |
| `split.json` | Copy of the split used |
| `model.pt` / `model.joblib` | `state_dict` (M2–M4) or fitted sklearn pipeline (M1) |
| `config.json`, `preprocess.json`, `threshold.json` | As in [models.md §8](models.md#8-model-artifacts); `threshold.json` maps score name → τ |
| `history.csv` | Per-epoch train/val loss, val macro-F1, learning rate, λ_t (EDL) |
| `preds_{set}.parquet` | `set ∈ {val, test_known, test_unknown, test_nonex, shift_mmfit, …}`: `window_id`, `y_true` (−1 for unknown), `y_pred`, `p_max`, one column per score (`msp`, `entropy`, `vacuity`, `mi`, `varratio`, …) and `probs` as a fixed-length list column |
| `metrics.json` | Every metric in [evaluation.md §4](evaluation.md#4-metrics), keyed by test set and score |
| `cost.json` | Parameters, MACs, CPU and GPU latency, `T` if applicable |

Because `preds_*.parquet` are saved, `make_tables.py` and all plots run without models or GPUs.

## 4. Module specifications

Owner = the person who writes it, reviews it with the other, and can explain it. Signatures are the intended public interface; private helpers are free.

### 4.1 `uaar.data` (Robert; RecGym loader Sirena)

| Module | Public interface | Notes |
|---|---|---|
| `recofit.py` | `download(dest: Path) -> None`; `load_single(path) -> Iterator[Recording]`; `load_multi(path) -> Iterator[Recording]`; `to_recordings(out_dir) -> None` | LFS media URL from [datasets.md §1](datasets.md#1-recofit-primary). `loadmat` with `struct_as_record=False, squeeze_me=True`; process one subject at a time and append to Parquet with PyArrow to keep RAM under 4 GB. Drop junk labels listed in a module constant. Keep only the master `rightArm` device. |
| `mmfit.py` | `download(dest)`; `to_recordings(out_dir, wrist: Literal["sw_r","sw_l"])` | Unit conversion, duplicate-timestamp removal, interpolation to uniform 100 Hz, `resample_poly(x, 1, 2)`, label assignment from `wXX_labels.csv` by frame index. Workout → participant map from [datasets.md §2](datasets.md#2-mm-fit-device-shift-test). |
| `recgym.py` | `download(dest)` (Kaggle CLI); `to_recordings(out_dir, position="wrist")` | No timestamps: synthesize `t` at 20 Hz per (subject, session) run; drop `C_1`. |
| `wearable.py` | `parse_serial_csv(path) -> Recording`; `parse_ble_log(path) -> Recording`; `to_recordings(session_dir, out_dir)` | Raw counts → g and °/s; `seq` gap detection; labels from `events.csv` written by `record_wearable.py`. |
| `windows.py` | `make_windows(recordings: Path, out_dir: Path, window_s=2.0, hop_s=1.0, purity=0.8) -> None`; `window_recording(df, ...) -> tuple[np.ndarray, pd.DataFrame]` | Pure NumPy; the per-recording function is what the tests exercise. |
| `splits.py` | `eligible_classes(meta, min_subjects, min_minutes) -> list[str]`; `make_subject_folds(meta, n_folds=5, val_frac=0.15, seed=0)`; `make_class_split(classes, frac_unknown=0.25, seed)`; `near_unknown_split(classes, pairs: dict)`; `write_splits(out_dir)`; `load_split(path) -> Split`; `select(meta, split, part: Literal["train","val","test_known","test_unknown"]) -> np.ndarray` (window ids); `encode_labels(labels, split) -> np.ndarray` | `GroupKFold` from scikit-learn. `near` split pairs are a hand-written dict committed in `configs/near_unknown_recofit.yaml` after the label list is printed in W0. |
| `registry.py` | `load_windows(dataset) -> tuple[np.ndarray, pd.DataFrame]`; `load_features(dataset)`; `paths(dataset)` | Single place that knows the directory layout. |

`Recording` is a small dataclass (`subject`, `recording_id`, `source`, `t`, `acc`, `gyro`, `labels`) used only inside loaders.

### 4.2 `uaar.features.handcrafted` (Both; Sirena writes, Robert reviews)

`compute(X: np.ndarray, fs: float = 50.0) -> pd.DataFrame` implementing [models.md §3](models.md#3-m1--logistic-regression-in-class-baseline). Vectorized over windows; FFT with `np.fft.rfft`. Column names are generated from two lists so the test can check the count and the absence of NaN.

### 4.3 `uaar.models` (Robert: `backbone`, `edl`; Sirena: `softmax`, `mcdropout`; Both: `logreg`)

Common interface **(decision)**:

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
| `backbone.py` | `ConvBackbone(nn.Module)` exactly as [models.md §1](models.md#1-shared-backbone-m2m4), `head` left to the subclass; `count_params()`, `count_macs()` (analytic, by layer) | — |
| `logreg.py` | sklearn `Pipeline(StandardScaler, LogisticRegression(multinomial, l2))`; C chosen by `GridSearchCV` with `GroupKFold(3)` on training subjects over C ∈ {0.01, 0.1, 1, 10} | `msp`, `entropy` |
| `softmax.py` | `SoftmaxNet` = backbone + linear; cross-entropy with class weights; training loop in `train_loop.py` (shared with EDL): AdamW, cosine schedule, early stopping on val macro-F1, `history.csv` | `msp`, `entropy` |
| `edl.py` | `EDLNet` = backbone + evidence head (`softplus` default); `edl_loss(alpha, y_onehot, lam) -> Tensor` (Bayes-risk MSE + λ·KL(Dir(α̃)‖Dir(1)), closed-form KL); `edl_loss_digamma` for the ablation; λ_t annealing in the loop | `vacuity`, `one_minus_max`, optional `dissonance` |
| `mcdropout.py` | `MCDropout(softmax_run_dir, T)`: loads a `SoftmaxNet`, `enable_dropout_only()` (sets `nn.Dropout` to train, keeps `BatchNorm` in eval, asserted in a test), T passes, `p̄`, entropy, mutual information, variation ratio | `entropy`, `mi`, `varratio` |
| `train_loop.py` | `fit(model, loss_fn, loaders, cfg, run_dir) -> History`; class-weighted loss (`inv_sqrt`) by default, `balanced_sampler` as the alternative | — |
| `normalize.py` | `fit_stats(X_train) -> dict`; `apply(X, stats)`; written to `preprocess.json` | — |

MC Dropout **does not train**: `train.py --method mcdropout` is an error; `evaluate.py --method mcdropout --from <softmax run>` is the entry point.

### 4.4 `uaar.eval` (Sirena; `cost.py` and `stats.py` Robert)

| Module | Public interface |
|---|---|
| `metrics.py` | `closed_set(y_true, y_pred) -> dict` (accuracy, macro-F1); `unknown_detection(score_known, score_unknown) -> dict` (AUROC, AUPR, FPR@95, ratio); `open_set(y_true, y_pred, score, tau) -> dict` (open macro-F1 over K+1, unknown recall); `misclassification_auroc(score, correct)`; `risk_coverage(score, correct) -> (coverage, risk, aurc)` |
| `calibration.py` | `ece(probs, y, n_bins=15)`, `nll`, `brier`, `reliability_bins(...)` |
| `threshold.py` | `select_tau(score_val_known, accept=0.95) -> float` (the 95th percentile of validation known-class scores); `apply(score, tau) -> bool mask` |
| `cost.py` | `latency_cpu(method, batch=1, n=1000) -> ms`, `latency_gpu(method, batch=256)`, `summary(method) -> dict` |
| `stats.py` | `aggregate(runs: list[dict]) -> DataFrame` (mean ± sd over fold × split × seed); `paired_wilcoxon(a, b)`; `holm(pvals)`; `bootstrap_subjects(preds, metric_fn, n=1000) -> (lo, hi)` |
| `plots.py` | One function per figure F1–F7 in [evaluation.md §7](evaluation.md#7-planned-figures-and-tables), each taking DataFrames read from `preds_*.parquet` and writing to `reports/figures/`. Matplotlib only, one style file `uaar.mplstyle` |

### 4.5 `uaar.app` (Sirena; wearable live tab Robert, stretch)

Streamlit app per [software-and-firmware.md §3](software-and-firmware.md#3-demo-app-streamlit). Structure **(decision)**: `main.py` (layout only), `io.py` (CSV parsing, unit conversion, rate detection, resampling to 50 Hz, reuse of `windows.window_recording`), `inference.py` (load a run directory, cache with `st.cache_resource`, run all four methods), `views.py` (the five output panels). Bundled examples and the "About this model" metrics are copied into `src/uaar/app/assets/` by `scripts/export_app_assets.py` from chosen P1 runs, so the app never reads `reports/` or `data/` at runtime and works from a fresh clone.

### 4.6 `uaar.utils`

`seed_everything`, `git_sha()`, `Config` (dataclass loaded from YAML with dotted-key CLI overrides, `to_yaml()`), `timer`, `logging` setup.

## 5. Scripts and configs

| Script | Does | Example |
|---|---|---|
| `prepare_data.py` | `download` → `recordings` → `windows` → `splits` → `features` for one dataset; idempotent (skips existing outputs unless `--force`); prints T1 statistics | `uv run scripts/prepare_data.py recofit --steps all` |
| `train.py` | One run: resolves config, loads split, normalizes, fits, writes the run directory including `preds_val.parquet` and `threshold.json` | `uv run scripts/train.py --config configs/p1/edl.yaml --fold 0 --class-split rand0 --seed 0` |
| `evaluate.py` | Loads a run, predicts on the requested test sets (P1 known/unknown, P1-b, P2, P4), writes `preds_*.parquet`, `metrics.json`, `cost.json` | `uv run scripts/evaluate.py reports/runs/p1_recofit_edl_fold0_rand0_seed0 --sets test_known test_unknown test_nonex` |
| `sweep.py` | Runs `train` + `evaluate` over a sweep file (folds × splits × seeds × methods), sequentially or `--jobs N`; skips completed runs | `uv run scripts/sweep.py configs/sweeps/p1_full.yaml --jobs 2` |
| `make_tables.py` | Aggregates every `metrics.json`, runs stats, writes T1–T4 as Markdown and CSV, writes F1–F7 | `uv run scripts/make_tables.py` |
| `export_app_assets.py` | Copies chosen runs and example CSVs into the app's assets folder | `uv run scripts/export_app_assets.py --runs <ids>` |
| `record_wearable.py` | Serial or BLE capture, block prompts, `events.csv`, `meta.json`, immediate `seq`-gap report and quick plot | `uv run scripts/record_wearable.py --port /dev/ttyACM0 --subject W01` |

Config resolution: `configs/base.yaml` ← `configs/<protocol>/<method>.yaml` ← CLI `--set key=value` overrides; the resolved result is what gets saved. Minimal `base.yaml`:

```yaml
dataset: recofit
protocol: p1
data: {window_s: 2.0, hop_s: 1.0, purity: 0.8, channels: [ax, ay, az, gx, gy, gz], augment: false}
split: {n_folds: 5, val_frac: 0.15, frac_unknown: 0.25, min_subjects: 20, min_minutes: 5.0}
train: {epochs: 60, batch_size: 256, lr: 1.0e-3, weight_decay: 1.0e-4, patience: 8, class_weighting: inv_sqrt}
model: {dropout: 0.3}
eval: {accept_rate: 0.95}
```

Method files add only what differs, for example `configs/p1/edl.yaml`: `method: edl`, `model.edl: {evidence: softplus, loss: mse, kl_anneal_epochs: 10}`.

## 6. Experiment plan and compute budget

| Protocol | Runs | Training cost |
|---|---|---|
| P1 | 5 folds × 4 class splits × 3 seeds = **60** per trained method; M1 (logreg), M2 (softmax), M3 (EDL) → 180 fits; M4 reuses M2 | M2/M3: ~2–5 min each on the RTX 3070 (~30 k parameters, ~100 k windows, ≤ 60 epochs) → about 4–10 GPU-hours total. M1: seconds each. |
| P1-b, P2, P4 | Evaluation only on the 60 P1 runs per method | Minutes |
| P3 | 5 folds × 3 splits × 1 seed on RecGym = 15 per method | ~1 GPU-hour |
| Cost measurements | One per method on one run, CPU and GPU, MC Dropout at T ∈ {5, 10, 30, 50} | Minutes |

Order of execution: fold 0, `rand0`, seed 0 for every method first (the "smoke sweep," `configs/sweeps/p1_smoke.yaml`), inspected by both members before the full sweep is launched. The full sweep runs on Robert's GPU overnight in the W4 window; Sirena's CPU can run M1 and the evaluation-only protocols.

Seeds: 3 per cell is the plan; if the full sweep exceeds one night, drop to seeds {0, 1} and say so in the report.

## 7. Testing

Unit tests run on synthetic data, never on downloads, so CI needs no data. Tests marked `slow` need `data/processed` and run locally.

| Test file | Invariants checked |
|---|---|
| `test_windows.py` | Window count for a known-length signal; majority-label rule at exactly 80 %; dropped windows when purity < 80 %; last partial window dropped; channel order |
| `test_splits.py` | **No subject in two of train/val/test**; **no unknown-class window in train or val**; test windows come only from test subjects; eligibility thresholds; known-class encoding is stable across folds; the `near` split uses only eligible classes |
| `test_loaders.py` | Tiny synthetic `.mat` (written with `scipy.io.savemat`), `.npy`+labels and CSV fixtures round-trip through each loader with the correct units, rate and labels; MM-Fit duplicate-timestamp removal; wearable `seq` gap count |
| `test_features.py` | Feature count, no NaN/inf, a sine window has the expected dominant frequency |
| `test_backbone.py` | Output shape `(N, K)`; parameter count within 10 % of 30 k; MAC count positive and stable |
| `test_edl.py` | KL(Dir(1)‖Dir(1)) = 0; loss is finite for extreme evidence; gradient check with `torch.autograd.gradcheck` on a tiny batch in float64; the `mse` and `digamma` variants agree in ordering on a fixed batch; vacuity ∈ (0, 1]; vacuity decreases as total evidence increases |
| `test_mcdropout.py` | After `enable_dropout_only()`, every `Dropout` is in train mode and every `BatchNorm` in eval mode; with T = 1 and dropout p = 0 the output equals the softmax net's; MI ≥ 0; variation ratio ∈ [0, 1] |
| `test_metrics.py` | AUROC/AUPR agree with scikit-learn on random data; FPR@95 on a hand-built case; ECE of a perfectly calibrated synthetic predictor ≈ 0; AURC of a perfect ranking = 0; `select_tau` accepts 95 % of the known validation scores |
| `test_stats.py` | Wilcoxon on identical inputs gives p = 1; Holm never lowers a p-value; bootstrap resamples subjects, not windows |
| `test_app_io.py` | CSV in m/s² and rad/s converts to g and °/s; sample rate detected from `t`; a 100 Hz file is resampled to 50 Hz with the right length |
| `test_train_smoke.py` (`slow`) | `train.py` on 500 synthetic windows for 2 epochs writes every file in §3.5 |

Definition of done for a module: public functions documented, tests above passing, reviewed per §9.2.

## 8. Documentation plan

The instructor's requirement is that the submission reflects the team's understanding and documents the project clearly. The repository is the primary evidence, and the final report is distilled from the repository docs, not written from scratch in December.

### 8.1 Documentation set

| Document | Audience | Content | Owner | Written during |
|---|---|---|---|---|
| `README.md` (repo root) | Grader, future us | One-paragraph problem statement; the four methods in two sentences each; quickstart (`uv sync`, `prepare_data`, `train`, `streamlit run`) that is tested from a fresh clone before each milestone; link map to all other docs; team, course, AI-use statement pointer | Robert | W0 skeleton, updated every milestone |
| `docs/design/*` | Both | The design docs from this folder; edited as decisions change, with the change noted in `decisions.md` | Both | continuous |
| `docs/architecture.md` | Grader, Sirena ↔ Robert | Pipeline diagram (Mermaid: downloads → recordings → windows → splits → train → evaluate → tables/app); the data contracts of §3; module map of §4 with owners | Robert | W1 |
| `docs/data-pipeline.md` | Anyone reproducing | Exact commands; download sizes and times; what the T1 numbers are and where they come from; EDA findings (class counts, subject counts, example plots) that feed report §5 | Robert | W1–W2 |
| `docs/methods.md` | Grader (report §4) | For each of M1–M4: the equations from [models.md](models.md) placed next to the code that implements them (function names and line links), the hyperparameters and why, the failure modes seen and how they were handled | Robert (M3), Sirena (M4), both (M1, M2) | W3–W4 |
| `docs/experiments.md` | Grader (report §5–6) | One subsection per table and figure: the command that regenerates it, the runs it aggregates, and a short interpretation written when the result first appears | Sirena (figures), Robert (tables T1, T3), both (T2) | W4–W6 |
| `docs/app.md` | Grader (report §7), users | Screenshots, input format, what each panel shows, how τ is chosen, limitations | Sirena | W5–W7 |
| `docs/wearable-recording.md` | Anyone repeating the session | Hardware and firmware version, protocol as run (deviations from [software-and-firmware.md §5](software-and-firmware.md#5-recording-protocol-stretch-goal-3)), session log table (date, subject code, block, duration, dropped samples) | Robert | W6 |
| `docs/decisions.md` | Both | Dated entries: what changed, why, which design doc was updated | whoever decides | continuous |
| `docs/contributions.md` | Report §11 | Per member, per week: what was built, reviewed, written. Fed by the PR log | Both, each writes their own | weekly |
| `docs/ai-use-log.md` | Report disclosure | Continued from [ai-use-log.md](ai-use-log.md): one line per AI contribution (date, tool, what it produced, what the human did with it) | whoever used the tool | every PR |
| Docstrings | Readers of the code | Every public function and class: one-line summary, arguments with units and shapes, return value, reference to the paper equation where applicable (for example `edl_loss`: "Eq. 5 and 7 of Sensoy et al. 2018") | module owner | with the code |
| Module headers | Grader | Each file in `src/uaar/models/` opens with: owner, the paper it implements, and the sentence "Implemented from the paper by <name>; AI assistance recorded in docs/ai-use-log.md" | module owner | with the code |

### 8.2 Standards

- **Plain language first.** Each doc opens with what the thing is and why it exists before any command or equation.
- **Own words.** The owner writes the module docstring and the `docs/methods.md` subsection for their method themselves, without AI, after the code works. This is the artifact that demonstrates understanding and it is the thing we would be asked about in the presentation.
- **Commands are copy-pasteable** and are re-run before each milestone. A doc that lists a command that does not run is a bug.
- **Generated, not edited:** everything in `reports/` is produced by `make_tables.py`. Docs link to those files rather than pasting numbers that can go stale.
- **Verification marks:** the design docs' **(verify)** items are resolved in `decisions.md` with the measured value and the date, then the mark is removed from the design doc.

### 8.3 Traceability to the final report

| Report section (Project_instruction.pdf §7) | Points | Source documents |
|---|---|---|
| 1 Introduction | 5 | `README.md`, [README.md §1](README.md#1-problem) |
| 2 Related work | 7 | [references.md](references.md); `docs/methods.md` difference paragraphs |
| 3 Problem formulation | 10 | [evaluation.md §1](evaluation.md#1-formal-problem-definition) |
| 4 Methods | 10 | `docs/methods.md` |
| 5 Data description and experimental setup | 10 | `docs/data-pipeline.md` (T1, EDA), `docs/experiments.md` (settings), resolved `config.yaml` files |
| 6 Results and analyses | 20 | `reports/tables/`, `reports/figures/`, interpretations in `docs/experiments.md` |
| 7 Application | 10 | `docs/app.md` |
| 8 Conclusion | 3 | written last |
| 9 Video | 10 | link |
| 10 References | — | [references.md](references.md) |
| 11 Individual contributions | required | `docs/contributions.md` |
| AI-use disclosure | required | `docs/ai-use-log.md`, summarized into a paragraph |

The report draft is assembled in W7 by pulling from these sources; each section's owner is the owner of the source doc.

### 8.4 AI-use disclosure process

1. Every pull request description has an **"AI assistance"** line: `none`, or the tool and what it produced (for example "Claude Code drafted `windows.py` and its tests; I rewrote the purity rule and the docstrings").
2. On merge, the same line is appended to `docs/ai-use-log.md`.
3. In W7 the log is summarized into the report's disclosure paragraph, in the style the instructor gave ("Used X to help Y"). Robert writes it; Sirena checks it against the log.

## 9. Working agreements

### 9.1 Git

- `main` is always green (CI) and always runnable from a fresh clone.
- Feature branches named `<area>/<short-topic>` (`data/recofit-loader`, `models/edl-loss`). One PR per work item in §10.
- Commit messages: imperative summary line; body explains why when not obvious.
- Tags at milestones: `m1-data`, `m2-baselines`, `m3-methods`, `m4-shift`, `m5-app`, `final`.

### 9.2 Review rule (the "explain it" rule)

Every PR is reviewed by the other member before merge. The reviewer's checklist:

1. Tests exist for the invariants in §7 and pass in CI.
2. The author can walk through the code in a 5-minute call, or in the PR description, in their own words.
3. Docstrings and the relevant `docs/*.md` were updated.
4. The "AI assistance" line is present.

Small fixes (typos, config values) can be merged by the author with a note.

### 9.3 Communication

- Weekly 30-minute sync (proposed: after class Thursday) with three items: what landed, what is blocked, what is next; outcomes go in `docs/contributions.md`.
- Blocking questions go to the shared chat with the `@` of the other member; non-blocking ones become GitHub issues.
- Run results are shared as links to run directories or to `reports/tables/`, never as screenshots of terminals.

## 10. Schedule

Revised 2026-11-03. The original W0–W3 plan (Oct 8 – Nov 1) did not run: the wearable hardware came first. Dates below run from Nov 3 to the Dec 8 submission; course dates are unchanged ([README.md §5](README.md#5-plan)). R = Robert, S = Sirena, B = both. The live demo (D8) and fine-tuning (D4) decided on Nov 3 are folded in; the original item numbers are not reused.

**Critical path:** RecoFit pipeline → fold-0 runs of all four methods → wearable recordings → fine-tuning → live demo rehearsal. Everything on the wearable side (firmware, recorder, browser page) runs in parallel with the pipeline and never blocks it.

### W4 · Nov 3–8 · Bootstrap, firmware, data pipeline (Exam 2 is Nov 5)

| # | Item | Owner | Done when |
|---|---|---|---|
| 4.1 | Repository bootstrap (§2.3); firmware v0.1.0 flashed and verified; `wearable.py` + tests; `record_wearable.py`; 60-s test recording | R | **done Nov 3** |
| 4.1b | `windows.py`, `splits.py`, `registry.py`, `Config`, `train_loop.py`, `backbone.py`, `softmax.py`, `edl.py`, `normalize.py`, `threshold.py`, `prepare_data.py`, `train.py` written and tested on synthetic data (ahead of 4.4, 5.2, 5.3); `uaar.live` backend, `live_replay.py` and the `web/` page (4.6) working with a placeholder predictor | R | **done Nov 3**, pending the real-data run |
| 4.2 | GitHub remote `uaar` (public), CI green, Sirena's clone passes `pytest` | R, S | first CI-green commit |
| 4.3 | `recofit.py` single- and multi-activity loaders → `recordings.parquet`; the 75 labels printed to `docs/data-pipeline.md` | R | label list committed |
| 4.4 | `windows.py`, `splits.py` + leakage tests; eligibility thresholds decided; `rand0–2` and `near` splits; T1 | R (near pairs B) | `splits/*.json`, T1 generated |
| 4.5 | `metrics.py`, `calibration.py`, `threshold.py` + tests (§7); read Gal & Ghahramani 2016 | S | tests pass |
| 4.6 | Browser page `web/`: Web Bluetooth connect, live charts, session state from the button, record mode that saves the session CSV | R | written and simulator-tested Nov 3; **to verify against a kit in Windows Chrome** |
| 4.7 | Sirena's kit assembled and flashed; both kits gyro-calibrated; exercise set confirmed against the label list (§12) | B | `docs/wearable-recording.md` device table |

**Milestone M1 (Nov 8, tag `m1-data`):** windows and splits on disk, T1 generated, both wearables streaming.

### W5 · Nov 9–15 · All four methods on fold 0; recording session

| # | Item | Owner | Done when |
|---|---|---|---|
| 5.1 | `handcrafted.py`, `logreg.py`, `evaluate.py` with P1 known/unknown sets, `metrics.json` | S writes, R reviews | M1 row of T2, fold 0 |
| 5.2 | `backbone.py`, `softmax.py`, `train_loop.py`, `normalize.py`, `Config`, `train.py`, run-directory writer | R writes, S reviews | M2 row of T2, fold 0 |
| 5.3 | `edl.py` + `test_edl.py`; fold-0 run; vacuity histograms | R | val macro-F1 within 2 points of M2 |
| 5.4 | `mcdropout.py` + `test_mcdropout.py`; fold-0 evaluation at T = 30 | S | score histograms |
| 5.5 | **Recording session 1**: both subjects, ~20 min each, protocol in [software-and-firmware.md §5](software-and-firmware.md#5-recording-protocol-stretch-goal-3) as revised in `decisions.md` | B | labelled sessions in `data/wearable/`, zero-gap check |
| 5.6 | Live backend: FastAPI WebSocket in WSL, 2-s windowing at 1-s hop, inference for all four methods from exported run directories, side-by-side panel | R | demo runs with fold-0 models on one kit |
| 5.7 | `docs/methods.md` for M1–M4 in own words (§8.2); `docs/architecture.md` | R (M3), S (M4), B | reviewed |

**Milestone M2 (Nov 15, tag `m2-methods`):** four methods on fold 0, all splits; wearable data on disk; live demo works end to end with one method.

### W6 · Nov 16–22 · Full sweep, fine-tuning, app

| # | Item | Owner | Done when |
|---|---|---|---|
| 6.1 | Full P1 sweep (M2, M3 on the GPU; M1 and M4 evaluations on CPU); `stats.py`; `make_tables.py` → T2 | R (GPU), S (CPU, tables) | T2 with mean ± sd and CIs |
| 6.2 | `finetune.py`: freeze backbone, adapt head and normalization on wearable sets; leave-one-subject-out (train R → test S and the reverse); **P4 zero-shot vs P4-ft** table | R | T3 wearable rows, before and after |
| 6.3 | P1-b and P2 (MM-Fit) evaluation on all runs; F2–F6 | R (eval), S (figures) | T3 shift rows; figures committed |
| 6.4 | Streamlit app panels 1–5 from exported assets (`export_app_assets.py`); fresh-clone test | S | `docs/app.md` draft |
| 6.5 | Live demo polish: τ from `threshold.json`, "unknown" flagging, LED feedback commands to the kit, method side-by-side, replay mode for a recorded session (the fallback if BLE misbehaves on stage) | R | rehearsed once with both kits |
| 6.6 | **Decision point Nov 18:** if EDL or MC Dropout is not separating unknowns better than softmax on fold 0, freeze methods and spend the remaining time on analysis | B | logged in `decisions.md` |
| 6.7 | Freeze: tag `m3-results`; no new experiments after Nov 22 except bug fixes | B | tag pushed |

### W7 · Nov 23–29 · Report draft, video, rehearsal (Thanksgiving Nov 26; plan for three working days)

| # | Item | Owner | Done when |
|---|---|---|---|
| 7.1 | Report draft assembled from §8.3 sources; AI-use paragraph (R) checked against the log (S); `contributions.md` | B | draft in `report/` |
| 7.2 | Video 1–3 min: goal, live demo, key result | B | uploaded |
| 7.3 | Final slides 10–15 on the SDSU template; demo script with the fallback | B | reviewed |
| 7.4 | RecGym P3 replication, only if 6.1–6.5 are done | S | T4, or dropped and said so |

### W8 · Nov 30 – Dec 8 · Final

| # | Item | Owner | Done when |
|---|---|---|---|
| 8.1 | Rehearse twice with both kits charged; verify BLE in the presentation room if possible | B | Dec 2 |
| 8.2 | Present Dec 3 or Dec 8 | B | — |
| 8.3 | Report final pass: 5–7 pages; every number traced to a table | B | Dec 7 |
| 8.4 | Submission zip: `git archive` of the tagged `final` commit, report PDF, both decks | R | Dec 8 |

## 11. Implementation risks

Beyond [README.md §8](README.md#8-risks):

| Risk | Mitigation |
|---|---|
| `loadmat` on the 1.5 GB files needs ~3 GB RAM and 14 s per file; the multi-activity file's nested structs are awkward | Process per subject, append to Parquet; write the loader against a tiny synthetic `.mat` first; keep the raw `.mat` only in `data/external/` |
| GPU not available under WSL2 when the eGPU is detached | `train.py` falls back to CPU with a warning; the smoke sweep is sized to run on CPU in an hour |
| Sirena's machine cannot run the full sweep | Split by method, not by fold: Robert runs M2/M3 training; Sirena runs M1 and all evaluation-only work (M4, P1-b, P2, tables) |
| Class-eligibility thresholds leave too few known classes | Decide thresholds from the W1 counts with a target of 10–16 known classes; record in `decisions.md` |
| Streamlit app slow on upload (MC Dropout T = 30 per window) | Cache models; batch all windows in one forward pass per stochastic sample; cap uploads at 10 minutes of signal |
| Two people editing `docs/design/*.md` and the proposal-era copies in this repository drift apart | After bootstrap, this folder is frozen with a note pointing to the `uaar` repository |
| Wearable firmware differs from the FormCoach design docs (units, columns) | Item 5.6 confirms before the session; the parser is table-driven by firmware version in `meta.json` |
| Report over 7 pages | Figures F2–F5 as a 2×2 panel; ablations and P3 to an appendix if the instructor allows, else cut |

## 12. Open decisions

To be settled at the dates shown and logged in `decisions.md`.

| Decision | By | Owner |
|---|---|---|
| Role split confirmed with Sirena (this plan assumes [README.md §6](README.md#6-roles)) | Nov 6 | B |
| Known-class eligibility thresholds and the resulting known/unknown lists | Nov 7 (item 4.4) | R |
| Hand-picked `near` split pairs | Nov 7 | B |
| Class weighting vs balanced sampling (choose on fold 0, freeze) | Nov 13 (item 5.2) | B |
| MM-Fit → RecoFit label mapping | Nov 17 (item 6.3) | R |
| Report format (LaTeX vs Word) and who holds the master file | Nov 15 | B |
| Which model the app demonstrates by default (the instructor requires one; we show four with one default) | Nov 18 | S |
| Seeds per cell (3 planned; 2 if the sweep exceeds one night) | Nov 16 | R |
| Exercise set for our recordings: shortlist bicep curl, dumbbell shoulder press, lateral raise, squat, jumping jacks (in RecoFit and MM-Fit, wrist-visible, classroom-safe) plus 2–3 unknown movements; confirm names against the printed label list | Nov 7 (item 4.7) | B |
| Fine-tuning recipe (what is frozen, epochs, how τ is re-selected) | Nov 16 (item 6.2) | R |

Settled decisions are logged with their reasons in [`../decisions.md`](../decisions.md).
