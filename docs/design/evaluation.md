# Evaluation

## 1. Formal problem definition

**Given**

- Windows x ∈ ℝ^{6×100} (2 s of 6-axis IMU at 50 Hz) with labels y ∈ 𝒦 = {1, …, K} (known exercises) from a set of training subjects.
- Optionally, windows of *background* classes ℬ (other exercises and non-exercise motion) from the same training subjects, with no exercise label; the taught and Other-class models use them (§2).
- At test time, windows from **new subjects** whose labels may lie in 𝒦, in ℬ, or in 𝒰, the held-out exercises that no model sees in any form. On our own wearable (P4, pending) the device also changes.

**Goal:** learn a classifier f: x → 𝒦 and an uncertainty score s(x) ∈ ℝ such that

1. f is accurate on known-class windows (closed-set performance);
2. s is higher on unknown windows than on known ones (unknown detection);
3. s tracks the chance that f is wrong (calibration and selective prediction).

The open-set decision is ŷ = f(x) if s(x) ≤ τ, otherwise "unknown," with τ chosen from known validation windows only (§3).

**Constraints**

- No held-out-class data in training, validation or threshold selection. τ uses known validation windows only; taught models also use background validation windows to choose their checkpoint.
- No subject appears in both training and test.
- Single-window decisions only (no smoothing across windows).
- Inference cost matters: the target setting is a wearable-class device (not measured, §6).

## 2. Protocols

The headline benchmark is **K4**: four known exercises, on 5 subject folds × 3 background divisions (`bg0`–`bg2`) × 1 seed (15 runs per result), in three tiers that differ only in what the model sees in training:

| ID | Tier | Train | Test (known) | Test (unknown) |
|---|---|---|---|---|
| `k4plain` | plain | the four known classes, training subjects | known classes, test subjects | 16 held-out exercises; non-exercise; background classes; look-alikes (all test subjects) |
| `k4taught` | taught | the same plus background windows with a "be uncertain" target | same | same |
| `k4other` | Other class (reference) | the same background windows as a fifth class | same | same |

Other protocols:

| ID | Protocol | Train | Test (known) | Test (unknown) | Status |
|---|---|---|---|---|---|
| P1 | RecoFit 28-class "hard case" | 28 known classes, training subjects | Known classes, test subjects | 10 held-out exercises, test subjects | Fold 0 only, four class splits (`rand0`–`rand2`, `near`), one seed |
| P1-b | RecoFit non-exercise | same models | same | `Non-Exercise`, `Rest`, `Walk` windows from the multi-activity file, test subjects | Reported with P1 and K4 |
| P4 | Our wearable | K4 models (zero-shot), and the same fine-tuned on one person (P4-ft) | The four known exercises we record | `waving`, `torso_twists`; placement shifts `curl_rotated90`, `curl_loose` (`configs/wearable.yaml`) | Pending the recording session |

P1 was the planned headline and was cut to the hard case on 2026-11-23; P2 (MM-Fit) and P3 (RecGym) were dropped the same day ([../decisions.md](../decisions.md)).

Headline numbers (decided 2026-11-24): the share of windows accepted at τ (§3), for unknown exercises and for non-exercise, with known accuracy next to it. FPR@95, AUROC and ECE are supporting. Results and wording for the report: [../methods.md](../methods.md), section "Background windows in training".

## 3. Threshold selection

For each method and score, τ is the value at which **95 % of the validation subjects' known-class windows are accepted**. This uses known data only, which is what a deployed system could do. The same τ is used in the zero-shot wearable test (P4); a fine-tuned model gets a new τ from the last set of each label of the training person (decisions 2026-11-23).

## 4. Metrics

| Group | Metric | Definition / note |
|---|---|---|
| Headline | **Accepted at τ** | Share of a test set's windows with s ≤ τ, per set: unknown exercises, non-exercise, background, look-alikes (lower is better, except look-alikes) |
| | **Known kept** | The same share for the known test windows |
| Closed-set | **Accuracy**, macro-F1 | On known-class test windows; accuracy is reported with the headline, macro-F1 in `reports/tables/t2_headline.md` |
| Unknown detection | **AUROC** | Unknown windows = positive class, score s; threshold-free |
| | AUPR-unknown | Precision–recall with unknown as positive; sensitive to the unknown:known ratio (report the ratio) |
| | **FPR@95** | Fraction of unknown windows accepted as known when 95 % of the known *test* windows are accepted (lower is better) |
| Open-set classification | Open-set macro-F1 over K + 1 classes; unknown recall | Using τ from §3 |
| Calibration | **ECE** (15 equal-width bins), NLL, Brier score | On known-class test windows |
| Selective prediction | Risk–coverage curve, **AURC** (and excess AURC) | Error rate among retained windows as the rejection threshold moves |
| Misclassification detection | AUROC (correct vs. wrong known predictions) | Does s flag the model's own mistakes? |
| Shift response (P4) | Δ accuracy, Δ mean s | Compared with the K4 values; pending |

## 5. Statistics

Results are mean ± standard deviation over the 15 runs (5 folds × 3 background divisions × 1 seed). Several seeds, significance tests and bootstrap confidence intervals were not done and are not planned before submission unless time allows. Differences inside the run-to-run spread are not claimed ([../methods.md](../methods.md)).

## 6. Inference cost

Not measured. The only timing is MC Dropout against one softmax pass in [../methods.md](../methods.md) (M4).

## 7. Tables and figures

| Table | Content | Source |
|---|---|---|
| T1 | Dataset and split statistics | [../data-pipeline.md](../data-pipeline.md) |
| T2 | Headline: methods × (macro-F1, AUROC, FPR@95, accepted at τ, ECE, AURC) for P1 and the three K4 tiers | `reports/tables/t2_headline.md` |
| T3 | Every score of every method, with known kept and the non-exercise, background and look-alike sets | `reports/tables/scores_by_protocol.md` |

Both tables are written by `uv run scripts/make_tables.py` from the saved predictions.

No figure has been generated yet. The plotting functions in `uaar.eval.plots` (score histograms, ROC curves, reliability diagrams, risk–coverage curves) exist; no script calls them. An uncertainty timeline on a wearable recording is pending the recording session.

## 8. Reproducibility

- Every run writes `config.yaml`, `config.json`, `split.json`, `metrics.json` (with the git commit), `preds_*.parquet`, `history.csv`, `threshold.json`, `preprocess.json`, `train_summary.json` and the model to `reports/runs/<run-id>/`. `reports/runs/` is not in git; the final models are copied to [`models/`](../../models/README.md) with `scripts/export_model.py`.
- Seeds are set for Python, NumPy and PyTorch (`uaar.utils.seed_everything`).
- The tables are regenerated by one command from saved predictions, not edited by hand.
