# Evaluation

## 1. Formal problem definition

**Given**

- Windows x ∈ ℝ^{6×100} (2 s of 6-axis IMU at 50 Hz) with labels y ∈ 𝒦 = {1, …, K} (known exercises) from a set of training subjects.
- At test time, windows from **new subjects** whose labels may lie in 𝒦 or in 𝒰 (unknown exercises and non-exercise motion never seen in training), possibly recorded by a **different device**.

**Goal:** learn a classifier f: x → 𝒦 and an uncertainty score s(x) ∈ ℝ such that

1. f is accurate on known-class windows (closed-set performance);
2. s is higher on unknown windows than on known ones (unknown detection);
3. s tracks the chance that f is wrong (calibration and selective prediction).

The open-set decision is ŷ = f(x) if s(x) ≤ τ, otherwise "unknown," with τ chosen **without any unknown data** (§3).

**Constraints**

- No unknown-class data in training, validation or threshold selection.
- No subject appears in both training and test.
- Single-window decisions only (no smoothing across windows) for the headline results; temporal smoothing is an ablation.
- Inference cost matters: the target setting is a wearable-class device.

## 2. Protocols

| ID | Protocol | Train | Test (known) | Test (unknown) | Goal |
|---|---|---|---|---|---|
| P1 | **RecoFit open-set** (headline) | Known classes, training subjects | Known classes, test subjects | Held-out exercises, test subjects | Minimum |
| P1-b | RecoFit non-exercise | same models | same | Idle/transition windows from the multi-activity file, test subjects | Minimum |
| P2 | Device shift | RecoFit (P1 models) | MM-Fit exercises mapped to known classes | MM-Fit exercises not in the known set | Stretch 1 |
| P3 | RecGym open-set | RecGym wrist, training subjects | RecGym known, test subjects | RecGym held-out workouts | Stretch 2 |
| P4 | Our wearable | RecoFit (P1 models) | Known exercises we record | New movements; rotated strap | Stretch 3 |

P1 runs over 5 subject folds × (3 random + 1 near-unknown) class splits × up to 3 seeds.

## 3. Threshold selection

For each method and score, τ is the value at which **95 % of the validation subjects' known-class windows are accepted**. This uses known data only, which is what a deployed system could do. The same τ is used in P2–P4 without re-tuning, which is itself part of the shift test.

## 4. Metrics

| Group | Metric | Definition / note |
|---|---|---|
| Closed-set | Accuracy, **macro-F1** | On known-class test windows; macro-F1 is primary because of imbalance |
| Unknown detection | **AUROC** | Unknown windows = positive class, score s; threshold-free |
| | AUPR-unknown | Precision–recall with unknown as positive; sensitive to the unknown:known ratio (report the ratio) |
| | **FPR@95** | Fraction of unknown windows accepted as known when 95 % of known windows are accepted (lower is better) |
| Open-set classification | Open-set macro-F1 over K + 1 classes; unknown recall | Using τ from §3 |
| Calibration | **ECE** (15 equal-width bins), NLL, Brier score | On known-class test windows; reliability diagrams |
| Selective prediction | Risk–coverage curve, **AURC** | Error rate among retained windows as the rejection threshold moves |
| Misclassification detection | AUROC (correct vs. wrong known predictions) | Does s flag the model's own mistakes? |
| Shift response (P2, P4) | Δ accuracy, Δ mean s, P2-AUROC | Compared with P1 values |
| Cost | Parameters, latency per window, passes | §6 |

## 5. Statistics

- Report mean ± standard deviation over folds × class splits × seeds.
- Paired comparisons between methods on the same fold/split/seed: Wilcoxon signed-rank test on AUROC, FPR@95, ECE and macro-F1; Holm correction across the pairwise tests.
- 95 % bootstrap confidence intervals (resampling test **subjects**, not windows, 1,000 resamples) for the headline table.
- A difference counts as meaningful only if it is significant and larger than the fold-to-fold standard deviation.

## 6. Inference cost

- Parameters and multiply-accumulate operations per window for each method.
- Latency per window on the laptop CPU at batch size 1 (median of 1,000 runs, after warm-up), and on the GPU at batch size 256.
- MC Dropout reported at each T in {5, 10, 30, 50}, so detection quality can be plotted against cost.
- On-device (ESP32-S3) latency is **out of scope**; estimate from MAC counts only.

## 7. Planned figures and tables

| # | Content | Owner |
|---|---|---|
| T1 | Dataset and split statistics (subjects, windows, classes per split) | Robert |
| T2 | **Headline:** four methods × (macro-F1, AUROC, FPR@95, ECE, AURC, latency) on P1 | Both |
| T3 | P1-b non-exercise and P2 shift results | Robert |
| T4 | P3 RecGym replication | Sirena |
| F1 | Example windows: known vs. unknown exercise, with each method's score | Robert |
| F2 | Score histograms, known vs. unknown, per method | Sirena |
| F3 | ROC curves for unknown detection | Sirena |
| F4 | Reliability diagrams | Sirena |
| F5 | Risk–coverage curves | Sirena |
| F6 | MC Dropout AUROC vs. T (cost/quality trade-off), with EDL as a single-pass reference point | Sirena |
| F7 | Uncertainty timeline on a wearable recording (P4) | Robert |

## 8. Reproducibility

- Every run writes `config.yaml`, the split file, metrics JSON and figures to `reports/runs/<run-id>/`.
- Seeds are set for NumPy, PyTorch and the data loaders; the git commit is recorded with each run.
- The headline table is regenerated by one command from saved predictions, not edited by hand.
