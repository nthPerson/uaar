# Models

Four methods are compared on identical windows and splits. The three neural methods share one backbone so that the only difference between them is how uncertainty is produced. Code and equations: [../methods.md](../methods.md).

| ID | Method | Course status | Uncertainty score(s); primary score first | Forward passes |
|---|---|---|---|---|
| M1 | Multinomial logistic regression on handcrafted features | Taught in class | 1 − max p (`msp`); entropy | 1 (tiny) |
| M2 | Softmax neural network (1D-CNN backbone) | Taught in class (neural networks) | 1 − max softmax (`msp`); entropy | 1 |
| M3 | **Evidential deep learning (EDL)**, Robert | **Not covered** | Vacuity u = K / S; 1 − max p̂ | 1 |
| M4 | **Monte Carlo Dropout**, Sirena | **Not covered** | Mutual information for plain networks, predictive entropy for taught networks (§9); variation ratio | T = 30 (the trunk runs once, the head T times) |

The primary score of each method is set in `uaar.models.factory.primary_score`.

## 1. Shared backbone (M2–M4)

Input: 6 × 100 (channels × samples, one 2-s window at 50 Hz).

| Layer | Output | Parameters |
|---|---|---|
| Conv1d(6 → 32, k = 5) + BatchNorm + ReLU + MaxPool(2) | 32 × 50 | ≈ 1.0 k |
| Conv1d(32 → 64, k = 5) + BatchNorm + ReLU + MaxPool(2) | 64 × 25 | ≈ 10.3 k |
| Conv1d(64 → 64, k = 3) + BatchNorm + ReLU | 64 × 25 | ≈ 12.4 k |
| Global average pool | 64 | — |
| Dropout(p) → Linear(64 → 64) + ReLU → Dropout(p) | 64 | ≈ 4.2 k |
| Linear(64 → K) (K = number of known classes) | K | ≈ 65 K |

About **30 k parameters**, small enough to train many folds on a laptop GPU and, in principle, to run on a microcontroller. Dropout p = 0.3 (`model.dropout` in `configs/base.yaml`; not tuned).

## 2. Training setup (M2–M4)

| Setting | Value |
|---|---|
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4, cosine decay |
| Batch size | 256 |
| Epochs | up to 60, early stopping (patience 8) on validation-subject macro-F1; taught models on open-set validation macro-F1 (§9) |
| Class imbalance | Class-weighted loss, inverse square-root frequency (`train.class_weighting: inv_sqrt`) |
| Seeds | 1 (seed 0) |
| Hardware | Laptop CPU; RTX 3070 eGPU when attached |

## 3. M1 — Logistic regression (in-class baseline)

- **Features (102 per window):** for each of ax, ay, az, gx, gy, gz, \|a\|, \|g\|: mean, standard deviation, min, max, 10th/50th/90th percentiles, RMS, zero-crossing rate of the mean-removed signal, dominant frequency and its relative power (FFT), spectral entropy (12 × 8). Plus the 6 within-sensor axis correlations.
- **Model:** multinomial logistic regression with L2 penalty (`sklearn.linear_model.LogisticRegression`), features standardized on training subjects; C from {0.01, 0.1, 1, 10} by `GroupKFold(3)` over the training subjects.
- **Why:** a transparent, classical point of comparison. Its probabilities are often better calibrated than a neural network's, which makes it a fair test for Q2.

## 4. M2 — Softmax neural network (in-class baseline)

- Backbone + softmax, class-weighted cross-entropy loss, trained with dropout active.
- **Scores:** maximum softmax probability, MSP (Hendrycks & Gimpel 2017), and predictive entropy.

## 5. M4 — Monte Carlo Dropout (Sirena)

Gal & Ghahramani (2016): a network trained with dropout, with dropout kept on at test time, approximates Bayesian inference over the weights. Averaging T stochastic passes gives a predictive distribution whose spread reflects model (epistemic) uncertainty.

- **Weights:** the M2 network (same training run). This isolates the effect of the inference procedure.
- **Inference:** T = 30 stochastic passes. **Only the Dropout modules are switched to training mode; BatchNorm stays in eval mode** (a common implementation bug otherwise).
- **Predictive mean:** p̄ = (1/T) Σ_t softmax(z_t).
- **Scores:**
  - predictive entropy H[p̄] (total uncertainty);
  - mutual information MI = H[p̄] − (1/T) Σ_t H[softmax(z_t)] (model uncertainty, BALD);
  - variation ratio 1 − (count of the modal class)/T.
- **Cost:** the backbone has Dropout only after the global pooling, so the convolutional trunk runs once and only the two-layer head runs T times (`share_trunk`); timing in [../methods.md](../methods.md) (M4).

## 6. M3 — Evidential deep learning (Robert)

Sensoy, Kaplan & Kandemir (2018): the network outputs non-negative **evidence** for each class, which parameterizes a Dirichlet distribution over class probabilities. Little total evidence means high uncertainty ("vacuity"), from a single forward pass.

- **Head:** evidence e = softplus(z); α = e + 1; S = Σ_k α_k.
- **Prediction:** p̂_k = α_k / S; label = argmax p̂.
- **Uncertainty:** vacuity **u = K / S** (in (0, 1]); also 1 − max p̂.
- **Loss (per sample, one-hot y):**
  - expected cross-entropy, the digamma form (Eq. 4 of Sensoy et al.): Σ_k y_k (ψ(S) − ψ(α_k));
  - plus λ_t · KL( Dir(α̃) ‖ Dir(1) ), where α̃ = y + (1 − y) ⊙ α removes the true class's evidence, so only misleading evidence is penalized;
  - annealing λ_t = `kl_weight` · min(1, epoch / 10), `kl_weight` 1.
- **Why not the squared-error loss (Eq. 5):** it did not train on the 28 known classes (decided 2026-11-05, [../decisions.md](../decisions.md)).
- **Variants tried** on fold 0 only (`kl_weight` 0.1, 0.01 and 0; the log marginal likelihood, Eq. 3; exponential evidence): table in [../decisions.md](../decisions.md), 2026-11-05.
- **Known failure modes:** evidence collapse early in training; over-regularization from the KL term, which lowers known-class accuracy; sensitivity to class imbalance. Watch known-class macro-F1 against M2 on the validation subjects.
- **Link to Robert's research:** Robert already works with EDL (vacuity-based uncertainty in a perception proof of concept); this project applies it to wearable time series.

## 7. Stretch methods

None were built (deep ensembles, SVM, random forest, Mahalanobis distance and temperature scaling were listed in the proposal).

## 8. Model artifacts

Each run directory, and each folder in [`models/`](../../models/README.md), holds `model.pt` (`model.joblib` for logistic regression), `config.json`, `config.yaml`, `preprocess.json` (channel means and standard deviations), `threshold.json` (τ per score from validation), `split.json`, `metrics.json`, `history.csv` and `train_summary.json`, and loads with `uaar.models.factory.load`. The final models are copied to `models/` with `scripts/export_model.py` and tracked in git; the live page takes them from `configs/demo.yaml`.

## 9. Background windows: plain, taught and Other

M2 and M3 can be trained with *background* windows, windows that belong to no known class (other RecoFit exercises and non-exercise motion from the training subjects), chosen by `background.mode`:

- **Plain** (`none`, protocol `k4plain`): the four known classes only. All four methods.
- **Taught** (`uncertain`, protocol `k4taught`): M2 adds the cross-entropy to the uniform distribution on background windows, weight 0.5 (outlier exposure, Hendrycks et al. 2019); M3 adds KL(Dir(α) ‖ Dir(1)) on them, weight 1. M4 runs on the taught M2 network with entropy as its score, because mutual information fails there. A taught model is selected on open-set validation macro-F1 (four classes plus "rejected", with background validation windows as the rejected class).
- **Other class** (`other`, protocol `k4other`): M2 with the background windows as a fifth class, as a reference.
- M1 has plain only.

Configs: `configs/k4plain`, `configs/k4taught`, `configs/k4other`. Write-up and results: [../methods.md](../methods.md), section "Background windows in training".

## 10. Fine-tuning on our own recordings

`uaar.models.finetune`, `scripts/finetune.py`; label roles in `configs/wearable.yaml` (decided 2026-11-23, [../decisions.md](../decisions.md)). Start from a RecoFit four-exercise model; freeze the convolutional trunk; re-fit the input normalization on the training person's windows; train the two output layers on their known sets. A taught model keeps its "be uncertain" term on their background movements. EDL re-initializes the last layer and repeats the KL ramp. `--full` also trains the trunk, as a fallback. The last set of every label is validation, for early stopping and a new threshold; the test is on the other person. Tested on synthetic sessions only; the recording session has not happened yet.
