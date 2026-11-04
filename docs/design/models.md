# Models

Four methods are compared on identical windows and splits. The three neural methods share one backbone so that the only difference between them is how uncertainty is produced.

| ID | Method | Course status | Owner | Uncertainty score(s) | Forward passes |
|---|---|---|---|---|---|
| M1 | Multinomial logistic regression on handcrafted features | Taught in class | Both | 1 − max p; entropy | 1 (tiny) |
| M2 | Softmax neural network (1D-CNN backbone) | Taught in class (neural networks) | Both | 1 − max softmax (MSP); entropy | 1 |
| M3 | **Evidential deep learning (EDL)** | **Not covered** | **Robert** | Vacuity u = K / S; 1 − max p̂ | 1 |
| M4 | **Monte Carlo Dropout** | **Not covered** | **Sirena** | Predictive entropy; mutual information; variation ratio | T (≈ 30) |

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

About **30 k parameters**, small enough to train many folds on a laptop GPU and, in principle, to run on a microcontroller. Default dropout p = 0.3 (tuned for M4, see §5).

## 2. Training setup (M2–M4)

| Setting | Value |
|---|---|
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4, cosine decay |
| Batch size | 256 |
| Epochs | up to 60, early stopping on validation-subject macro-F1 (patience 8) |
| Class imbalance | Class-weighted loss (inverse square-root frequency) or class-balanced sampling (choose one on the first fold, then freeze) |
| Seeds | 3 per fold × class split (fewer if compute runs short) |
| Hardware | Laptop CPU; RTX 3070 eGPU when attached |

## 3. M1 — Logistic regression (in-class baseline)

- **Features (≈ 90 per window):** for each of ax, ay, az, gx, gy, gz, \|a\|, \|g\|: mean, standard deviation, min, max, 10th/50th/90th percentiles, RMS, zero-crossing rate of the mean-removed signal, dominant frequency and its relative power (FFT), spectral entropy. Plus the 6 within-sensor axis correlations.
- **Model:** multinomial logistic regression with L2 penalty (`sklearn.linear_model.LogisticRegression`), features standardized on training subjects; C chosen by grouped inner cross-validation on the training subjects.
- **Why:** a transparent, classical point of comparison. Its probabilities are often better calibrated than a neural network's, which makes it a fair test for Q2.

## 4. M2 — Softmax neural network (in-class baseline)

- Backbone + softmax, cross-entropy loss, trained with dropout active.
- **Scores:** maximum softmax probability, MSP (Hendrycks & Gimpel 2017), and predictive entropy.
- **Optional:** post-hoc temperature scaling fitted on validation NLL (calibration-only variant, reported separately as M2+TS).

## 5. M4 — Monte Carlo Dropout (Sirena)

Gal & Ghahramani (2016): a network trained with dropout, with dropout kept on at test time, approximates Bayesian inference over the weights. Averaging T stochastic passes gives a predictive distribution whose spread reflects model (epistemic) uncertainty.

- **Weights:** the M2 network (same training run). This isolates the effect of the inference procedure. A separately tuned dropout rate is an ablation.
- **Inference:** T = 30 stochastic passes; ablate T ∈ {5, 10, 30, 50}. **Only the Dropout modules are switched to training mode; BatchNorm stays in eval mode** (a common implementation bug otherwise).
- **Predictive mean:** p̄ = (1/T) Σ_t softmax(z_t).
- **Scores:**
  - predictive entropy H[p̄] (total uncertainty);
  - mutual information MI = H[p̄] − (1/T) Σ_t H[softmax(z_t)] (model uncertainty, BALD);
  - variation ratio 1 − (count of the modal class)/T.
- **Tuning:** dropout rate p ∈ {0.1, 0.3, 0.5} by validation NLL on known classes only.
- **Cost:** T × the forward-pass cost of M2; measured in [evaluation.md §6](evaluation.md#6-inference-cost).

## 6. M3 — Evidential deep learning (Robert)

Sensoy, Kaplan & Kandemir (2018): the network outputs non-negative **evidence** for each class, which parameterizes a Dirichlet distribution over class probabilities. Little total evidence means high uncertainty ("vacuity"), from a single forward pass.

- **Head:** evidence e = softplus(z) (ablation: ReLU, exp with clamping); α = e + 1; S = Σ_k α_k.
- **Prediction:** p̂_k = α_k / S; label = argmax p̂.
- **Uncertainty:** vacuity **u = K / S** (in (0, 1]); also 1 − max p̂. Dissonance is optional.
- **Loss (per sample, one-hot y):**
  - Bayes risk with squared error: Σ_k [(y_k − p̂_k)² + p̂_k (1 − p̂_k) / (S + 1)]
  - plus λ_t · KL( Dir(α̃) ‖ Dir(1) ), where α̃ = y + (1 − y) ⊙ α removes the true class's evidence, so only misleading evidence is penalized;
  - annealing λ_t = min(1, epoch / 10).
- **Ablations:** the log-likelihood (digamma) variant of the loss; annealing length; interaction with class weights.
- **Known failure modes:** evidence collapse early in training; over-regularization from the KL term, which lowers known-class accuracy; sensitivity to class imbalance. Watch known-class macro-F1 against M2 on the validation subjects.
- **Link to Robert's research:** Robert already works with EDL (vacuity-based uncertainty in a perception proof of concept); this project applies it to wearable time series.

## 7. Stretch methods

Only if the minimum goals are met by Nov 8.

| Method | Status in course | Note |
|---|---|---|
| Deep ensemble (5 × M2 with different seeds) | Ensemble learning is taught Dec 1 | Strong uncertainty baseline; 5 × training and inference cost |
| SVM with Platt scaling | Taught Nov 17 | Classical margin-based comparison on M1 features |
| Random forest (vote entropy) | Decision trees / ensembles taught | Classical ensemble comparison |
| Mahalanobis distance on penultimate features | Not covered | Feature-space detector; no retraining |
| Temperature scaling (M2+TS) | Not covered | Calibration only, does not change rankings |

## 8. Model artifacts

Each trained model is saved as `state_dict` + `config.json` (architecture, K, class names, dropout p, seed) + `preprocess.json` (channel means and standard deviations, window length, rate) + `threshold.json` (τ per score from validation). The demo app loads these three files.
