# Methods

Equation-to-code map for M1–M4. Each owner writes their section in their own words after the
code works (development-spec §8.2); the tables below point at the implementation.

## M2 softmax network (both)

- Backbone: `uaar.models.backbone.ConvBackbone` (models.md §1), 28,908 parameters for K = 12.
- Loss: class-weighted cross-entropy (`uaar.models.base.class_weights`, inverse square-root).
- Scores: `msp = 1 - max p`, predictive entropy (`uaar.models.base.entropy`).
- Training: `uaar.models.train_loop.fit` (AdamW, cosine, early stopping on validation macro-F1).

## M3 evidential deep learning (Robert)

- Evidence and Dirichlet: `uaar.models.edl.EDLNet` (`alpha = softplus(z) + 1`), `dirichlet_summary`.
- Loss Eq. 4 (digamma form, the default since 2026-11-05) + λ_t · Eq. 7: `edl_loss_digamma`; squared-error variant Eq. 5: `edl_loss`; log marginal likelihood Eq. 3: `edl_loss_log`; KL closed form: `kl_dirichlet_uniform`; annealing `kl_anneal`, λ_t = `kl_weight` · min(1, epoch / `kl_anneal_epochs`).
- Why not Eq. 5, and the variants tried: `decisions.md`, 2026-11-05, "EDL trains with the digamma loss".
- Scores: vacuity `K / S`, `1 - max p̂`.
- Section text: to be written by Robert after the fold-0 run.

## M1 logistic regression (Sirena writes, Robert reviews)

- Features: `uaar.features.handcrafted.compute`, 102 per window: 12 statistics (mean, std, min,
  max, p10, p50, p90, rms, zcr, domfreq, dompow, specent) for each of ax, ay, az, gx, gy, gz,
  |a|, |g|, plus the 6 within-sensor axis correlations. Flat signals get 0 for the undefined
  ones (pinned by `tests/test_features.py`).
- Model: `uaar.models.logreg.LogRegMethod`, `StandardScaler` + multinomial L2
  `LogisticRegression` (lbfgs, tol 1e-4, up to 2000 iterations so every fit converges), the same
  inverse-square-root class weights as the neural methods.
- Choosing C: `GroupKFold(3)` over the training subjects on the grid {0.01, 0.1, 1, 10}, mean
  macro-F1, then a refit on all training windows; `history.csv` has one row per C. The grid is
  fitted in ascending C with warm starts and the three inner folds run in parallel, which
  brought a fold-0 run from about 14 minutes to about 4.5.
- Scores: `msp = 1 - max p`, predictive entropy.
- Seen on fold 0 (first run, both data copies): validation accuracy 0.754, macro-F1 0.729 with
  C = 1. C = 1 and C = 10 cannot be told apart at this data size (inner-fold macro-F1 0.649
  against 0.647; the warm-started C = 10 fit on the full training set stops after 0 iterations).
- Section text: to be written by Sirena.

## M4 Monte Carlo dropout (Sirena)

- Weights: a trained M2 run, loaded by `uaar.models.mcdropout.MCDropoutMethod.from_softmax_run`;
  nothing is trained (`fit` raises).
- Stochastic passes: `enable_dropout_only` (Dropout in train mode, BatchNorm in eval mode,
  pinned by `tests/test_mcdropout.py`), T = 30 passes by default, masks seeded per `predict`
  call so results repeat.
- Predictive mean and scores: `mc_scores` (`entropy` = H[p̄], `mi` = H[p̄] − mean_t H[p_t],
  `varratio` = 1 − modal votes / T).
- Cost: the backbone has Dropout only after the global pooling, so the convolutional part is
  identical in every pass. `share_trunk=True` (default) runs it once and repeats only the
  two-layer head; `share_trunk=False` runs the whole network T times. The outputs are equal
  (checked on the fold-0 test windows: maximum difference 0); the cost is not (2.8 s against
  19.2 s for 36,404 windows at T = 30 on the laptop CPU; one softmax pass takes 1.3 s). The
  cost table should report both.
- Section text: to be written by Sirena.

## Background windows in training: plain, taught and "Other" (2026-11-23)

This section is worded so that it can go into the report and the slides. Code:
`uaar.models.softmax` (`taught_cross_entropy`), `uaar.models.edl` (`taught_edl_loss`),
`uaar.data.splits` (`write_background_splits`, `select_background`), configs `configs/k4plain`,
`configs/k4taught`, `configs/k4other`, sweep `configs/sweeps/k4.yaml`. Decision record:
`decisions.md`, 2026-11-23. Splits: `data-pipeline.md`, "Background splits".

### The problem

A recognizer for four exercises (dumbbell curl, shoulder press, lateral raise, jumping jacks)
will also see every other movement its wearer makes. We want it to answer "unknown" for those.
The usual recipe is to compute an uncertainty score for each 2 s window and to reject the
window when the score is above a threshold τ, where τ is set so that 95 % of the known-class
validation windows are kept.

With that recipe the four methods, trained on the four exercises only, recognize the known
exercises well (accuracy 0.96 on subjects never seen in training) and still accept most of what
they should reject: about 60 to 77 % of the windows of 16 other exercises, and 54 to 86 % of
non-exercise windows (resting, walking, moving between machines). We call these the **plain**
models.

### Why a four-class model accepts unknown movements

A classifier with four outputs divides the whole input space among its four classes. Training
only teaches it where the borders between those four lie; it says nothing about the rest of
the space, and there the network extrapolates with high confidence. In our runs the plain
softmax network calls more than half of the windows "lateral raise" for 26 of the 32 other
exercises, and static poses are the worst case: 94 % of the windows of a squat hold with the arms in front,
95 % of the boat pose and 82 % of the plank are accepted as a known exercise. The uncertainty
scores inherit this: softmax confidence is high, the dropout passes agree with each other
(low mutual information), and the evidential network reports plenty of evidence, because
nothing in training ever asked it for less.

### Three tiers

| Tier | Protocol | What the model sees in training | How a window is rejected |
|---|---|---|---|
| 1. Plain | `k4plain` | windows of the four known exercises | uncertainty score > τ |
| 2. Taught | `k4taught` | the same, plus *background* windows with the target "be uncertain here" | uncertainty score > τ |
| 3. Other class (reference) | `k4other` | the same background windows as a fifth class, "Other" | 1 − largest known-class probability > τ |

*Background* windows are windows of other exercises (16 of the 32 remaining RecoFit classes)
and non-exercise windows, all from the training subjects: 8,000 + 4,000 windows per run.
The other 16 exercises are **held out**: no model sees them in any form, and they are the
"unknown" test set. Two labels for one exercise, and listed look-alikes, are kept on the same
side of this division, so a held-out exercise never has a twin among the background ones. The
division is drawn three times (`bg0`, `bg1`, `bg2`) and everything is run on five subject
folds: 15 runs per row of the table below.

**Taught** means that the network keeps its four outputs and the loss gets one more term:

- softmax: for a background window, the cross-entropy between the uniform distribution and
  the prediction, `−(1/K) Σ_k log softmax(z)_k`, weight 0.5. It is smallest when all four
  probabilities are 1/4. This is outlier exposure (Hendrycks, Mazeika & Dietterich 2019).
- EDL: for a background window, `KL(Dir(α) ‖ Dir(1, …, 1))`, weight 1. It is zero when the
  network gives no evidence for any class (α = 1, vacuity 1). It is the regularizer EDL already
  has, applied to all outputs because a background window has no true class.
- MC Dropout: no training of its own; T = 30 dropout passes through the taught softmax network.
- Logistic regression has no taught version and stays a plain baseline.

The difference from the **Other class** is what the model says about a plank. The Other-class
model says "this is class Other" with high confidence; the decision is ordinary classification
and no uncertainty score is involved. The taught model says "I cannot tell which of my four
exercises this is"; the decision is still made by the uncertainty score, so the methods'
scores can still be compared.

### Choosing the checkpoint of a taught model

Training stops early and keeps the weights of the best validation epoch. For the plain models
"best" is macro-F1 on the known validation windows. That criterion is useless for a taught
model: with four classes it reaches 1.0 in the first epochs, before the background term has
had any effect, and says nothing about whether the model has learned to be uncertain. A taught
model is therefore selected on the **open-set macro-F1 of the validation windows**: the four
known classes plus "rejected", with background validation windows (1,200 exercise + 600
non-exercise, from the validation subjects) as the rejected class and the threshold at the 95 %
quantile of the known validation scores (`uaar.models.base.open_macro_f1`). The Other-class
model is selected on macro-F1 over its five classes, which is the same idea.

This was found the hard way. The first sweep selected the taught models on known-class
macro-F1; they accepted 12.9 ± 11.4 % (softmax) and 17.0 ± 16.3 % (EDL) of unknown exercise
windows, with 68 % in the worst run, where early stopping had kept the weights of the first
epoch. With the open-set criterion the same runs give 6.3 ± 3.8 % and 5.2 ± 1.8 %. The earlier
runs are kept in `reports/runs/k4taught-f1-selection/` and are not in the tables.

### Results (RecoFit, 5 subject folds × 3 background divisions, one seed)

Share of test windows **accepted** as a known exercise at τ (lower is better, except "known
kept"). Mean ± sd over the 15 runs.

| Tier | Method | Score | Known accuracy | Known kept | Unknown exercises accepted | Non-exercise accepted | Unknown AUROC |
|---|---|---|---|---|---|---|---|
| Plain | Logistic regression | 1 − max p | 0.957 ± 0.032 | 0.918 ± 0.069 | 0.599 ± 0.124 | 0.540 ± 0.165 | 0.797 ± 0.034 |
| Plain | Softmax | 1 − max p | 0.955 ± 0.038 | 0.932 ± 0.047 | 0.647 ± 0.141 | 0.736 ± 0.107 | 0.760 ± 0.070 |
| Plain | MC Dropout | mutual information | 0.955 ± 0.037 | 0.932 ± 0.067 | 0.774 ± 0.141 | 0.855 ± 0.114 | 0.731 ± 0.048 |
| Plain | MC Dropout | entropy | 0.955 ± 0.037 | 0.933 ± 0.050 | 0.631 ± 0.145 | 0.729 ± 0.110 | 0.766 ± 0.071 |
| Plain | EDL | vacuity | 0.961 ± 0.037 | 0.933 ± 0.043 | 0.630 ± 0.141 | 0.666 ± 0.161 | 0.794 ± 0.052 |
| Taught | Softmax | 1 − max p | 0.962 ± 0.031 | 0.884 ± 0.080 | 0.063 ± 0.038 | 0.029 ± 0.020 | 0.965 ± 0.024 |
| Taught | MC Dropout | entropy | 0.962 ± 0.031 | 0.883 ± 0.078 | 0.061 ± 0.037 | 0.027 ± 0.019 | 0.965 ± 0.025 |
| Taught | MC Dropout | mutual information | 0.962 ± 0.031 | 0.935 ± 0.066 | 0.970 ± 0.027 | 0.991 ± 0.008 | 0.336 ± 0.061 |
| Taught | EDL | vacuity | 0.956 ± 0.032 | 0.894 ± 0.066 | 0.052 ± 0.018 | 0.021 ± 0.011 | 0.965 ± 0.025 |
| Other class | Softmax | 1 − max known p | 0.958 ± 0.034 | 0.897 ± 0.054 | 0.051 ± 0.026 | 0.026 ± 0.015 | 0.972 ± 0.019 |

The full table, with every score, FPR@95 and the background and look-alike sets, is
`reports/tables/scores_by_protocol.md` (`uv run scripts/make_tables.py`).

What the table shows:

1. **Plain models accept most unknown movement**, whichever uncertainty score is used: 60 to
   77 % of unknown exercise windows. No plain method is clearly better than another.
2. **Teaching cuts the accepted share of never-seen exercises from about 63 % to about 6 %**
   (softmax 6.3 %, MC Dropout entropy 6.1 %, EDL vacuity 5.2 %), and of non-exercise windows
   from about 70 % to 2 to 3 %. The taught model accepted fewer unknown windows than its plain
   version in all 15 runs, for softmax and for EDL. Known accuracy is unchanged.
3. **The taught models and the Other class are level on this benchmark** (5.1 % for the Other
   class). The Other class had the lower share in 9 of the 15 runs against taught softmax and
   in 10 of 15 against taught EDL; the differences are inside the run-to-run spread. No
   significance test has been run.
4. **Among the taught models, EDL's vacuity is the steadiest score** (sd 0.018 against 0.038
   for softmax), softmax and MC Dropout entropy are indistinguishable, and EDL remains the
   worst calibrated on the known classes (ECE 0.15 against 0.05).
5. **Mutual information fails on a taught network** (97 % accepted, AUROC 0.34, worse than
   chance). The taught network gives the same flat output in every dropout pass; the passes
   agree, so their disagreement, which is what mutual information measures, is near zero for
   exactly the windows that should be rejected. The entropy of the averaged prediction carries
   the information instead.
6. **The threshold does not transfer perfectly between subjects.** τ keeps 95 % of the known
   windows of the validation subjects but only 88 to 90 % of those of the test subjects for the
   taught and Other-class models (76 % in the worst fold): about one known window in ten is
   wrongly rejected. FPR@95, which sets the threshold on the test subjects' own known windows,
   is therefore much worse than the share accepted at τ in the folds where a test subject
   performs a known exercise unusually (fold 4: known accuracy 0.90, FPR@95 0.55 to 1.0). A
   deployed system would have to choose τ on the wearer's own data.
7. **Variants of a known exercise are accepted by every model** (88 to 95 % of the windows of
   `Bicep Curl` and `Squat Rack Shoulder Press`, depending on the model). They are the known exercises under another
   label, so this is the wanted behaviour and they are not counted as unknown.

### Limitations to state

- One training seed per cell; the spread shown is over subject folds and background divisions.
- For the taught and Other-class models, non-exercise is a category seen in training (from
  other subjects). Only the "unknown exercises" column measures classes never seen in any form.
- Rejection is learned from the background data and reaches as far as that data does. The
  held-out exercises that still pass most often are the ones closest to a known exercise:
  seated back fly (35 to 37 % of its windows for the taught models), squat jump (26 to 31 %)
  and overhead triceps extension (26 to 27 %).
- The checkpoint criterion was changed after the first sweep had been evaluated on the test
  subjects (see above). The new criterion uses validation data only and was fixed before the
  re-run, but the decision to change it was prompted by test results.
- RecoFit was recorded with an armband on the forearm. Nothing here has been tested on our
  wrist wearable yet.
- τ is a 95th percentile of about 650 validation windows from 12 subjects, which makes it the
  least stable number in the pipeline (point 6).

### Short form for a slide

"A four-class model must answer one of four: trained on the four exercises alone, it accepts
about 63 % of windows from 16 exercises it never saw, whichever uncertainty score we use.
Showing it other movement during training with the target 'be uncertain here' brings that to
about 6 %, with the same accuracy on the known exercises, and level with an explicit 'Other'
class (5 %). Mutual information from MC Dropout stops working on the taught network, because
the dropout passes now agree on 'I don't know'."
