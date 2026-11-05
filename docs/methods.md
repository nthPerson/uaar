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
- Loss Eq. 5 + λ_t · Eq. 7: `edl_loss`; digamma variant Eq. 4: `edl_loss_digamma`; KL closed form: `kl_dirichlet_uniform`; annealing `kl_anneal`.
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
