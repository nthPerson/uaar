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

To be written with `uaar.features.handcrafted` and `uaar.models.logreg`.

## M4 Monte Carlo dropout (Sirena)

To be written with `uaar.models.mcdropout`.
