# Contributions

Per member, per week: what was built, reviewed and written. Summarized into the report's
individual-contributions section. Each member writes their own entries. AI assistance is
recorded separately in [ai-use-log.md](ai-use-log.md); the items below list the work and the
decisions, whoever typed the code.

## Week of Nov 3

**Robert:** project re-planning (live demo, fine-tuning, own firmware); wearable firmware v0.1.0
and bench verification; serial recorder; browser page and live backend; repository bootstrap;
RecoFit loader, windows, splits, Config, training loop, softmax and EDL implementations; schedule
revision and decisions log. AI assistance throughout, recorded in `ai-use-log.md`.

Nov 4–5, with AI assistance: the fix for RecoFit's duplicated exercise data (class windows
from the single-exercise file only); MC Dropout, the evaluation module and scripts, the
handcrafted features and logistic regression; the decisions on class eligibility (25 subjects,
10 minutes), the demo labels, `Walk` as non-exercise, the `near` split and tied twin labels,
and class-stratified subject folds; the fold-0 sweep of the 28-class benchmark; the choice of
the digamma loss for EDL after the squared-error loss did not train (`decisions.md`,
2026-11-04 and 2026-11-05).

**Sirena:** wearable K2 assembly (in progress).

## Week of Nov 23

**Robert**, with AI assistance (`decisions.md` 2026-11-23 and 2026-11-24): the decision to cut
the benchmark to four known exercises in three tiers (plain, taught, Other class) and to drop
MM-Fit and RecGym; the background splits, the taught losses, the Other-class model and the
120-run sweep on all folds; acceptance of open-set checkpoint selection for the taught models;
the choice of headline metric; the live demo page as the course app (`scripts/demo.py`, three
cards); the guided recording page; the fine-tuning recipe and `scripts/finetune.py`; final
models tracked in `models/`; Bluetooth bench checks with kit K1; the report outline
(`report/outline.docx`) and the seven-slide deck (`report/slides.pptx`).

**Sirena:** *(to be completed by Sirena)*
