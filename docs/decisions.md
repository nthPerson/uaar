# Decisions

Dated log of design and plan changes. Each entry says what changed, why, and which design document was updated. The design docs under `design/` are the source of truth for *what*; `design/development-spec.md` for *how*.

## 2026-11-03

**Own firmware, FormCoach-compatible wire format.** The wearable runs UAAR firmware v0.1.0 (`firmware/`), a 50 Hz IMU streamer with a button-driven session marker and no on-device inference, instead of the FormCoach (BDA 696) firmware the design docs assumed. Reasons: the live demo needs a device with no sleep mode and no gate chatter; a firmware written for this course keeps the course boundary clean; both kits are flashed from this repository. The packet layout and CSV columns are deliberately identical to FormCoach's so one recorder reads either device and FormCoach's browser viewer can sanity-check a UAAR kit. Updated: `design/software-and-firmware.md` §4 (annotated), `firmware/README.md`.

**BMI160 configuration.** The range registers must be written after the sensors are in normal power mode, with a short delay between writes (datasheet interface idle time, 450 µs in suspend). Verified on kit UAAR-8428: 50.00 Hz, 20 ms spacing, zero dropped samples over 60 s, |a| = 0.987 g at rest, so the **(verify)** on units in `design/software-and-firmware.md` §4 is resolved: 4096 LSB/g and 32.8 LSB per °/s.

**Live demo (new deliverable D8).** The final presentation includes a live demonstration: a kit streams over BLE to a browser page (Web Bluetooth in Windows Chrome, because WSL2 has no Bluetooth stack), the page forwards samples over a WebSocket to a FastAPI backend in WSL that runs the trained models, and the page shows per-window predictions, probabilities and the uncertainty score for the four methods side by side, with "unknown" flagged above τ. A replay mode that feeds a recorded session through the same path is the on-stage fallback. The Streamlit upload app required by the course is kept and shares the inference code.

**Fine-tuning on our own recordings (new experiment P4-ft).** Besides the zero-shot wearable test (P4), the backbone pretrained on RecoFit is fine-tuned on our recordings (head and normalization statistics; backbone frozen unless ablated) and evaluated leave-one-subject-out between the two of us. One model is fine-tuned on both subjects for the demo. Updated: `design/evaluation.md` (P4-ft to be added with the recipe at item 6.2), `design/models.md` §6.

**Exercise set (shortlist, to confirm against the printed RecoFit label list).** Known: bicep curl, dumbbell shoulder press, lateral raise, squat, jumping jacks (all in RecoFit and MM-Fit, visible from the wrist, doable in a classroom with light weights or a water bottle). Unknown: two or three of arm circles, shadow boxing, waving, towel wringing. Plus one minute each of standing still and walking. About 20 minutes per subject in one sitting.

**Repository.** `~/academics/COMPE510_uaar` on the WSL ext4 filesystem, GitHub `uaar` (public). The design folder of the proposal repository is frozen and copied to `docs/design/`; the OneDrive repository keeps the proposal deck and course handouts.

**Datasets.** The public downloads were copied from the FormCoach machine's `data/external/` (same public files; download facts in `design/datasets.md` are cited from FormCoach's profiling). Only data files, no code.

**Schedule.** The Oct 8 week plan did not run; §10 of the development spec was rewritten from Nov 3 to Dec 8 with the decision point moved to Nov 18.

**RecoFit junk labels and the class pool (first real run).** Dropped as junk: the device-tap labels
("Tap Left Device", "Tap Right Device", "Tap IMU Device"), "Arm Band Adjustment", "Device on
Table", "<Initial Activity>", "Note", "Invalid" (650 single-exercise recordings, about 11 h).
Kept: "Non-Exercise" and "Rest" as the non-exercise set (P1-b), never as classes
(`split.nonexercise_labels`); "Unlisted Exercise" (4 subjects, never eligible) and "Walk" as
ordinary labels for now. The eligibility thresholds are still the spec defaults; the counts per
alternative are in `data-pipeline.md` and the decision is item 4.4.

## 2026-11-04

**RecoFit exercise windows come from the single file only (bug fix).** The single-exercise file
is a copy of the labelled segments of the multi-activity file (65 of 68 labels have identical
sample counts in both; one recording compared bit for bit), and the first pipeline run took
class windows from both, so every exercise sample was counted twice. There was no subject
leakage (splits are by subject), but per-class minutes and training time were doubled. New
config key `split.class_source: single`, stored in each split file and honoured by
`splits.select` and `splits.eligible_classes`; the non-exercise set still comes from the multi
file. This is what `design/datasets.md` §1 specified. Unique data: 132,787 exercise windows.
Split files written before this date have no `class_source` and select from both sources; they
are regenerated with the eligibility decision. Updated: `data-pipeline.md` (tables in unique
minutes), `design/development-spec.md` §3.3.

**MC Dropout shares the convolutional trunk across passes.** The backbone's only Dropout layers
sit after the global pooling, so the T stochastic passes differ only in the two-layer head.
`MCDropoutMethod` computes the convolutional features once per batch and repeats the head T
times by default (`share_trunk=True`); `share_trunk=False` runs the full network T times with
identical outputs. Consequence for Q3: with this backbone MC Dropout does not cost T full
passes (2.8 s against 19.2 s for the fold-0 test windows at T = 30; softmax 1.3 s), so the cost
comparison with EDL must report both the general T-pass cost and the shared-trunk cost.
To confirm with Sirena (owner of M4).

**Evaluation and logistic regression: where the code differs from the spec's wording.**
(1) Logistic regression has 102 features, not "about 90" (the spec's own list gives 8 x 12 + 6).
(2) C is chosen with an explicit `GroupKFold` loop over a warm-started C path instead of
`GridSearchCV` (same model, scoring and refit; about three times faster). (3) `metrics.json`
holds the score-independent metrics (closed-set, calibration) once under `test_known` and the
per-score metrics under `scores`; the layout is documented in `uaar/eval/runner.py`.
(4) Standard AURC is 0 only when there are no errors, so the spec's "AURC of a perfect ranking
= 0" test is done on the excess AURC (`eaurc`), which is also reported. (5) `train.py` now
passes the training subjects to methods whose `fit` accepts `groups`, and indexes windows by
row position instead of assuming `window_id` equals the row. (6) No `uaar.mplstyle` file; the
plot style is a dict in `uaar/eval/plots.py`.

## 2026-11-05

Decided by Robert; the options and counts are in `data-pipeline.md`.

**Class eligibility: at least 25 subjects and 10 unique minutes** (`split.min_subjects: 25`,
`split.min_minutes: 10`, counted in the single file). 38 classes are eligible; each split has
28 known and 10 unknown. This is the tightest cut that keeps the demo exercises (Jumping Jacks
has 25 subjects and 11.4 minutes) and it drops the 21-subject cohort, whose classes would have
about four test subjects per fold. The plan's target of 10–16 known classes is replaced by 28:
it cannot be reached with thresholds without losing the demo exercises, and hand-picking a
smaller pool would be harder to defend.

**Demo and recording exercise set (RecoFit label strings).**
`Two-arm Dumbbell Curl (both arms, not alternating)`, `Shoulder Press (dumbbell)`,
`Lateral Raise`, `Squat (arms in front of body, parallel to ground)`, `Jumping Jacks`. The
specific curl and squat labels are used instead of the generic `Bicep Curl` and `Squat`: they
have 44 and 43 subjects instead of 25, and they fix the arm position, which is what a wrist
sensor measures. We perform them as named (both arms together; arms held out in front).
Updated: `wearable-recording.md` (protocol labels).

**Walk is non-exercise.** `Walk` (12,128 windows, the largest label by far) joins `Non-Exercise`
and `Rest` in `split.nonexercise_labels`: never a class, part of the P1-b test set, taken from
the multi file. Everyday motion is something the recognizer should reject, and the recording
protocol already has walking in its rest block.

**Near split and twin labels** (`configs/near_unknown_recofit.yaml`). The `near` split holds
out ten look-alikes of known classes: Bicep Curl, Squat, Dumbbell Squat (hands at side), Squat
Rack Shoulder Press, Sit-ups, Pushups, Side Plank Right side, Walking lunge, Static Stretch (at
your own pace), Dumbbell Deadlift Row. All five demo exercises are known in it, so the `near`
models are the ones used for the wearable tests (P4) and the live demo. In the random splits,
five pairs of labels that name the same exercise in two cohorts (Bicep Curl / Two-arm Dumbbell
Curl, Squat / Squat (arms in front ...), Sit-ups / Sit-up (hands positioned behind head),
Pushups / Pushup (knee or foot variation), Static stretch / Static Stretch (at your own pace))
are kept on the same side (`tied`), so a random split never holds out an exercise whose twin is
known; the `near` split does that on purpose.

**Subject folds are stratified by exercise** (`split.stratify_folds: true`; decided by Robert
the same day, after the counts below were shown). *Problem:* RecoFit is four participant
cohorts with their own exercise lists, so an exercise has data from only 25 to 61 of the 94
subjects, and folds made by shuffling subjects left some classes almost without validation or
test subjects: in fold 3, 24 of the 38 eligible classes had one or two validation subjects and
11 had two test subjects. *Why it matters here:* the rejection threshold τ and early stopping
come from the validation subjects, the headline metrics are macro-averaged over classes with
confidence intervals that resample test subjects, and methods are compared fold by fold; a
class resting on one person makes all of these depend on that person. *Change:* subjects are
dealt into folds, and then into validation and training, by iterative stratification on the
subject x exercise table (Sechidis, Tsoumakas & Vlahavas 2011), implemented in
`uaar.data.splits.stratified_assign`. *Result:* every class has at least 3 validation, 5 test
and 17 training subjects in every fold. *Unchanged:* subjects are disjoint between train,
validation and test; fold sizes; the class splits. Only the list of exercises each subject
performed is used, no signals and no results. *Cost:* the split files were regenerated and the
fold-0 runs restarted; the runs made on the old folds (two class splits) are kept in
`reports/runs/unstratified-fold0/` and are not used. This replaces the plain `GroupKFold` of
`design/datasets.md` §6. Full write-up with the before and after tables, worded for the
report and slides: `data-pipeline.md`, "Subject folds are stratified by exercise".

**EDL trains with the digamma loss, not the squared-error loss** (`configs/p1/edl.yaml`:
`loss: digamma`; Robert asked for this to be tested on 2026-11-05 after the diagnostics below,
and for alternatives to be tried). *Problem:* with the paper's preferred loss (Eq. 5, squared
error) and its 10-epoch KL ramp, EDL did not train on the 28 known classes. On the four fold-0
class splits the test macro-F1 was 0.19 to 0.31 (softmax: 0.67 to 0.77), and in three of the
four runs early stopping kept the weights of the first epoch, because validation macro-F1 fell
as the KL weight ramped in. With early stopping off and a 60-epoch ramp the squared-error loss
still reached only 0.55 macro-F1 on `near`, and more than half of the windows had zero evidence
for every class (vacuity exactly 1, every probability 1/28). *Change:* the expected
cross-entropy loss (Eq. 4 of Sensoy et al. 2018, the digamma form) with the same KL term and
the same 10-epoch ramp. *Result (fold 0, seed 0, four class splits):* macro-F1 0.72, 0.64, 0.66,
0.70 (mean 0.68, softmax 0.72); unknown AUROC by vacuity 0.59, 0.62, 0.70, 0.58 (mean 0.62,
softmax 0.59, MC Dropout 0.66); training takes about two minutes per run. *Weak spot:* the
expected probabilities are far too low for the accuracy, ECE 0.36 to 0.41 (softmax 0.08).

Variants tried on fold 0, seed 0 (`near` / `rand0`; runs in `reports/runs/edl-debug/`, not in
any table). `kl_weight` is the value the KL weight ramps to (the paper uses 1).

| EDL variant | macro-F1 | ECE | unknown AUROC (vacuity) | non-exercise AUROC | misclassification AUROC |
|---|---|---|---|---|---|
| digamma, `kl_weight` 1 (default) | 0.723 / 0.643 | 0.409 / 0.358 | 0.585 / 0.624 | 0.768 / 0.754 | 0.855 / 0.784 |
| digamma, `kl_weight` 0.1 | 0.752 / 0.642 | 0.213 / 0.220 | 0.571 / 0.576 | 0.769 / 0.717 | 0.818 / 0.749 |
| digamma, `kl_weight` 0.01 | 0.730 / 0.647 | 0.114 / 0.096 | 0.566 / 0.618 | 0.720 / 0.667 | 0.782 / 0.674 |
| digamma, `kl_weight` 0 (no KL term) | 0.676 / 0.634 | 0.107 / 0.073 | 0.524 / 0.624 | 0.659 / 0.628 | 0.649 / 0.586 |
| log marginal likelihood (Eq. 3), `kl_weight` 1 | 0.713 / 0.593 | 0.374 / 0.362 | 0.597 / 0.599 | 0.776 / 0.744 | 0.866 / 0.800 |
| digamma, exponential evidence | 0.715 / 0.620 | 0.217 / 0.278 | 0.574 / 0.568 | 0.790 / 0.743 | 0.872 / 0.802 |
| softmax, 1 − max p (for reference) | 0.774 / 0.667 | 0.076 / 0.078 | 0.586 / 0.542 | 0.799 / 0.730 | 0.857 / 0.793 |

What the variants show: (1) no variant detects unknowns better than the default; (2) the KL
weight trades calibration against the quality of vacuity: lowering it brings ECE down to the
softmax level, but vacuity then separates wrong from right predictions, and exercise from
non-exercise, worse; (3) with the full KL weight the Dirichlet has evidence for one class only,
so vacuity, 1 − max p̂, entropy, mutual information and differential entropy rank the windows
the same way (unknown AUROC equal to within 0.002), and a different score cannot help; with a
low KL weight they differ, and 1 − max p̂ is then the better score for misclassification.
One fold, one seed and two class splits: differences of a few points between variants are not
established. *Open for Robert:* keep `kl_weight` 1 for the main comparison (the published
schedule, best vacuity, poor ECE) and report the KL weight as an ablation, or adopt 0.1.
Code: `kl_weight` and `loss: log` were added to `uaar.models.edl` for these runs. The runs made
with the squared-error loss are kept in `reports/runs/edl-mse-fold0/` and are not used.

