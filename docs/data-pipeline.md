# Data pipeline

How to get from downloads to windows and splits, what the numbers are, and where they come from.
Commands are copy-pasteable and re-run before each milestone.

## Commands

```bash
uv run scripts/prepare_data.py recofit --steps recordings windows splits
uv run scripts/prepare_data.py recofit --steps labels        # per-label minutes and subjects
uv run scripts/prepare_data.py wearable --steps recordings windows
```

Inputs live under `data/external/<dataset>/` (gitignored); outputs under `data/processed/<dataset>/`
in the layout of [design/development-spec.md §3](design/development-spec.md#3-data-contracts).

## RecoFit

Source files (copied 2026-11-03 from the FormCoach machine; same public LFS objects):

| File | Bytes | SHA-256 |
|---|---|---|
| `exercise_data.50.0000_singleonly.mat` | 1,562,300,284 | `70eea039362555daccb00065e3775ce37fff421f18d533c60933a9d859f33f8a` |
| `exercise_data.50.0000_multionly.mat` | 1,571,881,721 | `d14a8fa3a6ddb6740ff09f7aa4a3039d3ee5524513a8d4b65ac66e00d14ab509` |

First run 2026-11-03 (`recordings` 55 s, `windows` 96 s, `splits` 0.1 s on the ThinkPad):

| Stage | Result |
|---|---|
| recordings | 4157 recordings (26,411,908 samples, 146.7 h): 4,037 single-exercise recordings after dropping 650 junk ones (device taps, band adjustment, "Note", "Invalid") and 126 multi-activity visits from 94 subjects |
| windows | 473,859 kept, 48,192 dropped by the 80 % purity rule; `X.npy` is 1.1 GB |
| splits | 5 folds x 3 random class splits with the spec defaults (20 subjects, 5 min): **52 eligible classes**, 39 known / 13 unknown per split; `Non-Exercise` and `Rest` are excluded from the class pool (`split.nonexercise_labels`) and form the P1-b set |

The window count above includes both copies of the exercise data; see the next section.

### The two files hold the same exercise data (found 2026-11-04)

The single-exercise file is a copy of the labelled segments of the multi-activity file: 65 of
the 68 labels have exactly the same number of samples in both sources (`Non-Exercise` differs by
4 samples, `Walk` by 1, and the unlabelled stretches exist only in the multi file), and a
200-sample stretch of one recording compared bit for bit matched exactly. Taking exercise windows from both sources, as the first run did,
counts every sample twice (no subject leakage, since splits are by subject, but double the
minutes and double the training time). Since 2026-11-04 `split.class_source: single` restricts
class eligibility and the train/val/test selections to the single file, as
[design/datasets.md §1](design/datasets.md#1-recofit-primary) intended; the non-exercise set
(P1-b) still comes from the multi file. Unique data: 132,787 exercise windows (single) and
104,171 non-exercise windows (multi).

### Labels (from the windows, one source each; `prepare_data.py recofit --steps labels`)

Minutes are `windows x hop / 60`, the quantity the eligibility rule uses.

| Label | Minutes | Subjects | Windows |
|---|---|---|---|
| Non-Exercise *(non-exercise set)* | 1703.7 | 94 | 102219 |
| Walk *(non-exercise set)* | 202.2 | 25 | 12134 |
| Static stretch | 149.5 | 37 | 8970 |
| Running (treadmill) | 119.1 | 21 | 7148 |
| Elliptical machine | 108.5 | 22 | 6508 |
| Static Stretch (at your own pace) | 107.4 | 30 | 6444 |
| Rowing machine | 103.4 | 22 | 6202 |
| Dynamic Stretch (at your own pace) | 99.8 | 29 | 5986 |
| Jump Rope | 59.9 | 30 | 3595 |
| Plank | 51.0 | 61 | 3061 |
| Butterfly Sit-up | 45.5 | 29 | 2729 |
| Lunge (alternating both legs, weight optional) | 42.9 | 43 | 2573 |
| Repetitive Stretching | 40.5 | 16 | 2430 |
| Wall Squat | 39.6 | 30 | 2374 |
| Squat (arms in front of body, parallel to ground) | 38.1 | 43 | 2285 |
| Burpee | 37.5 | 43 | 2253 |
| Triceps Kickback (knee on bench) (label spans both arms) | 35.3 | 30 | 2119 |
| Dumbbell Row (knee on bench) (label spans both arms) | 33.8 | 29 | 2025 |
| Two-arm Dumbbell Curl (both arms, not alternating) | 33.5 | 44 | 2011 |
| Sit-ups | 33.2 | 25 | 1993 |
| Sit-up (hands positioned behind head) | 33.2 | 43 | 1991 |
| V-up | 32.8 | 43 | 1970 |
| Rest *(non-exercise set)* | 32.5 | 9 | 1952 |
| Russian Twist | 32.5 | 33 | 1951 |
| Dumbbell Squat (hands at side) | 32.4 | 30 | 1944 |
| Walking lunge | 32.1 | 25 | 1928 |
| Crunch | 31.9 | 30 | 1911 |
| Squat Rack Shoulder Press | 30.8 | 29 | 1850 |
| Dumbbell Deadlift Row | 29.3 | 25 | 1759 |
| Bicep Curl | 28.8 | 25 | 1729 |
| Overhead Triceps Extension | 27.5 | 41 | 1649 |
| Shoulder Press (dumbbell) | 25.7 | 43 | 1543 |
| Squat | 25.0 | 25 | 1499 |
| Triceps extension (lying down) | 24.4 | 22 | 1467 |
| Fast Alternating Punches | 24.4 | 30 | 1464 |
| Dip | 24.1 | 44 | 1447 |
| Seated Back Fly | 23.7 | 30 | 1422 |
| Medicine Ball Slam | 22.8 | 22 | 1369 |
| Kettlebell Swing | 22.6 | 30 | 1354 |
| Lateral Raise | 22.5 | 30 | 1351 |
| Chest Press (rack) | 21.6 | 21 | 1294 |
| Biceps Curl (band) | 19.7 | 22 | 1181 |
| Squat (kettlebell / goblet) | 19.4 | 21 | 1165 |
| Band Pull-Down Row | 19.3 | 21 | 1157 |
| Pushup (knee or foot variation) | 19.3 | 44 | 1157 |
| Squat (hands behind head) | 19.0 | 21 | 1138 |
| Power Boat pose | 18.0 | 25 | 1081 |
| Side Plank Left side | 18.0 | 25 | 1081 |
| Side Plank Right side | 17.8 | 25 | 1068 |
| Squat Jump | 17.8 | 25 | 1065 |
| Wall Ball | 17.4 | 23 | 1041 |
| Lawnmower (right arm) | 17.1 | 21 | 1028 |
| Lawnmower (left arm) | 15.9 | 21 | 953 |
| Pushups | 11.5 | 25 | 690 |
| Jumping Jacks | 11.4 | 25 | 685 |
| Unlisted Exercise | 7.2 | 4 | 433 |
| Triceps Kickback (knee on bench) (right arm) | 6.9 | 14 | 413 |
| Dumbbell Row (knee on bench) (right arm) | 6.7 | 15 | 402 |
| Triceps Kickback (knee on bench) (left arm) | 6.2 | 13 | 374 |
| Dumbbell Row (knee on bench) (left arm) | 6.2 | 15 | 372 |
| Box Jump (on bench) | 2.8 | 3 | 170 |
| Lawnmower (label spans both arms) | 2.3 | 3 | 139 |
| Overhead Triceps Extension (label spans both arms) | 2.3 | 2 | 138 |
| Alternating Dumbbell Curl | 1.1 | 1 | 69 |
| Triceps extension (lying down) (left arm) | 0.5 | 1 | 28 |
| Triceps extension (lying down) (right arm) | 0.5 | 1 | 27 |
| Arm straight up | 0.1 | 1 | 6 |

RecoFit is four subject cohorts with almost no overlap, each with its own exercise list: about
44 subjects (Two-arm Dumbbell Curl, Dip, Burpee, Shoulder Press (dumbbell), ...), 30 (Lateral
Raise, Jump Rope, Crunch, Kettlebell Swing, ...), 25 (the generic names: Bicep Curl, Squat,
Sit-ups, Pushups, Jumping Jacks, Walk, ...) and 21 (Running, Elliptical, Rowing, the band and
rack exercises). The 25-subject cohort's generic labels duplicate better-supported ones in the
44-subject cohort (`Bicep Curl` / `Two-arm Dumbbell Curl ...`, `Squat` / `Squat (arms in front
...)`, `Sit-ups` / `Sit-up (hands positioned behind head)`, `Pushups` / `Pushup (knee or foot
variation)`), which matters for the class splits: a random split can hold out one twin while
the other is known.

### Eligibility (decided 2026-11-05, item 4.4)

Eligible exercise classes (`Non-Exercise`, `Rest` and `Walk` excluded, single file only) by
threshold:

| min subjects | >= 5 min | >= 10 min | >= 20 min | >= 30 min |
|---|---|---|---|---|
| 20 | 51 | 51 | 37 | 24 |
| 21 | 51 | 51 | 37 | 24 |
| 25 | 38 | 38 | 31 | 21 |
| 29 | 27 | 27 | 26 | 19 |
| 30 | 23 | 23 | 22 | 15 |
| 40 | 11 | 11 | 10 | 7 |
38 51 dropped at 25/10: ['Band Pull-Down Row', 'Biceps Curl (band)', 'Chest Press (rack)', 'Elliptical machine', 'Lawnmower (left arm)', 'Lawnmower (right arm)', 'Medicine Ball Slam', 'Rowing machine', 'Running (treadmill)', 'Squat (hands behind head)', 'Squat (kettlebell / goblet)', 'Triceps extension (lying down)', 'Wall Ball']
total class windows single: 120659 nonex windows: 116305

**Chosen: 25 subjects, 10 minutes** (`configs/base.yaml`): 38 classes, of which each class split
holds out 10 as unknown and keeps 28 known. With `Walk` in the non-exercise set the unique data are
120,659 class windows (single file) and 116,305 non-exercise windows (multi file). Reasons and the alternatives are in `decisions.md`.

### Splits (T1 input)

`uv run scripts/prepare_data.py recofit --steps splits --force` writes 20 files: 5 subject folds
x (`rand0`, `rand1`, `rand2`, `near`). The `near` pairs and the tied twin labels are in
`configs/near_unknown_recofit.yaml`; the folds are class-stratified (next section). Windows per
part, fold 0:

| Split | Known | Unknown | Train | Val | Test known | Test unknown | Test non-exercise |
|---|---|---|---|---|---|---|---|
| fold0_near | 28 | 10 | 43,371 | 7,254 | 12,478 | 3,750 | 26,341 |
| fold0_rand0 | 28 | 10 | 44,847 | 7,880 | 12,563 | 3,665 | 26,341 |
| fold0_rand1 | 28 | 10 | 37,011 | 6,470 | 10,931 | 5,297 | 26,341 |
| fold0_rand2 | 28 | 10 | 44,560 | 7,652 | 12,451 | 3,777 | 26,341 |

Fold 0 has 63 training, 12 validation and 19 test subjects. Its non-exercise test set is 23,849
`Non-Exercise`, 2,375 `Walk` and 117 `Rest` windows from the multi file. Demo exercises in
`fold0_near` (subjects / windows): train Two-arm Dumbbell Curl 30 / 1,391, Shoulder Press
(dumbbell) 29 / 1,042, Lateral Raise 20 / 976, Squat (arms in front) 29 / 1,540, Jumping Jacks
17 / 469; validation 5, 5, 4, 5 and 3 subjects; test 9, 9, 6, 9 and 5 subjects.

### Subject folds are stratified by exercise (decided 2026-11-05)

This section is written to be reused in the report (data description and experimental setup)
and in the presentation.

**The problem.** RecoFit was collected in four groups of participants, and each group did its
own list of exercises (the cohort note above). An exercise therefore has data from only 25 to
61 of the 94 subjects (median 30). The usual way to make subject-independent folds, shuffling
the subjects and cutting the list into five, ignores this, so by chance one fold can receive
most of a cohort's subjects as training subjects and almost none as validation or test
subjects. With the unstratified folds of the first run (seed 0), counting subjects per
eligible class:

| Fold | Val subjects per class: min / median | Classes with < 3 val subjects (of 38) | Test subjects per class: min / median | Classes with < 3 test subjects |
|---|---|---|---|---|
| 0 | 2 / 4 | 13 | 5 / 7 | 0 |
| 1 | 2 / 4 | 3 | 3 / 6 | 0 |
| 2 | 2 / 3 | 2 | 6 / 7 | 0 |
| 3 | **1** / 2 | **24** | **2** / 7 | **11** |
| 4 | 3 / 5 | 0 | 4 / 6 | 0 |

In fold 3, 24 of the 38 classes had one or two validation subjects (Lateral Raise had one,
Jumping Jacks two) and 11 classes had two test subjects.

**Why it matters for this project in particular.**

1. *The rejection threshold comes from the validation subjects.* τ is the 95th percentile of
   the validation known-class scores, and it is never re-tuned. If a class is represented by
   one validation subject, τ for every method depends on how that one person moves.
2. *Model selection uses validation macro-F1*, which weights every class equally. A class with
   one validation subject contributes as much to early stopping as a class with eight.
3. *The headline metrics are macro-averaged and their confidence intervals resample test
   subjects.* A class with two test subjects has an unstable per-class F1 and can disappear
   from a bootstrap resample altogether.
4. *Method comparisons are paired by fold.* If folds differ mainly in which classes are
   under-represented, fold-to-fold variation reflects fold composition and not the methods,
   which weakens the Wilcoxon tests.

**What we do instead.** Subjects are assigned to folds by *iterative stratification*
(Sechidis, Tsoumakas & Vlahavas 2011, a standard method for multi-label data), treating each
subject as an example and "this subject performed exercise c" as its labels. The exercise
with the fewest unassigned subjects is dealt out first; each of its subjects goes to the fold
that still lacks the most subjects of that exercise (ties: the fold lacking the most subjects
overall, then at random with the seed). The same procedure then separates the validation
subjects (15 %) from the training subjects inside each fold. Code:
`uaar.data.splits.stratified_assign` and `make_subject_folds(stratify_classes=...)`, switched
on by `split.stratify_folds: true`, recorded in every split file as
`"fold_strategy": "class_stratified"`, tested in `tests/test_splits.py`.

**Result** (same seed, same data):

| Fold | Train / val / test subjects | Val subjects per class: min / median / max | Test subjects per class: min / median / max | Classes with < 3 val or test subjects |
|---|---|---|---|---|
| 0 | 63 / 12 / 19 | 3 / 4 / 8 | 5 / 6 / 11 | 0 |
| 1 | 64 / 11 / 19 | 3 / 4 / 7 | 5 / 6 / 12 | 0 |
| 2 | 64 / 11 / 19 | 3 / 4 / 7 | 5 / 6 / 13 | 0 |
| 3 | 64 / 11 / 19 | 3 / 4 / 8 | 5 / 6 / 12 | 0 |
| 4 | 65 / 11 / 18 | 3 / 4 / 7 | 5 / 6 / 13 | 0 |

Every class now has at least 3 validation subjects, at least 5 test subjects and at least 17
training subjects in every fold, and the five folds look alike. In all 20 split files every
known class has at least 3 validation and 5 test subjects.

**What did not change, and why this is still a fair test.**

- A subject is still in exactly one of train, validation and test; every subject is a test
  subject in exactly one fold. The evaluation is as subject-independent as before.
- The fold sizes are the same (18 or 19 test subjects, 11 or 12 validation subjects).
- The only information used is *which exercises each subject performed*, which is a fact about
  how the dataset was collected. No signal, no model output and no result was used.
- The known/unknown class splits are unchanged; they do not depend on the folds.
- It was decided from the subject counts above, before any full results were compared. The
  fold-0 runs made on the unstratified folds (two class splits) are kept aside in
  `reports/runs/unstratified-fold0/` and are not used in any table.

**Limitation to state.** Three validation subjects per class is still few, because the data
have only 25 subjects for the smallest classes. Stratification makes the folds even; it cannot
add subjects. This is one reason results are reported as a mean and spread over five folds.

**Short form for a slide.** "RecoFit's participants did different exercise lists, so random
subject folds left some exercises with 1–2 validation subjects (24 of 38 classes in one fold).
We stratify the subject folds by exercise: every class now has ≥ 3 validation and ≥ 5 test
subjects in every fold. Subjects never cross between train and test."

## Our wearable

See [wearable-recording.md](wearable-recording.md) for the sessions. The `wearable` dataset is
built from `data/wearable/<subject>/<session>/` by `uaar.data.wearable.to_recordings`.
