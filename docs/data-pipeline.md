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
`configs/near_unknown_recofit.yaml`. Windows per part, fold 0:

| Split | Known | Unknown | Train | Val | Test known | Test unknown | Test non-exercise |
|---|---|---|---|---|---|---|---|
| fold0_near | 28 | 10 | 43,128 | 7,971 | 12,004 | 4,168 | 19,804 |
| fold0_rand0 | 28 | 10 | 44,811 | 7,996 | 12,483 | 3,689 | 19,804 |
| fold0_rand1 | 28 | 10 | 37,270 | 6,456 | 10,686 | 5,486 | 19,804 |
| fold0_rand2 | 28 | 10 | 43,648 | 8,375 | 12,640 | 3,532 | 19,804 |

The non-exercise test set of fold 0 is 16,850 `Non-Exercise`, 2,441 `Walk` and 513 `Rest`
windows from the multi file. Demo exercises in `fold0_near` (subjects / windows): train
Two-arm Dumbbell Curl 32 / 1,458, Shoulder Press (dumbbell) 32 / 1,141, Lateral Raise 19 / 795,
Squat (arms in front) 32 / 1,637, Jumping Jacks 18 / 504; validation 5, 4, 4, 4 and 2 subjects;
test 7, 7, 7, 7 and 5 subjects.

## Our wearable

See [wearable-recording.md](wearable-recording.md) for the sessions. The `wearable` dataset is
built from `data/wearable/<subject>/<session>/` by `uaar.data.wearable.to_recordings`.
