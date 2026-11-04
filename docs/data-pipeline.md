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

### Labels (from `recordings.meta.json`, after junk removal)

| Label | Minutes | Subjects | Windows |
|---|---|---|---|
| Non-Exercise | 3492.2 | 94 | 203803 |
| (unlabelled) | 726.4 | 94 | 0 |
| Walk | 405.5 | 25 | 24262 |
| Static stretch | 300.8 | 37 | 17948 |
| Running (treadmill) | 239.7 | 21 | 14309 |
| Elliptical machine | 218.3 | 22 | 13026 |
| Static Stretch (at your own pace) | 216.4 | 30 | 12899 |
| Rowing machine | 208.0 | 22 | 12412 |
| Dynamic Stretch (at your own pace) | 201.1 | 29 | 11986 |
| Jump Rope | 122.8 | 30 | 7205 |
| Plank | 105.5 | 61 | 6151 |
| Butterfly Sit-up | 92.5 | 29 | 5467 |
| Lunge (alternating both legs, weight optional) | 87.9 | 43 | 5157 |
| Repetitive Stretching | 82.4 | 16 | 4869 |
| Wall Squat | 82.3 | 30 | 4767 |
| Squat (arms in front of body, parallel to ground) | 78.4 | 43 | 4584 |
| Burpee | 77.4 | 43 | 4520 |
| Triceps Kickback (knee on bench) (label spans both arms) | 72.2 | 30 | 4249 |
| Two-arm Dumbbell Curl (both arms, not alternating) | 69.3 | 44 | 4031 |
| Dumbbell Row (knee on bench) (label spans both arms) | 68.9 | 29 | 4054 |
| Sit-up (hands positioned behind head) | 68.5 | 43 | 3994 |
| V-up | 67.9 | 43 | 3954 |
| Sit-ups | 67.7 | 25 | 3995 |
| Russian Twist | 67.0 | 33 | 3919 |
| Dumbbell Squat (hands at side) | 66.4 | 30 | 3898 |
| Walking lunge | 65.5 | 25 | 3865 |
| Rest | 65.5 | 9 | 3903 |
| Crunch | 65.3 | 30 | 3831 |
| Squat Rack Shoulder Press | 63.3 | 29 | 3711 |
| Dumbbell Deadlift Row | 60.0 | 25 | 3527 |
| Bicep Curl | 58.9 | 25 | 3463 |
| Overhead Triceps Extension | 57.1 | 41 | 3312 |
| Shoulder Press (dumbbell) | 53.8 | 43 | 3104 |
| Squat | 51.2 | 25 | 3003 |
| Dip | 50.6 | 44 | 2910 |
| Fast Alternating Punches | 50.6 | 30 | 2936 |
| Triceps extension (lying down) | 50.3 | 22 | 2941 |
| Seated Back Fly | 49.3 | 30 | 2858 |
| Medicine Ball Slam | 47.2 | 22 | 2751 |
| Kettlebell Swing | 46.8 | 30 | 2722 |
| Lateral Raise | 46.8 | 30 | 2713 |
| Chest Press (rack) | 44.6 | 21 | 2595 |
| Pushup (knee or foot variation) | 40.8 | 44 | 2328 |
| Biceps Curl (band) | 40.7 | 22 | 2370 |
| Squat (kettlebell / goblet) | 40.0 | 21 | 2333 |
| Band Pull-Down Row | 39.9 | 21 | 2319 |
| Squat (hands behind head) | 39.2 | 21 | 2284 |
| Side Plank Left side | 37.3 | 25 | 2171 |
| Power Boat pose | 37.3 | 25 | 2166 |
| Side Plank Right side | 36.9 | 25 | 2143 |
| Squat Jump | 36.8 | 25 | 2140 |
| Wall Ball | 36.1 | 23 | 2091 |
| Lawnmower (right arm) | 35.5 | 21 | 2062 |
| Lawnmower (left arm) | 33.0 | 21 | 1920 |
| Pushups | 24.2 | 25 | 1383 |
| Jumping Jacks | 24.1 | 25 | 1380 |
| Unlisted Exercise | 15.1 | 4 | 867 |
| Triceps Kickback (knee on bench) (right arm) | 14.5 | 14 | 830 |
| Dumbbell Row (knee on bench) (right arm) | 14.1 | 15 | 808 |
| Dumbbell Row (knee on bench) (left arm) | 13.2 | 15 | 749 |
| Triceps Kickback (knee on bench) (left arm) | 13.2 | 13 | 750 |
| Box Jump (on bench) | 5.8 | 3 | 342 |
| Lawnmower (label spans both arms) | 4.8 | 3 | 281 |
| Overhead Triceps Extension (label spans both arms) | 4.7 | 2 | 277 |
| Alternating Dumbbell Curl | 2.4 | 1 | 138 |
| Triceps extension (lying down) (left arm) | 1.0 | 1 | 57 |
| Triceps extension (lying down) (right arm) | 0.9 | 1 | 54 |
| Arm straight up | 0.3 | 1 | 12 |

### Eligibility decision (open, item 4.4)

The spec's defaults keep far more classes than the 10–16 known classes the plan targets. Eligible
exercise classes (`Non-Exercise` and `Rest` excluded) for other thresholds:

| min subjects | >= 5 min | >= 20 min | >= 40 min | >= 60 min |
|---|---|---|---|---|
| 20 | 52 | 52 | 38 | 25 |
| 25 | 39 | 39 | 32 | 22 |
| 29 | 27 | 27 | 26 | 19 |
| 30 | 23 | 23 | 22 | 15 |
| 40 | 11 | 11 | 10 | 7 |

Constraints on the choice: the demo exercises must stay eligible (Bicep Curl 25 subjects,
Jumping Jacks 25, Squat 25, Lateral Raise 30, Shoulder Press (dumbbell) 43), and the `near`
split needs look-alike pairs (curl variants, squat variants, row variants, sit-up variants).
Re-run with, for example, `--set split.min_subjects=25 split.min_minutes=20 --force` once decided
and record the choice in `decisions.md`.

## Our wearable

See [wearable-recording.md](wearable-recording.md) for the sessions. The `wearable` dataset is
built from `data/wearable/<subject>/<session>/` by `uaar.data.wearable.to_recordings`.
