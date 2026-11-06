# Datasets

Three public wrist/forearm IMU datasets plus a small recording session with our own wearable. Facts marked "verified" were checked against the downloaded files by the FormCoach (BDA 696) data profiling on 2026-09-24 (`reports/data_profile.md` and `docs/04-datasets.md` in that repository); they are facts about the files, not code, and should be re-checked by our own loaders. Everything else is marked **(verify)**.

## Summary

| Dataset | Subjects | Classes | Sensor / placement | Rate | Units | Role in UAAR | License |
|---|---|---|---|---|---|---|---|
| **RecoFit** | 94 in released files | 75 activity labels | Arm band, right forearm | 50 Hz | g, °/s | Training; main open-set benchmark | CDLA-Permissive-2.0 |
| **MM-Fit** | 10 (21 workouts) | 10 exercises + non-activity | Smartwatch, left and right wrist | ~100 Hz | m/s², rad/s | Device-shift test (stretch 1) | MIT (sensor/pose) |
| **RecGym** | 10 (5 sessions each) | 11 workouts + Null | Wrist, pocket, calf (use wrist) | 20 Hz | min-max normalized [0, 1] | Second open-set benchmark (stretch 2) | CC BY 4.0 |
| **Our wearable** | 2+ | Chosen from known set + new movements | XIAO ESP32-S3 + BMI160, left wrist | 50 Hz | raw LSB → g, °/s | Real-device test (stretch 3) | ours |

## 1. RecoFit (primary)

- **Source:** https://github.com/microsoft/Exercise-Recognition-from-Wearable-Sensors (archived, read-only). Paper: Morris, Saponas, Guillory & Kelner, *RecoFit*, CHI 2014, https://doi.org/10.1145/2556288.2557116.
- **Files:** two MATLAB v5 files, `exercise_data.50.0000_multionly.mat` (1,571,881,721 B) and `exercise_data.50.0000_singleonly.mat` (1,562,300,284 B), plus `readme.html` and `load_exercise_data.m`. They are **Git LFS objects**: the raw GitHub URL returns a 135-byte pointer; download from `https://media.githubusercontent.com/media/microsoft/Exercise-Recognition-from-Wearable-Sensors/main/<file>`. (verified)
- **Loading:** `scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)`; about 14 s and 3 GB RAM per file; no `h5py` needed. (verified)
- **Content (verified):**
  - The README says "over 200 participants"; the released files hold **94 subjects**, **126 lab visits** with activity labels (multi-activity file, 79.4 h) and **4,687 single-exercise recordings** (single-activity file, 78 h, a cell matrix of 94 subjects × 75 activities).
  - **75 activity labels** in `exerciseConstants.activities`.
  - Each recording: `data.accelDataMatrix` = `[t_s, x, y, z]` in **g**; `data.gyroDataMatrix` = `[t_s, x, y, z]` in **°/s**; exactly 0.02 s spacing (50 Hz). The master sensor is `rightArm` (forearm band); a second "slave" device in most recordings is ignored.
  - Multi-activity file: `activityStartMatrix` with 7 columns (name, start_s, end_s, notes, reps or −1, start_s again, a sequence struct).
  - Strong imbalance: in the multi-activity file there are 2,642 idle minutes, and single exercises range from hours down to about 20 minutes (for example, lateral raise 23.4 min). About 78 minutes are "junk" labels (device taps, band adjustment) and are dropped.
- **How UAAR uses it:**
  - **Known/unknown exercises** come from the single-activity file (clean, one exercise per recording).
  - **Non-exercise windows** (idle, transitions) from the multi-activity file form a separate "non-exercise" unknown set.
  - **Class eligibility:** an exercise can be a known or unknown class only if it has data from at least 20 subjects and at least 5 minutes in total. Thresholds are tentative until counted. **(verify)**

## 2. MM-Fit (device-shift test)

- **Source:** https://mmfit.github.io/ ; sensor + pose zip `https://s3.eu-west-2.amazonaws.com/vradu.uk/mm-fit.zip` (1,742,309,258 B; extracts to `mm-fit/w00 … w20`, 3.6 GB). Code: https://github.com/KDMStromback/mm-fit. Paper: Strömbäck, Huang & Radu, IMWUT 4(4), 2020, https://doi.org/10.1145/3432701. (verified)
- **Content (verified):** 10 subjects, 21 workouts (about 800 minutes), 10 exercises × 3 sets × 10 reps per workout: squats, lunges, bicep_curls, situps, pushups, tricep_extensions, dumbbell_rows, jumping_jacks, dumbbell_shoulder_press, lateral_shoulder_raises, plus non-activity. 616 labelled sets.
- **Files (verified):** one directory per workout with `wXX_labels.csv` (`start_frame, end_frame, repetition_count, activity`, no header) and `.npy` arrays. Use `wXX_sw_r_acc.npy`, `wXX_sw_r_gyr.npy` (right-wrist smartwatch; the closest match to RecoFit's right forearm) and `sw_l_*` as a second placement. Arrays are `(n, 5)`: column 0 = video frame index, column 1 = Unix time in ms, columns 2–4 = x, y, z.
- **Units (verified):** accelerometer **m/s²**, gyroscope **rad/s**, about 100 Hz with a few hundred duplicated timestamps per workout (drop them).
- **Subjects (verified):** workouts map to 10 participants (P0 = w01 w03 w06 w08 w10 w14; P1 = w02 w04 w07 w09 w11 w15; P2 = w00 w05; P3–P9 = w12, w13, w16, w17, w18, w19, w20).
- **How UAAR uses it:** test only; models are never trained on MM-Fit. Its exercises that map to a RecoFit known class are "known under shift"; the rest are "unknown under shift." The label mapping to RecoFit names is built by hand once the RecoFit label list is printed. **(verify)**

## 3. RecGym (second benchmark)

- **Source:** https://zhaxidele.github.io/RecGym/ ; paper arXiv:2210.14794. The UCI archive (dataset 1128) is **served corrupt** (first 60 MB are zero bytes); use the authors' Kaggle mirror `zhaxidelebsz/10-gym-exercises-with-615-abstracted-features`, which extracts to `RecGym.csv` (475,013,586 B). (verified)
- **Content (verified):** 10 participants × 5 sessions × 3 positions (`wrist`, `pocket`, `leg`); 4,703,320 rows; columns `Subject, Position, Session, A_x, A_y, A_z, G_x, G_y, G_z, C_1, Workout`; 20 Hz, no timestamps. Workouts: Adductor, ArmCurl, BenchPress, LegCurl, LegPress, Riding, RopeSkipping, Running, Squat, StairClimber, Walking, Null.
- **Units (verified):** every signal column is **min-max normalized to [0, 1]**; gravity cannot be recovered. RecGym is therefore **never mixed** with RecoFit or MM-Fit for training or testing.
- **How UAAR uses it:** a self-contained second open-set benchmark on the `wrist` rows: its own subject folds, its own known/unknown split, the same four methods. It answers "does the method ranking hold on a second dataset?" Drop `C_1` (body capacitance).

## 4. Our wearable (real-device test)

See [software-and-firmware.md §4–5](software-and-firmware.md#4-wearable-hardware-and-firmware-interface) for the hardware, data format and recording protocol. In short: 6-axis IMU at 50 Hz on the left wrist, streamed over USB serial or BLE, converted to g and °/s, and labelled by set with the wearable's button.

## 5. Preprocessing pipeline (all datasets)

| Step | Detail |
|---|---|
| 1. Units | Convert to **g** and **°/s**: MM-Fit acc ÷ 9.80665, gyro × 180/π; wearable acc ÷ 4096 LSB/g (±8 g), gyro ÷ 32.8 LSB/(°/s) (±1000 °/s). RecGym stays normalized. |
| 2. Clean | Drop duplicate/non-monotonic timestamps; drop junk-labelled RecoFit segments. |
| 3. Resample | Common rate **50 Hz**: MM-Fit 100 → 50 Hz with `scipy.signal.resample_poly(x, 1, 2)` after linear interpolation onto a uniform 100 Hz grid. RecGym stays at 20 Hz inside its own benchmark. |
| 4. Window | **2 s windows (100 samples at 50 Hz), 1 s hop.** A window takes the majority label if ≥ 80 % of its samples carry that label; otherwise it is dropped. |
| 5. Channels | 6 channels (ax, ay, az, gx, gy, gz). Optional 7th/8th channels \|a\| and \|g\| for orientation robustness (ablation). |
| 6. Normalize | Per-channel z-score with mean and standard deviation from the **training subjects of each fold only**. |
| 7. Augment (ablation) | Random 3-D rotation (±30°) applied identically to the accel and gyro triplets; jitter; ±10 % magnitude scaling; ±10 % time warp. |

Approximate size: RecoFit exercise data are about 2,000 minutes, so about 120 k windows at a 1 s hop, before eligibility filtering. **(verify)**

## 6. Splits

- **Subjects:** 5 subject folds on RecoFit (about 19 test subjects per fold). Inside each training fold, 15 % of subjects are held out for validation (early stopping, threshold selection, hyperparameters). **Revised 2026-11-05:** the folds and the validation subsets are stratified by exercise with iterative stratification (Sechidis et al. 2011) instead of a plain `GroupKFold`, because RecoFit's participants did different exercise lists and unstratified folds left some classes with one or two validation subjects; see `../data-pipeline.md` and `../decisions.md`. Subjects remain disjoint between train, validation and test.
- **Known / unknown classes:** from the eligible exercises, hold out about 25 % as unknown. Repeat with **3 random class splits** (seeds 0–2; 5 if time allows), plus **1 hand-picked "near-unknown" split** where each unknown exercise resembles a known one (for example, a curl variant held out while another curl is known), chosen after reading the label list. **(verify)**
- **Leakage rules:** unknown-class windows never appear in training or validation, in any fold. Test windows (known and unknown) come only from the test subjects of that fold.
- **RecGym:** 5 folds of 2 subjects each; its own known/unknown class splits by the same rules.
- **Saved artifacts:** every split is written to disk (subject lists, class lists, seed) so every method sees identical data.
