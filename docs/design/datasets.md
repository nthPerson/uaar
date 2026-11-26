# Datasets

One public dataset, RecoFit, plus our own recordings with the wearable kit. Facts marked "verified" were checked against the downloaded files by the FormCoach (BDA 696) data profiling on 2026-09-24 (`reports/data_profile.md` and `docs/04-datasets.md` in that repository); they are facts about the files, not code, and were re-checked by our loader ([../data-pipeline.md](../data-pipeline.md)).

## Summary

| Dataset | Subjects | Classes | Sensor / placement | Rate | Units | Role in UAAR | License |
|---|---|---|---|---|---|---|---|
| **RecoFit** | 94 in released files | 75 activity labels | Arm band, right forearm | 50 Hz | g, °/s | Training; main open-set benchmark | CDLA-Permissive-2.0 |
| **Our wearable** | 2 (W01, W02) | 4 known exercises, 4 background and 2 unknown movements, 2 placement shifts (`configs/wearable.yaml`) | XIAO ESP32-S3 + BMI160, left wrist | 50 Hz | raw LSB → g, °/s | Zero-shot test (P4) and fine-tuning; recording session pending | ours |

## 1. RecoFit (primary)

- **Source:** https://github.com/microsoft/Exercise-Recognition-from-Wearable-Sensors (archived, read-only). Paper: Morris, Saponas, Guillory & Kelner, *RecoFit*, CHI 2014, https://doi.org/10.1145/2556288.2557116.
- **Files:** two MATLAB v5 files, `exercise_data.50.0000_multionly.mat` (1,571,881,721 B) and `exercise_data.50.0000_singleonly.mat` (1,562,300,284 B), plus `readme.html` and `load_exercise_data.m`. They are **Git LFS objects**: the raw GitHub URL returns a 135-byte pointer; download from `https://media.githubusercontent.com/media/microsoft/Exercise-Recognition-from-Wearable-Sensors/main/<file>`. (verified)
- **Loading:** `scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)`; about 14 s and 3 GB RAM per file; no `h5py` needed. (verified)
- **Content (verified):**
  - The README says "over 200 participants"; the released files hold **94 subjects**, **126 lab visits** with activity labels (multi-activity file, 79.4 h) and **4,687 single-exercise recordings** (single-activity file, 78 h, a cell matrix of 94 subjects × 75 activities). The single file is a copy of the labelled segments of the multi file ([../data-pipeline.md](../data-pipeline.md), found 2026-11-04).
  - **75 activity labels** in `exerciseConstants.activities`.
  - Each recording: `data.accelDataMatrix` = `[t_s, x, y, z]` in **g**; `data.gyroDataMatrix` = `[t_s, x, y, z]` in **°/s**; exactly 0.02 s spacing (50 Hz). The master sensor is `rightArm` (forearm band); a second "slave" device in most recordings is ignored.
  - Multi-activity file: `activityStartMatrix` with 7 columns (name, start_s, end_s, notes, reps or −1, start_s again, a sequence struct).
  - Strong imbalance: in the multi-activity file there are 2,642 idle minutes, and single exercises range from hours down to about 20 minutes (for example, lateral raise 23.4 min in the FormCoach count; 22.5 min as windows × hop in [../data-pipeline.md](../data-pipeline.md)). About 78 minutes are "junk" labels (device taps, band adjustment) and are dropped.
- **How UAAR uses it:**
  - **Known/unknown exercises** come from the single-activity file (clean, one exercise per recording).
  - **Non-exercise windows** (`Non-Exercise`, `Rest`, `Walk`) from the multi-activity file form a separate "non-exercise" set.
  - **Class eligibility:** an exercise can be a known or unknown class only if it has data from at least 25 subjects and 10 unique minutes in the single file: 38 classes (decided 2026-11-05; [../data-pipeline.md](../data-pipeline.md), section "Eligibility").

## 2. MM-Fit (dropped)

Planned as a device-shift test (stretch goal 1) and dropped on 2026-11-23 ([../decisions.md](../decisions.md)); no loader was written.

## 3. RecGym (dropped)

Planned as a second open-set benchmark (stretch goal 2) and dropped on 2026-11-23 ([../decisions.md](../decisions.md)); no loader was written.

## 4. Our wearable (real-device test)

Hardware and data format: [software-and-firmware.md §4](software-and-firmware.md#4-wearable-hardware-and-firmware-interface). Recording protocol and session log: [../wearable-recording.md](../wearable-recording.md); label roles: `configs/wearable.yaml`. In short: 6-axis IMU at 50 Hz on the left wrist, streamed over USB serial or Bluetooth, converted to g and °/s, and recorded set by set with the kit's button.

## 5. Preprocessing pipeline

| Step | Detail |
|---|---|
| 1. Units | Convert to **g** and **°/s**: wearable acc ÷ 4096 LSB/g (±8 g), gyro ÷ 32.8 LSB/(°/s) (±1000 °/s). RecoFit is already in g and °/s. |
| 2. Clean | Drop duplicate/non-monotonic timestamps; drop junk-labelled RecoFit segments. |
| 3. Resample | None needed: RecoFit and the wearable are both 50 Hz. |
| 4. Window | **2 s windows (100 samples at 50 Hz), 1 s hop.** A window takes the majority label if ≥ 80 % of its samples carry that label; otherwise it is dropped. |
| 5. Channels | 6 channels (ax, ay, az, gx, gy, gz). |
| 6. Normalize | Per-channel z-score with mean and standard deviation from the **training subjects of each fold only**. |

Size: 120,659 class windows (single file) and 116,305 non-exercise windows (multi file); [../data-pipeline.md](../data-pipeline.md), section "Eligibility".

## 6. Splits

- **Subjects:** 5 subject folds on RecoFit (18 or 19 test subjects per fold). Inside each training fold, 15 % of subjects are held out for validation (early stopping, threshold selection, hyperparameters). **Revised 2026-11-05:** the folds and the validation subsets are stratified by exercise with iterative stratification (Sechidis et al. 2011) instead of a plain `GroupKFold`, because RecoFit's participants did different exercise lists and unstratified folds left some classes with one or two validation subjects; see [../data-pipeline.md](../data-pipeline.md) and [../decisions.md](../decisions.md). Subjects remain disjoint between train, validation and test.
- **Known / unknown classes:**
  - *Main benchmark* (`bg0`–`bg2`, `configs/known4_recofit.yaml`): 4 known classes; 2 look-alikes (`Bicep Curl`, `Squat Rack Shoulder Press`) left out of training and of the unknown set and tested as their own set; the other 32 eligible classes divided at random into 16 background and 16 held-out classes, three times. See [../data-pipeline.md](../data-pipeline.md), section "Background splits".
  - *Hard case* (protocol `p1`, fold 0 only): 28 known / 10 unknown classes, three random class splits (`rand0`–`rand2`) and one hand-picked `near` split of look-alikes (`configs/near_unknown_recofit.yaml`).
- **Leakage rules:** held-out (unknown) classes never appear in training or validation, in any fold. Background classes may be used in training by the taught and Other-class models. Twin labels and look-alikes stay on one side of each division. Test windows come only from the test subjects of that fold.
- **Saved artifacts:** every split is written to disk (subject lists, class lists, seed) so every method sees identical data.
