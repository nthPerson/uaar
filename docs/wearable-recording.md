# Wearable recording

How we record labelled sets from our own wearables, which devices exist, and the log of every
session. The hardware and wire format are in [design/software-and-firmware.md](design/software-and-firmware.md) §4; the firmware is `firmware/`.

## Devices

| Kit | Owner | BLE name | IMU | Firmware | Gyro calibrated | Notes |
|---|---|---|---|---|---|---|
| K1 | Robert | `UAAR-8428` | BMI160 at 0x69 | 0.1.0 (2026-11-03) | not yet | bench-verified 2026-11-03: 50.00 Hz, 0 drops, \|a\| = 0.987 g at rest; USB and Bluetooth checked 2026-11-23 (`decisions.md`) |
| K2 | Sirena | | | | | to be assembled and flashed |

Flash: `uv tool run platformio run -d firmware -t upload --upload-port /dev/ttyACM0` (WSL: run `firmware/tools/attach-xiao.sh` first). Calibrate once per kit: hold the button 3 s with the kit flat and still; the LED blinks fast for 3 s and the banner then reports `gyro bias applied`.

## Recording a session (USB serial)

1. Wear the kit on the **left wrist, case arrow toward the elbow**, strap snug.
2. Attach the port, then start the recorder with your subject code (Robert `W01`, Sirena `W02`):

   ```bash
   uv run scripts/record_wearable.py --port /dev/ttyACM0 --subject W01
   ```

3. For each set: type the label at the prompt, press the kit's button to start, do the set,
   press the button to stop. The recorder prints the set's duration, sample count and dropped
   samples right away. If a set shows dropped samples or the wrong duration, record it again
   with the same label.
4. `q` ends the session. Check `quicklook.png` before taking the kit off.

## Recording a session (Bluetooth, the recording page)

This is the way to record the protocol; it needs no cable and one person can do it alone.

1. `uv run scripts/demo.py` (`--no-models` serves the pages without loading models), then open
   `http://localhost:8000/record.html` in Edge or Chrome on Windows.
2. **Connect wearable**, set the person (Robert `W01`, Sirena `W02`), **Start session**.
3. For each set the page says what is next. Press the kit's button: after a three-second
   countdown the set is recorded and, by default, stops by itself (beeps mark start and end).
   The page then moves to the next set of the checklist.
4. Watch the "Lost" column: a set marked in red is discarded and recorded again.
5. **Finish session**. The session is already in `data/wearable/<person>/<session>/`; commit it.

Details of the page: `web/README.md`. The USB recorder above stays as the fallback.

## Protocol (revised 2026-11-23, about 20 minutes per subject)

This protocol replaces the planned one in
[design/software-and-firmware.md §5](design/software-and-firmware.md#5-recording-protocol).
The models know four exercises (`decisions.md`, 2026-11-23). Type the short labels below;
`configs/wearable.yaml` maps them to the model classes and says what each one is used for.
The recording page shows this table as its checklist (the `protocol` list of the same file).

| Block | Sets | Label | Used for |
|---|---|---|---|
| Known 1 | 3 × 30–40 s | `curl` | two-arm dumbbell curl, both arms together; light dumbbells or filled water bottles |
| Known 2 | 3 × 30–40 s | `shoulder_press` | dumbbell shoulder press, same weight |
| Known 3 | 3 × 30–40 s | `lateral_raise` | same weight |
| Known 4 | 3 × 30–40 s | `jumping_jacks` | over Bluetooth |
| Rest | 2 × 45 s each | `standing_still`, `walking` | fine-tuning: the taught models learn to be uncertain on these |
| Taught movements | 2 × 30 s each | `arm_circles`, `shadow_boxing` | fine-tuning: same |
| Unknown movements | 2 × 30 s each | `waving`, `torso_twists` | **test only**, never trained on: the models should say unknown |
| Placement shift | 1 × 30 s each | `curl_rotated90`, `curl_loose` | test only: the curl with the case rotated 90° on the wrist; with a loose strap |

Record at least two sets of every label that is used for fine-tuning: the last set of each
label is held out for validation (early stopping and the rejection threshold). Rest 20–30 s
between sets; the recorder keeps streaming, only button-delimited sets are labelled.

## After the session: fine-tune and test across the two of us

```bash
uv run scripts/finetune.py --run models/k4taught_recofit_edl_fold0_bg0_seed0 \
    --train data/wearable/W01 --test data/wearable/W02      # and again with W01 and W02 swapped
uv run scripts/finetune.py --run models/k4taught_recofit_softmax_fold0_bg0_seed0 \
    --train data/wearable/W01 --test data/wearable/W02      # also saves the MC Dropout model
uv run scripts/finetune.py --run models/k4plain_recofit_softmax_fold0_bg0_seed0 \
    --train data/wearable/W01 --test data/wearable/W02      # the plain softmax card
```

Each call prints and saves a before/after table (`report.md` in the new run directory): known
accuracy, the share of known windows kept, and the share of unknown-movement windows accepted,
for the RecoFit-only model and the fine-tuned one. The fine-tuned models are written to
`models/` (tracked in git: commit them after the session) and load in the
live page: put them in `configs/demo.yaml`. Recipe: `uaar.models.finetune`.

## Session log

| Date | Subject | Kit | Session | Transport | Sets | Samples | Dropped | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-11-03 | W01 | K1 | `test-60s` | serial | 1 (`still_on_desk`, 18.5 s) | 3091 | 0 | bench test; the kit was handled and the button pressed mid-test, which ended the set as designed |
