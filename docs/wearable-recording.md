# Wearable recording

How we record labelled sets from our own wearables, which devices exist, and the log of every
session. The hardware and wire format are in [design/software-and-firmware.md](design/software-and-firmware.md) §4; the firmware is `firmware/`.

## Devices

| Kit | Owner | BLE name | IMU | Firmware | Gyro calibrated | Notes |
|---|---|---|---|---|---|---|
| K1 | Robert | `UAAR-8428` | BMI160 at 0x69 | 0.1.0 (2026-11-03) | not yet | bench-verified 2026-11-03: 50.00 Hz, 0 drops, \|a\| = 0.987 g at rest |
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

Over BLE: open the browser page (`uv run scripts/serve_web.py`, Chrome or Edge on Windows),
connect, enter subject and labels there, and download the three session files into
`data/wearable/<subject>/<session>/`. The folder layout is identical.

## Protocol (revised 2026-11-03, about 20 minutes per subject)

The known sets are labelled with the exact RecoFit activity names (decided 2026-11-05,
`decisions.md`), so a recorded set maps to a model class without a lookup table. Type the label
exactly as written; the recorder repeats the previous label on Enter.

| Block | Sets | Label | Notes |
|---|---|---|---|
| Known 1 | 3 × 30–40 s | `Two-arm Dumbbell Curl (both arms, not alternating)` | both arms together; light dumbbells or filled water bottles |
| Known 2 | 3 × 30–40 s | `Shoulder Press (dumbbell)` | same weight |
| Known 3 | 3 × 30–40 s | `Lateral Raise` | same weight |
| Known 4 | 3 × 30–40 s | `Squat (arms in front of body, parallel to ground)` | body weight, arms held straight out in front |
| Known 5 | 3 × 30–40 s | `Jumping Jacks` | |
| Unknown | 2 × 30 s each | `arm_circles`, `shadow_boxing`, `waving` | movements not in RecoFit |
| Placement shift | 1 × 30 s each | `curl_rotated90`, `curl_loose` | the Known 1 exercise with the case rotated 90° on the wrist; with a loose strap |
| Rest | 1 × 60 s each | `standing_still`, `walking` | non-exercise: the models should call these unknown |

Labels in lower case with underscores are ours and are not model classes.

Rest 20–30 s between sets; the recorder keeps streaming, only button-delimited sets are labelled.

## Session log

| Date | Subject | Kit | Session | Transport | Sets | Samples | Dropped | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-11-03 | W01 | K1 | `test-60s` | serial | 1 (`still_on_desk`, 18.5 s) | 3091 | 0 | bench test; the kit was handled and the button pressed mid-test, which ended the set as designed |
