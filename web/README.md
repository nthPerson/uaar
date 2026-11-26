# UAAR web pages (`web/`)

Two pages, served together with the predictions by one command:

```bash
uv run scripts/demo.py                                      # http://localhost:8000
uv run scripts/demo.py --replay data/wearable/W01/test-60s  # plays a recording on load
```

| page | for | what it does |
| --- | --- | --- |
| `/` (`index.html`) | the presentation | upload a sensor stream, show each model's answer and uncertainty |
| `/record.html` | data collection | guided recording of labelled sets, saved by the server into `data/wearable/` |

Vanilla JavaScript ES modules, no build step, no CDN or network dependencies (the pages work
with no internet in the room). Open them in **Edge or Chrome**; Web Bluetooth needs
`http://localhost` or https, and `file://` does not work.

| file | role |
| --- | --- |
| `index.html`, `demo.css`, `demo.js` | the demo page |
| `record.html`, `record.css`, `record.js` | the recording page |
| `link.js` | shared: Bluetooth link, simulated wearable, signal trace |
| `protocol.js` | wire protocol shared by both (constants, `decodeBatch`, `encodeBatch`, `decodeStatus`, `toPhysical`, `seqGap`, `parseCsvLine`); mirrors `firmware/include/protocol.h` and `src/uaar/data/wearable.py` |
| `test/protocol.test.mjs` | `node:test` suite for `protocol.js` (run by `tests/test_web_protocol.py` when `node` exists) |

## Demo page (`/`)

One screen, no scrolling, light and large for a projector. It meets the course requirement
"upload data, show prediction, explain output" with two steps:

1. **Upload.** Either **Connect wearable** (the kit streams over Bluetooth; the page shows the
   device name, an arrow into the app, samples per second, samples uploaded and samples lost) or
   **Upload a recording** (a `raw.csv` written by either recorder; it is sent at the pace it was
   recorded). The trace shows the last 10 s of the accelerometer.
2. **Prediction and uncertainty.** One card per model in `configs/demo.yaml` (the plain softmax
   network with 1 − max p, taught MC Dropout with predictive entropy, taught EDL with
   vacuity; fold 0, background split `bg0`): the model's
   answer (an exercise, or **Unknown** in red), its uncertainty on a 0-100 % bar with the
   model's limit marked, and the last 30 answers as a strip. A model answers unknown when its
   uncertainty is over its limit. The bar is the model's own score divided by the largest value
   that score can take (`uaar.live.predictors.score_scale`), so the three bars are comparable
   in direction but not in meaning: each model's limit is set from its own validation data.

Nothing needs clicking once a source is connected. The page connects to the prediction server
by itself and reconnects if the server restarts; a dropped Bluetooth link is retried ten times.
With `--replay` the recording starts as soon as the page opens, which is the fallback on stage.
`?speed=4` in the address plays an uploaded recording four times faster (for testing).

For the presentation, open the page before the talk and connect the wearable; an Edge app
window (`msedge --app=http://localhost:8000`) has no address bar or tabs.

Exercise colours are fixed by class order (`--c0` ... in `demo.css`); with more than eight
classes the chips are grey.

## Recording page (`/record.html`)

For collecting our own labelled sets. One person can run it alone with the kit's button.

1. **Connect wearable**, choose the person (`W01`, `W02`), and **Start session**. From here
   the server writes every sample to `data/wearable/<person>/<session>/raw.csv` as it arrives,
   so a reload or a closed tab loses nothing (the page picks the session up again).
2. The strip in the middle says what is next ("Next: curl (set 1 of 3)"). Press the **kit's
   button** (or Space, or **Start set**): after a short countdown the set is recorded. With
   "stop each set by itself" it ends after its seconds and the page moves on to the next set
   on the checklist; without it, the set runs until the button is pressed again. Pressing the
   button during a set always ends it early; during the countdown it cancels.
3. Every kept set appears on the right with its length and the samples lost. A set that lost
   more than 1 % of its samples, or is shorter than 10 s, is marked: **Discard** it and record
   it again. Click a row of the checklist to record that label next (also for extra sets).
4. **Finish session** writes `events.csv` and `meta.json` and shows what is still missing from
   the checklist and the commands to run next. A session without a kept set leaves no folder.

The two traces show the accelerometer and the gyroscope of the last 10 s; the red band is
what is being saved under the label. Beeps mark the countdown, the start and the end of a set.
**Calibrate gyro** stores the gyro offsets in the kit (lay it still on a table first).

The checklist (labels, sets, seconds, what each label is used for) is the `protocol` list of
`configs/wearable.yaml`; the seconds of a row can be changed on the page. The page writes
`raw.csv`, `events.csv` and `meta.json`, the same core files as `scripts/record_wearable.py`
(which also writes `app.csv`, a plain `t,ax..gz` CSV, and `quicklook.png`), and
`uaar.data.wearable.load_session` reads them. Server side: `uaar.live.recorder`; messages: `uaar.live.server`.

`/record.html?sim` runs the page with a simulated wearable (Space is then its button), for
trying it without a kit; sessions recorded that way have `"transport": "sim"` in `meta.json`.

## Shared code

`link.js` holds what both pages use: the Web Bluetooth link (`Wearable`: connect, reconnect
up to ten times after a dropped link, one-byte commands), the simulated wearable and the
scrolling trace. Web Bluetooth cannot request an MTU; every notification is decoded by its
`n` header field, so whatever size the link negotiates works. On Windows the wearable must not
be connected by another app at the same time.

## Tests

```bash
uv run pytest tests/test_web_protocol.py   # pins protocol.js to uaar.data.wearable; runs node tests if node >= 18
node --test web/test/protocol.test.mjs      # the JS suite alone
```
