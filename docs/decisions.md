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
