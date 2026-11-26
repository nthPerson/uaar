# Uncertainty-Aware Activity Recognition (UAAR)

CompE 510 · Machine Learning for Engineers · Fall 2026 · Term project
Team: **Robert Ashe**, **Sirena Rocca**

This folder holds the design documents. They were written for the proposal (brainstorm "Option C", 2026-09-27) and are kept up to date with the project; changes and their dates are in [../decisions.md](../decisions.md).

| File | Contents |
|---|---|
| [README.md](README.md) | Problem, questions, scope, goals, plan, roles, course-requirement map, risks |
| [datasets.md](datasets.md) | RecoFit and our own wearable recordings: sources, formats, units, splits, preprocessing |
| [models.md](models.md) | Shared backbone, the four compared methods, training setup, background-window tiers, fine-tuning |
| [evaluation.md](evaluation.md) | Protocols (four-exercise benchmark, 28-class hard case, wearable), metrics, threshold selection, tables |
| [software-and-firmware.md](software-and-firmware.md) | Repository layout, software stack, live demo and recording pages, wearable hardware/firmware interface and recording protocol |
| [development-spec.md](development-spec.md) | Build plan: repository and tooling, data contracts, module interfaces, scripts and configs, tests, documentation plan, working agreements, week-by-week schedule |
| [references.md](references.md) | Related work and dataset citations |
| [../ai-use-log.md](../ai-use-log.md) | Running record of AI assistance, for the report's disclosure section |

Other documents: [../decisions.md](../decisions.md) (dated decisions), [../methods.md](../methods.md) (equation-to-code map and results), [../data-pipeline.md](../data-pipeline.md) (data counts and splits), [../architecture.md](../architecture.md), [../wearable-recording.md](../wearable-recording.md), [../contributions.md](../contributions.md), [../../web/README.md](../../web/README.md), [../../models/README.md](../../models/README.md).

---

## 1. Problem

A wrist-worn exercise recognizer is trained on a fixed list of exercises. In use it will see movements that were never in that list: a new exercise, a variant of a known one, everyday motion between sets, or the same exercise measured by a different device or a strap worn differently. A standard classifier must still pick one of its known labels, and its softmax confidence is often high when it is wrong. Downstream, that produces wrong rep counts, wrong form feedback and misleading workout logs.

**Engineering goal:** give the recognizer a usable way to say "I don't know," and measure which method does that best, at what accuracy, on data from people it was not trained on (and, once recorded, on our own wrist wearable).

## 2. Research questions and hypotheses

| # | Question | Hypothesis (to test, not assumed) |
|---|---|---|
| Q1 | How well does softmax confidence separate unknown exercises from known ones? | Poorly: the maximum softmax probability stays high on many unknown windows. |
| Q2 | Do evidential deep learning (EDL) and Monte Carlo (MC) Dropout detect unknown exercises better than softmax and logistic regression, without lowering known-class accuracy? | Yes, higher AUROC for unknown detection; known-class macro-F1 within about 2 points of the softmax baseline. |
| Q3 | What does each method cost at inference time? | EDL needs one forward pass; MC Dropout needs T passes (T ≈ 30), so EDL is the better fit for a wearable if detection is comparable. |
| Q4 | Does uncertainty rise when the device changes (forearm band in RecoFit → our wrist wearable)? | Yes, and the windows with the highest uncertainty are the ones most often misclassified. |

Results for Q1–Q2: [../methods.md](../methods.md), section "Background windows in training". Q3 was not measured beyond the MC Dropout timing in [../methods.md](../methods.md) (M4: the convolutional trunk runs once, the head T times). Q4 is the wearable test (P4), pending the recording session.

## 3. Scope and goals

**Minimum goals:**

1. An open-set benchmark on **RecoFit**: test subjects never seen in training, and exercises held out entirely to act as "unknown." Delivered as the four-exercise benchmark: four known exercises (two-arm dumbbell curl, dumbbell shoulder press, lateral raise, jumping jacks); two look-alikes (`Bicep Curl`, `Squat Rack Shoulder Press`) left out of training and of the unknown set; the other 32 eligible exercises divided into 16 background and 16 held-out classes, three times (`bg0`–`bg2`), on five subject folds: 15 runs per result, one seed. The earlier 28-class benchmark was run on fold 0 only and is kept as the "hard case."
2. Four methods compared on the same windows: **logistic regression** and a **softmax neural network** (both taught in class), **EDL** (Robert) and **MC Dropout** (Sirena). Known-class accuracy, unknown detection and calibration are reported in three tiers: plain models, models taught to be uncertain on background windows, and a softmax with an explicit "Other" class as a reference.
3. **An app:** the live web page. `uv run scripts/demo.py` serves the page and the predictions on one port; the user uploads a live Bluetooth stream from the kit or a recorded `raw.csv`, and the page shows one card per model (plain softmax, taught MC Dropout, taught EDL), each with its answer and a 0–100 % uncertainty bar against its limit.

**Stretch goals:**

1. **Device-shift test on MM-Fit** and 2. **replication on RecGym**: dropped on 2026-11-23 ([../decisions.md](../decisions.md)).
3. **Our wearable:** zero-shot test of the four-exercise models and fine-tuning on our own recordings (P4), pending the recording session. The kit is also the live input of the demo page.
4. **Extra methods** (deep ensembles, temperature scaling, SVM): not done.

**Out of scope:** on-device (microcontroller) inference, rep counting, form feedback, and anything camera-based. Those belong to the BDA 696 project.

## 4. Relationship to the BDA 696 project (FormCoach)

Robert is also on a BDA 696 team building **FormCoach**, a wrist wearable (XIAO ESP32-S3 + BMI160) that decides when to wake a webcam pose pipeline and gives rule-based form feedback. The CompE 510 syllabus lists "submitting a piece of work you did for one class to another class" as plagiarism, so the boundary is explicit:

| | FormCoach (BDA 696) | UAAR (CompE 510) |
|---|---|---|
| Question | When should the camera run, and is the rep's form correct? | Can the recognizer tell when a movement is unfamiliar? |
| ML task | Binary active/idle gate, closed-set exercise class, pose rules | Open-set recognition and uncertainty estimation |
| Shared | Public dataset RecoFit; the kit hardware (XIAO ESP32-S3 + BMI160) | Same; UAAR runs its own firmware (decisions 2026-11-03) |
| Not shared | Code, models, results, reports | Separate repository, written for this course |

UAAR does not import FormCoach code. Where FormCoach's documentation recorded facts about the dataset files (download URLs, units, column layouts), those facts are reused in [datasets.md](datasets.md) and cited. State the overlap in both proposals and final reports.

## 5. Plan

The phase plan of the proposal was replaced by the week-by-week schedule in [development-spec.md §10](development-spec.md#10-schedule). What remains as of 2026-11-25:

| Item | Status |
|---|---|
| Recording session with both people (week of Nov 23) | Pending; kit K1 works over USB and Bluetooth, kit K2 is not flashed yet |
| Wearable test P4 (zero-shot and fine-tuned) | Pending the recording session; fine-tuning tested on synthetic sessions only |
| Video demo, 1–3 min | To do |
| Final presentation, Dec 3 2026 | 7 slides, 6–7 minutes (`report/slides.pptx`) |
| Report, due Dec 8 2026 | Outline in `report/outline.docx`; text to do |
| Final submission | Zip with code, report, proposal slides and final slides |

## 6. Roles

The course requires each member to implement one algorithm not covered in class: **EDL** is Robert's and **MC Dropout** is Sirena's. Actual contributions are recorded in [../contributions.md](../contributions.md).

## 7. Course-requirement map

| Requirement (Project_instruction.pdf / syllabus) | Where it is met |
|---|---|
| Real engineering problem + dataset | §1; [datasets.md](datasets.md) |
| Related work: at least two papers, summary + differences | [references.md](references.md) (core papers marked) |
| Formal problem definition: given, goal, constraints | [evaluation.md §1](evaluation.md#1-formal-problem-definition) |
| Each member implements an algorithm not covered in class | EDL (Robert), MC Dropout (Sirena) — neither is on the syllabus topic list |
| Compare with at least one in-class algorithm | Logistic regression and softmax neural network (in-class topics) |
| App: upload data, show prediction, explain output | Live page, `uv run scripts/demo.py`; [software-and-firmware.md §3](software-and-firmware.md#3-demo-page-the-course-app), [../../web/README.md](../../web/README.md) |
| Video demo 1–3 min | To do (§5) |
| Report 5–7 pages incl. individual contributions and AI-use disclosure | Due Dec 8; [../ai-use-log.md](../ai-use-log.md) |
| Proposal 3–5 slides, 3 min (Sep 29); final presentation | Handout: 10–15 slides, 10 min; set by the instructor to 6–7 minutes on Dec 3, deck of 7 slides (decisions 2026-11-24) |

## 8. Risks

Risks as seen at the proposal, with what happened.

| Risk | Likelihood | Mitigation | Outcome |
|---|---|---|---|
| RecoFit class imbalance (a few exercises have minutes of data, idle has hours) | High | Eligibility threshold for known classes; class-weighted loss; report macro-F1 | Applied (25 subjects, 10 unique minutes; inverse square-root class weights) |
| EDL training unstable (evidence collapse, sensitivity to KL annealing) | Medium | Follow Sensoy et al. loss and annealing; fall back to another loss variant | Happened: the squared-error loss did not train; the digamma loss is the default (decisions 2026-11-05) |
| Unknown classes too easy (very different from known ones) | Medium | Add a hand-picked "near-unknown" split (variants of known exercises) | `near` split in the 28-class hard case; in the four-exercise benchmark the look-alikes are a separate test set |
| Wearable not ready in November | Medium | Wearable test is a stretch goal; nothing else depends on it | K1 works over USB and Bluetooth; K2 not flashed; recording pending |
| Training time on CPU | Medium | ~30 k-parameter network; RTX 3070 eGPU available; fewer seeds if needed | One seed per cell |
